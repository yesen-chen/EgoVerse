"""Offline, native-resolution verification of a calibrated SDK end frame.

All transforms map the named source frame into the destination frame and use
metres. Board predictions depend only on robot feedback and the fitted rigid
transforms; no per-image PnP pose is used by this module.
"""

from __future__ import annotations

import cv2
import numpy as np

# These are RGB values: both the input and returned images are RGB arrays.
_AXIS_COLORS = ((255, 65, 65), (60, 235, 85), (65, 115, 255))
_PREDICTED = (255, 70, 235)
_DETECTED = (30, 235, 255)
_RESIDUAL = (255, 220, 45)
_TEXT = (235, 237, 240)
_MIN_DEPTH_M = 1e-9


def _transform(value, name):
    value = np.asarray(value, dtype=np.float64)
    if (
        value.shape != (4, 4)
        or not np.isfinite(value).all()
        or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7)
    ):
        raise ValueError(f"{name} must be a finite 4x4 homogeneous transform")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError(f"{name} rotation must be orthonormal with determinant +1")
    return value


def _project(camera_points, K, D):
    """Return NaN for behind-camera/unprojectable points, never mirrored pixels."""
    pixels = np.full((len(camera_points), 2), np.nan)
    valid = np.isfinite(camera_points).all(axis=1) & (
        camera_points[:, 2] > _MIN_DEPTH_M
    )
    if np.any(valid):
        with np.errstate(over="ignore", invalid="ignore"):
            projected = cv2.projectPoints(
                camera_points[valid], np.zeros(3), np.zeros(3), K, D
            )[0].reshape(-1, 2)
        pixels[valid] = projected
    pixels[~np.isfinite(pixels).all(axis=1)] = np.nan
    return pixels


def _inside(pixel, width, height):
    return bool(
        np.isfinite(pixel).all() and 0 <= pixel[0] < width and 0 <= pixel[1] < height
    )


def _clipped_segment(first, second, width, height):
    """Clip in floating point before converting to OpenCV's bounded integers."""
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return None
    # Extended precision keeps subtraction safe even for enormous finite pixels.
    first = np.asarray(first, dtype=np.longdouble)
    delta = np.asarray(second, dtype=np.longdouble) - first
    start, stop = np.longdouble(0), np.longdouble(1)
    for axis, maximum in ((0, width - 1), (1, height - 1)):
        if delta[axis] == 0:
            if not 0 <= first[axis] <= maximum:
                return None
        else:
            a = -first[axis] / delta[axis]
            b = (maximum - first[axis]) / delta[axis]
            start, stop = max(start, min(a, b)), min(stop, max(a, b))
            if start > stop:
                return None
    # Rounding can put a clipped point one pixel outside; only clipped endpoints
    # are bounded here. The SDK origin itself is never clamped into the image.
    lower = np.array([0, 0])
    upper = np.array([width - 1, height - 1])
    return tuple(
        tuple(np.rint(np.clip(first + t * delta, lower, upper)).astype(int))
        for t in (start, stop)
    )


def _line(image, first, second, color, thickness=1):
    segment = _clipped_segment(first, second, image.shape[1], image.shape[0])
    if segment is None:
        return False
    cv2.line(image, *segment, color, thickness, cv2.LINE_AA)
    return True


def _visible_point(pixel, width, height):
    if not _inside(pixel, width, height):
        return None
    return tuple(np.clip(np.rint(pixel), [0, 0], [width - 1, height - 1]).astype(int))


def _text(image, value, position, color=_TEXT, scale=0.5):
    cv2.putText(
        image, value, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
    )


def _canvas(image, lines):
    """Put all explanatory text outside the untouched native image extent."""
    # Fit long lines to the available width without changing the source pixels.
    height, width = image.shape[:2]
    result = np.full((height + 132, width, 3), 23, dtype=np.uint8)
    result[:height] = image
    for index, (value, color) in enumerate(lines):
        scale = 0.51
        extent = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if extent > width - 20:
            scale *= max(width - 20, 1) / extent
        _text(result, value, (10, height + 21 + index * 22), color, scale)
    return result


def _nullable_rows(values):
    return [row.tolist() if np.isfinite(row).all() else None for row in values]


def render_verification(
    rgb,
    *,
    T_camera_end,
    T_end_board,
    object_points,
    corner_ids,
    corners_px,
    K,
    D,
    sample_id,
    arm,
    split,
    axis_length_m=0.04,
):
    """Return SDK-end axes, independent board comparison and JSON-safe metrics.

    ``object_points`` contains all board corners; ``corner_ids`` selects the
    corresponding observations in ``corners_px``. Input pixels are never resized
    or mutated. All corner residuals use ``T_camera_end @ T_end_board`` and the
    supplied K/D. An incomplete forward projection yields a null RMS rather than
    a misleading score computed from a convenient subset.
    """
    rgb = np.asarray(rgb)
    if (
        rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.dtype != np.uint8
        or min(rgb.shape[:2]) < 1
    ):
        raise ValueError("rgb must be a nonempty uint8 HxWx3 image")
    T_camera_end = _transform(T_camera_end, "T_camera_end")
    T_end_board = _transform(T_end_board, "T_end_board")
    if not np.isfinite(axis_length_m) or axis_length_m <= 0:
        raise ValueError("axis_length_m must be positive and finite")
    if arm not in ("left", "right") or split not in ("train", "validation"):
        raise ValueError("arm must be left/right and split must be train/validation")
    K, D = np.asarray(K, dtype=float), np.asarray(D, dtype=float).reshape(-1)
    if (
        K.shape != (3, 3)
        or not np.isfinite(K).all()
        or K[0, 0] <= 0
        or K[1, 1] <= 0
        or not np.allclose(K[2], [0, 0, 1])
        or not np.allclose([K[0, 1], K[1, 0]], 0)
        or D.size not in (4, 5, 8, 12, 14)
        or not np.isfinite(D).all()
    ):
        raise ValueError("K/D must contain valid finite OpenCV camera parameters")
    points = np.asarray(object_points, dtype=float)
    ids_value = np.asarray(corner_ids)
    observed = np.asarray(corners_px, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("object_points must be a finite Nx3 array of board corners")
    if (
        ids_value.ndim != 1
        or not np.issubdtype(ids_value.dtype, np.number)
        or not np.isfinite(ids_value).all()
        or np.any(ids_value < 0)
        or np.any(ids_value >= len(points))
        or np.any(ids_value != np.floor(ids_value))
        or len(np.unique(ids_value)) != len(ids_value)
    ):
        raise ValueError("corner_ids must be unique integer indices into object_points")
    ids = ids_value.astype(int)
    if observed.size == 0:
        observed = observed.reshape(0, 2)
    if observed.shape != (len(ids), 2) or not np.isfinite(observed).all():
        raise ValueError("corners_px must have one finite xy observation per corner_id")

    height, width = rgb.shape[:2]
    local_axes = np.vstack([np.zeros(3), np.eye(3) * axis_length_m])
    axes_camera = local_axes @ T_camera_end[:3, :3].T + T_camera_end[:3, 3]
    with np.errstate(over="ignore", invalid="ignore"):
        camera_board = T_camera_end @ T_end_board
        board_camera = points[ids] @ camera_board[:3, :3].T + camera_board[:3, 3]
    if (
        not np.isfinite(axes_camera).all()
        or not np.isfinite(camera_board).all()
        or not np.isfinite(board_camera).all()
    ):
        raise ValueError("Transformed SDK axes/board coordinates must be finite")
    axes_pixels = _project(axes_camera, K, D)
    predicted = _project(board_camera, K, D)
    axes_visible = [_inside(pixel, width, height) for pixel in axes_pixels]
    corners_visible = [_inside(pixel, width, height) for pixel in predicted]
    observed_visible = [_inside(pixel, width, height) for pixel in observed]
    with np.errstate(over="ignore", invalid="ignore"):
        # hypot avoids overflow from squaring large but finite pixel differences.
        delta = predicted - observed
        errors = np.hypot(delta[:, 0], delta[:, 1])
    all_valid = bool(len(ids) and np.isfinite(errors).all())
    if all_valid:
        maximum = float(errors.max())
        rms = (
            maximum * float(np.sqrt(np.mean((errors / maximum) ** 2)))
            if maximum
            else 0.0
        )
    else:
        rms = None

    if not np.isfinite(axes_pixels[0]).all():
        origin_status = (
            "behind_camera" if axes_camera[0, 2] <= _MIN_DEPTH_M else "unprojectable"
        )
    else:
        origin_status = "in_image" if axes_visible[0] else "outside_image"
    notes = [
        "SDK end frame; no TCP offset. Axes are X red, Y green, Z blue.",
        "Board prediction uses robot pose and fitted transforms, not image PnP.",
        "This image is a visual check, not a physical accuracy certification.",
    ]
    if origin_status != "in_image":
        notes.append(
            f"SDK_END origin is {origin_status}; no substitute origin is drawn."
        )
    if not all_valid:
        notes.append(
            "No full board RMS: at least one projection is invalid, or no corners are available."
        )

    axes_image = rgb.copy()
    drawn_axes = []
    for i, color in enumerate(_AXIS_COLORS, start=1):
        drawn_axes.append(_line(axes_image, axes_pixels[0], axes_pixels[i], color, 2))
    origin_point = _visible_point(axes_pixels[0], width, height)
    if origin_point is not None:
        cv2.circle(axes_image, origin_point, 3, (255, 255, 255), -1, cv2.LINE_AA)
    # Labels stay in the caption so they cannot obscure measured scene features.
    comparison = axes_image.copy()
    for predicted_pixel, observed_pixel in zip(predicted, observed):
        _line(comparison, predicted_pixel, observed_pixel, _RESIDUAL)
        predicted_point = _visible_point(predicted_pixel, width, height)
        observed_point = _visible_point(observed_pixel, width, height)
        if predicted_point is not None:
            cv2.drawMarker(
                comparison,
                predicted_point,
                _PREDICTED,
                cv2.MARKER_CROSS,
                9,
                1,
                cv2.LINE_AA,
            )
        if observed_point is not None:
            cv2.circle(comparison, observed_point, 4, _DETECTED, 1, cv2.LINE_AA)

    arm_label = "L" if arm == "left" else "R"
    split_label = (
        "HELD-OUT VALIDATION" if split == "validation" else "TRAIN / fitted sample"
    )
    caption = [
        (f"{arm_label} SDK_END | sample {sample_id} | {split_label}", _TEXT),
        (
            f"Axis length {axis_length_m * 100:.1f} cm | X red | Y green | Z blue | origin {origin_status}",
            _TEXT,
        ),
        (
            "SDK feedback origin; no TCP offset. Native image pixels preserved above.",
            _TEXT,
        ),
    ]
    axes_caption = caption + [
        (
            "Axes predicted from actual robot pose and calibrated camera-to-base transform.",
            _TEXT,
        ),
        (
            "Judge at stationary poses; this view alone does not certify physical accuracy.",
            _TEXT,
        ),
    ]
    rms_label = (
        f"{rms:.3f} px" if rms is not None else "unavailable (invalid/no corners)"
    )
    comparison_caption = caption + [
        ("Board: + magenta predicted | o cyan detected | yellow residual", _TEXT),
        (
            f"Robot-predicted board RMS {rms_label} | {len(ids)} observed corners | no per-frame PnP",
            _TEXT,
        ),
    ]
    metrics = {
        "sample_id": str(sample_id),
        "arm": arm,
        "split": split,
        "frame": "SDK_END",
        "tcp_offset_applied": False,
        "image_size": [width, height],
        "caption_height_px": 132,
        "axis_length_m": float(axis_length_m),
        "T_camera_end": T_camera_end.tolist(),
        "T_camera_board_predicted": camera_board.tolist(),
        "axes_order": ["origin", "x", "y", "z"],
        "axes_camera_xyz": axes_camera.tolist(),
        "axes_pixels_xy": _nullable_rows(axes_pixels),
        "axes_points_in_image": axes_visible,
        "axes_segments_drawn": drawn_axes,
        "origin_status": origin_status,
        "corner_count": len(ids),
        "corner_ids": ids.tolist(),
        "predicted_corners_camera_xyz": board_camera.tolist(),
        "predicted_corners_px": _nullable_rows(predicted),
        "detected_corners_px": observed.tolist(),
        "predicted_corners_in_image": corners_visible,
        "detected_corners_in_image": observed_visible,
        "corner_errors_px": [
            float(error) if np.isfinite(error) else None for error in errors
        ],
        "reprojection_rms_px": rms,
        "all_corner_projections_valid": all_valid,
        "notes": notes,
    }
    return (
        _canvas(axes_image, axes_caption),
        _canvas(comparison, comparison_caption),
        metrics,
    )
