"""Offline UVC pinhole calibration from original ChArUco PNG images.

No camera or robot is opened here. Only board construction is shared with the
PICO calibration code; PICO intrinsics are never used. Validation estimates a
fresh board pose with fixed training K/D, so its residual is not independent
geometric ground truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from egomimic.robot.piper_pico.calibration_vision import (
    _make_board,
    default_board_config,  # noqa: F401 -- shared measured-print default
)

PARAMETER_ORDER = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
DISTORTION_ORDER = PARAMETER_ORDER[4:]


def _image_size(value) -> tuple[int, int]:
    size = np.asarray(value, dtype=float)
    if (
        size.shape != (2,)
        or not np.isfinite(size).all()
        or np.any(size < 1)
        or np.any(size != size.astype(int))
    ):
        raise ValueError("image_size must be [width, height] in integer pixels")
    return tuple(int(v) for v in size)


def _hull_fraction(points: np.ndarray, size: tuple[int, int]) -> float:
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32).reshape(-1, 2))
    return float(cv2.contourArea(hull) / (size[0] * size[1]))


def _corner_arrays(ids, points, object_points, size, min_corners=8):
    ids_array = np.asarray(ids)
    points = np.asarray(points, dtype=np.float64)
    if (
        ids_array.ndim != 1
        or not np.issubdtype(ids_array.dtype, np.integer)
        or points.shape != (len(ids_array), 2)
        or not np.isfinite(points).all()
        or len(set(ids_array.tolist())) != len(ids_array)
        or np.any(ids_array < 0)
        or np.any(ids_array >= len(object_points))
        or np.any(points < 0)
        or np.any(points >= np.asarray(size))
    ):
        raise ValueError("invalid_corners")
    if len(ids_array) < min_corners:
        raise ValueError("insufficient_corners")
    obj = object_points[ids_array]
    for coordinates in (points, obj[:, :2]):
        spread = np.linalg.svd(coordinates - coordinates.mean(axis=0), compute_uv=False)
        if spread[0] <= 0 or spread[1] < 0.01 * spread[0]:
            raise ValueError("collinear_corners")
    return obj.astype(np.float32), points.astype(np.float32)


class CharucoCorners:
    """Detect native BGR corners without supplying any candidate camera model."""

    def __init__(self, board_config: dict, min_corners: int = 8):
        self.board = _make_board(board_config)
        self.object_points = np.asarray(
            self.board.getChessboardCorners(), dtype=np.float32
        ).reshape(-1, 3)
        if (
            isinstance(min_corners, bool)
            or int(min_corners) != min_corners
            or not 8 <= min_corners <= len(self.object_points)
        ):
            raise ValueError("min_corners must be between 8 and the board corner count")
        self.min_corners = int(min_corners)
        self.detector = cv2.aruco.CharucoDetector(
            self.board, cv2.aruco.CharucoParameters()
        )

    def detect(self, bgr: np.ndarray) -> dict:
        result = {
            "accepted": False,
            "reason": "no_board",
            "ids": [],
            "corners_px": [],
            "corner_count": 0,
            "coverage_fraction": 0.0,
        }
        if (
            not isinstance(bgr, np.ndarray)
            or bgr.dtype != np.uint8
            or bgr.ndim != 3
            or bgr.shape[2] != 3
            or min(bgr.shape[:2]) < 1
        ):
            return {**result, "reason": "image_format_mismatch"}
        try:
            corners, ids, _, _ = self.detector.detectBoard(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            )
            if ids is None or corners is None or not np.asarray(ids).size:
                return result
            ids = np.asarray(ids, dtype=int).reshape(-1)
            points = np.asarray(corners, dtype=float).reshape(-1, 2)
            result.update(
                ids=ids.tolist(), corners_px=points.tolist(), corner_count=len(ids)
            )
            size = (bgr.shape[1], bgr.shape[0])
            _corner_arrays(ids, points, self.object_points, size, self.min_corners)
            result.update(
                accepted=True,
                reason="accepted",
                coverage_fraction=_hull_fraction(points, size),
            )
        except ValueError as exc:
            result["reason"] = str(exc)
        except cv2.error as exc:
            result.update(reason="opencv_detection_error", detail=str(exc)[-300:])
        return result


def _coverage(observations, size):
    points = np.concatenate([np.asarray(o["corners_px"]) for o in observations])
    normalized = points / np.asarray(size)
    cells = np.minimum((normalized * [8, 6]).astype(int), [7, 5])
    return {
        "corner_hull_fraction": _hull_fraction(points, size),
        "grid_8x6_fraction": len(set(map(tuple, cells))) / 48,
        "bounds_fraction": [
            normalized.min(axis=0).tolist(),
            normalized.max(axis=0).tolist(),
        ],
        "mean_view_hull_fraction": float(
            np.mean([_hull_fraction(o["corners_px"], size) for o in observations])
        ),
    }


def _evaluate(observations, arrays, poses, K, D, size):
    views, square_error, count = [], 0.0, 0
    for observation, (obj, points), (rvec, tvec) in zip(observations, arrays, poses):
        R = cv2.Rodrigues(rvec)[0]
        if np.min((obj @ R.T + np.asarray(tvec).reshape(3))[:, 2]) <= 0:
            raise ValueError("Estimated board pose has nonpositive depth")
        projected = cv2.projectPoints(obj, rvec, tvec, K, D)[0].reshape(-1, 2)
        errors = np.linalg.norm(projected.astype(float) - points, axis=1)
        if not np.isfinite(errors).all():
            raise ValueError("Nonfinite reprojection error")
        square_error += float(np.sum(errors**2))
        count += len(points)
        views.append(
            {
                "file": observation.get("file"),
                "corner_count": len(points),
                "reprojection_rms_px": float(np.sqrt(np.mean(errors**2))),
                "max_corner_error_px": float(errors.max()),
                "coverage_fraction": _hull_fraction(points, size),
                "rvec_board_to_camera": np.asarray(rvec).reshape(3).tolist(),
                "tvec_board_to_camera_m": np.asarray(tvec).reshape(3).tolist(),
            }
        )
    return {
        "count": len(views),
        "corner_count": count,
        "rms_px": float(np.sqrt(square_error / count)),
        "max_view_rms_px": max(v["reprojection_rms_px"] for v in views),
        "coverage": _coverage(observations, size),
        "views": views,
    }


def fit_observations(
    train: list[dict],
    validation: list[dict],
    board_config: dict,
    image_size,
    *,
    min_train: int = 15,
    min_validation: int = 5,
    max_rms_px: float = 1.0,
) -> dict:
    """Fit Brown5 only on training corners; evaluate held-out pose residuals.

    Each observation contains ``ids`` and ``corners_px`` (N x 2), with optional
    ``file`` for reporting. This lower-level geometry API does not authenticate
    image provenance; use :func:`solve_session` for actual sessions.
    """
    size = _image_size(image_size)
    if (
        isinstance(min_train, bool)
        or isinstance(min_validation, bool)
        or int(min_train) != min_train
        or int(min_validation) != min_validation
        or min_train < 3
        or min_validation < 1
        or not np.isfinite(max_rms_px)
        or max_rms_px <= 0
    ):
        raise ValueError(
            "Require min_train >= 3, min_validation >= 1 and positive max_rms_px"
        )
    if len(train) < min_train or len(validation) < min_validation:
        raise ValueError(
            f"Insufficient accepted samples: train={len(train)} (need {min_train}), "
            f"validation={len(validation)} (need {min_validation})"
        )
    object_points = np.asarray(_make_board(board_config).getChessboardCorners())
    train_arrays = [
        _corner_arrays(o["ids"], o["corners_px"], object_points, size) for o in train
    ]
    val_arrays = [
        _corner_arrays(o["ids"], o["corners_px"], object_points, size)
        for o in validation
    ]
    # No intrinsic guess, fixed focal length, rational, thin-prism or fisheye flags.
    try:
        _, K, D, rvecs, tvecs, intrinsic_std, extrinsic_std, _ = (
            cv2.calibrateCameraExtended(
                [a[0] for a in train_arrays],
                [a[1] for a in train_arrays],
                size,
                None,
                None,
                flags=0,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-10),
            )
        )
    except cv2.error as exc:
        raise ValueError(f"OpenCV intrinsic calibration failed: {exc}") from exc
    D = np.asarray(D).reshape(-1)
    if (
        D.shape != (5,)
        or not np.isfinite(K).all()
        or not np.isfinite(D).all()
        or K[0, 0] <= 0
        or K[1, 1] <= 0
    ):
        raise ValueError(
            "Calibration did not produce finite positive-focal Brown5 parameters"
        )
    train_metrics = _evaluate(train, train_arrays, zip(rvecs, tvecs), K, D, size)
    validation_poses = []
    for obj, points in val_arrays:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj, points, K, D, flags=cv2.SOLVEPNP_ITERATIVE
            )
        except cv2.error as exc:
            raise ValueError(f"Validation pose estimation failed: {exc}") from exc
        if not ok:
            raise ValueError("Validation pose estimation failed")
        validation_poses.append((rvec, tvec))
    val_metrics = _evaluate(validation, val_arrays, validation_poses, K, D, size)
    val_metrics["interpretation"] = (
        "Held-out images with training K/D fixed and one fitted board pose per image; "
        "not independent geometric ground truth. No validation-based refit or filtering."
    )
    normals = np.array([cv2.Rodrigues(r)[0][:, 2] for r in rvecs])
    centers = np.array([np.mean(o["corners_px"], axis=0) for o in train]) / size
    depths = np.array([np.asarray(t).reshape(3)[2] for t in tvecs])
    diversity = {
        "max_board_normal_separation_deg": float(
            np.degrees(np.arccos(np.clip(normals @ normals.T, -1, 1))).max()
        ),
        "board_center_span_fraction": np.ptp(centers, axis=0).tolist(),
        "max_to_min_depth_ratio": float(depths.max() / depths.min()),
    }
    warnings = []
    if len(train) < 15 or len(validation) < 5:
        warnings.append(
            "Fewer than the recommended 15 training and 5 validation images."
        )
    for label, metrics in (("train", train_metrics), ("validation", val_metrics)):
        metrics["rms_within_threshold"] = metrics["rms_px"] <= max_rms_px
        if metrics["max_view_rms_px"] > max_rms_px:
            warnings.append(
                f"{label}: at least one view exceeds {max_rms_px:g} px RMS."
            )
    if diversity["max_board_normal_separation_deg"] < 15:
        warnings.append(
            "Training board tilts lack diversity (<15 degrees between normals)."
        )
    if diversity["max_to_min_depth_ratio"] < 1.2:
        warnings.append("Training board distances lack diversity (depth ratio <1.2).")
    if min(diversity["board_center_span_fraction"]) < 0.25:
        warnings.append(
            "Training board centers cover too little of at least one image axis."
        )
    if train_metrics["coverage"]["corner_hull_fraction"] < 0.35:
        warnings.append(
            "Training corners cover <35% of the image; edge distortion is weakly constrained."
        )
    std_values = np.asarray(intrinsic_std).reshape(-1)[:9]
    if not np.isfinite(std_values).all():
        warnings.append(
            "Some parameter uncertainty estimates are nonfinite; inspect degeneracy."
        )
    elif np.any(std_values[:2] > 0.1 * np.array([K[0, 0], K[1, 1]])):
        warnings.append(
            "Estimated focal-length uncertainty exceeds 10%; collect more diverse views."
        )
    if (
        not 0 <= K[0, 2] < size[0]
        or not 0 <= K[1, 2] < size[1]
        or not 0.1 * max(size) < min(K[0, 0], K[1, 1])
        or max(K[0, 0], K[1, 1]) > 10 * max(size)
    ):
        warnings.append(
            "Intrinsic parameters are unusual for this image size; inspect calibration geometry."
        )
    for view, std in zip(
        train_metrics["views"], np.asarray(extrinsic_std).reshape(-1, 6)
    ):
        view["pose_parameter_std"] = [float(v) if np.isfinite(v) else None for v in std]
    return {
        "camera_model": "opencv_pinhole_brown5",
        "image_size": list(size),
        "K": K.tolist(),
        "D": D.tolist(),
        "distortion_parameter_order": DISTORTION_ORDER,
        "parameter_order": PARAMETER_ORDER,
        "parameter_std": {
            name: float(value) if np.isfinite(value) else None
            for name, value in zip(PARAMETER_ORDER, std_values)
        },
        "parameter_std_note": "OpenCV linearized training-fit estimates; not certified measurement uncertainty.",
        "train": train_metrics,
        "validation": val_metrics,
        "training_pose_diversity": diversity,
        "max_rms_px": float(max_rms_px),
        "quality_warnings": warnings,
        "production_approved": False,
        "approval_note": "Low reprojection RMS alone does not establish physical calibration accuracy.",
    }


def _write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write {path}")


def _session_warnings(manifest):
    warnings = []
    if manifest.get("closed_cleanly") is not True:
        warnings.append(
            "Capture was not recorded as closed cleanly; inspect session completeness."
        )
    camera = manifest["camera"]
    note = camera.get("focus_note", "")
    if not note or note.lower().startswith("not specified"):
        warnings.append(
            "Lens focus was not documented; manually confirm fixed focus throughout capture."
        )
    snapshots = [camera.get("controls")] + [
        s.get("controls") for s in manifest["samples"]
    ]
    signatures, unknown = [], False
    for controls in snapshots:
        if not isinstance(controls, dict) or not controls.get("available", False):
            unknown = True
            continue
        values = controls.get("values", {})
        signature = {k: v for k, v in values.items() if "focus" in k or "zoom" in k}
        if not signature:
            unknown = True
        for key, value in signature.items():
            if "focus" in key and "auto" in key:
                if isinstance(value, (int, float)) and value > 0:
                    raise ValueError(
                        "Autofocus was enabled in a recorded control snapshot; lock focus and recapture"
                    )
                if value != 0:
                    unknown = True
        if signature:
            signatures.append(signature)
    if signatures and any(s != signatures[0] for s in signatures[1:]):
        raise ValueError(
            "Focus/zoom controls changed during the session; use a fixed lens setting and recapture"
        )
    if unknown:
        warnings.append(
            "Focus/zoom controls are unavailable or incomplete for some images; manually confirm a fixed lens and unchanged zoom."
        )
    return warnings


def _write_previews(output, images, observations, K, D, K_new, size):
    comparison = output / "comparison"
    comparison.mkdir()
    for index in np.linspace(
        0, len(observations) - 1, min(10, len(observations)), dtype=int
    ):
        observation = observations[index]
        original = images[observation["file"]]
        rectified = cv2.undistort(original, K, D, None, K_new)
        pair = np.concatenate([original, rectified], axis=1)
        pair = cv2.copyMakeBorder(pair, 32, 0, 0, 0, cv2.BORDER_CONSTANT)
        for x, label in (
            (5, "Original: raw K / Brown5 D"),
            (size[0] + 5, "Undistorted: K_new / D=0, no crop"),
        ):
            cv2.putText(
                pair, label, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        _write_image(
            comparison / f"{index:06d}_{Path(observation['file']).stem}.jpg", pair
        )
    coverage = np.full((size[1], size[0], 3), 25, np.uint8)
    for observation in observations:
        color = (70, 210, 70) if observation["split"] == "train" else (0, 160, 255)
        for point in observation["corners_px"]:
            cv2.circle(coverage, tuple(np.rint(point).astype(int)), 2, color, -1)
    coverage = cv2.copyMakeBorder(coverage, 32, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(
        coverage,
        "Training: green | Validation: orange",
        (5, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    _write_image(output / "coverage.png", coverage)


def solve_session(
    session: Path,
    output: Path,
    *,
    min_train: int = 15,
    min_validation: int = 5,
    max_rms_px: float = 1.0,
    alpha: float = 0.0,
) -> dict:
    """Redetect original PNGs, fit training views, and write a new output directory."""
    session, output = Path(session).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    manifest_path = session / "session.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "uvc_intrinsics_capture_v1":
        raise ValueError("Unsupported UVC capture session schema")
    camera = manifest["camera"]
    if not isinstance(camera.get("name"), str) or not camera["name"].strip():
        raise ValueError("camera.name must be a nonempty string")
    size = _image_size(camera["image_size"])
    preprocessing = manifest.get("preprocessing", {})
    if (
        preprocessing.get("crop") is not None
        or preprocessing.get("resize") is not None
        or preprocessing.get("mirror") is not False
    ):
        raise ValueError(
            "Session must contain original images: crop=None, resize=None, mirror=False"
        )
    detector = CharucoCorners(manifest["board"], manifest.get("min_corners", 8))
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Session samples must be a list")
    session_warnings = _session_warnings(manifest)
    observations = {"train": [], "validation": []}
    rejected, input_files, images, seen = [], [], {}, {}
    for sample in samples:
        split, filename = sample["split"], sample["file"]
        if split not in observations or not isinstance(filename, str):
            raise ValueError("Each sample needs file and split=train|validation")
        relative = Path(filename)
        path = (session / relative).resolve()
        if (
            relative.is_absolute()
            or not path.is_relative_to(session)
            or relative.suffix.lower() != ".png"
        ):
            raise ValueError(f"Sample must name a PNG inside the session: {filename}")
        record = {"file": filename, "split": split, "sha256": None}
        input_files.append(record)
        try:
            raw = path.read_bytes()
            record["sha256"] = hashlib.sha256(raw).hexdigest()
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        except (OSError, cv2.error):
            bgr = None
        if bgr is None:
            rejected.append({**record, "reason": "unreadable_image"})
            continue
        if (bgr.shape[1], bgr.shape[0]) != size:
            raise ValueError(
                f"Image size mismatch for {filename}: got {bgr.shape[1::-1]}, expected {size}; no resizing is allowed"
            )
        digest = hashlib.sha256(
            str((bgr.shape, bgr.dtype.str)).encode() + bgr.tobytes()
        ).hexdigest()
        record["decoded_image_sha256"] = digest
        if digest in seen:
            previous_split, previous_file = seen[digest]
            if previous_split != split:
                raise ValueError(
                    f"Duplicate image across train/validation: {previous_file} and {filename}"
                )
            rejected.append(
                {
                    **record,
                    "reason": "duplicate_image_within_split",
                    "duplicate_of": previous_file,
                }
            )
            continue
        seen[digest] = (split, filename)
        detection = detector.detect(bgr)
        record["accepted"] = detection["accepted"]
        record["reason"] = detection["reason"]
        if not detection["accepted"]:
            rejected.append({**record, "reason": detection["reason"]})
            continue
        observations[split].append({**detection, "file": filename, "split": split})
        images[filename] = bgr
    result = fit_observations(
        observations["train"],
        observations["validation"],
        manifest["board"],
        size,
        min_train=min_train,
        min_validation=min_validation,
        max_rms_px=max_rms_px,
    )
    result.update(
        schema="uvc_intrinsics_calibration_v1",
        opencv_version=cv2.__version__,
        camera_name=camera["name"],
        type="uvc",
        camera=camera,
        board=manifest["board"],
        preprocessing=preprocessing,
        input_session=str(session),
        session_json_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        input_files=input_files,
        rejected_samples=rejected,
        intrinsics_usage="K/D apply to the original, unmirrored, uncropped native images only.",
        rectified_camera_file="rectified_camera.json",
    )
    result["quality_warnings"].extend(session_warnings)
    K, D = np.asarray(result["K"]), np.asarray(result["D"])
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, size, alpha, size)
    rectified = {
        "schema": "uvc_rectified_camera_v1",
        "camera_name": camera["name"],
        "type": "uvc",
        "camera_model": "opencv_pinhole_brown5",
        "image_size": list(size),
        "K": K_new.tolist(),
        "K_new": K_new.tolist(),
        "D": [0.0] * 5,
        "distortion_parameter_order": DISTORTION_ORDER,
        "valid_roi": [int(v) for v in roi],
        "alpha": float(alpha),
        "roi_cropped": False,
        "source_calibration": "calibration.json",
        "intrinsics_usage": "K_new applies only after undistortion with raw calibration K/D; output is full-size, with no ROI crop.",
        "production_approved": False,
    }
    # Validate serialization before creating anything; never overwrite an old result.
    calibration_text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    rectified_text = json.dumps(rectified, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    _write_previews(
        output,
        images,
        observations["train"] + observations["validation"],
        K,
        D,
        K_new,
        size,
    )
    (output / "calibration.json").write_text(calibration_text)
    (output / "rectified_camera.json").write_text(rectified_text)
    return result
