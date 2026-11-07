"""One reproducible native-RGB -> policy-image transform for conversion/deployment.

Raw capture is never modified. Rectification keeps the optical frame (R=I),
uses a saved K_new, does not crop its ROI, then resizes with INTER_AREA.
Processed dataset images must be identified explicitly to avoid rectifying twice.
"""

from __future__ import annotations

import copy
import hashlib
import json

import cv2
import numpy as np


def _size(value, label):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{label} must be [width,height]")
    if any(
        isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 1
        for x in value
    ):
        raise ValueError(f"{label} must contain positive integers")
    return tuple(map(int, value))


def _K(value):
    a = np.asarray(value, dtype=float)
    if a.shape == (3, 4) and np.all(a[:, 3] == 0):
        a = a[:, :3]
    if (
        a.shape != (3, 3)
        or not np.isfinite(a).all()
        or a[0, 0] <= 0
        or a[1, 1] <= 0
        or a[0, 1] != 0
        or a[1, 0] != 0
        or not np.array_equal(a[2], [0, 0, 1])
    ):
        raise ValueError("K must be a finite zero-skew pinhole matrix")
    return a.copy()


def _D(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (5,) or not np.isfinite(a).all():
        raise ValueError("D must contain five finite Brown5 coefficients")
    return a.copy()


def _raw_pixels(snapshot, name):
    preprocessing = snapshot.get("preprocessing", {})
    if (
        any(
            preprocessing.get(k) != v
            for k, v in {"crop": None, "resize": None, "mirror": False}.items()
        )
        or any(
            preprocessing.get(k, False) is not False
            for k in ("undistort", "undistorted", "rectified", "roi_cropped")
        )
        or any(preprocessing.get(k) not in (None, 0) for k in ("rotate", "rotation"))
    ):
        raise ValueError(f"{name}: calibration must correspond to raw native pixels")


def _same_camera(a, b, name):
    a, b = a.get("camera", {}), b.get("camera", {})
    keys = ("ID_SERIAL",) if a.get("ID_SERIAL") or b.get("ID_SERIAL") else ("ID_PATH",)
    for key in keys:
        if a.get(key) is not None and b.get(key) is not None and a[key] != b[key]:
            raise ValueError(f"{name}: raw calibration camera identity {key} disagrees")


def _lens_controls(values):
    return {
        "focus_automatic_continuous" if key == "focus_auto" else key: value
        for key, value in values.items() if "focus" in key or "zoom" in key
    }


def _capture_matches_snapshot(spec, snapshot, name):
    _same_camera({"camera": spec.get("expected_identity", {})}, snapshot, name)
    requested = _lens_controls(spec.get("controls", {}))
    known = snapshot.get("camera", {}).get("controls", {})
    known = _lens_controls(known.get("values", known))
    for key in requested.keys() & known.keys():
        if requested[key] != known[key]:
            raise ValueError(f"{name}: recorded {key} differs from raw calibration")


def _digest(contract):
    data = {k: v for k, v in contract.items() if k != "sha256"}
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def make_image_geometry(source_size, output_size, *, K=None, D=None, alpha=0.0):
    """D supplied explicitly selects rectification; omitted D preserves legacy resize."""
    source, target = (
        _size(source_size, "source_size"),
        _size(output_size, "output_size"),
    )
    if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite in [0,1]")
    if D is not None and K is None:
        raise ValueError("distortion requires raw intrinsics")
    raw_K = None if K is None else _K(K)
    raw_D = None if D is None else _D(D)
    new_K = None if raw_K is None else raw_K.copy()
    roi = None
    if raw_D is not None:
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            raw_K, raw_D, source, float(alpha), source
        )
    output_K = None if new_K is None else new_K.copy()
    if output_K is not None:
        output_K[0] *= target[0] / source[0]
        output_K[1] *= target[1] / source[1]
    result = {
        "schema": "camera_image_geometry_v1",
        "operation": "undistort_resize" if raw_D is not None else "resize",
        "input_space": "raw",
        "output_space": "processed",
        "source_size": list(source),
        "output_size": list(target),
        "K_raw": None if raw_K is None else raw_K.tolist(),
        "D_raw": None if raw_D is None else raw_D.tolist(),
        "K_rectified": None if new_K is None else new_K.tolist(),
        "K_output": None if output_K is None else output_K.tolist(),
        "D_output": [0.0] * 5 if raw_D is not None else None,
        "alpha": float(alpha) if raw_D is not None else None,
        "valid_roi": None if roi is None else list(map(int, roi)),
        "roi_cropped": False,
        "rectification_rotation": "identity_optical_frame_unchanged",
        "resize_interpolation": "INTER_AREA",
        "remap_interpolation": "INTER_LINEAR",
        "border_mode": "BORDER_CONSTANT_0",
        "opencv_version": cv2.__version__,
    }
    result["sha256"] = _digest(result)
    validate_image_geometry(result)
    return result


def validate_image_geometry(contract):
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != "camera_image_geometry_v1"
    ):
        raise ValueError("unsupported image geometry schema")
    if contract.get("sha256") != _digest(contract):
        raise ValueError("image geometry hash mismatch")
    source, target = (
        _size(contract["source_size"], "source_size"),
        _size(contract["output_size"], "output_size"),
    )
    constants = {
        "input_space": "raw",
        "output_space": "processed",
        "roi_cropped": False,
        "rectification_rotation": "identity_optical_frame_unchanged",
        "resize_interpolation": "INTER_AREA",
        "remap_interpolation": "INTER_LINEAR",
        "border_mode": "BORDER_CONSTANT_0",
    }
    if any(contract.get(k) != v for k, v in constants.items()):
        raise ValueError("unsupported image geometry operation, crop or interpolation")
    mode = contract.get("operation")
    if mode not in ("resize", "undistort_resize"):
        raise ValueError("unsupported image operation")
    if mode == "undistort_resize":
        _K(contract["K_raw"])
        _D(contract["D_raw"])
        if (
            contract.get("D_output") != [0.0] * 5
            or not isinstance(contract.get("alpha"), (int, float))
            or not 0 <= contract["alpha"] <= 1
        ):
            raise ValueError("invalid rectification metadata")
    elif contract.get("D_raw") is not None or contract.get("D_output") is not None:
        raise ValueError("resize cannot claim a distortion correction")
    if contract.get("K_raw") is None:
        if (
            contract.get("K_rectified") is not None
            or contract.get("K_output") is not None
        ):
            raise ValueError(
                "image geometry invents intrinsics without raw calibration"
            )
    else:
        raw_K, new_K, output_K = (
            _K(contract[k]) for k in ("K_raw", "K_rectified", "K_output")
        )
        if mode == "resize" and not np.array_equal(raw_K, new_K):
            raise ValueError("resize must retain the raw camera matrix before scaling")
        scaled = new_K.copy()
        scaled[0] *= target[0] / source[0]
        scaled[1] *= target[1] / source[1]
        if not np.allclose(scaled, output_K, atol=1e-10, rtol=0):
            raise ValueError("K_output disagrees with saved rectification and resize")


def camera_image_geometry(
    name, spec, calibration_data, source_size, output_size, *, alpha=0.0
):
    """Resolve raw geometry from immutable capture and per-batch calibration copies.

    Both copies must agree. Legacy data without measured distortion retain their
    old resize path; this never marks their lens distortion as measured zero.
    """
    spec, data = spec or {}, calibration_data or {}
    if "width" in spec and "height" in spec:
        if (spec["width"], spec["height"]) != tuple(source_size):
            raise ValueError(f"{name}: native video size differs from capture config")
    snapshot = spec.get("calibration")
    extra_snapshot = (
        data.get("raw_camera_calibration")
        if name == data.get("reference_camera")
        else None
    )
    if snapshot is not None and extra_snapshot is not None:
        _raw_pixels(extra_snapshot, name)
        _same_camera(snapshot, extra_snapshot, name)
        for key in ("K", "D", "image_size"):
            if np.shape(snapshot.get(key)) != np.shape(
                extra_snapshot.get(key)
            ) or not np.allclose(
                snapshot[key], extra_snapshot[key], atol=1e-10, rtol=0
            ):
                raise ValueError(
                    f"{name}: capture and batch raw calibration {key} disagree"
                )
        if snapshot.get("camera_model") != extra_snapshot.get("camera_model"):
            raise ValueError(f"{name}: raw camera models disagree")
    if snapshot is None and name == data.get("reference_camera"):
        snapshot = data.get("raw_camera_calibration")
    K, D = None, None
    if snapshot is not None:
        if (
            snapshot.get("schema") != "uvc_intrinsics_calibration_v1"
            or snapshot.get("camera_model") != "opencv_pinhole_brown5"
        ):
            raise ValueError(f"{name}: expected a raw Brown5 intrinsic snapshot")
        if tuple(snapshot["image_size"]) != tuple(source_size):
            raise ValueError(f"{name}: raw calibration size differs from source video")
        _raw_pixels(snapshot, name)
        _capture_matches_snapshot(spec, snapshot, name)
        K, D = _K(snapshot["K"]), _D(snapshot["D"])
    if name in data.get("intrinsics", {}):
        other_K = _K(data["intrinsics"][name])
        if tuple(data.get("image_sizes", {}).get(name, ())) != tuple(source_size):
            raise ValueError(
                f"{name}: calibration image size differs from source video"
            )
        if K is not None and not np.allclose(K, other_K, atol=1e-8, rtol=0):
            raise ValueError(f"{name}: capture and extrinsic calibration K disagree")
        K = other_K
    if name in data.get("distortion_coefficients", {}):
        if data.get("camera_models", {}).get(name) != "opencv_pinhole_brown5":
            raise ValueError(f"{name}: unsupported or missing distortion camera model")
        other_D = _D(data["distortion_coefficients"][name])
        if D is not None and not np.allclose(D, other_D, atol=1e-10, rtol=0):
            raise ValueError(f"{name}: capture and extrinsic calibration D disagree")
        D = other_D
    if (
        name == data.get("reference_camera")
        and data.get("image_geometry") == "raw_distorted"
        and D is None
    ):
        raise ValueError(f"{name}: raw_distorted calibration is missing distortion")
    return make_image_geometry(source_size, output_size, K=K, D=D, alpha=alpha)


def validate_dataset_image_geometry(metadata):
    """Reject contradictory processed pixels, intrinsics, or raw provenance."""
    contracts = metadata.get("image_processing")
    data = metadata.get("calibration", {}).get("data", {})
    specs = metadata.get("source_capture_config", {}).get("cameras") or []
    specs = {spec["name"]: spec for spec in specs}
    if contracts is None:
        measured = (
            bool(data.get("distortion_coefficients"))
            or data.get("raw_camera_calibration") is not None
            or data.get("image_geometry") == "raw_distorted"
            or any(spec.get("calibration") is not None for spec in specs.values())
        )
        if measured:
            raise ValueError(
                "measured raw lens calibration requires image_processing metadata"
            )
        return  # Original PICO/uncalibrated legacy exports keep their resize path.
    if not isinstance(contracts, dict) or not contracts:
        raise ValueError("image_processing must be a nonempty camera mapping")
    if metadata.get("image_space") != "processed":
        raise ValueError("converted dataset image_space must be processed")
    intrinsics = metadata.get("intrinsics") or {}
    features = metadata.get("features", {})
    image_names = {key[7:] for key in features if key.startswith("images.")}
    if image_names and image_names != set(contracts):
        raise ValueError("image processing must cover exactly the stored camera views")
    for name, contract in contracts.items():
        validate_image_geometry(contract)
        shape = features.get(f"images.{name}", {}).get("shape")
        size = contract["output_size"]
        if shape is not None and list(shape) != [size[1], size[0], 3]:
            raise ValueError(
                f"{name}: stored image feature shape differs from image processing"
            )
        output_K = contract["K_output"]
        if output_K is not None:
            if name not in intrinsics or not np.allclose(
                _K(intrinsics[name]), output_K, atol=1e-8, rtol=0
            ):
                raise ValueError(
                    f"{name}: root intrinsics differ from processed image K_output"
                )
        elif name in intrinsics:
            raise ValueError(f"{name}: root intrinsics lack image geometry provenance")
        if contract["operation"] == "undistort_resize":
            if metadata.get("distortion_coefficients", {}).get(name) != [0.0] * 5:
                raise ValueError(
                    f"{name}: rectified images must declare zero output distortion"
                )
        # Resolve raw K/D independently from provenance, but use the stored
        # K_rectified for rendering even if an OpenCV upgrade changes K_new.
        expected = camera_image_geometry(
            name,
            specs.get(name),
            data,
            contract["source_size"],
            size,
            alpha=contract["alpha"] or 0.0,
        )
        for key in ("K_raw", "D_raw"):
            a, b = expected[key], contract[key]
            if ((a is None) != (b is None)) or (
                a is not None and not np.allclose(a, b, atol=1e-8, rtol=0)
            ):
                raise ValueError(
                    f"{name}: {key} differs from raw calibration provenance"
                )


class RgbImageProcessor:
    def __init__(self, contract):
        validate_image_geometry(contract)
        self.contract = copy.deepcopy(contract)
        self.maps = None
        if contract["operation"] == "undistort_resize":
            self.maps = cv2.initUndistortRectifyMap(
                _K(contract["K_raw"]),
                _D(contract["D_raw"]),
                np.eye(3),
                _K(contract["K_rectified"]),
                tuple(contract["source_size"]),
                cv2.CV_32FC1,
            )

    def apply(self, rgb, *, image_space="raw"):
        rgb = np.asarray(rgb)
        if image_space not in ("raw", "processed"):
            raise ValueError("image_space must be raw or processed")
        expected = self.contract[
            "source_size" if image_space == "raw" else "output_size"
        ]
        if rgb.dtype != np.uint8 or rgb.shape != (expected[1], expected[0], 3):
            raise ValueError(
                f"{image_space} image must be RGB uint8 at size {expected}"
            )
        if image_space == "processed":
            return np.ascontiguousarray(rgb)
        if self.maps is not None:
            rgb = cv2.remap(
                rgb,
                *self.maps,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        target = tuple(self.contract["output_size"])
        if (rgb.shape[1], rgb.shape[0]) != target:
            rgb = cv2.resize(rgb, target, interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(rgb)
