"""Hardware-free camera-rig validation and immutable intrinsic snapshots.

``camera_rig_v1`` describes native capture modes individually. Loading never
opens a camera, rescales intrinsics, undistorts pixels, or supplies extrinsics.
Calibration snapshots retain the complete standalone solver result alongside
the exact source-file SHA-256, so a recording does not depend on a mutable path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_CONTROL = re.compile(r"[a-z][a-z0-9_]*\Z")
_RAW_PREPROCESSING = {"crop": None, "resize": None, "mirror": False}
_DISTORTION_ORDER = ["k1", "k2", "p1", "p2", "k3"]
DEFAULT_UVC_BRIGHTNESS = 48


def _number(value, label: str, *, integer=False, positive=False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    if not math.isfinite(value) or (integer and not isinstance(value, int)):
        raise ValueError(
            f"{label} must be a finite {'integer' if integer else 'number'}"
        )
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'}")


def _raw_preprocessing(value, label: str, *, required=False) -> None:
    if value is None and not required:
        return
    if not isinstance(value, dict) or any(
        key not in value or value[key] != expected
        for key, expected in _RAW_PREPROCESSING.items()
    ):
        raise ValueError(
            f"{label} must describe raw uncropped, unresized, unmirrored images"
        )
    if value.get("mirror") is not False:
        raise ValueError(f"{label}.mirror must be false")
    supported = set(_RAW_PREPROCESSING) | {
        "undistort",
        "undistorted",
        "rectified",
        "roi_cropped",
        "rotate",
        "rotation",
    }
    if value.keys() - supported:
        raise ValueError(f"{label} contains unsupported preprocessing operations")
    for key in ("undistort", "undistorted", "rectified", "roi_cropped"):
        if value.get(key, False) is not False:
            raise ValueError(f"{label} cannot use undistorted/rectified images")
    for key in ("rotate", "rotation"):
        if value.get(key) not in (None, 0):
            raise ValueError(f"{label} cannot rotate images")


def _controls(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping of V4L2 control names to integers")
    normalized = {}
    for key, val in value.items():
        if not isinstance(key, str) or not _CONTROL.fullmatch(key):
            raise ValueError(f"{label} has invalid V4L2 control name: {key!r}")
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError(f"{label}.{key} must be an integer in driver units")
        if key in ("focus_auto", "focus_automatic_continuous") and val not in (0, 1):
            raise ValueError(f"{label}.{key} must be 0 or 1")
        if key == "focus_absolute" and val < 0:
            raise ValueError(f"{label}.focus_absolute must be nonnegative")
        canonical = "focus_automatic_continuous" if key == "focus_auto" else key
        if canonical in normalized and normalized[canonical] != val:
            raise ValueError(f"{label} has conflicting focus-auto aliases")
        normalized[canonical] = val
    return normalized


def uvc_controls_with_defaults(controls: dict | None = None) -> dict:
    """Resolve live-capture defaults without mutating the caller's settings."""
    if controls is None:
        controls = {}
    _controls(controls, "camera controls")
    return {"brightness": DEFAULT_UVC_BRIGHTNESS, **controls}


def _calibration(spec: dict) -> None:
    calibration = spec.get("calibration")
    if calibration is None:
        if spec.get("calibration_file"):
            raise ValueError(
                "calibration_file requires a loaded calibration snapshot; use load_camera_config"
            )
        return
    if not isinstance(calibration, dict):
        raise ValueError("calibration must be a standalone intrinsic snapshot")
    if spec["type"] != "uvc" or calibration.get("type", "uvc") != "uvc":
        raise ValueError("UVC intrinsics can only describe a UVC camera")
    if calibration.get("schema") != "uvc_intrinsics_calibration_v1":
        raise ValueError(
            "calibration must use raw uvc_intrinsics_calibration_v1, not rectified intrinsics"
        )
    if calibration.get("camera_model") != "opencv_pinhole_brown5":
        raise ValueError(
            "unsupported calibration camera_model; expected opencv_pinhole_brown5"
        )
    size = calibration.get("image_size")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("calibration image_size must be [width, height]")
    for value in size:
        _number(value, "calibration image_size", integer=True, positive=True)
    if size != [spec["width"], spec["height"]]:
        raise ValueError(
            "calibration image_size must exactly match the native camera width/height"
        )
    try:
        for matrix in (calibration.get("K"), calibration.get("D")):
            values = np.asarray(matrix, dtype=object).flat
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in values
            ):
                raise ValueError("nonnumeric intrinsic value")
        K = np.asarray(calibration.get("K"), dtype=float)
        D = np.asarray(calibration.get("D"), dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("calibration K/D must contain finite numeric values") from exc
    if (
        K.shape != (3, 3)
        or not np.isfinite(K).all()
        or K[0, 0] <= 0
        or K[1, 1] <= 0
        or not np.allclose(K[2], [0, 0, 1], atol=1e-12, rtol=0)
        or K[0, 1] != 0
        or K[1, 0] != 0
    ):
        raise ValueError(
            "calibration K must be a finite 3x3 pinhole matrix with positive focal lengths"
        )
    if D.shape != (5,) or not np.isfinite(D).all():
        raise ValueError("calibration D must contain five finite Brown5 coefficients")
    if (
        calibration.get("distortion_parameter_order", _DISTORTION_ORDER)
        != _DISTORTION_ORDER
    ):
        raise ValueError(
            "calibration distortion_parameter_order must be k1,k2,p1,p2,k3"
        )
    _raw_preprocessing(
        calibration.get("preprocessing"), "calibration preprocessing", required=True
    )
    if "K_new" in calibration or calibration.get("roi_cropped", False) is not False:
        raise ValueError(
            "calibration must describe raw images, not rectified K_new or an ROI crop"
        )
    camera = calibration.get("camera", {})
    if not isinstance(camera, dict):
        raise ValueError("calibration camera metadata must be an object")
    if "image_size" in camera and camera["image_size"] != size:
        raise ValueError(
            "calibration camera image_size contradicts calibration image_size"
        )
    for key in ("requested_mode", "actual_mode"):
        mode = camera.get(key, {})
        if not isinstance(mode, dict):
            raise ValueError(f"calibration camera {key} must be an object")
        for dimension, expected in zip(("width", "height"), size):
            if dimension in mode and mode[dimension] != expected:
                raise ValueError(
                    f"calibration {key} {dimension} contradicts calibration image_size"
                )
    control_snapshot = camera.get("controls", {})
    if not isinstance(control_snapshot, dict):
        raise ValueError("calibration controls must be an object")
    if "values" in control_snapshot:
        known = _controls(control_snapshot["values"], "calibration controls")
    else:
        # Early standalone fixtures stored the controls directly.
        known = _controls(control_snapshot, "calibration controls")
    requested = _controls(spec.get("controls", {}), "camera controls")
    if (
        known.get("focus_automatic_continuous") == 1
        or requested.get("focus_automatic_continuous") == 1
    ):
        raise ValueError(
            "calibration requires fixed focus; autofocus cannot be enabled"
        )
    for key in known.keys() & requested.keys():
        if ("focus" in key or "zoom" in key) and known[key] != requested[key]:
            raise ValueError(f"camera control {key} differs from calibration")
    expected_identity = spec.get("expected_identity", {})
    for key in camera.keys() & expected_identity.keys():
        # Match verify_identity: a recorded serial survives a USB port move.
        # The explicitly configured current path is still checked at startup.
        if key == "ID_PATH" and camera.get("ID_SERIAL"):
            continue
        if camera[key] != expected_identity[key]:
            raise ValueError(f"camera expected_identity {key} differs from calibration")


def validate_camera_specs(specs: list[dict]) -> list[dict]:
    """Validate complete modes without hardware access; return independent dicts.

    Legacy callers may fill their historic defaults before calling this function.
    Unknown latency is explicit: omitted ``latency_status`` means ``unmeasured``;
    a numeric compensation alone never establishes that a latency was measured.
    """
    if not isinstance(specs, list):
        raise ValueError("cameras must be a list")
    result = copy.deepcopy(specs)
    names, devices = set(), set()
    pico_count = 0
    for spec in result:
        if not isinstance(spec, dict):
            raise ValueError("each camera spec must be an object")
        name = spec.get("name")
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(
                "camera name must be a safe nonempty filename without slashes"
            )
        if name in names:
            raise ValueError("camera names must be unique")
        names.add(name)
        source = spec.get("type")
        if source not in ("uvc", "pico"):
            raise ValueError(f"unknown camera type: {source!r}")
        for key in ("width", "height", "fps", "latency"):
            _number(
                spec.get(key),
                f"camera {name} {key}",
                integer=key in ("width", "height"),
                positive=key != "latency",
            )
        spec.setdefault("latency_status", "unmeasured")
        if spec["latency_status"] not in ("measured", "unmeasured"):
            raise ValueError("camera latency_status must be measured or unmeasured")
        _raw_preprocessing(spec.get("preprocessing"), "camera preprocessing")
        _controls(spec.get("controls", {}), "camera controls")
        identity = spec.get("expected_identity", {})
        if not isinstance(identity, dict) or any(
            not isinstance(key, str)
            or not key.startswith("ID_")
            or not isinstance(value, str)
            or not value.strip()
            for key, value in identity.items()
        ):
            raise ValueError(
                "expected_identity must map ID_* property names to nonempty strings"
            )
        if source == "uvc":
            spec["controls"] = uvc_controls_with_defaults(spec.get("controls"))
            device = spec.get("device")
            if (
                isinstance(device, bool)
                or not isinstance(device, (str, int))
                or not str(device).strip()
                or (isinstance(device, int) and device < 0)
            ):
                raise ValueError(
                    "UVC camera requires a device path or nonnegative index"
                )
            path = Path(
                f"/dev/video{int(device)}" if str(device).isdigit() else str(device)
            )
            resolved = str(path.resolve())
            if resolved in devices:
                raise ValueError("each UVC device may only be opened once")
            devices.add(resolved)
            fourcc = spec.get("fourcc")
            if (
                not isinstance(fourcc, str)
                or len(fourcc) != 4
                or not fourcc.isascii()
                or not fourcc.isprintable()
            ):
                raise ValueError(
                    "camera fourcc must contain exactly four printable ASCII characters"
                )
        else:
            pico_count += 1
            if pico_count > 1:
                raise ValueError("only one PICO image stream can use the video port")
            if (
                not isinstance(spec.get("headset_ip"), str)
                or not spec["headset_ip"].strip()
            ):
                raise ValueError("PICO camera requires a headset IP")
            if spec.get("eye", "left") != "left":
                raise ValueError("collector PICO camera supports only the left eye")
            if "bitrate" in spec:
                _number(spec["bitrate"], "PICO bitrate", integer=True, positive=True)
            if "startup_timeout" in spec:
                _number(spec["startup_timeout"], "PICO startup_timeout", positive=True)
            if spec.get("controls") or identity:
                raise ValueError(
                    "controls and expected_identity are only supported for UVC cameras"
                )
        _calibration(spec)
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "camera specs must contain finite JSON-compatible metadata"
        ) from exc
    return result


def load_camera_config(path: str | Path) -> list[dict]:
    """Load ``camera_rig_v1`` and snapshot raw calibration files relative to it."""
    path = Path(path).expanduser().resolve()
    try:
        config = json.loads(path.read_text())
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"cannot read camera config {path}: {exc}") from exc
    if not isinstance(config, dict) or config.get("schema") != "camera_rig_v1":
        raise ValueError("camera config schema must be camera_rig_v1")
    specs = config.get("cameras")
    if not isinstance(specs, list) or not specs:
        raise ValueError("camera config cameras must be a nonempty list")
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("each camera spec must be an object")
        if "calibration" in spec:
            raise ValueError(
                "camera config must reference calibration_file; embedded calibration is reserved for snapshots"
            )
        if "calibration_file" not in spec:
            continue
        relative = spec["calibration_file"]
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError("calibration_file must be a nonempty path")
        calibration_path = (path.parent / Path(relative).expanduser()).resolve()
        try:
            raw = calibration_path.read_bytes()
            calibration = json.loads(raw)
        except (OSError, UnicodeError, ValueError) as exc:
            raise ValueError(
                f"cannot read calibration_file {calibration_path}: {exc}"
            ) from exc
        if not isinstance(calibration, dict):
            raise ValueError(
                "calibration_file must contain a standalone intrinsic JSON object"
            )
        calibration["source_path"] = str(calibration_path)
        calibration["source_sha256"] = hashlib.sha256(raw).hexdigest()
        spec["calibration_file"] = str(calibration_path)
        spec["calibration"] = calibration
    return validate_camera_specs(specs)
