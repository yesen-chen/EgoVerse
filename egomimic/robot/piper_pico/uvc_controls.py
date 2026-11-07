"""Bounded V4L2 settings and identity checks for UVC collection.

No device is probed unless settings or identity verification was requested.
Missing readings remain unknown; they never satisfy a known expected value.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

CONTROL_TIMEOUT = 3.0
_AUTO_FOCUS = "focus_automatic_continuous"


def device_path(device: str) -> str:
    value = str(device)
    return f"/dev/video{int(value)}" if value.isdigit() else value


def validate_controls(controls: dict | None) -> dict:
    if controls is None:
        return {}
    if not isinstance(controls, dict):
        raise ValueError("UVC controls must be a mapping of V4L2 names to integers")
    for key, value in controls.items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise ValueError(f"invalid V4L2 control name: {key!r}")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"V4L2 control {key} must be an integer")
    canonical_controls(controls)
    return dict(controls)


def canonical_controls(values: dict) -> dict:
    result = {}
    for key, value in values.items():
        canonical = _AUTO_FOCUS if key == "focus_auto" else key
        if canonical in result and result[canonical] != value:
            raise ValueError(f"conflicting V4L2 control aliases: {canonical}")
        result[canonical] = value
    return result


def calibration_lens_controls(calibration: dict | None) -> dict:
    if calibration is None:
        return {}
    controls = calibration.get("camera", {}).get("controls", {})
    values = controls.get("values", controls) if isinstance(controls, dict) else {}
    lens = {
        key: value for key, value in values.items() if "focus" in key or "zoom" in key
    }
    return canonical_controls(validate_controls(lens))


def read_controls(device: str) -> dict:
    if shutil.which("v4l2-ctl") is None:
        return {"available": False, "reason": "v4l2-ctl not installed", "values": {}}
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device_path(device), "--list-ctrls"],
            capture_output=True,
            text=True,
            timeout=CONTROL_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": str(exc), "values": {}}
    if result.returncode:
        return {
            "available": False,
            "reason": result.stderr.strip() or "v4l2-ctl failed",
            "values": {},
        }
    values = {}
    for line in result.stdout.splitlines():
        match = re.match(r"\s*([a-z][a-z0-9_]*)\s+.*?\bvalue=(-?\d+)\b", line)
        if match:
            values[match.group(1)] = int(match.group(2))
    return {"available": True, "values": values}


def verify_identity(
    device: str,
    expected: dict | None = None,
    *,
    calibration: dict | None = None,
) -> dict:
    """Verify optional udev ID_* properties before VideoCapture opens the node."""
    node = device_path(device)
    actual = {"device": node, "resolved_device": str(Path(node).resolve())}
    if expected is None:
        expected = {}
    if not isinstance(expected, dict) or any(
        not isinstance(key, str)
        or not key.startswith("ID_")
        or not isinstance(value, str)
        or not value
        for key, value in expected.items()
    ):
        raise ValueError(
            "expected_identity must map udev ID_* properties to nonempty strings"
        )
    expected = dict(expected)
    calibrated_camera = calibration.get("camera", {}) if calibration else {}
    # Serial identity survives moving the same camera to a new physical port.
    # If no serial was recorded, its recorded USB path is the only stable guard.
    key = "ID_SERIAL" if calibrated_camera.get("ID_SERIAL") else "ID_PATH"
    calibrated_value = calibrated_camera.get(key)
    if calibrated_value:
        if key in expected and expected[key] != calibrated_value:
            raise ValueError(f"expected identity {key} conflicts with calibration")
        expected[key] = calibrated_value
    if not expected:
        return {"status": "not_requested", "actual": actual, "expected": {}}
    if shutil.which("udevadm") is None:
        raise RuntimeError("cannot verify UVC identity: udevadm not installed")
    try:
        result = subprocess.run(
            ["udevadm", "info", "--query=property", f"--name={node}"],
            capture_output=True,
            text=True,
            timeout=CONTROL_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot verify UVC identity for {node}: {exc}") from exc
    if result.returncode:
        raise RuntimeError(
            f"cannot verify UVC identity for {node}: {result.stderr.strip()}"
        )
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.startswith("ID_"):
            actual[key] = value
    for key, value in expected.items():
        if actual.get(key) != value:
            raise RuntimeError(
                f"UVC identity mismatch for {node}: {key} expected {value!r}, "
                f"read back {actual.get(key)!r}"
            )
    return {"status": "verified", "actual": actual, "expected": dict(expected)}


def verify_controls(
    device: str,
    requested: dict | None = None,
    calibration: dict | None = None,
    *,
    apply: bool = False,
) -> dict:
    """Optionally apply requested controls, then verify them and known lens values."""
    requested = validate_controls(requested)
    lens = calibration_lens_controls(calibration)
    expected = canonical_controls(requested)
    for key, value in lens.items():
        if key in expected and expected[key] != value:
            raise ValueError(f"requested {key} conflicts with calibration: {value}")
    if lens.get(_AUTO_FOCUS, 0) != 0:
        raise ValueError("calibrated autofocus must be disabled for collection")
    lens_status = "not_requested" if calibration is None else "unknown"
    if not requested and not lens:
        return {
            "status": "not_requested",
            "requested": {},
            "readback": {"available": False, "reason": "not requested", "values": {}},
            "calibration_lens": {"status": lens_status, "expected": lens},
        }
    snapshot = read_controls(device)
    if not snapshot["available"]:
        raise RuntimeError(
            f"cannot verify UVC controls for {device}: {snapshot.get('reason')}"
        )
    if requested and apply:
        available = snapshot["values"]
        current = canonical_controls(available)
        # Disable automatic controls before setting their corresponding manual values.
        ordered = sorted(requested, key=lambda key: ("auto" not in key, key))
        for key in ordered:
            actual_key = key
            if key not in available and key in ("focus_auto", _AUTO_FOCUS):
                actual_key = _AUTO_FOCUS if key == "focus_auto" else "focus_auto"
            if actual_key not in available:
                raise RuntimeError(f"UVC control unavailable: {key}")
            canonical = _AUTO_FOCUS if key == "focus_auto" else key
            if current.get(canonical) == requested[key]:
                continue
            try:
                result = subprocess.run(
                    [
                        "v4l2-ctl",
                        "-d",
                        device_path(device),
                        f"--set-ctrl={actual_key}={requested[key]}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=CONTROL_TIMEOUT,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(f"failed to set UVC control {key}: {exc}") from exc
            if result.returncode:
                snapshot = read_controls(device)
                current = canonical_controls(snapshot.get("values", {}))
                if snapshot.get("available") and current.get(canonical) == requested[key]:
                    continue
                raise RuntimeError(
                    f"failed to set UVC control {key}: {result.stderr.strip()}"
                )
            current[canonical] = requested[key]
        snapshot = read_controls(device)
    if not snapshot["available"]:
        raise RuntimeError(
            f"cannot read back UVC controls for {device}: {snapshot.get('reason')}"
        )
    actual = canonical_controls(snapshot["values"])
    if lens and actual.get(_AUTO_FOCUS, 0) != 0:
        raise RuntimeError(
            "UVC autofocus is enabled for a calibrated lens; lock focus before collection"
        )
    for key, value in {**expected, **lens}.items():
        if actual.get(key) != value:
            raise RuntimeError(
                f"UVC control mismatch: {key} expected {value}, read back {actual.get(key)!r}"
            )
    return {
        "status": "verified",
        "requested": requested,
        "readback": snapshot,
        "calibration_lens": {
            "status": "verified" if lens else lens_status,
            "expected": lens,
        },
    }
