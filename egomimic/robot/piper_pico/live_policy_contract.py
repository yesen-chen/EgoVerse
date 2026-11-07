"""Read-only calibration and capture checks for a live Piper policy.

The checkpoint does not embed the rig's measured calibration. The explicit
training-data selection supplies that evidence; it must describe one rig. No
image chunks, devices, or CAN interfaces are opened by this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from egomimic.rldb.zarr.piper_dataset import (
    observation_to_send,
    validate_piper_metadata,
)
from egomimic.robot.piper_pico.camera_config import load_camera_config
from egomimic.robot.piper_pico.image_geometry import camera_image_geometry
from egomimic.robot.piper_pico.policy_inference import CAMERA_KEYS, check_capture_config


def _read_json(path):
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _episode_paths(root):
    if not root.is_dir():
        raise ValueError(f"--training-data must be a directory: {root}")
    if root.suffix == ".zarr":
        yield root
        return
    for parent, dirs, _ in os.walk(root):
        # Prune stores: walking image chunks is unnecessary and can be expensive.
        stores = sorted(name for name in dirs if name.endswith(".zarr"))
        dirs[:] = sorted(name for name in dirs if not name.endswith(".zarr"))
        for name in stores:
            yield Path(parent) / name


def _metadata(path):
    if (path / "zarr.json").is_file():
        source = path / "zarr.json"
        value, digest = _read_json(source)
        if value.get("node_type") != "group":
            raise ValueError(f"{path}: expected a Zarr group")
        value = value.get("attributes", {})
    elif (path / ".zattrs").is_file():
        source = path / ".zattrs"
        value, digest = _read_json(source)
    else:
        raise ValueError(f"{path}: missing root Zarr metadata")
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root Zarr attributes must be an object")
    return value, {"path": str(source), "sha256": digest}


def _same(left, right, label):
    """Compare semantic geometry, with tolerance for harmless float formatting."""
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise ValueError(f"{label}: fields differ")
        for key in left:
            _same(left[key], right[key], f"{label}.{key}")
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(f"{label}: lengths differ")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _same(a, b, f"{label}[{index}]")
    elif (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        if not np.isfinite([left, right]).all() or not np.isclose(
            left, right, rtol=0, atol=1e-8
        ):
            raise ValueError(f"{label}: values differ ({left!r} != {right!r})")
    elif type(left) is not type(right) or left != right:
        raise ValueError(f"{label}: values differ ({left!r} != {right!r})")


def _profile(metadata):
    processing = metadata.get("image_processing")
    if not isinstance(processing, dict) or set(processing) != set(CAMERA_KEYS):
        raise ValueError(
            "Live inference requires saved image_processing for all three cameras; "
            "reconvert legacy episodes with the calibration JSON."
        )
    encoding = metadata["gripper_encoding"]
    width = encoding.get("open_m")
    if (
        isinstance(width, bool)
        or not isinstance(width, (int, float))
        or not np.isfinite(width)
        or width <= 0
        or encoding.get("closed_m", 0) != 0
    ):
        raise ValueError("gripper_encoding requires positive open_m and closed_m=0")
    fps = metadata.get("fps")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not np.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("recorded fps must be finite and positive")
    return {
        "fps": fps,
        "extrinsics_convention": metadata["extrinsics_convention"],
        "reference_camera": metadata["reference_camera"],
        "extrinsics": metadata["extrinsics"],
        "gripper_encoding": {
            "unit": encoding["unit"],
            "closed_m": encoding.get("closed_m", 0),
            "open_m": width,
        },
        "image_processing": {
            name: {
                key: value
                for key, value in contract.items()
                if key not in ("sha256", "opencv_version")
            }
            for name, contract in processing.items()
        },
    }


def _check_calibration(metadata, calibration, camera_specs):
    """Validate the supplied geometry against the saved training pixel contract."""
    data = calibration["data"]
    if data.get("translation_unit", "metre") not in ("metre", "meter", "m"):
        raise ValueError("--calibration translation_unit must be metres")
    _same(metadata["extrinsics"], data["extrinsics"], "calibration.extrinsics")
    specs = {spec["name"]: spec for spec in camera_specs}
    for name, saved in metadata["image_processing"].items():
        # Resolve raw K/D with the same rules as conversion, but keep the saved
        # rectification matrix: an OpenCV upgrade must not change model pixels.
        expected = camera_image_geometry(
            name,
            specs[name],
            data,
            saved["source_size"],
            saved["output_size"],
            alpha=saved["alpha"] or 0.0,
        )
        for key in ("operation", "K_raw", "D_raw"):
            _same(saved[key], expected[key], f"calibration.{name}.{key}")
    candidate = copy.deepcopy(metadata)
    candidate["calibration"] = copy.deepcopy(calibration)
    validate_piper_metadata(candidate)


def prepare_live_contract(training_data, calibration, camera_config):
    """Return ``metadata``, normalized ``camera_specs``, and a JSON ``audit``.

    All selected Piper episode profiles must agree. Non-Piper stores may coexist
    with them (e.g. mixed ego training), but multiple robot rigs require an
    explicit narrower --training-data selection. Per-episode send delays may
    differ; their median is used and recorded, rather than picking one episode.
    """
    # Reuse the converter's calibration validation and 3x3/3x4 normalization.
    from egomimic.scripts.piper_pico_to_zarr import Options, load_calibration

    root, calibration_path, config_path = (
        Path(path).expanduser().resolve()
        for path in (training_data, calibration, camera_config)
    )
    camera_specs = load_camera_config(config_path)
    supplied = load_calibration(calibration_path, Options())
    representative, profile = None, None
    episodes, ignored, delays = [], [], []
    for path in _episode_paths(root):
        metadata, evidence = _metadata(path)
        if metadata.get("embodiment") != "piper_bimanual" and not str(
            metadata.get("schema", "")
        ).startswith("piper_"):
            ignored.append(str(path))
            continue
        try:
            validate_piper_metadata(metadata, episode_path=path)
            current = _profile(metadata)
            if profile is not None:
                try:
                    _same(profile, current, "rig profile")
                except ValueError as exc:
                    raise ValueError(
                        "Selected Piper episodes contain multiple geometry/preprocessing "
                        "profiles; use --training-data to select the current rig's "
                        f"subdirectory or one episode. {exc}"
                    ) from exc
            check_capture_config(metadata, camera_specs)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        if representative is None:
            representative, profile = copy.deepcopy(metadata), current
        delays.append(observation_to_send(metadata))
        episodes.append(
            {
                "episode": str(path),
                "metadata": evidence,
                "recorded_calibration_sha256": metadata["calibration"].get("sha256"),
            }
        )
    if representative is None:
        raise ValueError(f"No Piper training episodes found under {root}")
    try:
        _check_calibration(representative, supplied, camera_specs)
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"--calibration does not match the selected training rig: {exc}"
        ) from exc
    representative["calibration"] = copy.deepcopy(supplied)
    representative["time_alignment"]["observation_to_send_s"] = float(np.median(delays))
    representative["time_alignment"]["observation_to_send_source"] = (
        "median_across_selected_training_episodes"
    )
    _, config_digest = _read_json(config_path)
    audit = {
        "schema": "live_piper_contract_v1",
        "training_data": str(root),
        "episode_count": len(episodes),
        "episodes": episodes,
        "ignored_non_piper_episode_count": len(ignored),
        "ignored_non_piper_episodes": ignored,
        "calibration": {
            "path": supplied["path"],
            "sha256": supplied["sha256"],
            "calibration_id": supplied["data"]["calibration_id"],
        },
        "camera_config": {"path": str(config_path), "sha256": config_digest},
        "geometry": profile,
        "observation_to_send_s": {
            "source": "median_across_selected_training_episodes",
            "min": float(min(delays)),
            "median": float(np.median(delays)),
            "max": float(max(delays)),
        },
        "calibration_check": "numeric_geometry_matches_selected_training_data",
        "checkpoint_calibration_binding": (
            "The checkpoint does not embed the rig calibration hash. This check "
            "uses the explicit --training-data selection, not a checkpoint binding."
        ),
        "capture_check": "all_selected_piper_episodes_match_runtime_camera_config",
    }
    return {"metadata": representative, "camera_specs": camera_specs, "audit": audit}
