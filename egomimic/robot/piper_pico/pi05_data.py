"""Portable Pi0.5 windows over existing Piper Zarr episodes, without Torch/JAX.

Conversion stores native control ticks and processed JPEGs once. This adapter
selects a current-inclusive window, repeats its terminal row, applies stride,
changes each arm's base pose to the reference camera, then SLERPs to 100 points.
Neither conversion nor interpolation changes the chunk's physical time span.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import zarr
from scipy.spatial.transform import Rotation, Slerp

from egomimic.rldb.action_timing import ActionTiming
from egomimic.robot.piper_pico.image_geometry import validate_dataset_image_geometry

SCHEMA = "egoverse_pi05_manifest_v1"
CAMERAS = ("front_1", "left_wrist", "right_wrist")
SIDES = ("left", "right")
NUMERIC_KEYS = tuple(
    f"{side}.{field}"
    for side in SIDES
    for field in ("obs_ee_pose", "obs_gripper", "cmd_ee_pose", "cmd_gripper")
) + ("timestamp",)
ACTION_CONTRACT = {
    "dimension": 14,
    "representation": "absolute_xyz_ypr_gripper",
    "rotation_convention": "intrinsic_ZYX",
    "position_unit": "m",
    "rotation_unit": "rad",
    "gripper_unit": "normalized_width",
    "arm_order": ["left", "right"],
    "per_arm_fields": ["x", "y", "z", "yaw", "pitch", "roll", "gripper"],
    "reference_camera": "front_1",
    "coordinate_transform": "inverse(T_base_camera) @ T_base_sdk_end",
    "state_source": "actual_robot_feedback",
    "action_source": "absolute_sent_command",
    "camera_slots": {
        "front_1": "base_0_rgb",
        "left_wrist": "left_wrist_0_rgb",
        "right_wrist": "right_wrist_0_rgb",
    },
    "images": "RGB_uint8_HWC_already_processed; no_second_undistortion",
}


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _manifest_hash(value):
    return canonical_hash({k: v for k, v in value.items() if k != "manifest_sha256"})


def _numeric_hash(arrays):
    digest = hashlib.sha256()
    for key in NUMERIC_KEYS:
        value = np.ascontiguousarray(arrays[key], dtype="<f8")
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _rigid(value, label):
    a = np.asarray(value, dtype=float)
    if (
        a.shape != (4, 4)
        or not np.isfinite(a).all()
        or not np.allclose(a[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
        or not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-5, rtol=0)
        or not np.isclose(np.linalg.det(a[:3, :3]), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError(f"{label}: expected a rigid T_base_camera")
    return a


def _episode(path, timing, allow_synthetic):
    """Validate source identity and contracts without importing training code."""
    store = zarr.open_group(str(path), mode="r")
    meta = dict(store.attrs)
    expected = {
        "schema": "piper_base_v3",
        "embodiment": "piper_bimanual",
        "reference_camera": "front_1",
        "pose_representation": "xyz_m+quat_wxyz",
        "extrinsics_convention": "T_base_camera",
        "pose_coordinates": {side: f"piper_{side}_base" for side in SIDES},
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"{path}: {key} must be {value!r}")
    count, fps = meta.get("total_frames"), meta.get("fps")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"{path}: invalid total_frames")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not np.isclose(fps, timing.source_fps, atol=1e-9, rtol=0)
    ):
        raise ValueError(f"{path}: fps disagrees with action timing")
    calibration = meta.get("calibration", {}).get("data", {})
    status = meta.get("calibration_status")
    if status not in ("provided", "synthetic") or not calibration.get("calibration_id"):
        raise ValueError(f"{path}: missing calibration")
    if (status == "synthetic" or calibration.get("synthetic")) and not allow_synthetic:
        raise ValueError(f"{path}: synthetic calibration requires explicit opt-in")
    for key in ("reference_camera", "extrinsics_convention"):
        if calibration.get(key) != expected[key]:
            raise ValueError(f"{path}: calibration {key} disagrees")
    for side in SIDES:
        root = _rigid(meta.get("extrinsics", {}).get(side), f"{path}/{side}")
        origin = _rigid(calibration.get("extrinsics", {}).get(side), f"{path}/{side}")
        if not np.allclose(root, origin, atol=1e-8, rtol=0):
            raise ValueError(f"{path}: extrinsics disagree with calibration provenance")
    grip = meta.get("gripper_encoding", {})
    if grip.get("unit") != "normalized_width" or not (
        isinstance(grip.get("open_m"), (int, float)) and 0 < grip["open_m"] < 1
    ):
        raise ValueError(f"{path}: expected normalized_width gripper with open_m")
    delay = meta.get("time_alignment", {}).get("observation_to_send_s")
    if (
        isinstance(delay, bool)
        or not isinstance(delay, (int, float))
        or not np.isfinite(delay)
        or delay < 0
    ):
        raise ValueError(f"{path}: invalid observation_to_send_s")
    arrays = {}
    features = meta.get("features", {})
    for key in NUMERIC_KEYS:
        shape = (
            () if key == "timestamp" else ((7,) if key.endswith("ee_pose") else (1,))
        )
        if key not in store or tuple(features.get(key, {}).get("shape", [])) != shape:
            raise ValueError(f"{path}: missing or invalid {key}")
        values = np.asarray(store[key][:count], dtype=np.float64)
        if values.shape != (count, *shape) or not np.isfinite(values).all():
            raise ValueError(f"{path}: invalid finite numeric array {key}")
        if key.endswith("ee_pose") and not np.allclose(
            np.linalg.norm(values[:, 3:], axis=1), 1, atol=1e-3, rtol=0
        ):
            raise ValueError(f"{path}: non-unit quaternion in {key}")
        if key == "timestamp" and np.any(np.diff(values) <= 0):
            raise ValueError(f"{path}: timestamps must increase")
        arrays[key] = values
    processing = meta.get("image_processing")
    validate_dataset_image_geometry(meta)
    if processing is not None:
        if set(processing) != set(CAMERAS) or meta.get("image_space") != "processed":
            raise ValueError(f"{path}: all camera image contracts must be processed")
    elif calibration.get("distortion_coefficients") or calibration.get(
        "raw_camera_calibration"
    ):
        raise ValueError(f"{path}: measured lens calibration requires image_processing")
    for camera in CAMERAS:
        key = f"images.{camera}"
        info = features.get(key, {})
        shape = info.get("shape", [])
        if (
            key not in store
            or info.get("dtype") != "jpeg"
            or len(shape) != 3
            or shape[-1] != 3
        ):
            raise ValueError(f"{path}: missing or invalid JPEG camera {camera}")
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in shape):
            raise ValueError(f"{path}: invalid camera shape {camera}")
        if store[key].ndim != 1 or store[key].shape[0] < count:
            raise ValueError(f"{path}: insufficient image rows {camera}")
        if processing:
            geometry = processing[camera]
            # Same serialization as image_geometry._digest; no OpenCV import here.
            geometry_hash = hashlib.sha256(
                json.dumps(
                    {k: v for k, v in geometry.items() if k != "sha256"},
                    sort_keys=True,
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            if geometry.get("sha256") != geometry_hash:
                raise ValueError(f"{path}: image geometry checksum mismatch")
            if geometry.get("output_size") != [shape[1], shape[0]]:
                raise ValueError(f"{path}: image geometry dimensions disagree")
            if geometry.get("output_space") != "processed":
                raise ValueError(f"{path}: expected processed output pixels")
    return store, meta, arrays


def _contract(meta):
    return {
        key: copy.deepcopy(meta.get(key))
        for key in (
            "calibration",
            "calibration_status",
            "extrinsics",
            "intrinsics",
            "gripper_encoding",
            "image_processing",
            "image_space",
            "time_alignment",
            "source_capture_config",
            "tool_frame",
        )
    }


def _training_contract(meta):
    result = _contract(meta)
    for key in ("time_alignment", "source_capture_config"):
        result.pop(key)
    # File location is provenance, not a different geometric calibration.
    result["calibration"] = {"data": result["calibration"]["data"]}
    return result


def export_manifest(
    episode_paths,
    output_path,
    *,
    source_fps=30.0,
    source_horizon=30,
    action_stride=1,
    validation_ratio=0.1,
    seed=42,
    prompt=None,
    allow_synthetic_calibration=False,
) -> Path:
    """Write a portable reference manifest; never copy or re-encode source images.

    Existing manifests are not overwritten. Train/validation splitting is by
    complete episode, independent of input enumeration order. All stored source
    paths are relative to the manifest; moving their common ancestor is safe.
    """
    timing = ActionTiming(source_fps, source_horizon, action_stride, 100)
    if not np.isfinite(validation_ratio) or not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0,1)")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    destination = Path(output_path).resolve()
    paths = sorted(Path(path).resolve() for path in episode_paths)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("episode_paths must be nonempty and unique")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    records = []
    contracts, common_shapes, common_gripper = [], None, None
    for path in paths:
        _, meta, arrays = _episode(path, timing, allow_synthetic_calibration)
        contract = _training_contract(meta)
        shapes = {name: meta["features"][f"images.{name}"]["shape"] for name in CAMERAS}
        if common_shapes is None:
            common_shapes, common_gripper = shapes, contract["gripper_encoding"]
        elif shapes != common_shapes or contract["gripper_encoding"] != common_gripper:
            raise ValueError(
                f"{path}: image shape/gripper contract differs across episodes"
            )
        if contract not in contracts:
            contracts.append(contract)
        text = prompt if prompt is not None else meta.get("task_description", "")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"{path}: task prompt required; set --pi05-prompt/--prompt"
            )
        records.append(
            {
                "path": os.path.relpath(path, destination.parent),
                "total_frames": meta["total_frames"],
                "metadata_sha256": canonical_hash(meta),
                "numeric_sha256": _numeric_hash(arrays),
                "contract": _contract(meta),
                "image_shapes": {
                    name: meta["features"][f"images.{name}"]["shape"]
                    for name in CAMERAS
                },
                "prompt": text.strip(),
            }
        )
    if validation_ratio and len(records) < 2:
        raise ValueError(
            "a validation split requires at least two episodes; use ratio 0 for a smoke test"
        )
    n_valid = (
        min(len(records) - 1, max(1, int(len(records) * validation_ratio)))
        if validation_ratio
        else 0
    )
    indices = np.random.default_rng(seed).permutation(len(records))
    validation = set(indices[:n_valid].tolist())
    for index, record in enumerate(records):
        record["split"] = "validation" if index in validation else "train"
    result = {
        "schema": SCHEMA,
        "action_timing": timing.to_dict(),
        "action_contract": copy.deepcopy(ACTION_CONTRACT),
        "training_contract": contracts[0],
        "training_contracts": contracts,
        "image_shapes": common_shapes,
        "split": {
            "validation_ratio": float(validation_ratio),
            "seed": seed,
            "unit": "whole_episode",
        },
        "allow_synthetic_calibration": bool(allow_synthetic_calibration),
        "episodes": records,
        "data_identity_sha256": canonical_hash(
            [
                {
                    k: record[k]
                    for k in ("metadata_sha256", "numeric_sha256", "total_frames")
                }
                for record in records
            ]
        ),
        "identity_scope": "metadata_and_numeric_rows; JPEG bytes are referenced, not re-encoded or content-hashed",
    }
    result["manifest_sha256"] = _manifest_hash(result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return destination


def load_manifest(path) -> dict:
    """Load an internally consistent manifest (sources checked by Pi05Dataset)."""
    return validate_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_manifest(value) -> dict:
    """Validate manifest dictionaries received in a model-server handshake."""
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported Pi0.5 manifest schema")
    if value.get("manifest_sha256") != _manifest_hash(value):
        raise ValueError("Pi0.5 manifest checksum mismatch")
    if value.get("action_contract") != ACTION_CONTRACT:
        raise ValueError("unsupported Pi0.5 action contract")
    raw = value.get("action_timing", {})
    timing = ActionTiming(
        **{
            k: raw[k]
            for k in (
                "source_fps",
                "source_horizon",
                "action_stride",
                "output_chunk_length",
                "includes_current",
            )
        }
    )
    if timing.output_chunk_length != 100 or raw != timing.to_dict():
        raise ValueError(
            "Pi0.5 timing must preserve the declared physical span and output 100 points"
        )
    episodes = value.get("episodes")
    if (
        not isinstance(episodes, list)
        or not episodes
        or not any(r.get("split") == "train" for r in episodes)
    ):
        raise ValueError("Pi0.5 manifest requires training episodes")
    paths = []
    for record in episodes:
        relative = record.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
        ):
            raise ValueError("episode path must be relative to manifest")
        paths.append(relative)
        if record.get("split") not in ("train", "validation"):
            raise ValueError("invalid episode split")
        if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
            raise ValueError("empty episode prompt")
        if (
            isinstance(record.get("total_frames"), bool)
            or not isinstance(record.get("total_frames"), int)
            or record["total_frames"] < 1
        ):
            raise ValueError("invalid episode total_frames")
        if record.get("image_shapes") != value.get("image_shapes"):
            raise ValueError("episode camera shapes disagree with training contract")
        if _training_contract(record["contract"]) not in value.get(
            "training_contracts", []
        ):
            raise ValueError(
                "episode deployment contract disagrees with training contract"
            )
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate episode paths")
    contracts = value.get("training_contracts")
    if not contracts or value.get("training_contract") != contracts[0]:
        raise ValueError("invalid training contracts")
    if any(
        c.get("gripper_encoding") != contracts[0].get("gripper_encoding")
        for c in contracts
    ):
        raise ValueError("gripper encoding must match across training contracts")
    identity = canonical_hash(
        [
            {
                k: record[k]
                for k in ("metadata_sha256", "numeric_sha256", "total_frames")
            }
            for record in episodes
        ]
    )
    if value.get("data_identity_sha256") != identity:
        raise ValueError("data identity checksum mismatch")
    return value


def _camera_pose(poses, T_base_camera):
    camera_base = np.linalg.inv(np.asarray(T_base_camera, dtype=float))
    rotations = Rotation.from_matrix(camera_base[:3, :3]) * Rotation.from_quat(
        poses[:, [4, 5, 6, 3]]
    )
    positions = poses[:, :3] @ camera_base[:3, :3].T + camera_base[:3, 3]
    return positions, rotations


class Pi05Dataset:
    """Map-style numpy dataset; returns unnormalized, unpadded 14D samples.

    OpenPI subsequently normalizes the 14 physical dimensions, resizes/pads
    RGB images, tokenizes the task plus normalized state, then pads to 32D.
    No Torch, CUDA, robot SDK or JAX is imported by this module.
    """

    def __init__(self, manifest_path, split="train", include_images=True):
        if split not in ("train", "validation", "all"):
            raise ValueError("split must be train, validation, or all")
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = load_manifest(self.manifest_path)
        raw = self.manifest["action_timing"]
        self.timing = ActionTiming(
            **{
                k: raw[k]
                for k in (
                    "source_fps",
                    "source_horizon",
                    "action_stride",
                    "output_chunk_length",
                    "includes_current",
                )
            }
        )
        self.include_images = include_images
        self.episodes = [
            r
            for r in self.manifest["episodes"]
            if split == "all" or r["split"] == split
        ]
        self._numeric, self._paths = [], []
        for record in self.episodes:
            path = (self.manifest_path.parent / record["path"]).resolve()
            _, meta, arrays = _episode(
                path, self.timing, self.manifest["allow_synthetic_calibration"]
            )
            if (
                canonical_hash(meta) != record["metadata_sha256"]
                or _numeric_hash(arrays) != record["numeric_sha256"]
            ):
                raise ValueError(f"{path}: source changed since manifest export")
            if (
                meta["total_frames"] != record["total_frames"]
                or _contract(meta) != record["contract"]
            ):
                raise ValueError(f"{path}: source contract changed")
            self._numeric.append(arrays)
            self._paths.append(path)
        self._ends = np.cumsum([r["total_frames"] for r in self.episodes])
        self._stores = {}

    def __len__(self):
        return int(self._ends[-1]) if len(self._ends) else 0

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self._ends, index, side="right"))
        local = int(index - (self._ends[episode - 1] if episode else 0))
        record, arrays = self.episodes[episode], self._numeric[episode]
        indices = np.minimum(
            local + np.asarray(self.timing.support_indices), record["total_frames"] - 1
        )
        old_time = (
            np.asarray(self.timing.support_indices, dtype=float)
            / self.timing.source_fps
        )
        new_time = np.asarray(self.timing.output_offsets_s)
        states, actions = [], []
        for side in SIDES:
            transform = record["contract"]["extrinsics"][side]
            xyz, rotation = _camera_pose(
                arrays[f"{side}.obs_ee_pose"][local : local + 1], transform
            )
            states.append(
                np.r_[
                    xyz[0],
                    rotation.as_euler("ZYX")[0],
                    arrays[f"{side}.obs_gripper"][local],
                ]
            )
            xyz, rotation = _camera_pose(
                arrays[f"{side}.cmd_ee_pose"][indices], transform
            )
            interp_xyz = np.column_stack(
                [np.interp(new_time, old_time, xyz[:, axis]) for axis in range(3)]
            )
            interp_rotation = Slerp(old_time, rotation)(new_time).as_euler("ZYX")
            gripper = np.interp(
                new_time, old_time, arrays[f"{side}.cmd_gripper"][indices, 0]
            )
            actions.append(np.column_stack([interp_xyz, interp_rotation, gripper]))
        result = {
            "observation.state": np.concatenate(states).astype(np.float32),
            "actions": np.concatenate(actions, axis=1).astype(np.float32),
            "prompt": record["prompt"],
        }
        if self.include_images:
            if episode not in self._stores:
                self._stores[episode] = zarr.open_group(
                    str(self._paths[episode]), mode="r"
                )
            for camera in CAMERAS:
                # Zarr VariableLengthBytes scalar indexing can return a buffer
                # wrapper; a one-row slice yields the actual JPEG bytes.
                encoded = self._stores[episode][f"images.{camera}"][local : local + 1][
                    0
                ]
                if isinstance(encoded, np.ndarray):
                    encoded = encoded.item()
                bgr = cv2.imdecode(
                    np.frombuffer(bytes(encoded), dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if bgr is None:
                    raise ValueError(f"{self._paths[episode]}: invalid JPEG bytes")
                pixels = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if list(pixels.shape) != record["image_shapes"][camera]:
                    raise ValueError(
                        f"{self._paths[episode]}: decoded image shape disagrees"
                    )
                result[f"observation.images.{camera}"] = pixels
        return result


def compute_norm_stats(manifest_path, max_windows=None):
    """Exact per-dimension train-only stats, pooling all 100 action timesteps.

    Explicit max_windows deterministically subsamples training windows for smoke
    tests. Full computation holds numeric samples in memory; images are not read.
    The returned JSON-friendly arrays have 14 entries, never (100,14).
    """
    dataset = Pi05Dataset(manifest_path, split="train", include_images=False)
    if max_windows is not None and (
        isinstance(max_windows, bool)
        or not isinstance(max_windows, int)
        or max_windows < 1
    ):
        raise ValueError("max_windows must be a positive integer")
    count = len(dataset) if max_windows is None else min(len(dataset), max_windows)
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    samples = [dataset[int(index)] for index in indices]
    arrays = {
        "state": np.stack([sample["observation.state"] for sample in samples]).astype(
            np.float64
        ),
        "actions": np.concatenate([sample["actions"] for sample in samples]).astype(
            np.float64
        ),
    }
    return {
        key: {
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
        for key, values in arrays.items()
    }
