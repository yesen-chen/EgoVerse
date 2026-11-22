"""NumPy-only PICO Ego windows for Pi0.5; source data is never retimed.

All future virtual TCP poses use the *current* optical camera frame. The
source window is selected before stride and endpoint-preserving interpolation
to 100 model points. Artificial endings can require complete future windows;
the retained tail remains available as supervision for earlier starts.
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
from egomimic.rldb.ego_boundary_policy import EgoBoundaryPolicy
from egomimic.robot.piper_pico.image_geometry import validate_image_geometry
from egomimic.robot.piper_pico.pi05_data import _camera_pose, canonical_hash

SCHEMA = "egoverse_pi05_ego_manifest_v1"
SIDES = ("left", "right")
CAMERAS = ("front_1",)
NUMERIC_KEYS = ("timestamp", "timestamp_ns", "obs_head_pose") + tuple(
    f"{side}.{field}"
    for side in SIDES
    for field in (
        "obs_ee_pose",
        "obs_gripper",
        "obs_gripper_width_m",
        "hand_valid",
        "hand_filled",
    )
)
PADDING = "per_episode_boundary_policy; all source rows retained"
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
    "coordinate_transform": "inverse(T_world_camera_at_current_observation) @ T_world_virtual_tcp",
    "state_source": "current_observed_virtual_tcp",
    "action_source": "future_observed_virtual_tcp",
    "virtual_tcp_axes": "piper_x",
    "camera_slots": {"front_1": "base_0_rgb"},
    "images": "RGB_uint8_HWC_already_processed; no_second_undistortion",
    "zero_gripper": "insert_zero_after_domain_normalization; raw_adapter_also_returns_zero",
}


def _manifest_hash(value):
    return canonical_hash({k: v for k, v in value.items() if k != "manifest_sha256"})


def _numeric_hash(arrays):
    digest = hashlib.sha256()
    for key in NUMERIC_KEYS:
        # Preserve integer nanoseconds; casting them to float loses source bits.
        dtype = (
            "<i8"
            if key == "timestamp_ns"
            else "?"
            if key.endswith(("hand_valid", "hand_filled"))
            else "<f8"
        )
        value = np.ascontiguousarray(arrays[key], dtype=dtype)
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _mode(value):
    if value not in ("zero", "width"):
        raise ValueError("gripper_mode must be zero or width")
    return value


def _timing(value):
    raw = value.get("action_timing", {})
    try:
        timing = ActionTiming(
            **{
                key: raw[key]
                for key in (
                    "source_fps",
                    "source_horizon",
                    "action_stride",
                    "output_chunk_length",
                    "includes_current",
                )
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid Ego action timing") from exc
    if timing.output_chunk_length != 100 or raw != timing.to_dict(
        episode_end_padding=PADDING
    ):
        raise ValueError("Ego timing must preserve source span and output 100 points")
    return timing


def _policy(value):
    raw = value.get("boundary_policy", {})
    try:
        policy = EgoBoundaryPolicy(
            raw["require_full_window_after_trim"],
            raw["require_full_window_after_inactivity"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid Ego boundary policy") from exc
    if raw != policy.to_dict():
        raise ValueError("inconsistent Ego boundary policy")
    return policy


def _contract(meta):
    pico = meta["pico_ego"]
    return {
        "intrinsics": copy.deepcopy(meta["intrinsics"]),
        "image_geometry": copy.deepcopy(meta["image_geometry"]),
        "image_space": meta["image_space"],
        "tool_frame": copy.deepcopy(pico["tool_frame"]),
        "gripper_encoding": copy.deepcopy(pico["gripper_encoding"]),
    }


def _episode(path, timing):
    store = zarr.open_group(str(path), mode="r")
    meta = dict(store.attrs)
    expected = {
        "schema": "pico_ego_world_v1",
        "embodiment": "human_bimanual",
        "pose_representation": "xyz_m+quat_wxyz",
        "reference_camera": "front_1",
        "pose_coordinates": {
            "left": "tracking_world",
            "right": "tracking_world",
            "obs_head_pose": "T_world_left_optical_camera",
        },
        "image_space": "processed",
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"{path}: {key} must be {value!r}")
    if meta.get("extrinsics") is not None:
        raise ValueError(
            f"{path}: Ego world data must not contain robot-base extrinsics"
        )
    pico = meta.get("pico_ego", {})
    if pico.get("tool_frame", {}).get("axis_convention") != "piper_x":
        raise ValueError(f"{path}: virtual TCP axis_convention must be piper_x")
    if not _sha(pico.get("source_tracking", {}).get("sha256")):
        raise ValueError(
            f"{path}: source_tracking.sha256 required for source recording split"
        )
    grip = pico.get("gripper_encoding", {})
    maximum = grip.get("open_m")
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not np.isfinite(maximum)
        or maximum <= 0
        or grip.get("normalized")
        != "clip(distance,0,open_m)/open_m; fixed across episodes"
    ):
        raise ValueError(f"{path}: invalid fixed-scale gripper_encoding")
    count, fps = meta.get("total_frames"), meta.get("fps")
    if not _positive_int(count):
        raise ValueError(f"{path}: invalid total_frames")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not np.isclose(fps, timing.source_fps, atol=1e-9, rtol=0)
    ):
        raise ValueError(f"{path}: fps disagrees with action timing")
    arrays, features = {}, meta.get("features", {})
    for key in NUMERIC_KEYS:
        shape = (
            (7,)
            if key.endswith("ee_pose") or key == "obs_head_pose"
            else (1,)
            if ".obs_gripper" in key
            else ()
        )
        if key not in store or tuple(features.get(key, {}).get("shape", [])) != shape:
            raise ValueError(f"{path}: missing or invalid {key}")
        values = np.asarray(store[key][:count])
        if (
            values.shape != (count, *shape)
            or values.dtype.kind not in "biuf"
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"{path}: invalid finite numeric array {key}")
        if shape == (7,) and not np.allclose(
            np.linalg.norm(values[:, 3:], axis=1), 1, atol=1e-3, rtol=0
        ):
            raise ValueError(f"{path}: non-unit quaternion in {key}")
        arrays[key] = values
    ns = arrays["timestamp_ns"]
    if (
        ns.dtype.kind not in "iu"
        or not np.allclose(
            arrays["timestamp"], ns.astype(float) / 1e9, atol=5e-7, rtol=0
        )
        or np.any(np.diff(ns) <= 0)
        or not np.allclose(np.diff(ns), 1e9 / fps, atol=2, rtol=0)
    ):
        raise ValueError(
            f"{path}: source timestamp grid disagrees with fps or timestamp_ns"
        )
    for side in SIDES:
        width, normalized = (
            arrays[f"{side}.obs_gripper_width_m"],
            arrays[f"{side}.obs_gripper"],
        )
        if (
            np.any(normalized < -1e-6)
            or np.any(normalized > 1 + 1e-6)
            or not np.allclose(normalized, width / maximum, atol=1e-6, rtol=1e-5)
        ):
            raise ValueError(f"{path}: {side} gripper disagrees with fixed open_m")
        valid, filled = arrays[f"{side}.hand_valid"], arrays[f"{side}.hand_filled"]
        if (
            valid.dtype.kind != "b"
            or filled.dtype.kind != "b"
            or not np.all(valid | filled)
            or np.any(valid & filled)
        ):
            raise ValueError(f"{path}: invalid {side} validity/held flags")
    info = features.get("images.front_1", {})
    shape = info.get("shape", [])
    if (
        "images.front_1" not in store
        or info.get("dtype") != "jpeg"
        or len(shape) != 3
        or shape[-1] != 3
        or not all(_positive_int(n) for n in shape)
        or store["images.front_1"].ndim != 1
        or store["images.front_1"].shape[0] < count
    ):
        raise ValueError(f"{path}: invalid front_1 JPEG rows or dimensions")
    geometry = meta.get("image_geometry", {}).get("front_1")
    validate_image_geometry(geometry)
    K = np.asarray(meta.get("intrinsics", {}).get("front_1"), dtype=float)
    if (
        geometry.get("K_output") is None
        or K.shape != (3, 3)
        or not np.allclose(K, geometry["K_output"], atol=1e-8, rtol=0)
        or geometry["output_size"] != [shape[1], shape[0]]
    ):
        raise ValueError(f"{path}: processed front_1 intrinsics or dimensions disagree")
    if (
        geometry["operation"] == "undistort_resize"
        and meta.get("distortion_coefficients", {}).get("front_1") != [0.0] * 5
    ):
        raise ValueError(f"{path}: undistorted output must declare zero distortion")
    return meta, arrays


def _identity(records):
    return canonical_hash(
        [
            {
                k: r[k]
                for k in (
                    "metadata_sha256",
                    "numeric_sha256",
                    "total_frames",
                    "eligible_starts",
                    "source_recording_sha256",
                )
            }
            for r in records
        ]
    )


def export_ego_manifest(
    episode_paths,
    output_path,
    *,
    source_fps=30.0,
    source_horizon=30,
    action_stride=1,
    validation_ratio=0.1,
    seed=42,
    prompt=None,
    gripper_mode="zero",
    require_full_window_after_trim=True,
    require_full_window_after_inactivity=True,
):
    """Export references only; split entire source recordings before training."""
    timing = ActionTiming(source_fps, source_horizon, action_stride, 100)
    policy = EgoBoundaryPolicy(
        require_full_window_after_trim, require_full_window_after_inactivity
    )
    mode = _mode(gripper_mode)
    if (
        isinstance(validation_ratio, bool)
        or not isinstance(validation_ratio, (int, float))
        or not np.isfinite(validation_ratio)
        or not 0 <= validation_ratio < 1
    ):
        raise ValueError("validation_ratio must be in [0,1)")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    paths = sorted(Path(p).resolve() for p in episode_paths)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("episode_paths must be nonempty and unique")
    records, contracts, common_shapes, common_gripper = [], [], None, None
    for path in paths:
        meta, arrays = _episode(path, timing)
        reason, required = policy.resolve(meta, episode_path=path)
        eligible = (
            meta["total_frames"] - timing.source_horizon + 1
            if required
            else meta["total_frames"]
        )
        if eligible < 1:
            raise ValueError(
                f"{path}: {reason} fragment has {meta['total_frames']} rows but full future window needs {timing.source_horizon}"
            )
        contract = _contract(meta)
        shapes = {"front_1": meta["features"]["images.front_1"]["shape"]}
        if common_shapes is None:
            common_shapes, common_gripper = shapes, contract["gripper_encoding"]
        elif shapes != common_shapes or contract["gripper_encoding"] != common_gripper:
            raise ValueError(
                f"{path}: image shape/gripper encoding differs across episodes"
            )
        if contract not in contracts:
            contracts.append(contract)
        text = prompt if prompt is not None else meta.get("task_description", "")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}: task prompt required; set --prompt")
        records.append(
            {
                "path": os.path.relpath(path, destination.parent),
                "total_frames": meta["total_frames"],
                "eligible_starts": eligible,
                "end_reason": reason,
                "requires_full_window": required,
                "source_recording_sha256": meta["pico_ego"]["source_tracking"][
                    "sha256"
                ],
                "metadata_sha256": canonical_hash(meta),
                "numeric_sha256": _numeric_hash(arrays),
                "contract": contract,
                "image_shapes": shapes,
                "prompt": text.strip(),
            }
        )
    groups = sorted({r["source_recording_sha256"] for r in records})
    if validation_ratio and len(groups) < 2:
        raise ValueError(
            "a validation split requires at least two source recordings; use ratio 0 for a smoke test"
        )
    n_valid = (
        min(len(groups) - 1, max(1, int(len(groups) * validation_ratio)))
        if validation_ratio
        else 0
    )
    validation = {
        groups[i]
        for i in np.random.default_rng(seed).permutation(len(groups))[:n_valid]
    }
    for record in records:
        record["split"] = (
            "validation" if record["source_recording_sha256"] in validation else "train"
        )
    result = {
        "schema": SCHEMA,
        "action_timing": timing.to_dict(episode_end_padding=PADDING),
        "action_contract": copy.deepcopy(ACTION_CONTRACT),
        "gripper_mode": mode,
        "boundary_policy": policy.to_dict(),
        "image_shapes": common_shapes,
        "training_contract": contracts[0],
        "training_contracts": contracts,
        "split": {
            "unit": "source_recording_sha256",
            "validation_ratio": float(validation_ratio),
            "seed": seed,
        },
        "episodes": records,
        "data_identity_sha256": _identity(records),
        "identity_scope": "metadata_and_numeric_rows; JPEG bytes are referenced, not re-encoded or content-hashed",
        "source_time_policy": "preserve stored timestamps; no physical slowdown or delay correction in adapter",
    }
    result["manifest_sha256"] = _manifest_hash(result)
    validate_ego_manifest(result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return destination


def validate_ego_manifest(value):
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported Ego Pi0.5 manifest schema")
    if value.get("manifest_sha256") != _manifest_hash(value):
        raise ValueError("Ego Pi0.5 manifest checksum mismatch")
    if value.get("action_contract") != ACTION_CONTRACT:
        raise ValueError("unsupported Ego Pi0.5 action contract")
    timing, policy = _timing(value), _policy(value)
    _mode(value.get("gripper_mode"))
    episodes = value.get("episodes")
    if (
        not isinstance(episodes, list)
        or not episodes
        or not all(isinstance(r, dict) for r in episodes)
        or not any(r.get("split") == "train" for r in episodes)
    ):
        raise ValueError("Ego manifest requires training episodes")
    if value.get("split", {}).get("unit") != "source_recording_sha256":
        raise ValueError("Ego split must use source recordings")
    contracts = value.get("training_contracts")
    if (
        not isinstance(contracts, list)
        or not contracts
        or value.get("training_contract") != contracts[0]
    ):
        raise ValueError("invalid Ego training contracts")
    shapes = value.get("image_shapes")
    if (
        not isinstance(shapes, dict)
        or set(shapes) != {"front_1"}
        or len(shapes["front_1"]) != 3
        or shapes["front_1"][-1] != 3
        or not all(_positive_int(n) for n in shapes["front_1"])
    ):
        raise ValueError("invalid Ego image shapes")
    paths, groups = set(), {}
    for record in episodes:
        relative = record.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or relative in paths
        ):
            raise ValueError("episode paths must be unique and relative to manifest")
        paths.add(relative)
        group, split = record.get("source_recording_sha256"), record.get("split")
        if not _sha(group) or split not in ("train", "validation"):
            raise ValueError("invalid Ego source identity or split")
        if group in groups and groups[group] != split:
            raise ValueError(
                "source recording fragments must not cross train/validation"
            )
        groups[group] = split
        reason = record.get("end_reason")
        meta = (
            {}
            if reason == "legacy"
            else {
                "preprocessing": {
                    "schema": "ego_preprocessing_v1",
                    "end_reason": reason,
                }
            }
        )
        _, required = policy.resolve(meta)
        count, eligible = record.get("total_frames"), record.get("eligible_starts")
        if (
            not _positive_int(count)
            or not _positive_int(eligible)
            or record.get("requires_full_window") is not required
            or eligible != (count - timing.source_horizon + 1 if required else count)
        ):
            raise ValueError(
                "Ego eligible starts disagree with source horizon/boundary policy"
            )
        if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
            raise ValueError("empty Ego task prompt")
        if not _sha(record.get("metadata_sha256")) or not _sha(
            record.get("numeric_sha256")
        ):
            raise ValueError("invalid Ego source checksums")
        if (
            record.get("image_shapes") != shapes
            or record.get("contract") not in contracts
        ):
            raise ValueError("Ego episode image/training contract differs")
    if value.get("data_identity_sha256") != _identity(episodes):
        raise ValueError("Ego data identity checksum mismatch")
    return value


def load_ego_manifest(path):
    return validate_ego_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


class EgoPi05Dataset:
    """Raw 14D samples; downstream normalizes by domain then zeros zero-mode grippers."""

    def __init__(self, manifest_path, split="train", include_images=True):
        if split not in ("train", "validation", "all"):
            raise ValueError("split must be train, validation, or all")
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = load_ego_manifest(self.manifest_path)
        self.timing = _timing(self.manifest)
        self.boundary_policy = _policy(self.manifest)
        self.gripper_mode = self.manifest["gripper_mode"]
        self.include_images = include_images
        self.episodes = [
            r
            for r in self.manifest["episodes"]
            if split == "all" or r["split"] == split
        ]
        self._numeric, self._paths, self._stores = [], [], {}
        for record in self.episodes:
            path = (self.manifest_path.parent / record["path"]).resolve()
            meta, arrays = _episode(path, self.timing)
            if (
                canonical_hash(meta) != record["metadata_sha256"]
                or _numeric_hash(arrays) != record["numeric_sha256"]
            ):
                raise ValueError(f"{path}: source changed since manifest export")
            reason, required = self.boundary_policy.resolve(meta, episode_path=path)
            if (
                meta["total_frames"] != record["total_frames"]
                or _contract(meta) != record["contract"]
                or reason != record["end_reason"]
                or required is not record["requires_full_window"]
                or meta["pico_ego"]["source_tracking"]["sha256"]
                != record["source_recording_sha256"]
            ):
                raise ValueError(f"{path}: source contract changed")
            self._paths.append(path)
            self._numeric.append(arrays)
        self._ends = np.cumsum([r["eligible_starts"] for r in self.episodes])

    def __len__(self):
        return int(self._ends[-1]) if len(self._ends) else 0

    def __getitem__(self, index):
        if (
            isinstance(index, bool)
            or not isinstance(index, (int, np.integer))
            or not 0 <= index < len(self)
        ):
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
        current_camera = arrays["obs_head_pose"][local]
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(current_camera[[4, 5, 6, 3]]).as_matrix()
        transform[:3, 3] = current_camera[:3]
        states, actions = [], []
        for side in SIDES:
            xyz, rotation = _camera_pose(
                arrays[f"{side}.obs_ee_pose"][local : local + 1], transform
            )
            width = (
                arrays[f"{side}.obs_gripper"][:, 0]
                if self.gripper_mode == "width"
                else np.zeros(record["total_frames"])
            )
            states.append(np.r_[xyz[0], rotation.as_euler("ZYX")[0], width[local]])
            xyz, rotation = _camera_pose(
                arrays[f"{side}.obs_ee_pose"][indices], transform
            )
            interpolated_xyz = np.column_stack(
                [np.interp(new_time, old_time, xyz[:, axis]) for axis in range(3)]
            )
            interpolated_rotation = Slerp(old_time, rotation)(new_time).as_euler("ZYX")
            actions.append(
                np.column_stack(
                    [
                        interpolated_xyz,
                        interpolated_rotation,
                        np.interp(new_time, old_time, width[indices]),
                    ]
                )
            )
        result = {
            "observation.state": np.concatenate(states).astype(np.float32),
            "actions": np.concatenate(actions, axis=1).astype(np.float32),
            "prompt": record["prompt"],
            "domain": "ego",
            "gripper_mode": self.gripper_mode,
        }
        if self.include_images:
            if episode not in self._stores:
                self._stores[episode] = zarr.open_group(
                    str(self._paths[episode]), mode="r"
                )
            encoded = self._stores[episode]["images.front_1"][local : local + 1][0]
            if isinstance(encoded, np.ndarray):
                encoded = encoded.item()
            bgr = cv2.imdecode(
                np.frombuffer(bytes(encoded), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if bgr is None:
                raise ValueError(f"{self._paths[episode]}: invalid JPEG bytes")
            pixels = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if list(pixels.shape) != record["image_shapes"]["front_1"]:
                raise ValueError(
                    f"{self._paths[episode]}: decoded image shape disagrees"
                )
            result["observation.images.front_1"] = pixels
        return result
