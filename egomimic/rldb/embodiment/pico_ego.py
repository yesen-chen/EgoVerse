"""PICO human recordings anchored to the current left optical-camera pose.

The converter has already applied tracking/fit timing and head-to-camera
extrinsics. This loader only changes world coordinates into the current camera;
it never applies the fit again or moves the action frame along future cameras.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import zarr

from egomimic.rldb.action_timing import ActionTiming
from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.embodiment.piper import coerce_timing
from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    ConcatKeys,
    DeleteKeys,
    InterpolateLinear,
    InterpolatePose,
    NumpyToTensor,
    PoseCoordinateFrameTransform,
    XYZWXYZ_to_XYZYPR,
)
from egomimic.rldb.zarr.piper_dataset import _intrinsics
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset


def _gripper_mode(value):
    if value not in ("zero", "width"):
        raise ValueError("gripper_mode must be 'zero' or 'width'")
    return value


class PicoEgo(Embodiment):
    @classmethod
    def get_keymap(
        cls,
        keymap_mode: str = "cartesian",
        *,
        timing: ActionTiming | Mapping,
        gripper_mode: str = "zero",
        norm_mode: bool = False,
        annotation_key: str | None = None,
    ) -> dict:
        if keymap_mode != "cartesian":
            raise ValueError("PicoEgo supports keymap_mode='cartesian' only")
        spec, width = coerce_timing(timing), _gripper_mode(gripper_mode) == "width"
        keys = {
            "obs_head_pose": {"key_type": "proprio_keys", "zarr_key": "obs_head_pose"}
        }
        if not norm_mode:
            keys[cls.VIZ_IMAGE_KEY] = {
                "key_type": "camera_keys",
                "zarr_key": "images.front_1",
            }
        for side in ("left", "right"):
            for field in ("ee_pose", "gripper") if width else ("ee_pose",):
                keys[f"{side}.obs_{field}"] = {
                    "key_type": "proprio_keys",
                    "zarr_key": f"{side}.obs_{field}",
                }
                keys[f"{side}.action_{field}"] = {
                    "key_type": "action_keys",
                    "zarr_key": f"{side}.obs_{field}",
                    "horizon": spec.source_horizon,
                }
        if annotation_key and not norm_mode:
            keys[annotation_key] = {
                "key_type": "annotation_keys",
                "zarr_key": annotation_key,
            }
        return keys

    @classmethod
    def _get_keymap(cls, keymap_mode: str):
        raise ValueError("PicoEgo.get_keymap requires an explicit timing argument")

    @staticmethod
    def get_transform_list(
        mode: str = "cartesian",
        *,
        timing: ActionTiming | Mapping,
        gripper_mode: str = "zero",
    ) -> list:
        if mode != "cartesian":
            raise ValueError("PicoEgo supports mode='cartesian' only")
        spec, width = coerce_timing(timing), _gripper_mode(gripper_mode) == "width"
        transforms, states, actions = [], [], []
        for side in ("left", "right"):
            obs, action = f"{side}.obs_ee_pose", f"{side}.action_ee_pose"
            transforms.extend(
                [
                    ActionChunkCoordinateFrameTransform(
                        "obs_head_pose", action, action, mode="xyzwxyz"
                    ),
                    PoseCoordinateFrameTransform(
                        "obs_head_pose", obs, obs, mode="xyzwxyz"
                    ),
                    InterpolatePose(
                        spec.output_chunk_length,
                        action,
                        action,
                        stride=spec.action_stride,
                        mode="xyzwxyz",
                    ),
                    XYZWXYZ_to_XYZYPR([obs, action]),
                ]
            )
            states.append(obs)
            actions.append(action)
            if width:
                obs_width, action_width = (
                    f"{side}.obs_gripper",
                    f"{side}.action_gripper",
                )
                transforms.append(
                    InterpolateLinear(
                        spec.output_chunk_length,
                        action_width,
                        action_width,
                        stride=spec.action_stride,
                    )
                )
                states.append(obs_width)
                actions.append(action_width)
        transforms.extend(
            [
                ConcatKeys(states, "observations.state.ee_pose", delete_old_keys=True),
                ConcatKeys(actions, "actions_cartesian", delete_old_keys=True),
                DeleteKeys(["obs_head_pose"]),
                NumpyToTensor(["observations.state.ee_pose", "actions_cartesian"]),
            ]
        )
        # zero keeps twelve pose dimensions until the shared head inserts zero
        # grippers AFTER pose normalization, as in the original Human path.
        return transforms


def validate_pico_ego_metadata(
    metadata, *, episode_path="<episode>", expected_axis_convention="piper_x"
):
    label = str(episode_path)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{label}: metadata must be a mapping")
    expected = {
        "embodiment": "human_bimanual",
        "schema": "pico_ego_world_v1",
        "pose_representation": "xyz_m+quat_wxyz",
        "reference_camera": "front_1",
        "pose_coordinates": {
            "left": "tracking_world",
            "right": "tracking_world",
            "obs_head_pose": "T_world_left_optical_camera",
        },
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{label}: {key} must be {value!r}")
    pico = metadata.get("pico_ego")
    if not isinstance(pico, Mapping):
        raise ValueError(f"{label}: missing pico_ego provenance")
    if expected_axis_convention not in ("piper_x", "piper"):
        raise ValueError(f"{label}: expected_axis_convention must be piper_x or piper")
    tool = pico.get("tool_frame")
    actual_axes = tool.get("axis_convention") if isinstance(tool, Mapping) else None
    if actual_axes != expected_axis_convention:
        raise ValueError(
            f"{label}: tool_frame.axis_convention {actual_axes!r} does not match "
            f"expected_axis_convention={expected_axis_convention!r}; reconvert with "
            "the intended axes rather than silently rotating training poses"
        )
    encoding = pico.get("gripper_encoding")
    maximum = encoding.get("open_m") if isinstance(encoding, Mapping) else None
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, Real)
        or not np.isfinite(maximum)
        or maximum <= 0
    ):
        raise ValueError(
            f"{label}: gripper_encoding.open_m must be finite and positive"
        )
    if (
        encoding.get("normalized")
        != "clip(distance,0,open_m)/open_m; fixed across episodes"
    ):
        raise ValueError(f"{label}: unsupported gripper normalization convention")
    if metadata.get("extrinsics") is not None:
        raise ValueError(
            f"{label}: world PICO data must not supply robot-base extrinsics"
        )
    if metadata.get("intrinsics") is None:
        raise ValueError(f"{label}: front_1 intrinsics are required")
    _intrinsics(metadata["intrinsics"], f"{label}: intrinsics")
    return float(maximum)


def _validate_pico_store(store, metadata, label, expected_axis_convention="piper_x"):
    maximum = validate_pico_ego_metadata(
        metadata, episode_path=label, expected_axis_convention=expected_axis_convention
    )
    count, fps = metadata.get("total_frames"), metadata.get("fps")
    if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
        raise ValueError(f"{label}: total_frames must be a positive integer")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, Real)
        or not np.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError(f"{label}: fps must be finite and positive")
    features = metadata.get("features")
    if not isinstance(features, Mapping):
        raise ValueError(f"{label}: features mapping is required")

    def read(key, shape):
        info = features.get(key)
        if key not in store or not isinstance(info, Mapping):
            raise ValueError(f"{label}: missing required array {key}")
        arr = store[key]
        if (
            arr.ndim != len(shape) + 1
            or arr.shape[0] < count
            or arr.shape[1:] != shape
            or tuple(info.get("shape", [])) != shape
        ):
            raise ValueError(f"{label}: invalid shape or insufficient rows for {key}")
        values = arr[:count]
        if values.dtype.kind not in "biuf" or not np.isfinite(values).all():
            raise ValueError(f"{label}: {key} must contain finite numeric values")
        return values

    timestamp, timestamp_ns = read("timestamp", ()), read("timestamp_ns", ())
    if timestamp_ns.dtype.kind not in "iu":
        raise ValueError(f"{label}: timestamp_ns must use integer nanoseconds")
    if not np.allclose(
        timestamp, timestamp_ns.astype(np.float64) / 1e9, rtol=0, atol=5e-7
    ):
        raise ValueError(f"{label}: timestamp and timestamp_ns disagree")
    if count > 1:
        delta = np.diff(timestamp_ns)
        if np.any(delta <= 0) or not np.allclose(delta, 1e9 / fps, rtol=0, atol=2):
            raise ValueError(f"{label}: timestamp grid disagrees with stored fps")
    for key in ("obs_head_pose", "left.obs_ee_pose", "right.obs_ee_pose"):
        values = read(key, (7,))
        if not np.allclose(np.linalg.norm(values[:, 3:], axis=1), 1, atol=1e-3, rtol=0):
            raise ValueError(f"{label}: {key} contains non-unit quaternions")
    for side in ("left", "right"):
        width = read(f"{side}.obs_gripper_width_m", (1,))
        normalized = read(f"{side}.obs_gripper", (1,))
        if (
            np.any(normalized < -1e-6)
            or np.any(normalized > 1 + 1e-6)
            or not np.allclose(normalized, width / maximum, rtol=1e-5, atol=1e-6)
        ):
            raise ValueError(
                f"{label}: {side} normalized width disagrees with fixed open_m"
            )
        valid, filled = read(f"{side}.hand_valid", ()), read(f"{side}.hand_filled", ())
        if (
            valid.dtype.kind != "b"
            or filled.dtype.kind != "b"
            or not np.all(valid | filled)
            or np.any(valid & filled)
        ):
            raise ValueError(
                f"{label}: {side} validity/held flags are invalid; reconvert and split gaps"
            )
    info = features.get("images.front_1", {})
    if (
        "images.front_1" not in store
        or not isinstance(info, Mapping)
        or info.get("dtype") != "jpeg"
    ):
        raise ValueError(f"{label}: missing front_1 JPEG image array")
    shape = info.get("shape", [])
    if (
        len(shape) != 3
        or shape[-1] != 3
        or any(not isinstance(x, Integral) or x <= 0 for x in shape)
    ):
        raise ValueError(f"{label}: invalid front_1 image shape")
    if store["images.front_1"].ndim != 1 or store["images.front_1"].shape[0] < count:
        raise ValueError(f"{label}: insufficient front_1 image rows")


def preflight_pico_ego_episode(path, *, expected_axis_convention="piper_x"):
    store = zarr.open_group(str(path), mode="r")
    metadata = dict(store.attrs)
    _validate_pico_store(store, metadata, str(path), expected_axis_convention)
    return metadata


class PicoEgoDataset(ZarrDataset):
    strict_read = True

    def __init__(
        self,
        Episode_path,
        key_map,
        transform_list=None,
        *,
        gripper_mode="zero",
        expected_axis_convention="piper_x",
    ):
        self.gripper_mode = _gripper_mode(gripper_mode)
        self.expected_axis_convention = expected_axis_convention
        super().__init__(
            Path(Episode_path), copy.deepcopy(key_map), copy.deepcopy(transform_list)
        )
        _validate_pico_store(
            self.episode_reader._store,
            self.metadata,
            str(Episode_path),
            expected_axis_convention,
        )
        width_keys = {
            f"{side}.{kind}_gripper"
            for side in ("left", "right")
            for kind in ("obs", "action")
        }
        present = width_keys & set(key_map)
        if (self.gripper_mode == "width" and present != width_keys) or (
            self.gripper_mode == "zero" and present
        ):
            raise ValueError(
                f"{Episode_path}: keymap disagrees with gripper_mode={gripper_mode!r}"
            )

    def __getitem__(self, idx, **kwargs):
        if (
            isinstance(idx, bool)
            or not isinstance(idx, Integral)
            or idx < 0
            or idx >= len(self)
        ):
            raise IndexError(
                f"{self.episode_path}: frame index {idx!r} outside episode"
            )
        sample = super().__getitem__(idx, **kwargs)
        if self.transform:
            dimensions = 12 if self.gripper_mode == "zero" else 14
            if (
                sample["observations.state.ee_pose"].shape != (dimensions,)
                or sample["actions_cartesian"].ndim != 2
                or sample["actions_cartesian"].shape[-1] != dimensions
            ):
                raise ValueError(
                    f"{self.episode_path}: transforms disagree with gripper_mode={self.gripper_mode!r}"
                )
        return sample
