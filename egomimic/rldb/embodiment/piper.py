"""Piper's base-frame recordings -> current front_1 optical-camera frame.

Calibration is supplied by each PiperDataset instance. No Eva calibration or
class-level mutable extrinsics participate in this pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping

from egomimic.rldb.action_timing import ActionTiming
from egomimic.rldb.embodiment.embodiment import Embodiment
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


def coerce_timing(timing: ActionTiming | Mapping) -> ActionTiming:
    """Accept explicit constructor settings, never infer a window or frequency."""
    if isinstance(timing, ActionTiming):
        return timing
    if isinstance(timing, Mapping):
        return ActionTiming(**timing)
    raise ValueError("timing must be an ActionTiming or explicit settings mapping")


class Piper(Embodiment):
    @classmethod
    def get_keymap(
        cls,
        keymap_mode: str = "cartesian",
        *,
        timing: ActionTiming | Mapping,
        norm_mode: bool = False,
        annotation_key: str | None = None,
    ) -> dict:
        if keymap_mode != "cartesian":
            raise ValueError("Piper supports keymap_mode='cartesian' only")
        spec = coerce_timing(timing)
        keys = {}
        if not norm_mode:
            for name, model_name in (
                ("front_1", cls.VIZ_IMAGE_KEY),
                ("left_wrist", "observations.images.left_wrist_img"),
                ("right_wrist", "observations.images.right_wrist_img"),
            ):
                keys[model_name] = {
                    "key_type": "camera_keys",
                    "zarr_key": f"images.{name}",
                }
        for side in ("left", "right"):
            for field in ("obs_ee_pose", "obs_gripper", "cmd_ee_pose", "cmd_gripper"):
                name = f"{side}.{field}"
                command = field.startswith("cmd_")
                keys[name] = {
                    "key_type": "action_keys" if command else "proprio_keys",
                    "zarr_key": name,
                }
                if command:
                    keys[name]["horizon"] = spec.source_horizon
        if annotation_key and not norm_mode:
            keys[annotation_key] = {
                "key_type": "annotation_keys",
                "zarr_key": annotation_key,
            }
        return keys

    @classmethod
    def _get_keymap(cls, keymap_mode: str):
        raise ValueError("Piper.get_keymap requires an explicit timing argument")

    @staticmethod
    def get_transform_list(
        mode: str = "cartesian", *, timing: ActionTiming | Mapping
    ) -> list:
        if mode != "cartesian":
            raise ValueError("Piper supports mode='cartesian' only")
        spec = coerce_timing(timing)
        transforms = []
        for side in ("left", "right"):
            camera_pose = f"piper.{side}.T_base_camera_pose"
            command = f"{side}.cmd_ee_pose"
            observation = f"{side}.obs_ee_pose"
            gripper = f"{side}.cmd_gripper"
            transforms.extend(
                [
                    ActionChunkCoordinateFrameTransform(
                        camera_pose, command, command, mode="xyzwxyz"
                    ),
                    PoseCoordinateFrameTransform(
                        camera_pose, observation, observation, mode="xyzwxyz"
                    ),
                    InterpolatePose(
                        spec.output_chunk_length,
                        command,
                        command,
                        stride=spec.action_stride,
                        mode="xyzwxyz",
                    ),
                    InterpolateLinear(
                        spec.output_chunk_length,
                        gripper,
                        gripper,
                        stride=spec.action_stride,
                    ),
                    XYZWXYZ_to_XYZYPR([command, observation]),
                ]
            )
        transforms.extend(
            [
                ConcatKeys(
                    [
                        "left.obs_ee_pose",
                        "left.obs_gripper",
                        "right.obs_ee_pose",
                        "right.obs_gripper",
                    ],
                    "observations.state.ee_pose",
                    delete_old_keys=True,
                ),
                ConcatKeys(
                    [
                        "left.cmd_ee_pose",
                        "left.cmd_gripper",
                        "right.cmd_ee_pose",
                        "right.cmd_gripper",
                    ],
                    "actions_cartesian",
                    delete_old_keys=True,
                ),
                DeleteKeys(
                    [f"piper.{side}.T_base_camera_pose" for side in ("left", "right")]
                ),
                NumpyToTensor(["observations.state.ee_pose", "actions_cartesian"]),
            ]
        )
        return transforms
