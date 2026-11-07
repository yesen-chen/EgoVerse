"""Explicitly synthetic recordings for the local Piper/PICO training recipes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from egomimic.rldb.zarr.zarr_writer import ZarrWriter


def _pose(xyz, ypr):
    return np.column_stack(
        [xyz, Rotation.from_euler("ZYX", ypr).as_quat()[:, [3, 0, 1, 2]]]
    )


def write_recording(root, kind, name, *, phase=0.0, offset=0.0, frames=24):
    """Independent deterministic synthetic source; never a copied real recording."""
    path = Path(root) / name / "episode_000000.zarr"
    t = np.linspace(0, 2 * np.pi, frames) + phase
    ns = np.int64(1700000000000000000) + np.rint(np.arange(frames) * 1e9 / 30).astype(
        np.int64
    )
    values = {"timestamp": ns.astype(np.float64) / 1e9}
    K = {"front_1": [[25, 0, 16, 0], [0, 25, 16, 0], [0, 0, 1, 0]]}
    metadata = {
        "pose_representation": "xyz_m+quat_wxyz",
        "reference_camera": "front_1",
        "conversion": {"source_dataset": f"synthetic-independent/{kind}/{name}"},
        "fixture": {"synthetic": True, "recording": name},
        "intrinsics": K,
    }
    if kind == "piper":
        matrices = {}
        for i, side in enumerate(("left", "right")):
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_euler(
                "ZYX", [0.2 * (i + 1), -0.1, 0.05]
            ).as_matrix()
            matrix[:3, 3] = [0.1 + i * 0.2, -0.1, 0.3]
            matrices[side] = matrix.tolist()
        calibration = {
            "calibration_id": f"synthetic-training-{name}",
            "synthetic": True,
            "reference_camera": "front_1",
            "extrinsics_convention": "T_base_camera",
            "extrinsics": matrices,
            "intrinsics": K,
        }
        metadata.update(
            {
                "schema": "piper_base_v3",
                "time_alignment": {"observation_to_send_s": 0.004},
                "embodiment": "piper_bimanual",
                "pose_coordinates": {
                    side: f"piper_{side}_base" for side in ("left", "right")
                },
                "extrinsics_convention": "T_base_camera",
                "extrinsics": matrices,
                "gripper_encoding": {"unit": "normalized_width", "open_m": 0.08},
                "calibration_status": "synthetic",
                "calibration": {"data": calibration, "path": "synthetic-fixture-only"},
            }
        )
    else:
        if kind != "pico_ego":
            raise ValueError(kind)
        values["timestamp_ns"] = ns
        values["obs_head_pose"] = _pose(
            np.column_stack([0.03 * np.sin(t), 0.02 * np.cos(t), np.zeros(frames)]),
            np.column_stack([0.04 * np.sin(t), 0.02 * np.cos(t), 0.01 * np.sin(t)]),
        )
        metadata.update(
            {
                "schema": "pico_ego_world_v1",
                "embodiment": "human_bimanual",
                "extrinsics": None,
                "pose_coordinates": {
                    "left": "tracking_world",
                    "right": "tracking_world",
                    "obs_head_pose": "T_world_left_optical_camera",
                },
                "pico_ego": {
                    "tool_frame": {"axis_convention": "piper_x"},
                    "source_tracking": {
                        "sha256": hashlib.sha256(
                            f"synthetic/{name}".encode()
                        ).hexdigest()
                    },
                    "gripper_encoding": {
                        "open_m": 0.08,
                        "normalized": "clip(distance,0,open_m)/open_m; fixed across episodes",
                    },
                },
            }
        )
    for side_index, side in enumerate(("left", "right")):
        fields = ("obs", "cmd") if kind == "piper" else ("obs",)
        for field in fields:
            lag = 0.1 if field == "cmd" else 0
            position = np.column_stack(
                [
                    offset + 0.2 * side_index + 0.08 * np.sin(t + lag),
                    0.04 * np.cos(t + lag),
                    0.9 + 0.03 * np.sin(2 * t + lag),
                ]
            )
            ypr = np.column_stack(
                [
                    0.1 * np.sin(t + lag),
                    0.08 * np.cos(t + lag),
                    0.06 * np.sin(2 * t + lag),
                ]
            )
            values[f"{side}.{field}_ee_pose"] = _pose(position, ypr)
            values[f"{side}.{field}_gripper"] = (
                0.4 + 0.2 * np.sin(t + side_index + lag)
            )[:, None]
        if kind == "pico_ego":
            values[f"{side}.obs_gripper_width_m"] = values[f"{side}.obs_gripper"] * 0.08
            values[f"{side}.hand_valid"] = np.ones(frames, dtype=bool)
            values[f"{side}.hand_filled"] = np.zeros(frames, dtype=bool)
    cameras = (
        ("front_1", "left_wrist", "right_wrist") if kind == "piper" else ("front_1",)
    )
    rng = np.random.default_rng(13)
    images = {
        f"images.{camera}": rng.integers(0, 255, (frames, 32, 32, 3), dtype=np.uint8)
        for camera in cameras
    }
    ZarrWriter(
        path,
        embodiment=metadata["embodiment"],
        fps=30,
        intrinsics=K,
        extrinsics=metadata["extrinsics"],
    ).write(
        numeric_data=values, image_data=images, metadata_override=metadata
    )
    return path


def write_training_roots(tmp_path):
    roots = {}
    for kind in ("piper", "pico_ego"):
        root = Path(tmp_path) / kind
        # Whole source groups are split by the recipe. Small phase differences
        # give independent fixtures without intentionally out-of-range validation.
        for index in range(4):
            write_recording(root, kind, f"recording_{index}", phase=index * 0.015)
        roots[kind] = root
    return roots
