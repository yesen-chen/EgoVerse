"""SI units to Piper SDK integers.

Joints are 0.001 deg, gripper and cartesian position are 0.001 mm, and
cartesian rotation is 0.001 deg. The pose vector is
``[x, y, z, roll, pitch, yaw]`` in the Piper base frame. Euler angles are
not wrapped: the firmware accepts a continuous branch.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

JOINT_FACTOR: float = 57324.840764
GRIPPER_FACTOR: float = 1_000_000.0
EE_POS_FACTOR: float = 1_000_000.0
EE_ROT_FACTOR: float = 1000.0 * 180.0 / math.pi
JOINT_VEL_FACTOR: float = 1000.0
EFFORT_FACTOR: float = 1000.0

GRIPPER_TORQUE_RAW: int = 1000
GRIPPER_MAX_M: float = 0.08


def joints_sdk_to_rad(joints_sdk) -> np.ndarray:
    return np.asarray(joints_sdk, dtype=np.float64) / JOINT_FACTOR


def rad_to_joints_sdk(joints_rad) -> list[int]:
    return [int(round(float(q) * JOINT_FACTOR)) for q in np.asarray(joints_rad)]


def gripper_sdk_to_m(gripper_sdk) -> float:
    return float(gripper_sdk) / GRIPPER_FACTOR


def m_to_gripper_sdk(gripper_m) -> int:
    return int(round(float(gripper_m) * GRIPPER_FACTOR))


def joint_vels_sdk_to_rad_s(vels_sdk) -> np.ndarray:
    return np.asarray(vels_sdk, dtype=np.float64) / JOINT_VEL_FACTOR


def efforts_sdk_to_nm(efforts_sdk) -> np.ndarray:
    return np.asarray(efforts_sdk, dtype=np.float64) / EFFORT_FACTOR


def endpose_sdk_to_pose(end_pose) -> np.ndarray:
    return np.array(
        [
            end_pose.X_axis / EE_POS_FACTOR,
            end_pose.Y_axis / EE_POS_FACTOR,
            end_pose.Z_axis / EE_POS_FACTOR,
            end_pose.RX_axis / EE_ROT_FACTOR,
            end_pose.RY_axis / EE_ROT_FACTOR,
            end_pose.RZ_axis / EE_ROT_FACTOR,
        ],
        dtype=np.float64,
    )


def pose_to_endpose_sdk(pose) -> list[int]:
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.shape != (6,):
        raise ValueError(f"pose must have 6 values, got {pose.shape}")
    return [
        int(round(pose[0] * EE_POS_FACTOR)),
        int(round(pose[1] * EE_POS_FACTOR)),
        int(round(pose[2] * EE_POS_FACTOR)),
        int(round(pose[3] * EE_ROT_FACTOR)),
        int(round(pose[4] * EE_ROT_FACTOR)),
        int(round(pose[5] * EE_ROT_FACTOR)),
    ]


def continuous_euler_xyz(rotation: Rotation, reference_rpy) -> np.ndarray:
    """XYZ Euler branch closest to ``reference_rpy``."""
    reference = np.asarray(reference_rpy, dtype=np.float64).reshape(-1)
    if reference.shape != (3,) or not np.all(np.isfinite(reference)):
        raise ValueError("reference_rpy must contain three finite values")
    canonical = rotation.as_euler("xyz")
    alternate = np.array(
        [canonical[0] + np.pi, np.pi - canonical[1], canonical[2] + np.pi],
        dtype=np.float64,
    )
    candidates = []
    for branch in (canonical, alternate):
        turns = np.round((reference - branch) / (2.0 * np.pi))
        candidates.append(branch + turns * (2.0 * np.pi))
    return min(candidates, key=lambda value: float(np.linalg.norm(value - reference)))


def trigger_to_gripper_m(
    trigger: float,
    gripper_min: float = 0.0,
    gripper_max: float = GRIPPER_MAX_M,
) -> float:
    """Released trigger is open. Pressed trigger is closed."""
    clipped = float(np.clip(trigger, 0.0, 1.0))
    return float(gripper_max + clipped * (gripper_min - gripper_max))
