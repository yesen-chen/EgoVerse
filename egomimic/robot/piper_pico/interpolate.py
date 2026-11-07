"""Causal alignment of one bimanual observation at a query time.

The query time is the moment the observation is taken; collection and live
inference use the same rule. Each camera contributes its latest frame stamped
at or before every query time, never a later one. Position is linear and
rotation is slerp on the rotation vector. Robot samples after the query time
are ignored, and query times outside the remaining samples are clipped.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


def _increasing(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    values = np.asarray(values)
    if times.shape[0] != values.shape[0]:
        raise ValueError("timestamps and values have different lengths")
    if times.shape[0] == 0:
        raise ValueError("no samples")
    keep = np.ones(times.shape[0], dtype=bool)
    keep[1:] = np.diff(times) > 1e-9
    times = times[keep]
    values = values[keep]
    if times.shape[0] == 1:
        times = np.array([times[0] - 1e-3, times[0]], dtype=np.float64)
        values = np.repeat(values, 2, axis=0)
    return times, values


class PoseInterpolator:
    def __init__(self, times: np.ndarray, poses: np.ndarray):
        times, poses = _increasing(times, np.asarray(poses, dtype=np.float64))
        if poses.shape[1] != 6:
            raise ValueError(f"pose must be (T, 6), got {poses.shape}")
        self.times = times
        self._pos = interp1d(times, poses[:, :3], axis=0, bounds_error=False, fill_value=(poses[0, :3], poses[-1, :3]))
        self._rot = Slerp(times, Rotation.from_rotvec(poses[:, 3:]))

    def __call__(self, times: np.ndarray) -> np.ndarray:
        query = np.clip(np.asarray(times, dtype=np.float64), self.times[0], self.times[-1])
        scalar = query.ndim == 0
        query = np.atleast_1d(query)
        pos = np.asarray(self._pos(query), dtype=np.float64)
        rotvec = self._rot(query).as_rotvec()
        pose = np.concatenate([pos, rotvec], axis=-1)
        return pose[0] if scalar else pose


def _interp_linear(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    times, values = _increasing(times, np.asarray(values, dtype=np.float64))
    query = np.clip(np.asarray(query, dtype=np.float64), times[0], times[-1])
    interpolator = interp1d(
        times,
        values,
        axis=0,
        bounds_error=False,
        fill_value=(values[0], values[-1]),
    )
    return np.asarray(interpolator(query), dtype=np.float64)


def rpy_to_rotvec_pose(pose_rpy: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_rpy, dtype=np.float64)
    single = pose.ndim == 1
    pose = np.atleast_2d(pose)
    rotvec = Rotation.from_euler("xyz", pose[:, 3:]).as_rotvec()
    out = np.concatenate([pose[:, :3], rotvec], axis=-1)
    return out[0] if single else out


def latest_at_or_before(stamps: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Index of the newest stamp <= each time; the earliest one if none is."""
    stamps = np.asarray(stamps, dtype=np.float64).reshape(-1)
    if stamps.size == 0:
        raise ValueError("no samples")
    indices = np.searchsorted(stamps, np.asarray(times, dtype=np.float64), side="right") - 1
    return np.clip(indices, 0, stamps.size - 1)


def align_bimanual_obs(
    camera_timestamps: list[np.ndarray],
    camera_colors: list[np.ndarray],
    robot_timestamps: list[np.ndarray],
    robot_poses_rotvec: list[np.ndarray],
    gripper_widths: list[np.ndarray],
    frequency: float,
    horizon: int,
    query_time: float,
    robot_joint_positions: list[np.ndarray] | None = None,
) -> dict:
    """Latest frames and arm state as available at ``query_time``.

    ``obs_times`` end at the query time and step back by one control period.
    ``camera{i}_timestamp`` holds the stamps of the selected frames, and
    ``robot{i}_source_timestamp`` the newest arm sample used.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    query_time = float(query_time)
    if not np.isfinite(query_time):
        raise ValueError("query_time must be finite")
    dt = 1.0 / float(frequency)
    obs_times = query_time - (np.arange(horizon)[::-1] * dt)
    obs = {"timestamp": obs_times}
    for camera_idx, (stamps, color) in enumerate(zip(camera_timestamps, camera_colors)):
        stamps = np.asarray(stamps, dtype=np.float64).reshape(-1)
        if stamps.size == 0 or stamps[0] > query_time:
            raise ValueError(f"camera {camera_idx} has no frame at or before the query time")
        indices = latest_at_or_before(stamps, obs_times)
        obs[f"camera{camera_idx}_rgb"] = np.asarray(color)[indices]
        obs[f"camera{camera_idx}_timestamp"] = stamps[indices]
    for robot_idx, (stamps, pose) in enumerate(zip(robot_timestamps, robot_poses_rotvec)):
        stamps = np.asarray(stamps, dtype=np.float64).reshape(-1)
        known = stamps <= query_time
        if not np.any(known):
            raise ValueError(f"robot {robot_idx} has no sample at or before the query time")
        stamps = stamps[known]
        interpolated = PoseInterpolator(stamps, np.asarray(pose)[known])(obs_times)
        obs[f"robot{robot_idx}_eef_pos"] = interpolated[..., :3]
        obs[f"robot{robot_idx}_eef_rot_axis_angle"] = interpolated[..., 3:]
        widths = np.asarray(gripper_widths[robot_idx]).reshape(-1, 1)[known]
        obs[f"robot{robot_idx}_gripper_width"] = _interp_linear(stamps, widths, obs_times)
        if robot_joint_positions is not None:
            joints = np.asarray(robot_joint_positions[robot_idx])[known]
            obs[f"robot{robot_idx}_joint_pos"] = _interp_linear(stamps, joints, obs_times)
        obs[f"robot{robot_idx}_source_timestamp"] = np.float64(stamps.max())
    return obs
