"""Absolute Pico clutch.

A rising grip edge stores the controller pose and the arm target. Later
motion is ``start + (controller - anchor)`` for position and
``q_now * inv(q_anchor) * q_start`` for orientation. Both are in the robot
frame after :func:`transform_pico_to_robot`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_pico.units import trigger_to_gripper_m

Q_FRAME_WXYZ = np.array([0.5, 0.5, -0.5, -0.5], dtype=np.float64)
_IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64).reshape(4)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    norm_sq = w * w + x * x + y * y + z * z
    if norm_sq < 1e-12:
        return _IDENTITY_WXYZ.copy()
    return np.array([w, -x, -y, -z], dtype=np.float64) / norm_sq


def normalize_wxyz(q: np.ndarray) -> np.ndarray:
    quat = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return _IDENTITY_WXYZ.copy()
    return quat / norm


def xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    return normalize_wxyz(np.array([quat[3], quat[0], quat[1], quat[2]]))


def wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    quat = normalize_wxyz(q_wxyz)
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def raw_pose_to_pos_quat(pose7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SDK pose ``[x, y, z, qx, qy, qz, qw]`` to position and wxyz."""
    pose = np.asarray(pose7, dtype=np.float64).reshape(-1)
    if pose.shape[0] < 7:
        raise ValueError(f"Pico pose must have 7 values, got {pose.shape[0]}")
    return pose[:3].copy(), xyzw_to_wxyz(pose[3:7])


def transform_pico_to_robot(
    pos: np.ndarray, quat_wxyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pico +X right, +Y up, +Z toward the user. Robot +X forward, +Y left, +Z up."""
    position = np.asarray(pos, dtype=np.float64).reshape(3)
    robot_pos = np.array([-position[2], -position[0], position[1]], dtype=np.float64)
    quat = normalize_wxyz(quat_wxyz)
    transformed = quaternion_multiply(
        quaternion_multiply(Q_FRAME_WXYZ, quat),
        quaternion_inverse(Q_FRAME_WXYZ),
    )
    return robot_pos, normalize_wxyz(transformed)


def _rotation_angle(q_from: np.ndarray, q_to: np.ndarray) -> float:
    dot = float(np.clip(abs(np.dot(normalize_wxyz(q_from), normalize_wxyz(q_to))), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def _limit_rotation(prev: np.ndarray, target: np.ndarray, max_angle: float) -> np.ndarray:
    if max_angle <= 0.0:
        return normalize_wxyz(target)
    start = Rotation.from_quat(wxyz_to_xyzw(prev))
    goal = Rotation.from_quat(wxyz_to_xyzw(target))
    relative = start.inv() * goal
    angle = float(relative.magnitude())
    if angle <= max_angle or angle < 1e-9:
        return normalize_wxyz(target)
    limited = start * Rotation.from_rotvec(relative.as_rotvec() * (max_angle / angle))
    return xyzw_to_wxyz(limited.as_quat())


def looks_like_tracking_origin(pos: np.ndarray, quat_wxyz: np.ndarray) -> bool:
    """Pico dropout: position at the origin and an identity quaternion.

    The SDK writes ``[0, 0, 0, 0, 0, 0, 1]``. After the robot-frame map that
    is still the origin with ``wxyz = [1, 0, 0, 0]``. A real controller is
    not at that point.
    """
    position = np.asarray(pos, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(position)) or float(np.linalg.norm(position)) > 1e-3:
        return False
    quat = normalize_wxyz(quat_wxyz)
    return abs(abs(float(quat[0])) - 1.0) < 1e-3 and float(np.linalg.norm(quat[1:])) < 1e-3


@dataclass
class HandTarget:
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    gripper_m: float
    engaged: bool
    event: str = "accepted"
    lost: bool = False


class HandClutch:
    """One controller. Call :meth:`seed` with the measured arm pose first.

    A position step above ``position_jump_threshold`` is ignored until
    ``jump_confirm_frames`` readings stay inside ``jump_stable_threshold``.
    The confirmed pose re-anchors, so the arm target does not chase the teleport.
    """

    def __init__(
        self,
        pos_sensitivity: float = 1.0,
        ori_sensitivity: float = 1.0,
        grip_enable_threshold: float = 0.5,
        grip_disable_threshold: float = 0.3,
        position_jump_threshold: float = 0.1,
        jump_confirm_frames: int = 4,
        jump_stable_threshold: float = 0.03,
        ema_alpha: float = 0.5,
        gripper_ema_alpha: float = 0.4,
        pos_ratio: float = 1.2,
        max_pos_velocity: float = 1.0,
        max_rot_velocity: float = 6.28,
        drift_max_deg: float = 45.0,
        gripper_max_m: float = 0.08,
        name: str = "",
    ):
        self.pos_sensitivity = float(pos_sensitivity)
        self.ori_sensitivity = float(ori_sensitivity)
        self.grip_enable_threshold = float(grip_enable_threshold)
        self.grip_disable_threshold = float(grip_disable_threshold)
        self.position_jump_threshold = float(position_jump_threshold)
        self.jump_confirm_frames = max(int(jump_confirm_frames), 1)
        self.jump_stable_threshold = float(jump_stable_threshold)
        for label, value in (("ema_alpha", ema_alpha), ("gripper_ema_alpha", gripper_ema_alpha)):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{label} must be in (0, 1]; got {value}")
        if float(pos_ratio) <= 0.0:
            raise ValueError(f"pos_ratio must be positive; got {pos_ratio}")
        self.ema_alpha = float(ema_alpha)
        self.gripper_ema_alpha = float(gripper_ema_alpha)
        self.pos_ratio = float(pos_ratio)
        self.max_pos_velocity = float(max_pos_velocity)
        self.max_rot_velocity = float(max_rot_velocity)
        self.drift_max_rad = float(np.deg2rad(drift_max_deg))
        self.gripper_max_m = float(gripper_max_m)
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.target_pos: np.ndarray | None = None
        self.target_quat: np.ndarray | None = None
        self.ref_pos: np.ndarray | None = None
        self.quat_offset: np.ndarray | None = None
        self.start_quat: np.ndarray | None = None
        self.enabled = False
        self.jump_reject_count = 0
        self.lost_count = 0
        self.last_event = "seed"
        self._trusted_pos: np.ndarray | None = None
        self._trusted_quat: np.ndarray | None = None
        self._candidate_pos: np.ndarray | None = None
        self._candidate_quat: np.ndarray | None = None
        self._candidate_count = 0
        self._was_lost = False
        self._ema_pos: np.ndarray | None = None
        self._ema_rot: Rotation | None = None
        self._ema_gripper: float | None = None
        self._prev_pos: np.ndarray | None = None
        self._prev_quat: np.ndarray | None = None
        self._last_time: float | None = None

    def seed(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> None:
        self.target_pos = np.asarray(position, dtype=np.float64).reshape(3).copy()
        self.target_quat = normalize_wxyz(quaternion_wxyz)
        self.ref_pos = None
        self.quat_offset = None
        self.start_quat = self.target_quat.copy()
        self.enabled = False
        self._reset_ema()
        self._prev_pos = None
        self._prev_quat = None
        self._last_time = None

    def update(
        self,
        robot_pos: np.ndarray,
        robot_quat_wxyz: np.ndarray,
        grip: float,
        trigger: float,
        measured_pos: np.ndarray,
        measured_quat_wxyz: np.ndarray,
        now: float,
    ) -> HandTarget:
        if self.target_pos is None or self.target_quat is None:
            raise RuntimeError("seed the clutch with the measured arm pose first")
        position = np.asarray(robot_pos, dtype=np.float64).reshape(3).copy()
        quat = normalize_wxyz(robot_quat_wxyz)
        position, quat, tracking_reset = self._gate_jump(position, quat)
        lost = self.last_event == "lost"
        if lost and self._trusted_pos is None:
            return HandTarget(
                position=self.target_pos.copy(),
                quaternion_wxyz=self.target_quat.copy(),
                gripper_m=trigger_to_gripper_m(trigger, gripper_max=self.gripper_max_m),
                engaged=self.enabled,
                event="lost",
                lost=True,
            )

        was_enabled = self.enabled
        if self.enabled:
            self.enabled = float(grip) > self.grip_disable_threshold
        else:
            self.enabled = float(grip) > self.grip_enable_threshold
        just_enabled = self.enabled and not was_enabled

        gripper_m = trigger_to_gripper_m(trigger, gripper_max=self.gripper_max_m)
        if self.enabled:
            if tracking_reset:
                self._reset_ema()
            position, quat, gripper_m = self._smooth(position, quat, gripper_m)
        else:
            self._reset_ema()

        if just_enabled and _rotation_angle(self.target_quat, measured_quat_wxyz) > self.drift_max_rad:
            self.target_pos = np.asarray(measured_pos, dtype=np.float64).reshape(3).copy()
            self.target_quat = normalize_wxyz(measured_quat_wxyz)
            self.enabled = False
            self._prev_pos = None
            self._prev_quat = None
            self._last_time = None

        if just_enabled or self.ref_pos is None or tracking_reset:
            self._set_anchor(position, quat)

        if self.enabled and self.ref_pos is not None and self.quat_offset is not None:
            scale = self.pos_sensitivity / self.pos_ratio
            self.target_pos = self._anchor_pos + (position - self.ref_pos) * scale
            full_quat = normalize_wxyz(quaternion_multiply(quat, self.quat_offset))
            if self.ori_sensitivity < 1.0 and self.start_quat is not None:
                start = Rotation.from_quat(wxyz_to_xyzw(self.start_quat))
                goal = Rotation.from_quat(wxyz_to_xyzw(full_quat))
                blended = (start.inv() * goal).as_rotvec() * self.ori_sensitivity
                self.target_quat = xyzw_to_wxyz((start * Rotation.from_rotvec(blended)).as_quat())
            else:
                self.target_quat = full_quat

        if self._prev_quat is not None and float(np.dot(self.target_quat, self._prev_quat)) < 0.0:
            self.target_quat = -self.target_quat
        self._apply_rate_limit(now)
        return HandTarget(
            position=self.target_pos.copy(),
            quaternion_wxyz=self.target_quat.copy(),
            gripper_m=gripper_m,
            engaged=self.enabled,
            event=self.last_event,
            lost=lost,
        )

    def _reset_ema(self) -> None:
        self._ema_pos = None
        self._ema_rot = None
        self._ema_gripper = None

    def _smooth(
        self, position: np.ndarray, quat: np.ndarray, gripper_m: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """EMA on position, rotation, and gripper. The first sample seeds it."""
        rot = Rotation.from_quat(wxyz_to_xyzw(quat))
        if self._ema_pos is None or self._ema_rot is None or self._ema_gripper is None:
            self._ema_pos = position.copy()
            self._ema_rot = rot
            self._ema_gripper = float(gripper_m)
        else:
            alpha = self.ema_alpha
            self._ema_pos = alpha * position + (1.0 - alpha) * self._ema_pos
            rotvec = (rot * self._ema_rot.inv()).as_rotvec()
            self._ema_rot = Rotation.from_rotvec(alpha * rotvec) * self._ema_rot
            self._ema_gripper = (
                self.gripper_ema_alpha * float(gripper_m)
                + (1.0 - self.gripper_ema_alpha) * self._ema_gripper
            )
        return (
            self._ema_pos.copy(),
            xyzw_to_wxyz(self._ema_rot.as_quat()),
            float(self._ema_gripper),
        )

    def _set_anchor(self, position: np.ndarray, quat: np.ndarray) -> None:
        self.ref_pos = position.copy()
        self._anchor_pos = self.target_pos.copy()
        self.start_quat = self.target_quat.copy()
        self.quat_offset = normalize_wxyz(quaternion_multiply(quaternion_inverse(quat), self.target_quat))

    def _accept_trusted(self, position: np.ndarray, quat: np.ndarray) -> None:
        self._trusted_pos = position.copy()
        self._trusted_quat = quat.copy()
        self._candidate_pos = None
        self._candidate_quat = None
        self._candidate_count = 0

    def _gate_jump(
        self, position: np.ndarray, quat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Hold the last trusted pose until a teleport stays put, then re-anchor.

        Same rule as ``PicoBimanualSession._gate_position_jump``: one bad frame
        is dropped, and a pose that stays within ``jump_stable_threshold`` for
        ``jump_confirm_frames`` becomes the new anchor instead of a command.
        """
        if looks_like_tracking_origin(position, quat):
            self.lost_count += 1
            self.last_event = "lost"
            self._candidate_pos = None
            self._candidate_quat = None
            self._candidate_count = 0
            if self._trusted_pos is None:
                return position, quat, False
            if not self._was_lost:
                label = self.name or "hand"
                print(
                    f"[Pico] {label} tracking lost (origin pose); "
                    "holding the last controller pose."
                )
            self._was_lost = True
            return self._trusted_pos.copy(), self._trusted_quat.copy(), False
        self._was_lost = False

        if self.position_jump_threshold <= 0.0 or self._trusted_pos is None:
            self._accept_trusted(position, quat)
            self.last_event = "accepted"
            return position, quat, False
        if float(np.linalg.norm(position - self._trusted_pos)) <= self.position_jump_threshold:
            self._accept_trusted(position, quat)
            self.last_event = "accepted"
            return position, quat, False

        self.jump_reject_count += 1
        if (
            self._candidate_pos is not None
            and float(np.linalg.norm(position - self._candidate_pos)) <= self.jump_stable_threshold
        ):
            self._candidate_count += 1
            self._candidate_pos = position.copy()
            self._candidate_quat = quat.copy()
        else:
            self._candidate_pos = position.copy()
            self._candidate_quat = quat.copy()
            self._candidate_count = 1
            label = self.name or "hand"
            print(
                f"[Pico] {label} position jump #{self.jump_reject_count}: "
                f"{float(np.linalg.norm(position - self._trusted_pos)):.3f}m "
                f"> {self.position_jump_threshold:.3f}m; holding until "
                f"{self.jump_confirm_frames} stable frames."
            )
        if (
            self._candidate_count >= self.jump_confirm_frames
            and not looks_like_tracking_origin(position, quat)
        ):
            self._accept_trusted(position, quat)
            self.last_event = "reanchor"
            label = self.name or "hand"
            print(
                f"[Pico] {label} new pose kept after "
                f"{self.jump_confirm_frames} stable frames; re-anchor."
            )
            return position, quat, True
        self.last_event = "held"
        return self._trusted_pos.copy(), self._trusted_quat.copy(), False

    def _apply_rate_limit(self, now: float) -> None:
        if (
            self._prev_pos is not None
            and self._prev_quat is not None
            and self._last_time is not None
        ):
            dt = float(now) - self._last_time
            if dt > 0.0:
                if self.max_pos_velocity > 0.0:
                    max_delta = self.max_pos_velocity * dt
                    delta = self.target_pos - self._prev_pos
                    distance = float(np.linalg.norm(delta))
                    if distance > max_delta > 0.0:
                        self.target_pos = self._prev_pos + delta * (max_delta / distance)
                if self.max_rot_velocity > 0.0:
                    self.target_quat = _limit_rotation(
                        self._prev_quat,
                        self.target_quat,
                        self.max_rot_velocity * dt,
                    )
        self._prev_pos = self.target_pos.copy()
        self._prev_quat = self.target_quat.copy()
        self._last_time = float(now)


def action_from_sample(
    left: HandClutch,
    right: HandClutch,
    sample: dict,
    measured_left: tuple[np.ndarray, np.ndarray],
    measured_right: tuple[np.ndarray, np.ndarray],
    now: float,
) -> np.ndarray:
    """14-D command: each arm is xyz, rotation vector, gripper metres."""
    action = np.zeros(14, dtype=np.float64)
    for offset, clutch, side, measured in (
        (0, left, "left", measured_left),
        (7, right, "right", measured_right),
    ):
        robot_pos, robot_quat = transform_pico_to_robot(
            sample[f"{side}_pos"],
            xyzw_to_wxyz(sample[f"{side}_quat_xyzw"]),
        )
        target = clutch.update(
            robot_pos,
            robot_quat,
            float(sample[f"{side}_grip"]),
            float(sample[f"{side}_trigger"]),
            measured[0],
            measured[1],
            now,
        )
        rotvec = Rotation.from_quat(wxyz_to_xyzw(target.quaternion_wxyz)).as_rotvec()
        action[offset : offset + 3] = target.position
        action[offset + 3 : offset + 6] = rotvec
        action[offset + 6] = target.gripper_m
    return action
