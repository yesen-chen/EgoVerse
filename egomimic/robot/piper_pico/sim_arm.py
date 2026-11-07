"""Cartesian stand-in for one Piper arm.

The arm process calls the same methods it uses on the CAN arm. A pose command
is stored immediately, so a jump in this pose is a jump that ``EndPoseCtrl``
would have sent. This plant does not filter commands.
"""

from __future__ import annotations

import numpy as np

from egomimic.robot.piper_pico.arm_process import HOME_QPOS

HOME_EE_POSE = np.array([0.30, 0.0, 0.25, 0.0, 0.80, 0.0], dtype=np.float64)
COMMAND_JUMP_M = 0.1


class SimPiperArm:
    def __init__(self, can_port: str, move_spd_rate: int = 50, name: str = "piper"):
        self.can_port = str(can_port)
        self.move_spd_rate = int(move_spd_rate)
        self.name = str(name)
        self._connected = False
        self._joints = HOME_QPOS.copy()
        self._ee = HOME_EE_POSE.copy()
        self._gripper = 0.0
        self.command_jumps = 0

    def connect(self, enable: bool = True, timeout: float = 10.0) -> "SimPiperArm":
        del enable, timeout
        self._connected = True
        print(f"[sim {self.name}] no CAN; cartesian target is shown directly")
        return self

    def disconnect(self, stop_motion: bool = True) -> None:
        del stop_motion
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def joint_positions(self) -> np.ndarray:
        return self._joints.copy()

    def joint_velocities(self) -> np.ndarray:
        return np.zeros(6, dtype=np.float64)

    def ee_pose(self) -> np.ndarray:
        return self._ee.copy()

    def gripper(self) -> float:
        return float(self._gripper)

    def send_ee_pose(self, pose: np.ndarray, gripper_m: float | None = None) -> None:
        command = np.asarray(pose, dtype=np.float64).reshape(6).copy()
        step = float(np.linalg.norm(command[:3] - self._ee[:3]))
        if step > COMMAND_JUMP_M:
            self.command_jumps += 1
            print(
                f"[sim {self.name}] 末端指令跳变 {step:.3f} m  "
                f"{np.round(self._ee[:3], 3).tolist()} -> {np.round(command[:3], 3).tolist()}"
            )
        self._ee = command
        if gripper_m is not None:
            self._gripper = float(gripper_m)

    def send_joints(self, qpos_rad: np.ndarray, gripper_m: float | None = None) -> None:
        """MOVE J home. Step toward the home pose so the return is not one teleport."""
        self._joints = np.asarray(qpos_rad, dtype=np.float64).reshape(6).copy()
        delta = HOME_EE_POSE[:3] - self._ee[:3]
        distance = float(np.linalg.norm(delta))
        if distance > 0.04:
            self._ee[:3] = self._ee[:3] + delta * (0.04 / distance)
        else:
            self._ee = HOME_EE_POSE.copy()
        if gripper_m is not None:
            self._gripper = float(gripper_m)

    def clear_trajectory(self) -> None:
        return None

    def hold_measured_pose(self, repeat: int = 3, settle_s: float = 0.02) -> None:
        del repeat, settle_s

    def stop_residual_motion(self) -> None:
        return None
