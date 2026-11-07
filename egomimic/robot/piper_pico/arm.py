"""One Piper arm on SocketCAN.

``disconnect`` clears the firmware plan and then releases the port. It does
not drop joint torque. MOVE P keeps the last ``EndPoseCtrl`` target until a
new target replaces it or the plan is cleared.
"""

from __future__ import annotations

import time

import numpy as np

from egomimic.robot.piper_pico.units import (
    GRIPPER_MAX_M,
    GRIPPER_TORQUE_RAW,
    endpose_sdk_to_pose,
    gripper_sdk_to_m,
    joint_vels_sdk_to_rad_s,
    joints_sdk_to_rad,
    m_to_gripper_sdk,
    pose_to_endpose_sdk,
    rad_to_joints_sdk,
)

CTRL_MODE_STANDBY = 0x00
CTRL_MODE_CAN = 0x01
CTRL_MODE_TEACH = 0x02

MOVE_P = 0x00
MOVE_J = 0x01

TRACK_OFF = 0x00
TRACK_CLEAR_CURRENT = 0x03
TRACK_CLEAR_ALL = 0x04

_ENABLE_TIMEOUT = 10.0


def _feedback_timestamp(message) -> float:
    """SDK CAN receive time in epoch seconds, or NaN when unavailable."""
    try:
        stamp = float(getattr(message, "time_stamp", np.nan))
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return stamp if np.isfinite(stamp) and stamp > 0.0 else float("nan")


class PiperArm:
    def __init__(self, can_port: str, move_spd_rate: int = 50, name: str = "piper"):
        self.can_port = str(can_port)
        self.move_spd_rate = int(move_spd_rate)
        self.name = str(name)
        self._piper = None

    def connect(
        self, enable: bool = True, timeout: float = _ENABLE_TIMEOUT, *,
        preserve_gripper_on_enable: bool = False,
    ) -> "PiperArm":
        from piper_sdk import C_PiperInterface_V2

        try:
            self._piper = C_PiperInterface_V2(self.can_port)
            if enable:
                self._piper.ConnectPort()
            else:
                # The SDK otherwise sends PiperInit query frames even when
                # EnableArm is skipped. Observation mode only receives CAN.
                self._piper.ConnectPort(piper_init=False)
        except Exception as exc:
            raise ConnectionError(
                f"[{self.name}] cannot open CAN interface {self.can_port!r}: {exc}\n"
                f"Bring the bus up first:\n"
                f"  sudo ip link set {self.can_port} type can bitrate 1000000\n"
                f"  sudo ip link set {self.can_port} up"
            ) from exc
        if enable:
            if preserve_gripper_on_enable:
                self.ensure_enabled(timeout=timeout, preserve_gripper_on_enable=True)
            else:
                self.ensure_enabled(timeout=timeout)
        return self

    def bind(self, sdk) -> "PiperArm":
        """Attach an already-open SDK object. Used by tests and the arm process."""
        self._piper = sdk
        return self

    def disconnect(self, stop_motion: bool = True) -> None:
        """Release the CAN port and leave the drivers energised."""
        if self._piper is None:
            return
        if stop_motion:
            self.stop_residual_motion()
        try:
            self._piper.DisconnectPort()
        except Exception as exc:
            print(f"[{self.name}] failed to disconnect {self.can_port}: {exc}")
        finally:
            self._piper = None

    @property
    def connected(self) -> bool:
        return self._piper is not None

    def _require(self):
        if self._piper is None:
            raise RuntimeError(f"[{self.name}] is not connected")
        return self._piper

    def ctrl_mode(self) -> int | None:
        try:
            return int(self._require().GetArmStatus().arm_status.ctrl_mode)
        except Exception:
            return None

    def is_estopped(self) -> bool:
        try:
            return int(self._require().GetArmStatus().arm_status.arm_status) == 0x01
        except Exception:
            return False

    def all_motors_enabled(self) -> bool:
        try:
            low = self._require().GetArmLowSpdInfoMsgs()
            return all(
                getattr(low, f"motor_{i}").foc_status.driver_enable_status for i in range(1, 7)
            )
        except Exception:
            return False

    def ensure_enabled(
        self, timeout: float = _ENABLE_TIMEOUT, *, preserve_gripper_on_enable: bool = False,
    ) -> None:
        piper = self._require()
        if self.is_estopped():
            raise RuntimeError(f"[{self.name}] reports E-STOP on {self.can_port}")
        start = time.monotonic()
        while not self.all_motors_enabled():
            if time.monotonic() - start > timeout:
                raise RuntimeError(
                    f"[{self.name}] did not enable on {self.can_port} within {timeout}s"
                )
            gripper_raw = 0
            if preserve_gripper_on_enable:
                # Do not issue the legacy close command when live inference
                # starts paused. An unknown/stale SDK cache is not a zero-width
                # measurement: wait for real feedback before any enable write.
                try:
                    message = piper.GetArmGripperMsgs()
                    stamp = _feedback_timestamp(message)
                    width = gripper_sdk_to_m(message.gripper_state.grippers_angle)
                    if (
                        not np.isfinite([stamp, width]).all()
                        or _feedback_timestamp(message) != stamp
                        or not -0.05 <= time.time() - stamp <= 0.5
                        or not 0.0 <= width <= GRIPPER_MAX_M + 1e-9
                    ):
                        raise ValueError("gripper feedback is unavailable or stale")
                    gripper_raw = m_to_gripper_sdk(width)
                except Exception:
                    time.sleep(0.05)
                    continue
            try:
                piper.EnableArm(7)
                piper.GripperCtrl(gripper_raw, GRIPPER_TORQUE_RAW, 0x01, 0)
            except Exception:
                pass
            time.sleep(0.3)

    def joint_positions(self) -> np.ndarray:
        js = self._require().GetArmJointMsgs().joint_state
        raw = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
        return joints_sdk_to_rad(raw)

    def joint_velocities(self) -> np.ndarray:
        hs = self._require().GetArmHighSpdInfoMsgs()
        raw = [getattr(hs, f"motor_{index}").motor_speed for index in range(1, 7)]
        return joint_vels_sdk_to_rad_s(raw)

    def ee_pose(self) -> np.ndarray:
        return endpose_sdk_to_pose(self._require().GetArmEndPoseMsgs().end_pose)

    def gripper(self) -> float:
        return gripper_sdk_to_m(self._require().GetArmGripperMsgs().gripper_state.grippers_angle)

    def feedback_snapshot(self) -> dict:
        """Copy feedback values with the timestamps of their SDK messages.

        The installed SDK derives ``time_stamp`` from python-can's receive
        timestamp in epoch seconds. Its getters return mutable cached objects,
        so copy values immediately and invalidate freshness if the timestamp
        changes during that copy. This is not an atomic six-axis measurement:
        the pose timestamp describes the most recently updated of three CAN
        partitions; this SDK exposes no per-partition timestamps. Calibration
        must still require a stationary window, not rely on this stamp alone.
        Missing timestamps remain NaN rather than becoming host polling time.
        """
        piper = self._require()
        joint_message = piper.GetArmJointMsgs()
        joint_stamp = _feedback_timestamp(joint_message)
        joints = joints_sdk_to_rad(
            [getattr(joint_message.joint_state, f"joint_{i}") for i in range(1, 7)]
        )
        if _feedback_timestamp(joint_message) != joint_stamp:
            joint_stamp = float("nan")

        pose_message = piper.GetArmEndPoseMsgs()
        pose_stamp = _feedback_timestamp(pose_message)
        pose = endpose_sdk_to_pose(pose_message.end_pose)
        if _feedback_timestamp(pose_message) != pose_stamp:
            pose_stamp = float("nan")

        gripper_message = piper.GetArmGripperMsgs()
        gripper_stamp = _feedback_timestamp(gripper_message)
        width = gripper_sdk_to_m(gripper_message.gripper_state.grippers_angle)
        if _feedback_timestamp(gripper_message) != gripper_stamp:
            gripper_stamp = float("nan")
        return {
            "joint_pos": joints,
            "ee_pose": pose,
            "gripper_position": float(width),
            "robot_joint_source_timestamp": joint_stamp,
            "robot_pose_source_timestamp": pose_stamp,
            "robot_gripper_source_timestamp": gripper_stamp,
        }

    def send_ee_pose(self, pose: np.ndarray, gripper_m: float | None = None) -> None:
        piper = self._require()
        x_mm, y_mm, z_mm, roll, pitch, yaw = pose_to_endpose_sdk(pose)
        piper.MotionCtrl_2(0x01, MOVE_P, self.move_spd_rate, 0x00)
        piper.EndPoseCtrl(x_mm, y_mm, z_mm, roll, pitch, yaw)
        if gripper_m is not None:
            piper.GripperCtrl(abs(m_to_gripper_sdk(gripper_m)), GRIPPER_TORQUE_RAW, 0x01, 0)

    def send_joints(self, qpos_rad: np.ndarray, gripper_m: float | None = None) -> None:
        piper = self._require()
        piper.MotionCtrl_2(0x01, MOVE_J, self.move_spd_rate, 0x00)
        piper.JointCtrl(*rad_to_joints_sdk(qpos_rad))
        if gripper_m is not None:
            piper.GripperCtrl(abs(m_to_gripper_sdk(gripper_m)), GRIPPER_TORQUE_RAW, 0x01, 0)

    def clear_trajectory(self) -> None:
        piper = self._require()
        piper.MotionCtrl_1(0x00, TRACK_CLEAR_CURRENT, 0x00)
        piper.MotionCtrl_1(0x00, TRACK_CLEAR_ALL, 0x00)
        piper.MotionCtrl_1(0x00, TRACK_OFF, 0x00)

    def hold_measured_pose(self, repeat: int = 3, settle_s: float = 0.02) -> None:
        for _ in range(max(int(repeat), 1)):
            try:
                self.send_ee_pose(self.ee_pose())
            except Exception as exc:
                print(f"[{self.name}] failed to hold the measured pose: {exc}")
                return
            time.sleep(max(float(settle_s), 0.0))

    def stop_residual_motion(self) -> None:
        if not self.connected:
            return
        self.clear_trajectory()
        self.hold_measured_pose()
