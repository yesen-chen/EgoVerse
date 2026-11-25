"""Piper arm process.

The queue timestamp only decides whether a waypoint is still fresh. A fresh
absolute pose is sent with ``EndPoseCtrl`` at the configured speed rate.
There is no host-side pose or joint trajectory. Every state sample carries
the id and host send time of the last waypoint actually sent, so a recording
can recover when each command left the host.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing.managers import SharedMemoryManager
from queue import Empty
from typing import Optional

import numpy as np

from egomimic.robot.piper_pico.arm import PiperArm
from egomimic.robot.piper_pico.shm import (
    SharedMemoryQueue,
    SharedMemoryRingBuffer,
    join_process,
)
from egomimic.robot.piper_pico.units import GRIPPER_MAX_M

CMD_WAYPOINT = 1
CMD_STOP = 2
CMD_HOME = 3

#HOME_QPOS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
HOME_QPOS = np.array([0.0, 0.82, -0.92, 0.84, 0.0, 0.0], dtype=np.float64)

@dataclass
class ArmCommand:
    kind: str
    pose: Optional[np.ndarray] = None
    gripper: float = 0.0
    joints: Optional[np.ndarray] = None
    command_id: int = -1


def select_fresh_index(target_times: np.ndarray, now: float) -> Optional[int]:
    """Last waypoint whose target time is still strictly in the future."""
    times = np.asarray(target_times, dtype=np.float64).reshape(-1)
    fresh = np.flatnonzero(times > float(now))
    if fresh.size == 0:
        return None
    return int(fresh[-1])


def reduce_arm_commands(batch: dict, now: float) -> Optional[ArmCommand]:
    """Collapse one queue read into the single command the firmware should see."""
    chosen: Optional[ArmCommand] = None
    count = len(batch["cmd"])
    for index in range(count):
        kind = int(batch["cmd"][index])
        if kind == CMD_STOP:
            return ArmCommand(kind="stop")
        if kind == CMD_HOME:
            chosen = ArmCommand(
                kind="home",
                joints=np.asarray(batch["joints"][index], dtype=np.float64).copy(),
                gripper=float(batch["gripper"][index]),
                command_id=int(batch["command_id"][index]) if "command_id" in batch else -1,
            )
        elif kind == CMD_WAYPOINT and float(batch["target_time"][index]) > float(now):
            chosen = ArmCommand(
                kind="waypoint",
                pose=np.asarray(batch["pose"][index], dtype=np.float64).copy(),
                gripper=float(batch["gripper"][index]),
                command_id=int(batch["command_id"][index]) if "command_id" in batch else -1,
            )
    return chosen


def command_examples() -> dict:
    return {
        "cmd": np.int64(0),
        "pose": np.zeros(6, dtype=np.float64),
        "gripper": np.float64(0.0),
        "target_time": np.float64(0.0),
        "joints": np.zeros(6, dtype=np.float64),
        "command_id": np.int64(-1),
    }


def state_examples() -> dict:
    return {
        "joint_pos": np.zeros(6, dtype=np.float64),
        "joint_vel": np.zeros(6, dtype=np.float64),
        "ee_pose": np.zeros(6, dtype=np.float64),
        "gripper_position": np.float64(0.0),
        "robot_receive_timestamp": np.float64(0.0),
        "robot_timestamp": np.float64(0.0),
        "robot_joint_source_timestamp": np.float64(np.nan),
        "robot_pose_source_timestamp": np.float64(np.nan),
        "robot_gripper_source_timestamp": np.float64(np.nan),
        "robot_source_timestamps_simulated": np.uint8(0),
        "feedback_id": np.int64(-1),
        "last_command_id": np.int64(-1),
        "last_command_send_timestamp": np.float64(np.nan),
    }


def read_arm_state(
    arm: PiperArm,
    receive_latency: float,
    last_command_id: int = -1,
    last_command_send_timestamp: float = np.nan,
    *,
    simulated: bool = False,
) -> dict:
    """Read feedback; keep CAN source freshness separate from host polling.

    Legacy/fake arms without ``feedback_snapshot`` still supply values, but
    source freshness is unknown. Only explicit simulation may use host time
    as a synthetic source timestamp, and those samples carry a separate flag.
    """
    snapshot_reader = getattr(arm, "feedback_snapshot", None)
    if callable(snapshot_reader):
        feedback = snapshot_reader()
    else:
        feedback = {
            "joint_pos": arm.joint_positions(),
            "ee_pose": arm.ee_pose(),
            "gripper_position": arm.gripper(),
            "robot_joint_source_timestamp": np.nan,
            "robot_pose_source_timestamp": np.nan,
            "robot_gripper_source_timestamp": np.nan,
        }
    joint_vel = arm.joint_velocities().astype(np.float64)
    # SDK objects can be updated while getters run. Stamp the completed read,
    # not its start, so a just-received source stamp does not appear future-dated.
    received = time.time()
    return {
        "joint_pos": np.asarray(feedback["joint_pos"], dtype=np.float64).copy(),
        "joint_vel": joint_vel,
        "ee_pose": np.asarray(feedback["ee_pose"], dtype=np.float64).copy(),
        "gripper_position": np.float64(feedback["gripper_position"]),
        "robot_receive_timestamp": np.float64(received),
        "robot_timestamp": np.float64(received - receive_latency),
        "robot_joint_source_timestamp": np.float64(
            received if simulated else feedback["robot_joint_source_timestamp"]
        ),
        "robot_pose_source_timestamp": np.float64(
            received if simulated else feedback["robot_pose_source_timestamp"]
        ),
        "robot_gripper_source_timestamp": np.float64(
            received if simulated else feedback["robot_gripper_source_timestamp"]
        ),
        "robot_source_timestamps_simulated": np.uint8(simulated),
        # Assigned by the publishing loop; this standalone read is not yet a
        # published sample. It never substitutes for the CAN source timestamps.
        "feedback_id": np.int64(-1),
        "last_command_id": np.int64(last_command_id),
        "last_command_send_timestamp": np.float64(last_command_send_timestamp),
    }


def arm_process_loop(
    can_port: str,
    move_spd_rate: int,
    name: str,
    ring: SharedMemoryRingBuffer,
    commands: SharedMemoryQueue,
    receive_latency: float,
    frequency: float,
    ready_event,
    stop_event,
    simulate: bool = False,
    enable_motion: bool = True,
    preserve_gripper_on_enable: bool = False,
    hold_event=None,
    held_event=None,
) -> None:
    if simulate:
        from egomimic.robot.piper_pico.sim_arm import SimPiperArm

        arm = SimPiperArm(can_port, move_spd_rate=move_spd_rate, name=name)
    else:
        arm = PiperArm(can_port, move_spd_rate=move_spd_rate, name=name)
    period = 1.0 / float(frequency)
    home_joints = None
    home_gripper = 0.0
    last_home_send = 0.0
    last_command_id = -1
    last_command_send = np.nan
    next_feedback_id = 0
    try:
        if preserve_gripper_on_enable and enable_motion and not simulate:
            arm.connect(enable=True, preserve_gripper_on_enable=True)
        else:
            arm.connect(enable=enable_motion)
        next_tick = time.monotonic()
        while not stop_event.is_set():
            if hold_event is not None and hold_event.is_set():
                # The child is the only queue reader. A hold supersedes every
                # pending waypoint and the persistent joint-home target.
                commands.clear()
                home_joints = None
                if enable_motion:
                    hold_fresh_measured_pose(arm, simulated=simulate)
                hold_event.clear()
                if held_event is not None:
                    held_event.set()
            try:
                state = read_arm_state(
                    arm, receive_latency, last_command_id, last_command_send,
                    simulated=simulate,
                )
                state["feedback_id"] = np.int64(next_feedback_id)
                ring.put(state)
                next_feedback_id += 1
                if not ready_event.is_set():
                    ready_event.set()
            except Exception as exc:
                print(f"[{name}] state read failed: {exc}")
            try:
                batch = commands.get_all()
            except Empty:
                batch = None
            decision = reduce_arm_commands(batch, time.time()) if batch is not None else None
            if decision is not None and decision.kind == "stop":
                break
            if not enable_motion:
                # Defense in depth: a direct queue write must never bypass
                # observation mode's parent-side command guards.
                decision = None
            if decision is not None and decision.kind == "home":
                home_joints = decision.joints
                home_gripper = decision.gripper
                arm.send_joints(home_joints, home_gripper)
                last_command_id = decision.command_id
                last_command_send = time.time()
                last_home_send = time.monotonic()
            elif decision is not None and decision.kind == "waypoint" and decision.pose is not None:
                home_joints = None
                arm.send_ee_pose(decision.pose, decision.gripper)
                last_command_id = decision.command_id
                last_command_send = time.time()
            elif home_joints is not None and time.monotonic() - last_home_send >= 0.04:
                arm.send_joints(home_joints, home_gripper)
                last_home_send = time.monotonic()
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            elif delay < -period:
                next_tick = time.monotonic()
    finally:
        try:
            if enable_motion and arm.connected:
                hold_fresh_measured_pose(arm, simulated=simulate)
        finally:
            # Never let disconnect perform an unchecked, cached-pose hold.
            arm.disconnect(stop_motion=False)


def hold_fresh_measured_pose(arm, *, simulated=False, max_source_age_s=0.5):
    """Clear the firmware plan, then hold fresh measured pose and gripper.

    When feedback is unavailable, the plan is still cleared but no old pose is
    sent and the caller gets an error instead of a successful hold receipt.
    """
    arm.clear_trajectory()
    state = read_arm_state(arm, 0.0, simulated=simulated)
    now = time.time()
    for field in ("pose", "gripper"):
        stamp = float(state[f"robot_{field}_source_timestamp"])
        age = now - stamp
        if not np.isfinite(age) or age < -0.05 or age > max_source_age_s:
            raise RuntimeError(f"cannot hold: {field} CAN feedback is stale ({age}s)")
    pose = np.asarray(state["ee_pose"], dtype=np.float64)
    width = float(state["gripper_position"])
    if pose.shape != (6,) or not np.isfinite(pose).all() or not np.isfinite(width):
        raise RuntimeError("cannot hold: invalid measured pose/gripper")
    arm.send_ee_pose(pose, width)


class PiperArmProcess:
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        can_port: str,
        name: str,
        move_spd_rate: int = 50,
        receive_latency: float = 0.0,
        frequency: float = 100.0,
        simulate: bool = False,
        enable_motion: bool = True,
        preserve_gripper_on_enable: bool = False,
        feedback_history_s: float = 1.28,
    ):
        import multiprocessing as mp

        if not np.isfinite(feedback_history_s) or feedback_history_s <= 0:
            raise ValueError("feedback_history_s must be finite and positive")
        context = mp.get_context("spawn")
        self.name = name
        self.can_port = can_port
        self.simulate = bool(simulate)
        self.enable_motion = bool(enable_motion)
        self.preserve_gripper_on_enable = bool(preserve_gripper_on_enable)
        self.ring = SharedMemoryRingBuffer.create_from_examples(
            shm_manager,
            state_examples(),
            get_max_k=max(128, int(np.ceil(feedback_history_s * frequency))),
            get_time_budget=0.01,
            put_desired_frequency=frequency,
        )
        self.commands = SharedMemoryQueue.create_from_examples(
            shm_manager,
            command_examples(),
            buffer_size=64,
        )
        self.ready_event = context.Event()
        self.stop_event = context.Event()
        self.hold_event = context.Event()
        self.held_event = context.Event()
        self.process = context.Process(
            target=arm_process_loop,
            kwargs={
                "can_port": can_port,
                "move_spd_rate": move_spd_rate,
                "name": name,
                "ring": self.ring,
                "commands": self.commands,
                "receive_latency": receive_latency,
                "frequency": frequency,
                "ready_event": self.ready_event,
                "stop_event": self.stop_event,
                "simulate": self.simulate,
                "enable_motion": self.enable_motion,
                "preserve_gripper_on_enable": self.preserve_gripper_on_enable,
                "hold_event": self.hold_event,
                "held_event": self.held_event,
            },
            daemon=False,
        )

    def start(self) -> None:
        self.process.start()

    def get_state(self) -> dict:
        return self.ring.get()

    def get_all_state(self) -> dict:
        return self.ring.get_all()

    def schedule_waypoint(
        self, pose: np.ndarray, gripper_m: float, target_time: float, command_id: int = -1
    ) -> None:
        self._require_motion()
        self.commands.put(
            {
                "cmd": np.int64(CMD_WAYPOINT),
                "pose": np.asarray(pose, dtype=np.float64),
                "gripper": np.float64(gripper_m),
                "target_time": np.float64(target_time),
                "joints": np.zeros(6, dtype=np.float64),
                "command_id": np.int64(command_id),
            }
        )

    def go_home(
        self, joints: np.ndarray | None = None, gripper_m: float = GRIPPER_MAX_M,
        command_id: int = -1,
    ) -> None:
        self._require_motion()
        qpos = HOME_QPOS if joints is None else np.asarray(joints, dtype=np.float64)
        self.commands.put(
            {
                "cmd": np.int64(CMD_HOME),
                "pose": np.zeros(6, dtype=np.float64),
                "gripper": np.float64(gripper_m),
                "target_time": np.float64(0.0),
                "joints": qpos,
                "command_id": np.int64(command_id),
            }
        )

    def _require_motion(self) -> None:
        if not self.enable_motion:
            raise RuntimeError(f"{self.name}: motion is disabled in observation mode")

    def request_hold(self) -> None:
        self._require_motion()
        self.held_event.clear()
        self.hold_event.set()

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        if self.process.pid is not None:
            try:
                self.commands.put(
                    {
                        "cmd": np.int64(CMD_STOP),
                        "pose": np.zeros(6, dtype=np.float64),
                        "gripper": np.float64(0.0),
                        "target_time": np.float64(time.time() + 10.0),
                        "joints": np.zeros(6, dtype=np.float64),
                    }
                )
            except Exception:
                pass
        join_process(self.process, timeout)
