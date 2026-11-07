"""Replay converted Piper commands in each arm's base frame, without a policy.

Only piper_base_v3 robot episodes are accepted. No camera calibration is used.
Hardware imports are confined to PiperReplayDevice.open(). Offline replay uses
the same approach / arrival check / timed playback state machine on a virtual
clock; it is not an IK, collision or physical tracking validation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation, Slerp

from egomimic.robot.piper_pico.units import (
    continuous_euler_xyz,
    endpose_sdk_to_pose,
    gripper_sdk_to_m,
)

SIDES = ("left", "right")


@dataclass(frozen=True)
class ReplayOptions:
    speed: float = 1.0
    approach_duration: float = 4.0
    approach_frequency: float = 30.0
    approach_linear_speed: float = 0.05
    approach_angular_speed: float = 0.5
    approach_gripper_speed: float = 0.04
    position_tolerance: float = 0.01
    rotation_tolerance: float = 0.08
    gripper_tolerance: float = 0.003
    settle_duration: float = 0.3
    settle_timeout: float = 10.0
    max_translation_step: float = 0.1
    max_rotation_step: float = 0.5
    max_tick_gap: float = 0.1
    max_schedule_slip: float = 0.1

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.speed > 1:
            raise ValueError("speed must be in (0, 1]; 0.5 means half-speed replay")
        if self.settle_timeout <= self.settle_duration:
            raise ValueError("settle_timeout must exceed settle_duration")


@dataclass(frozen=True)
class Episode:
    path: Path
    actions: np.ndarray  # [left xyz, rotvec, width_m, right xyz, rotvec, width_m]
    offsets: np.ndarray  # recorded decision time relative to first decision
    initial_state: np.ndarray
    move_spd_rate: int
    metadata: dict


def discover_episodes(inputs):
    """Explicit files or directories, sorted recursively; never skip non-Piper data."""
    paths = []
    for value in inputs:
        if "://" in str(value):
            raise ValueError("Only local converted Zarr directories are supported")
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise ValueError(f"Expected a directory: {path}")
        found = [path] if path.suffix == ".zarr" else sorted(path.rglob("*.zarr"))
        if not found:
            raise ValueError(f"No converted episode .zarr found in {path}")
        paths.extend(p.resolve() for p in found)
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Select at least one episode, with no duplicates")
    return paths


def _array(store, key, shape):
    if key not in store:
        raise ValueError(f"Missing field {key}")
    # ZarrWriter pads the final storage shard; total_frames is authoritative.
    array = np.asarray(store[key][: shape[0]], dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{key} must be finite with shape {shape}, got {array.shape}")
    return array


def _pose_and_gripper(store, side, kind, count, open_m):
    pose = _array(store, f"{side}.{kind}_ee_pose", (count, 7))
    quat = pose[:, [4, 5, 6, 3]]  # stored wxyz -> scipy xyzw
    if not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-4, rtol=0):
        raise ValueError(f"{side}.{kind}_ee_pose contains non-unit quaternions")
    grip = _array(store, f"{side}.{kind}_gripper", (count, 1))
    if kind == "cmd" and (np.any(grip < 0) or np.any(grip > 1)):
        raise ValueError(f"{side}.{kind}_gripper must be in [0, 1]; no silent clipping")
    width = grip * open_m
    physical_key = f"{side}.{kind}_gripper_width_m"
    if physical_key in store and not np.allclose(
        _array(store, physical_key, (count, 1)), width, atol=1e-7, rtol=0
    ):
        raise ValueError(f"{physical_key} disagrees with normalized gripper")
    return np.column_stack([pose[:, :3], Rotation.from_quat(quat).as_rotvec(), width])


def validate_steps(actions, options):
    for start, side in zip((0, 7), SIDES):
        distances = np.linalg.norm(
            np.diff(actions[:, start : start + 3], axis=0), axis=1
        )
        rotations = Rotation.from_rotvec(actions[:, start + 3 : start + 6])
        angles = (rotations[:-1].inv() * rotations[1:]).magnitude()
        if np.any(distances > options.max_translation_step + 1e-10):
            raise ValueError(
                f"{side} command translation step exceeds {options.max_translation_step} m"
            )
        if np.any(angles > options.max_rotation_step + 1e-10):
            raise ValueError(
                f"{side} command rotation step exceeds {options.max_rotation_step} rad"
            )


def load_episode(path, options):
    """Fully validate all commands before connecting CAN; leave images unread."""
    path = Path(path).resolve(strict=True)
    store = zarr.open_group(str(path), mode="r")
    metadata = dict(store.attrs)
    expected = {
        "embodiment": "piper_bimanual",
        "schema": "piper_base_v3",
        "pose_representation": "xyz_m+quat_wxyz",
        "pose_coordinates": {side: f"piper_{side}_base" for side in SIDES},
        "tool_frame": "piper_sdk_end_pose; no additional TCP offset applied",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"{path}: {key} must be {value!r}; Ego/Eva/camera-frame data cannot replay"
            )
    count = metadata.get("total_frames")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError(f"{path}: total_frames must be at least 2")
    encoding = metadata.get("gripper_encoding", {})
    if encoding.get("unit") != "normalized_width" or encoding.get("closed_m") != 0:
        raise ValueError(f"{path}: unsupported gripper encoding")
    open_m = float(encoding.get("open_m", math.nan))
    if not math.isfinite(open_m) or not 0 < open_m <= 0.08:
        raise ValueError(f"{path}: gripper open_m must be in (0, 0.08] metres")
    observations = _array(store, "timestamp", (count,))
    decisions = _array(store, "source.decision_timestamp", (count,))
    if np.any(decisions < observations) or np.any(np.diff(observations) <= 0):
        raise ValueError(f"{path}: invalid observation/decision timestamps")
    intervals = np.diff(decisions)
    if np.any(intervals <= 0) or np.any(intervals > options.max_tick_gap):
        raise ValueError(
            f"{path}: decision intervals must be in (0, {options.max_tick_gap}] seconds"
        )
    actions = np.column_stack(
        [_pose_and_gripper(store, side, "cmd", count, open_m) for side in SIDES]
    )
    initial = np.concatenate(
        [_pose_and_gripper(store, side, "obs", count, open_m)[0] for side in SIDES]
    )
    validate_steps(actions, options)
    rate = (
        metadata.get("source_capture_config", {})
        .get("replay_attributes", {})
        .get("move_spd_rate", 50)
    )
    if isinstance(rate, bool) or not isinstance(rate, int) or not 1 <= rate <= 100:
        raise ValueError(f"{path}: invalid recorded move_spd_rate")
    return Episode(path, actions, decisions - decisions[0], initial, rate, metadata)


def pose_errors(actual, target):
    actual = np.asarray(actual, dtype=float)
    if actual.shape != (14,) or not np.isfinite(actual).all():
        raise ValueError("Robot feedback must be a finite 14-D base-frame state")
    errors = []
    for start in (0, 7):
        errors.append(
            [
                float(
                    np.linalg.norm(
                        actual[start : start + 3] - target[start : start + 3]
                    )
                ),
                float(
                    (
                        Rotation.from_rotvec(actual[start + 3 : start + 6]).inv()
                        * Rotation.from_rotvec(target[start + 3 : start + 6])
                    ).magnitude()
                ),
                float(abs(actual[start + 6] - target[start + 6])),
            ]
        )
    return np.asarray(errors)


def approach_path(current, target, options):
    errors = pose_errors(current, target)
    # Cubic smoothstep has maximum derivative 1.5. Include that factor so the
    # requested approach speed bounds also hold during acceleration/deceleration.
    duration = max(
        options.approach_duration,
        1.5
        * np.max(
            errors
            / np.array(
                [
                    options.approach_linear_speed,
                    options.approach_angular_speed,
                    options.approach_gripper_speed,
                ]
            )
        ),
    )
    intervals = max(1, int(np.ceil(duration * options.approach_frequency)))
    offsets = np.linspace(0, duration, intervals + 1)
    u = offsets / duration
    fraction = u * u * (3 - 2 * u)
    rows = current[None] + fraction[:, None] * (target - current)[None]
    for start in (0, 7):
        rotations = Rotation.from_rotvec(
            np.stack(
                [
                    current[start + 3 : start + 6],
                    target[start + 3 : start + 6],
                ]
            )
        )
        rows[:, start + 3 : start + 6] = Slerp([0, 1], rotations)(fraction).as_rotvec()
    validate_steps(rows, options)
    return rows, offsets


class RealClock:
    monotonic = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)


class VirtualClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)


class SimulatedDevice:
    """Ideal instant-following device, initialized from the first recorded state."""

    def __init__(self, initial_state):
        self.state = np.asarray(initial_state, dtype=float).copy()
        self.sent = []
        self.holds = 0

    def read_state(self):
        return self.state.copy()

    def send(self, action):
        self.state = action.copy()
        self.sent.append(action.copy())

    def set_speed(self, rate):
        self.speed = rate

    def hold(self):
        self.holds += 1

    def close(self):
        pass


def _send_schedule(device, rows, offsets, options, clock, report, phase):
    start = clock.monotonic()
    last_send = None
    report[phase] = {
        "count": 0,
        "planned_span_s": float(offsets[-1]),
        "max_slip_s": 0.0,
    }
    stats = report[phase]
    for index, (row, offset) in enumerate(zip(rows, offsets)):
        planned = start + float(offset)
        interval = 0.0 if index == 0 else float(offset - offsets[index - 1])
        # Keep an absolute clock, so ordinary sleep jitter does not accumulate
        # over thousands of points. A delayed command cannot be followed by a
        # burst: subsequent sends remain at least half a source interval apart.
        due = planned if last_send is None else max(planned, last_send + interval * 0.5)
        while True:
            remaining = due - clock.monotonic()
            if remaining <= 0:
                break
            clock.sleep(min(0.01, remaining))
        device.read_state()  # Hardware checks fresh SDK feedback and E-STOP.
        now = clock.monotonic()
        slip = max(0.0, now - planned)
        if slip > options.max_schedule_slip:
            raise RuntimeError(
                f"{phase}: schedule slip {slip:.3f}s; stopping without catch-up burst"
            )
        device.send(row)
        last_send = now
        stats["count"] += 1
        stats["max_slip_s"] = max(stats["max_slip_s"], slip)
    stats["actual_span_s"] = float(last_send - start)


def _wait_arrival(device, target, options, clock):
    start, stable_since = clock.monotonic(), None
    tolerance = np.array(
        [
            options.position_tolerance,
            options.rotation_tolerance,
            options.gripper_tolerance,
        ]
    )
    while True:
        errors = pose_errors(device.read_state(), target)
        now = clock.monotonic()
        if np.all(errors <= tolerance):
            stable_since = now if stable_since is None else stable_since
            if now - stable_since >= options.settle_duration:
                return {"wait_s": now - start, "errors_m_rad_m": errors.tolist()}
        else:
            stable_since = None
        if now - start >= options.settle_timeout:
            raise RuntimeError(
                f"Arrival timed out; left/right position, rotation, gripper errors: {errors.tolist()}"
            )
        clock.sleep(0.02)


def replay_episode(
    episode, device, options, clock, report, *, move_spd_rate=None, notify=print
):
    """Every episode gets its own approach and measured arrival gate. Always hold."""
    report.update(
        episode=str(episode.path), status="running", frames=len(episode.actions)
    )
    failure = None
    try:
        rate = episode.move_spd_rate if move_spd_rate is None else move_spd_rate
        device.set_speed(rate)
        report["move_spd_rate"] = rate
        current = device.read_state()
        rows, offsets = approach_path(current, episode.actions[0], options)
        notify(f"{episode.path.name}: 移到第一条 command，过渡 {offsets[-1]:.2f} s")
        _send_schedule(device, rows, offsets, options, clock, report, "approach")
        report["start_arrival"] = _wait_arrival(
            device, episode.actions[0], options, clock
        )
        notify(
            f"{episode.path.name}: 双臂到位，开始回放 {len(episode.actions)} 点 / {episode.offsets[-1] / options.speed:.2f} s"
        )
        _send_schedule(
            device,
            episode.actions,
            episode.offsets / options.speed,
            options,
            clock,
            report,
            "playback",
        )
        report["end_arrival"] = _wait_arrival(
            device, episode.actions[-1], options, clock
        )
        report["status"] = "completed"
    except BaseException as exc:
        failure = exc
        report.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=str(exc),
        )
    finally:
        try:
            device.hold()
            report["hold"] = "completed"
        except BaseException as exc:
            report["hold_error"] = str(exc)
            report["status"] = "failed"
            if failure is None:
                failure = exc
    if failure is not None:
        raise failure


class PiperReplayDevice:
    """Direct paired SDK sends; no environment, PICO, camera or policy process."""

    def __init__(self, left_can, right_can, feedback_timeout=0.25):
        if not left_can or not right_can or left_can == right_can:
            raise ValueError("Left and right CAN ports must be nonempty and different")
        self.ports = (left_can, right_can)
        self.arms = []
        self.references = [None, None]
        if not math.isfinite(feedback_timeout) or feedback_timeout <= 0:
            raise ValueError("feedback_timeout must be finite and positive")
        self.feedback_timeout = feedback_timeout

    def open(self):
        from egomimic.robot.piper_pico.arm import PiperArm

        try:
            for side, port in zip(SIDES, self.ports):
                arm = PiperArm(port, name=f"replay {side}")
                self.arms.append(arm)  # Includes a partially connected arm on failure.
                arm.connect(enable=True)
            deadline = time.monotonic() + 3.0
            while True:
                try:
                    self.read_state()
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.02)
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup_error:
                exc.add_note(f"Cleanup after open failure: {cleanup_error}")
            raise
        return self

    def _read_arm(self, arm):
        sdk = arm._require()
        pose_msg, grip_msg, status = (
            sdk.GetArmEndPoseMsgs(),
            sdk.GetArmGripperMsgs(),
            sdk.GetArmStatus(),
        )
        now = time.time()
        for name, msg in (
            ("pose", pose_msg),
            ("gripper", grip_msg),
            ("status", status),
        ):
            stamp = float(msg.time_stamp)
            if (
                not math.isfinite(stamp)
                or stamp <= 0
                or not -0.05 <= now - stamp <= self.feedback_timeout
            ):
                raise RuntimeError(f"{arm.name}: missing/stale {name} CAN feedback")
        if int(status.arm_status.arm_status) == 0x01:
            raise RuntimeError(f"{arm.name}: E-STOP")
        pose = endpose_sdk_to_pose(pose_msg.end_pose)
        width = gripper_sdk_to_m(grip_msg.gripper_state.grippers_angle)
        if not np.isfinite(pose).all() or not math.isfinite(width):
            raise RuntimeError(f"{arm.name}: nonfinite feedback")
        return pose, width

    def read_state(self):
        if len(self.arms) != 2:
            raise RuntimeError("Both arms must be connected")
        result = []
        for index, arm in enumerate(self.arms):
            pose, width = self._read_arm(arm)
            if self.references[index] is None:
                self.references[index] = pose[3:].copy()
            result.extend(
                [*pose[:3], *Rotation.from_euler("xyz", pose[3:]).as_rotvec(), width]
            )
        return np.asarray(result)

    def set_speed(self, rate):
        if isinstance(rate, bool) or not isinstance(rate, int) or not 1 <= rate <= 100:
            raise ValueError("move_spd_rate must be an integer in [1, 100]")
        for arm in self.arms:
            arm.move_spd_rate = rate

    def send(self, action):
        for index, arm in enumerate(self.arms):
            start = 7 * index
            rpy = continuous_euler_xyz(
                Rotation.from_rotvec(action[start + 3 : start + 6]),
                self.references[index],
            )
            arm.send_ee_pose(
                np.r_[action[start : start + 3], rpy], float(action[start + 6])
            )
            self.references[index] = rpy

    def hold(self):
        errors = []
        # Clear both plans before reading/sending any hold, including on a
        # one-arm send failure. Never command a stale cached measured pose.
        for arm in self.arms:
            if arm.connected:
                try:
                    arm.clear_trajectory()
                except Exception as exc:
                    errors.append(f"{arm.name}: {exc}")
        for index, arm in enumerate(self.arms):
            if arm.connected:
                try:
                    pose, grip = self._read_arm(arm)
                    arm.send_ee_pose(pose, grip)
                    self.references[index] = pose[3:].copy()
                except Exception as exc:
                    errors.append(f"{arm.name}: {exc}")
        if errors:
            raise RuntimeError("Hold errors: " + "; ".join(errors))

    def close(self):
        errors = []
        try:
            if self.arms:
                self.hold()
        except Exception as exc:
            errors.append(str(exc))
        finally:
            for arm in self.arms:
                try:
                    arm.disconnect(stop_motion=False)
                except Exception as exc:
                    errors.append(f"{arm.name} disconnect: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))
