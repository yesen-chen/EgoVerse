"""Live policy adapters over the collector's shared-memory device processes.

No hardware is opened on import. A send receipt means both host workers called
their SDK send method, not that either robot reached the commanded position.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import replace

import numpy as np

from egomimic.robot.piper_pico.arm_process import HOME_QPOS
from egomimic.robot.piper_pico.policy_inference import ObservationFrame
from egomimic.robot.piper_pico.units import GRIPPER_MAX_M


def _alive(device):
    if device.process.pid is None or device.process.exitcode is not None:
        raise RuntimeError(
            f"{device.name} process is not running (exit={device.process.exitcode})"
        )


def _fresh(stamp, now, max_age, label):
    stamp = float(stamp)
    age = now - stamp
    if not np.isfinite(age) or age < -0.05 or age > max_age:
        raise RuntimeError(f"{label} source is stale or invalid: age={age:.6f}s")
    return stamp


def _arm_state(arm, *, now, max_age):
    _alive(arm)
    state = arm.get_state()
    _fresh(state["robot_receive_timestamp"], now, max_age, f"{arm.name} worker")
    for field in ("joint", "pose", "gripper"):
        _fresh(
            state.get(f"robot_{field}_source_timestamp", np.nan),
            now,
            max_age,
            f"{arm.name} {field} CAN",
        )
    for key, shape in (
        ("joint_pos", (6,)),
        ("ee_pose", (6,)),
        ("gripper_position", ()),
    ):
        value = np.asarray(state[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise RuntimeError(f"{arm.name}: invalid {key}")
    return state


class LiveObservationProvider:
    def __init__(self, env, *, max_source_age_s=0.5):
        self.env = env
        self.max_source_age_s = float(max_source_age_s)
        if not np.isfinite(self.max_source_age_s) or self.max_source_age_s <= 0:
            raise ValueError("max_source_age_s must be positive")
        self.camera_names = [spec["name"] for spec in env.camera_specs]
        self._camera_buffers = {}

    def check_health(self):
        """Check current frame/feedback freshness without another device reader."""
        now = time.time()
        sources = {}
        for camera in self.env.cameras:
            _alive(camera)
            ring = camera.ring
            before = ring.count
            started = time.monotonic()
            state = ring.get(out=self._camera_buffers.get(camera.name))
            self._camera_buffers[camera.name] = state
            # Reuse owned snapshots, never lend live shared memory to consumers.
            # A slow/wrapped copy is rejected; the camera writer never waits on
            # this reader and these checks never touch its read/write counters.
            if (
                time.monotonic() - started > ring.get_time_budget
                or ring.count - before >= ring.buffer_size - 1
            ):
                raise RuntimeError(
                    f"{camera.name}: shared-memory snapshot overran its read budget"
                )
            sources[camera.name] = _fresh(
                np.asarray(state["timestamp"]).item(),
                now,
                self.max_source_age_s,
                camera.name,
            )
        for arm in self.env.arms:
            state = _arm_state(arm, now=now, max_age=self.max_source_age_s)
            sources[arm.name] = min(
                float(state[f"robot_{field}_source_timestamp"])
                for field in ("joint", "pose", "gripper")
            )
        return sources

    def latest(self):
        # The state/image alignment and raw RGB are exactly the collection path.
        self.check_health()
        obs = self.env.get_obs()
        frame = ObservationFrame.from_aligned_obs(obs, camera_names=self.camera_names)
        sources = dict(frame.source_timestamps)
        now = time.time()
        for index, side in enumerate(("left", "right")):
            sources[side] = min(
                _fresh(
                    obs.get(f"robot{index}_{field}_source_timestamp", np.nan),
                    now,
                    self.max_source_age_s,
                    f"{side} {field} CAN",
                )
                for field in ("joint", "pose", "gripper")
            )
        frame = replace(frame, source_timestamps=sources)
        frame.check_freshness(now, self.max_source_age_s)
        return frame


class LiveActionSink:
    def __init__(
        self,
        env,
        *,
        execute=False,
        command_timeout_s=0.2,
        max_source_age_s=0.5,
    ):
        self.env = env
        self.execute = bool(execute)
        self.command_timeout_s = float(command_timeout_s)
        self.max_source_age_s = float(max_source_age_s)
        if (
            not np.isfinite([self.command_timeout_s, self.max_source_age_s]).all()
            or min(self.command_timeout_s, self.max_source_age_s) <= 0
        ):
            raise ValueError("command timeout and source age must be positive")
        if self.execute and not env.enable_motion:
            raise ValueError("execute requires an environment with motion enabled")
        self.records = deque(maxlen=128)
        self.hold_reasons = deque(maxlen=128)

    @property
    def last_record(self):
        return self.records[-1] if self.records else None

    def _wait_receipts(self, command_ids, *, queued_at, should_stop=None):
        deadline = time.monotonic() + self.command_timeout_s
        while True:
            if should_stop is not None and should_stop():
                raise InterruptedError("home cancelled")
            now = time.time()
            receipts = []
            for arm, expected in zip(self.env.arms, command_ids):
                state = _arm_state(arm, now=now, max_age=self.max_source_age_s)
                actual = int(state["last_command_id"])
                if actual > expected:
                    raise RuntimeError(
                        f"{arm.name}: command {expected} was superseded by {actual}"
                    )
                sent_at = float(state["last_command_send_timestamp"])
                if actual == expected and np.isfinite(sent_at) and sent_at >= queued_at:
                    receipts.append(
                        {
                            "arm": arm.name,
                            "command_id": actual,
                            "host_send_timestamp": sent_at,
                        }
                    )
            if len(receipts) == len(self.env.arms):
                return receipts
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"no SDK-send receipt for commands {list(command_ids)}"
                )
            time.sleep(0.002)

    def send_action(self, action, deadline_walltime):
        row = np.asarray(action, dtype=np.float64)
        deadline = float(deadline_walltime)
        if (
            row.shape != (14,)
            or not np.isfinite(row).all()
            or not np.isfinite(deadline)
        ):
            raise ValueError("action must be finite (14,) with a finite deadline")
        queued_at = time.time()
        if deadline <= queued_at:
            raise TimeoutError("action deadline already expired")
        record = {
            "action": row.tolist(),
            "deadline_walltime": deadline,
            "queued_at": queued_at,
            "execute": self.execute,
        }
        if self.execute:
            for arm in self.env.arms:
                _arm_state(arm, now=queued_at, max_age=self.max_source_age_s)
            ids = self.env.exec_actions(row, np.array([deadline]))
            if len(ids) != 1 or int(ids[0]) < 0:
                raise RuntimeError("environment rejected an expired action")
            record["receipts"] = self._wait_receipts(
                [int(ids[0])] * len(self.env.arms), queued_at=queued_at
            )
            if any(
                receipt["host_send_timestamp"] > deadline
                for receipt in record["receipts"]
            ):
                raise TimeoutError("SDK send completed after the command deadline")
        record["confirmation_elapsed_s"] = time.time() - queued_at
        self.records.append(record)
        return record

    def hold(self, reason):
        self.hold_reasons.append(str(reason))
        if not self.execute:
            return
        waiting, errors = [], []
        # Request both holds before waiting on either; a failed left arm must
        # never stop us from clearing the right arm's pending commands.
        for arm in self.env.arms:
            try:
                _alive(arm)
                arm.request_hold()
                waiting.append(arm)
            except Exception as exc:
                errors.append(f"{arm.name}: {exc}")
        deadline = time.monotonic() + max(1.0, self.command_timeout_s)
        while waiting and time.monotonic() < deadline:
            for arm in list(waiting):
                try:
                    _alive(arm)
                    if arm.held_event.is_set():
                        waiting.remove(arm)
                except Exception as exc:
                    errors.append(f"{arm.name}: {exc}")
                    waiting.remove(arm)
            if waiting:
                time.sleep(0.005)
        errors.extend(f"{arm.name}: hold was not acknowledged" for arm in waiting)
        if errors:
            raise RuntimeError("; ".join(errors))

        # MOVE J home or a manual change while paused may have crossed an Euler
        # branch. Resume continuity from feedback, never the prior policy target.
        self.env.seed_clutches_from_measured()

    def home(self, *, timeout_s=20.0, joint_tolerance_rad=0.05, should_stop=None):
        """Return to collector HOME_QPOS, open grippers, then hold measured pose."""
        if not self.execute:
            raise RuntimeError("home is unavailable in observation mode")
        if (
            not np.isfinite([timeout_s, joint_tolerance_rad]).all()
            or min(timeout_s, joint_tolerance_rad) <= 0
        ):
            raise ValueError("home timeout and tolerance must be positive")
        outcome = "home failed"
        try:
            queued_at = time.time()
            for arm in self.env.arms:
                _arm_state(arm, now=queued_at, max_age=self.max_source_age_s)
            ids = self.env.home()
            self._wait_receipts(ids, queued_at=queued_at, should_stop=should_stop)
            deadline = time.monotonic() + timeout_s
            stable_since = None
            while time.monotonic() < deadline:
                if should_stop is not None and should_stop():
                    raise InterruptedError("home cancelled")
                now = time.time()
                states = [
                    _arm_state(arm, now=now, max_age=self.max_source_age_s)
                    for arm in self.env.arms
                ]
                arrived = all(
                    np.max(np.abs(np.asarray(s["joint_pos"]) - HOME_QPOS))
                    <= joint_tolerance_rad
                    and abs(float(s["gripper_position"]) - GRIPPER_MAX_M) <= 0.005
                    for s in states
                )
                if arrived:
                    if stable_since is None:
                        stable_since = time.monotonic()
                    if time.monotonic() - stable_since >= 0.2:
                        outcome = "home complete"
                        return {
                            "status": "completed",
                            "home_joints_rad": HOME_QPOS.tolist(),
                        }
                else:
                    stable_since = None
                time.sleep(0.01)
            raise TimeoutError(
                "home did not reach measured joint tolerance before timeout"
            )
        finally:
            self.hold(outcome)
