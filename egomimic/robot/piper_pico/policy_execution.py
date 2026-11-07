"""Pure, synchronous policy-block execution. No hardware or CAN is imported.

Actions are base-frame [left xyz, rotvec, width_m, right xyz, rotvec, width_m].
The two endpoints retain ActionTiming's physical span. If span * requested Hz
is not an integer, the nearest positive number of equal intervals is used and
``actual_control_frequency`` reports that small difference explicitly.

A sink's ``deadline_walltime`` is only a command freshness deadline, never a
promise of robot execution or arrival at that time. Every send contains one
row. Pause ends the old block: a subsequent block requires a fresh observation.

In the training data a chunk's first command was sent a recorded delay
(``observation_to_send_s``) after its observation. When that delay is given,
execution maps send wall time ``s`` to chunk offset ``s - observation - delay``:
an early block waits, and a late block starts inside the chunk.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Protocol

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from egomimic.rldb.action_timing import ActionTiming


class SafetyViolation(ValueError):
    """An observation or target does not meet the configured execution limits."""


class ScheduleOverrun(RuntimeError):
    """The block cannot follow its declared timing within the allowed slip."""


class BlockExecutionError(RuntimeError):
    """Execution stopped and attempted hold; ``report`` contains its audit log."""

    def __init__(self, report: dict):
        self.report = report
        error = report.get("error") or report.get("hold_error") or {}
        super().__init__(
            f"Policy block failed: {error.get('message', 'unknown error')}"
        )


class ActionSink(Protocol):
    def send_action(self, action: np.ndarray, deadline_walltime: float) -> None: ...

    def hold(self, reason: str) -> None: ...


def _positive(value, name, *, zero_allowed=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not zero_allowed):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if zero_allowed else 'positive'}"
        )
    return value


def _row(value, label):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (14,) or not np.isfinite(value).all():
        raise SafetyViolation(f"{label} must be a finite (14,) action")
    return value


@dataclass(frozen=True)
class SafetyLimits:
    """Optional per-arm workspace/step bounds plus required data validity checks.

    These are command checks, not a collision checker or a firmware guarantee.
    Set both workspace bounds to None to omit the workspace check. Either step
    limit may independently be None; finite actions and gripper limits remain
    mandatory even when all optional geometric limits are disabled.
    Observation age is checked when a block starts, not against every future
    target: the block intentionally predicts beyond that observation's time.
    """

    workspace_min: np.ndarray | None
    workspace_max: np.ndarray | None
    max_translation_step_m: float | None = 0.05
    max_rotation_step_rad: float | None = 0.35
    gripper_min_m: float = 0.0
    gripper_max_m: float = 0.08
    max_observation_age_s: float = 0.5
    future_timestamp_tolerance_s: float = 0.05

    def __post_init__(self):
        if (self.workspace_min is None) != (self.workspace_max is None):
            raise ValueError(
                "workspace_min and workspace_max must both be None or arrays"
            )
        for name in ("workspace_min", "workspace_max"):
            if getattr(self, name) is None:
                continue
            value = np.asarray(getattr(self, name), dtype=np.float64).copy()
            if value.shape != (2, 3) or not np.isfinite(value).all():
                raise ValueError(
                    f"{name} must be a finite (2, 3) array, left then right"
                )
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if self.workspace_min is not None and np.any(
            self.workspace_min >= self.workspace_max
        ):
            raise ValueError("workspace_min must be smaller than workspace_max")
        for name in (
            "max_translation_step_m",
            "max_rotation_step_rad",
        ):
            if getattr(self, name) is None:
                continue
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(
            self,
            "max_observation_age_s",
            _positive(self.max_observation_age_s, "max_observation_age_s"),
        )
        for name in ("gripper_min_m", "future_timestamp_tolerance_s"):
            object.__setattr__(
                self, name, _positive(getattr(self, name), name, zero_allowed=True)
            )
        object.__setattr__(
            self, "gripper_max_m", _positive(self.gripper_max_m, "gripper_max_m")
        )
        if self.gripper_max_m <= self.gripper_min_m:
            raise ValueError("gripper_max_m must exceed gripper_min_m")

    def check_observation(
        self, observation_timestamp: float, now_walltime: float
    ) -> float:
        stamp, now = float(observation_timestamp), float(now_walltime)
        if not math.isfinite(stamp) or not math.isfinite(now):
            raise SafetyViolation(
                "observation and current wall timestamps must be finite"
            )
        age = now - stamp
        if age > self.max_observation_age_s:
            raise SafetyViolation(
                f"stale observation: age {age:.6f}s exceeds {self.max_observation_age_s:.6f}s"
            )
        if age < -self.future_timestamp_tolerance_s:
            raise SafetyViolation(
                f"observation timestamp is {-age:.6f}s in the future; check clock alignment"
            )
        return age

    def check_action(self, action, previous=None) -> None:
        action = _row(action, "action")
        previous = None if previous is None else _row(previous, "previous action")
        for index, side in enumerate(("left", "right")):
            start = index * 7
            position, rotation, width = (
                action[start : start + 3],
                action[start + 3 : start + 6],
                action[start + 6],
            )
            if self.workspace_min is not None and (
                np.any(position < self.workspace_min[index])
                or np.any(position > self.workspace_max[index])
            ):
                raise SafetyViolation(f"{side} target is outside configured workspace")
            if not self.gripper_min_m <= width <= self.gripper_max_m:
                raise SafetyViolation(
                    f"{side} gripper width {width:.6f}m is outside configured range"
                )
            if previous is not None:
                distance = np.linalg.norm(position - previous[start : start + 3])
                angle = (
                    Rotation.from_rotvec(previous[start + 3 : start + 6]).inv()
                    * Rotation.from_rotvec(rotation)
                ).magnitude()
                if (
                    self.max_translation_step_m is not None
                    and distance > self.max_translation_step_m + 1e-12
                ):
                    raise SafetyViolation(
                        f"{side} translation step {distance:.6f}m exceeds configured limit"
                    )
                if (
                    self.max_rotation_step_rad is not None
                    and angle > self.max_rotation_step_rad + 1e-12
                ):
                    raise SafetyViolation(
                        f"{side} rotation step {angle:.6f}rad exceeds configured limit"
                    )


@dataclass(frozen=True)
class ResampledActionBlock:
    actions: np.ndarray
    offsets_s: np.ndarray
    requested_control_frequency: float
    actual_control_frequency: float
    full_span_s: float
    full_point_count: int

    def to_report(self) -> dict:
        return {
            "requested_control_frequency": self.requested_control_frequency,
            "actual_control_frequency": self.actual_control_frequency,
            "full_span_s": self.full_span_s,
            "full_point_count": self.full_point_count,
            "selected_point_count": len(self.actions),
            "start_offset_s": float(self.offsets_s[0]),
            "selected_span_s": float(self.offsets_s[-1] - self.offsets_s[0]),
            "nominal_offsets_s": self.offsets_s.tolist(),
        }


def resample_action_block(
    actions,
    timing: ActionTiming,
    control_frequency: float,
    *,
    execute_steps: int | None = None,
    start_offset_s: float = 0.0,
) -> ResampledActionBlock:
    """Preserve duration, use linear xyz/width and shortest-path rotational SLERP.

    Samples start at ``start_offset_s`` into the chunk and keep the control
    interval. ``execute_steps`` selects a prefix after resampling; it cannot
    extend the trajectory. One step selects the first target only.
    """
    frequency = _positive(control_frequency, "control_frequency")
    start = _positive(start_offset_s, "start_offset_s", zero_allowed=True)
    if not isinstance(timing, ActionTiming):
        raise TypeError("timing must be an ActionTiming")
    source = np.asarray(actions, dtype=np.float64)
    if (
        source.shape != (timing.output_chunk_length, 14)
        or not np.isfinite(source).all()
    ):
        raise SafetyViolation(
            f"actions must be finite ({timing.output_chunk_length}, 14)"
        )
    if execute_steps is not None and (
        isinstance(execute_steps, bool)
        or not isinstance(execute_steps, Integral)
        or execute_steps < 1
    ):
        raise ValueError("execute_steps must be a positive integer or None")
    span = timing.effective_span_s
    intervals = max(1, int(round(span * frequency)))
    if start > span + 1e-9:
        raise ScheduleOverrun(
            f"chunk start offset {start:.6f}s is beyond its {span:.6f}s span"
        )
    if start == 0.0:
        offsets = np.linspace(0.0, span, intervals + 1)
    else:
        step = span / intervals
        count = int(np.floor((span - start) / step + 1e-9)) + 1
        offsets = start + step * np.arange(count)
    source_offsets = np.asarray(timing.output_offsets_s)
    result = np.empty((len(offsets), 14), dtype=np.float64)
    for start in (0, 7):
        for column in (start, start + 1, start + 2, start + 6):
            result[:, column] = np.interp(offsets, source_offsets, source[:, column])
        rotations = Rotation.from_rotvec(source[:, start + 3 : start + 6])
        result[:, start + 3 : start + 6] = Slerp(source_offsets, rotations)(
            offsets
        ).as_rotvec()
    count = (
        len(offsets) if execute_steps is None else min(int(execute_steps), len(offsets))
    )
    return ResampledActionBlock(
        result[:count].copy(),
        offsets[:count].copy(),
        frequency,
        intervals / span,
        span,
        len(offsets),
    )


class RecordingActionSink:
    """No-CAN sink for dry runs; records copies, not mutable caller arrays."""

    def __init__(self):
        self.records: list[dict] = []
        self.hold_reasons: list[str] = []

    def send_action(self, action: np.ndarray, deadline_walltime: float) -> None:
        action = _row(action, "sent action")
        deadline = float(deadline_walltime)
        if not math.isfinite(deadline):
            raise ValueError("freshness deadline must be finite")
        self.records.append({"action": action.tolist(), "deadline_walltime": deadline})

    def hold(self, reason: str) -> None:
        self.hold_reasons.append(str(reason))


def execute_action_block(
    actions,
    timing: ActionTiming,
    sink: ActionSink,
    *,
    control_frequency: float,
    initial_action,
    observation_timestamp: float,
    limits: SafetyLimits,
    execute_steps: int | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    wall_time: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    control_callback: Callable[[], str | None] | None = None,
    max_schedule_slip_s: float = 0.1,
    command_freshness_s: float = 0.1,
    max_wall_clock_skew_s: float = 0.1,
    poll_interval_s: float = 0.01,
    observation_to_send_s: float | None = None,
    hold_on_completion: bool = True,
) -> dict:
    """Send a synchronous prefix, always holding on pause, stop or failure.

    ``hold_on_completion=False`` leaves a completed block's last command active
    for a caller continuing with the next block. That caller owns final cleanup.
    The default preserves the standalone executor's hold on completion.

    ``observation_to_send_s`` is the recorded delay from observation to the
    first send; None sends the chunk from offset zero as soon as possible.

    The callback returns ``run``/None, ``pause`` or ``stop``. Pause/stop return
    immediately with that status after hold. They never resume an old block.
    A late but tolerated send shifts the next wait forward, so there is no
    burst catch-up. Accumulated slip against the original schedule is bounded.
    Errors raise BlockExecutionError with a JSON-serializable report; interrupts
    propagate after hold. No physical home, motor disable or CAN operation occurs.
    """
    report = {
        "status": "running",
        "sent_count": 0,
        "samples": [],
        "hold_attempted": False,
        "hold_succeeded": False,
        "deadline_semantics": "freshness_only_not_execution_time",
    }
    failure = None
    try:
        if not isinstance(hold_on_completion, bool):
            raise ValueError("hold_on_completion must be a boolean")
        slip_limit = _positive(
            max_schedule_slip_s, "max_schedule_slip_s", zero_allowed=True
        )
        ttl = _positive(command_freshness_s, "command_freshness_s")
        skew_limit = _positive(
            max_wall_clock_skew_s, "max_wall_clock_skew_s", zero_allowed=True
        )
        poll_interval = _positive(poll_interval_s, "poll_interval_s")
        start_mono, start_wall = float(monotonic()), float(wall_time())
        if not math.isfinite(start_mono):
            raise ValueError("monotonic clock must be finite")
        report["observation_age_s"] = limits.check_observation(
            observation_timestamp, start_wall
        )
        start_offset, start_delay = 0.0, 0.0
        if observation_to_send_s is not None:
            target = float(observation_timestamp) + _positive(
                observation_to_send_s, "observation_to_send_s", zero_allowed=True
            )
            start_offset = max(0.0, start_wall - target)
            start_delay = max(0.0, target - start_wall)
        report.update(
            observation_to_send_target_s=observation_to_send_s,
            start_delay_s=start_delay,
        )
        block = resample_action_block(
            actions,
            timing,
            control_frequency,
            execute_steps=execute_steps,
            start_offset_s=start_offset,
        )
        report.update(block.to_report())
        previous = _row(initial_action, "initial measured action").copy()
        limits.check_action(previous)
        # Reject invalid geometry anywhere in the selected block before motion.
        for row in block.actions:
            limits.check_action(row)
        report.update(start_monotonic=start_mono, start_walltime=start_wall)
        last_clock = start_mono
        last_send = None
        last_sink_end = None
        for index, (row, offset) in enumerate(zip(block.actions, block.offsets_s)):
            iteration_started = float(monotonic())
            stages = {
                "post_previous_sink_s": None
                if last_sink_end is None
                else iteration_started - last_sink_end,
                "control_callback_s": 0.0,
                "max_control_callback_s": 0.0,
                "sleep_requested_s": 0.0,
                "sleep_elapsed_s": 0.0,
                "max_sleep_overshoot_s": 0.0,
                "poll_count": 0,
            }
            # Kept even for an unsent/failed action, where samples has no row.
            attempt = {"index": index, "phase": "wait", "stage_timings": stages}
            report["last_attempt"] = attempt
            planned = start_mono + start_delay + float(offset - start_offset)
            # A tolerated delay must not make the next command follow sooner
            # than its resampled interval; do not race through missed points.
            interval = 0.0 if index == 0 else float(offset - block.offsets_s[index - 1])
            planned_wall = start_wall + start_delay + float(offset - start_offset)
            wait_until = (
                planned if last_send is None else max(planned, last_send + interval)
            )
            while True:
                callback_started = float(monotonic())
                command = control_callback() if control_callback is not None else "run"
                callback_elapsed = float(monotonic()) - callback_started
                stages["control_callback_s"] += callback_elapsed
                stages["max_control_callback_s"] = max(
                    stages["max_control_callback_s"], callback_elapsed
                )
                stages["poll_count"] += 1
                if command in ("pause", "stop"):
                    report["status"] = "paused" if command == "pause" else "stopped"
                    break
                if command not in (None, "run"):
                    raise ValueError(
                        "control_callback must return run, pause, stop or None"
                    )
                now = float(monotonic())
                if not math.isfinite(now) or now < last_clock:
                    raise ScheduleOverrun(
                        "monotonic clock moved backwards or is non-finite"
                    )
                last_clock = now
                if now >= wait_until:
                    break
                requested_sleep = min(poll_interval, wait_until - now)
                sleep_started = float(monotonic())
                sleep(requested_sleep)
                sleep_elapsed = float(monotonic()) - sleep_started
                stages["sleep_requested_s"] += requested_sleep
                stages["sleep_elapsed_s"] += sleep_elapsed
                stages["max_sleep_overshoot_s"] = max(
                    stages["max_sleep_overshoot_s"], sleep_elapsed - requested_sleep
                )
            if report["status"] != "running":
                break
            slip = max(0.0, now - planned)
            stages["wait_elapsed_s"] = now - iteration_started
            attempt.update(phase="schedule_check", schedule_slip_s=slip)
            if slip > slip_limit + 1e-9:
                report["schedule_overrun"] = {
                    "kind": "schedule_slip",
                    "index": index,
                    "slip_s": slip,
                    "limit_s": slip_limit,
                    "planned_monotonic": planned,
                    "scheduled_monotonic": wait_until,
                    "checked_monotonic": now,
                    "since_previous_send_s": None
                    if last_send is None
                    else now - last_send,
                }
                raise ScheduleOverrun(
                    f"schedule slip {slip:.6f}s exceeds {slip_limit:.6f}s; no catch-up sends"
                )
            sent_wall = float(wall_time())
            if (
                not math.isfinite(sent_wall)
                or abs((sent_wall - start_wall) - (now - start_mono)) > skew_limit
            ):
                raise ScheduleOverrun("wall clock changed relative to monotonic clock")
            attempt["phase"] = "validate_action"
            validation_started = float(monotonic())
            limits.check_action(row, previous)
            stages["action_validation_s"] = float(monotonic()) - validation_started
            deadline = sent_wall + ttl
            attempt["phase"] = "sink"
            sink_started = float(monotonic())
            try:
                sink.send_action(row.copy(), deadline)
            finally:
                last_sink_end = float(monotonic())
                stages["sink_elapsed_s"] = last_sink_end - sink_started
            attempt["phase"] = "sent"
            report["samples"].append(
                {
                    "index": index,
                    "nominal_offset_s": float(offset),
                    "planned_monotonic": planned,
                    "scheduled_monotonic": wait_until,
                    "planned_walltime": planned_wall,
                    "actual_monotonic": now,
                    "actual_walltime": sent_wall,
                    "observation_to_send_s": sent_wall - float(observation_timestamp),
                    "freshness_deadline_walltime": deadline,
                    "schedule_slip_s": slip,
                    "stage_timings": dict(stages),
                    "action": row.tolist(),
                }
            )
            report["sent_count"] += 1
            previous, last_send = row.copy(), now
        if report["status"] == "running":
            report["status"] = "completed"
    except BaseException as exc:
        failure = exc
        report["status"] = "failed" if isinstance(exc, Exception) else "interrupted"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if report["status"] != "completed" or hold_on_completion:
            report["hold_attempted"] = True
            try:
                sink.hold(report["status"])
                report["hold_succeeded"] = True
            except BaseException as exc:
                report["hold_error"] = {"type": type(exc).__name__, "message": str(exc)}
                if failure is None:
                    failure = exc
                    report["status"] = (
                        "failed" if isinstance(exc, Exception) else "interrupted"
                    )
    if len(report["samples"]) > 1:
        elapsed = (
            report["samples"][-1]["actual_monotonic"]
            - report["samples"][0]["actual_monotonic"]
        )
        report["actual_sent_span_s"] = elapsed
        report["observed_send_frequency"] = (report["sent_count"] - 1) / elapsed
    else:
        report["actual_sent_span_s"] = 0.0
        report["observed_send_frequency"] = None
    if failure is not None:
        if isinstance(failure, Exception):
            raise BlockExecutionError(report) from failure
        raise failure
    return report
