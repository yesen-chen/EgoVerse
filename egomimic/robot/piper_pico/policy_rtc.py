"""RTC execution with a fixed inference deadline and tick-aligned takeover.

The single worker reads an observation and predicts while the control thread
continues the previously committed plan. Guidance samples that plan on the NEW
prediction's 100-point time grid. A late result is never switched in: motion is
held, the result invalidated, and an explicit resume requires a fresh request.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_pico.policy_continuity import (
    ConstantPosePlan,
    ContinuousPlan,
    conditioned_plan,
    sample_plan,
    transition_disagreement,
)
from egomimic.robot.piper_pico.policy_streaming import ChunkPlan


def check_rtc_schedule(
    timing, control_frequency, replan_steps, rtc_delay_s, *, transition_s=0.0
):
    """Leave enough old-plan overlap for observation, inference and a switch."""
    frequency, delay = float(control_frequency), float(rtc_delay_s)
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("RTC control frequency must be finite and positive")
    if (
        isinstance(replan_steps, bool)
        or int(replan_steps) != replan_steps
        or replan_steps < 1
    ):
        raise ValueError("RTC execute_steps must be a positive integer")
    if not math.isfinite(delay) or delay <= 0:
        raise ValueError("RTC delay must be finite and positive")
    interval, replan = 1 / frequency, int(replan_steps) / frequency
    transition_s = float(transition_s)
    if not math.isfinite(transition_s) or transition_s < 0 or transition_s >= replan:
        raise ValueError(
            "RTC transition must be nonnegative and shorter than request period"
        )
    # One interval rounds the deadline to a send tick, another covers small
    # tick drift. The model point to the right must still belong to the old plan.
    grid_interval = float(np.max(np.diff(timing.output_offsets_s)))
    takeover_budget = delay + 2 * interval
    if replan <= takeover_budget:
        raise ValueError(
            "RTC execute_steps/frequency must exceed delay + two control intervals "
            "so one request can take over before the next request"
        )
    required = replan + takeover_budget + grid_interval + transition_s
    if required >= timing.effective_span_s:
        raise ValueError(
            f"RTC needs execute_steps/frequency + delay + two control intervals "
            f"+ one model interval + transition "
            f"({required:.3f}s) below chunk span {timing.effective_span_s:.3f}s"
        )
    return {
        "control_interval_s": interval,
        "replan_interval_s": replan,
        "rtc_delay_s": delay,
        "required_span_s": required,
        "takeover_guard_s": interval,
        "model_interval_s": grid_interval,
        "replan_clock": "request_start",
        "transition_s": transition_s,
    }


def build_rtc_prefix(
    plan,
    timing,
    observation_timestamp,
    observation_to_send_s,
    activation_walltime,
    request_id,
):
    """Sample committed base targets and the paper's discrete exponential mask.

    ``activation_walltime`` is the latest allowed first-target time, including
    the control-tick drift allowance. Both interpolation endpoints at that time
    receive weight one. The old plan endpoint is excluded from the overlap;
    no repeated endpoint values are constrained beyond its valid horizon.
    The remote adapter converts base -> camera; the server normalizes targets.
    """
    anchor = float(observation_timestamp) + float(observation_to_send_s)
    activation = float(activation_walltime)
    offsets = np.asarray(timing.output_offsets_s, dtype=np.float64)
    if not np.isfinite([anchor, activation]).all() or activation < anchor:
        raise ValueError("RTC observation arrived after its activation window")
    if activation >= plan.end_walltime:
        raise ValueError("RTC activation has no remaining committed-plan overlap")
    stamps = anchor + offsets
    valid = (stamps >= plan.anchor) & (stamps < plan.end_walltime)
    targets = np.zeros((len(offsets), 14), dtype=np.float64)
    targets[valid] = sample_plan(plan, stamps[valid])
    # Include the RIGHT interpolation point, not just points before takeover.
    right = int(np.searchsorted(stamps, activation, side="left"))
    if right >= len(stamps) or not valid[right]:
        raise ValueError("RTC activation lacks a valid right interpolation point")
    d = right + 1
    end = int(np.searchsorted(stamps, plan.end_walltime, side="left"))
    weights = np.zeros(len(offsets), dtype=np.float64)
    weights[:d] = 1.0
    if end > d:
        indices = np.arange(d, end)
        c = (end - indices) / (end - d + 1)
        weights[indices] = c * np.expm1(c) / np.expm1(1.0)
    weights[~valid] = 0.0
    return {
        "schema": "egoverse_rtc_v1",
        "base_actions": targets,
        "weights": weights,
        "request_id": request_id,
    }


@dataclass
class _Request:
    future: object
    request_id: int
    requested_monotonic: float
    deadline_monotonic: float | None
    activation_monotonic: float | None
    activation_walltime: float | None
    hard_until_walltime: float | None
    result: dict | None = None


def run_rtc_loop(
    policy,
    provider,
    sink,
    keys,
    limits,
    log,
    args,
    *,
    observation_to_send_s,
    executor=None,
    monotonic=time.monotonic,
    wall_time=time.time,
    sleep=time.sleep,
    max_schedule_slip_s=0.1,
    command_freshness_s=0.1,
    poll_interval_s=0.005,
    max_wall_clock_skew_s=0.05,
):
    """Run one asynchronous RPC at a time; hold before discarding pending work.

    A cancelled/running request is drained non-blockingly before another RPC
    may start. Worker shutdown never delays a physical hold. RPC transport must
    retain its own finite timeout; an already-running thread cannot be killed.
    """
    delay = float(observation_to_send_s)
    if not math.isfinite(delay) or delay < 0:
        raise ValueError("observation_to_send_s must be finite and nonnegative")
    rtc_delay = float(args.rtc_delay_ms) / 1000
    transition_s = float(getattr(args, "rtc_transition_ms", 0)) / 1000
    smoothing_s = float(getattr(args, "rtc_smoothing_ms", 0)) / 1000
    gap_position = float(getattr(args, "rtc_max_position_gap_mm", 20)) / 1000
    gap_rotation = np.deg2rad(float(getattr(args, "rtc_max_rotation_gap_deg", 5)))
    if (
        not np.isfinite([smoothing_s, gap_position, gap_rotation]).all()
        or not 0 <= smoothing_s < policy.timing.effective_span_s
        or min(gap_position, gap_rotation) <= 0
    ):
        raise ValueError("Invalid RTC continuity settings")
    schedule = check_rtc_schedule(
        policy.timing,
        args.control_frequency,
        args.execute_steps,
        rtc_delay,
        transition_s=transition_s,
    )
    interval, replan_s = schedule["control_interval_s"], schedule["replan_interval_s"]
    for value in (
        max_schedule_slip_s,
        command_freshness_s,
        poll_interval_s,
        max_wall_clock_skew_s,
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("RTC scheduling tolerances must be finite and positive")
    own_executor = executor is None
    if own_executor:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-rtc")
    epoch_mono, epoch_wall = monotonic(), wall_time()
    last_clock = epoch_mono
    plan = pending = retired = previous = None
    planned_tick = last_send = None
    used = serial = 0
    next_request = None
    first_send = None
    previous_state = None
    held = True
    next_health = 0.0

    def clock_pair():
        nonlocal last_clock
        now, wall = float(monotonic()), float(wall_time())
        if (
            not np.isfinite([now, wall]).all()
            or now < last_clock
            or abs((wall - epoch_wall) - (now - epoch_mono)) > max_wall_clock_skew_s
        ):
            raise RuntimeError("RTC wall clock changed relative to monotonic clock")
        last_clock = now
        return now, wall

    def invalidate():
        nonlocal pending, retired
        if pending is not None:
            if hasattr(pending.future, "cancel"):
                pending.future.cancel()
            retired, pending = pending, None

    def hold(reason):
        nonlocal plan, previous, held, planned_tick, last_send, next_request, first_send
        # This must precede any result(), cancel(), device teardown or logging.
        try:
            sink.hold(reason)
            held = True
        finally:
            invalidate()
            plan = previous = planned_tick = last_send = None
            next_request = first_send = None

    def pause(reason, **details):
        hold(reason)
        keys.pause()
        log.event("rtc_paused", reason=reason, **details)
        log.cycle(
            {
                "status": "paused",
                "timing_mode": "rtc",
                "pause_reason": reason,
                "hold_attempted": True,
                "discarded_remaining_actions": True,
                **details,
            }
        )
        message = reason
        if reason == "rtc_plan_disagreement":
            values = details["transition_disagreement"]
            gaps = "，".join(
                f"{side} {value['max_position_m'] * 1000:.1f} mm / "
                f"{np.rad2deg(value['max_rotation_rad']):.1f}°"
                for side, value in values.items()
            )
            message = f"新旧计划分歧过大（{gaps}），已拒绝接管"
        print(f"[推理] RTC 已保持：{message}；空格重新观察并推理，Q 退出。", flush=True)

    def worker(old_plan, hard_until_wall, request_id):
        started = monotonic()
        frame = provider.latest()
        frame.check_freshness(
            wall_time(),
            limits.max_observation_age_s,
            limits.future_timestamp_tolerance_s,
        )
        prefix = (
            None
            if old_plan is None
            else build_rtc_prefix(
                old_plan,
                policy.timing,
                frame.timestamp,
                delay,
                hard_until_wall,
                request_id,
            )
        )
        prediction = policy.predict(frame, rtc=prefix)
        if prediction["observation_timestamp"] != frame.timestamp:
            raise ValueError("RTC prediction timestamp differs from its observation")
        actions = np.asarray(prediction["base_actions"], dtype=np.float64)
        raw_plan = ChunkPlan(actions, policy.timing, frame.timestamp + delay)
        candidate = conditioned_plan(
            actions, policy.timing, frame.timestamp + delay, smoothing_s
        )
        for row in actions:
            limits.check_action(row)
        for row in candidate.actions:
            limits.check_action(row)
        transition_gap = {}
        if transition_s:
            if old_plan is None:
                # On resume, only compare the initial target to the held
                # feedback; intentional motion later in the plan is allowed.
                held_plan = ConstantPosePlan(frame.base_state, candidate.end_walltime)
                transition_gap = transition_disagreement(
                    candidate, held_plan, candidate.anchor, 0
                )
            else:
                audit_start = hard_until_wall - interval
                transition_gap = transition_disagreement(
                    candidate,
                    old_plan,
                    audit_start,
                    transition_s + interval + schedule["model_interval_s"],
                )
        prefix_errors = {}
        hard_until = None
        if prefix is not None:
            hard = prefix["weights"] == 1
            hard_until = candidate.anchor + candidate.offsets[np.flatnonzero(hard)[-1]]
            for side, k in (("left", 0), ("right", 7)):
                if np.any(hard):
                    target = prefix["base_actions"][hard, k : k + 6]
                    predicted = actions[hard, k : k + 6]
                    prefix_errors[side] = {
                        "max_position_m": float(
                            np.linalg.norm(
                                predicted[:, :3] - target[:, :3], axis=1
                            ).max()
                        ),
                        "max_rotation_rad": float(
                            (
                                Rotation.from_rotvec(target[:, 3:]).inv()
                                * Rotation.from_rotvec(predicted[:, 3:])
                            )
                            .magnitude()
                            .max()
                        ),
                    }
        return {
            "frame": frame,
            "prediction": prediction,
            "plan": candidate,
            "raw_plan": raw_plan,
            "prefix_hard_until_walltime": hard_until,
            "prefix_weights": None if prefix is None else prefix["weights"],
            "transition_disagreement": transition_gap,
            "elapsed_s": monotonic() - started,
            "completed_monotonic": monotonic(),
            "prefix_hard_count": 0
            if prefix is None
            else int(np.sum(prefix["weights"] == 1)),
            "prefix_overlap_count": 0
            if prefix is None
            else int(np.sum(prefix["weights"] > 0)),
            "hard_prefix_residual": prefix_errors,
        }

    def request(now, wall):
        nonlocal pending, serial, next_request
        serial += 1
        request_id = serial
        deadline = None if plan is None else now + rtc_delay
        activation = activation_wall = hard_until_wall = None
        if plan is not None:
            # Forecast from the current send phase. Actual sends can drift;
            # protect one extra interval and verify coverage at actual takeover.
            ticks = max(0, math.ceil((deadline - planned_tick) / interval - 1e-9))
            activation = planned_tick + ticks * interval
            activation_wall = wall + (activation - now)
            hard_until_wall = activation_wall + interval
            if (
                hard_until_wall + schedule["model_interval_s"] + transition_s
                >= plan.end_walltime
            ):
                pause("rtc_insufficient_overlap")
                return
        next_request = now + replan_s
        pending = _Request(
            executor.submit(worker, plan, hard_until_wall, request_id),
            request_id,
            now,
            deadline,
            activation,
            activation_wall,
            hard_until_wall,
        )
        log.event(
            "rtc_requested",
            request_id=request_id,
            requested_monotonic=now,
            requested_walltime=wall,
            deadline_monotonic=deadline,
            activation_monotonic=activation,
            activation_walltime=activation_wall,
            hard_until_walltime=hard_until_wall,
            next_request_monotonic=next_request,
        )

    def activate(item, now, wall):
        nonlocal plan, previous, planned_tick, last_send, held, first_send
        result = item.result
        frame, prediction = result["frame"], result["prediction"]
        frame.check_freshness(
            wall, limits.max_observation_age_s, limits.future_timestamp_tolerance_s
        )
        actions = np.asarray(prediction["base_actions"], dtype=np.float64)
        initial = item.activation_monotonic is None
        candidate = result["plan"]
        if initial:
            candidate.anchor = wall
            result["raw_plan"].anchor = wall
        remaining = candidate.end_walltime - wall
        needed = (
            max(0.0, next_request - now)
            + rtc_delay
            + 2 * interval
            + schedule["model_interval_s"]
            + transition_s
        )
        if not initial and remaining < needed:
            pause("rtc_insufficient_horizon", remaining_s=remaining)
            return False
        if initial:
            previous = np.asarray(frame.base_state, dtype=np.float64)
            planned_tick, last_send = now, None
        limits.check_action(candidate.sample(wall), previous)
        transition_gap = result["transition_disagreement"]
        if transition_s and any(
            values["max_position_m"] > gap_position
            or values["max_rotation_rad"] > gap_rotation
            for values in transition_gap.values()
        ):
            pause(
                "rtc_plan_disagreement",
                request_id=item.request_id,
                transition_disagreement=transition_gap,
            )
            return False
        execution_actions = candidate.actions
        if transition_s:
            previous_plan = (
                plan
                if plan is not None
                else ConstantPosePlan(frame.base_state, candidate.end_walltime)
            )
            if wall + transition_s >= min(
                previous_plan.end_walltime, candidate.end_walltime
            ):
                pause("rtc_insufficient_transition_overlap", request_id=item.request_id)
                return False
            candidate = ContinuousPlan(candidate, previous_plan, wall, transition_s)
        old_plan, plan = plan, candidate
        held = False
        # Publish only AFTER the first send, using its actual target time.
        # Diagnostics must not consume the protected takeover window.
        first_send = (
            item,
            old_plan,
            {
                "status": "completed",
                "timing_mode": "rtc",
                "hold_attempted": False,
                "observation_timestamp": float(frame.timestamp),
                "measured_base_state": np.asarray(frame.base_state),
                "source_timestamps": dict(frame.source_timestamps),
                "measured_inference_elapsed_s": result["elapsed_s"],
                "policy_inference_elapsed_s": float(prediction["inference_elapsed_s"]),
                "camera_actions": np.asarray(prediction["camera_actions"]),
                "base_actions": actions,
                "execution_base_actions": execution_actions,
                "execution_transition_s": transition_s,
                "execution_smoothing_s": smoothing_s,
                "transition_disagreement": transition_gap,
                "request_id": item.request_id,
                "requested_monotonic": item.requested_monotonic,
                "deadline_monotonic": item.deadline_monotonic,
                "activation_walltime": item.activation_walltime,
                "hard_until_walltime": item.hard_until_walltime,
                "prefix_hard_until_walltime": result["prefix_hard_until_walltime"],
                "start": "resume" if initial else "rtc",
                "prefix_hard_count": result["prefix_hard_count"],
                "prefix_overlap_count": result["prefix_overlap_count"],
                "hard_prefix_residual": result["hard_prefix_residual"],
            },
        )
        return True

    try:
        while True:
            state = keys.poll()
            if state != previous_state:
                if (
                    state == "pause"
                    and previous_state == "run"
                    and (not held or pending is not None)
                ):
                    hold("paused")
                log.event("control", state=state)
                print(
                    {
                        "pause": "[推理] RTC 已暂停；终端空格开始，Q 退出。",
                        "run": "[推理] RTC 运行中；终端空格/P 暂停，Q 退出。",
                        "stop": "[推理] RTC 正在退出。",
                    }[state],
                    flush=True,
                )
                previous_state = state
            if state == "stop":
                hold("stopped")
                return "stopped"
            # Do not queue another RPC behind an invalidated, still-running one.
            if retired is not None and retired.future.done():
                try:
                    retired.future.result()
                except BaseException:
                    pass
                retired = None
            now, wall = clock_pair()
            if state == "pause":
                if retired is None and now >= next_health:
                    provider.check_health()
                    next_health = now + 0.1
                sleep(poll_interval_s)
                continue
            if pending is not None and pending.result is None and pending.future.done():
                pending.result = pending.future.result()
                if (
                    pending.deadline_monotonic is not None
                    and pending.result["completed_monotonic"]
                    > pending.deadline_monotonic + 1e-9
                ):
                    pause("rtc_late_result", request_id=pending.request_id)
                    continue
            if (
                pending is not None
                and pending.deadline_monotonic is not None
                and now >= pending.deadline_monotonic - 1e-9
                and pending.result is None
            ):
                pause("rtc_deadline_missed", request_id=pending.request_id)
                continue
            if plan is None:
                if pending is not None and pending.result is not None:
                    item, pending = pending, None
                    if not activate(item, now, wall):
                        continue
                elif pending is None and retired is None:
                    if args.max_chunks is not None and used >= args.max_chunks:
                        hold("completed")
                        return "completed"
                    log.check()
                    now, wall = clock_pair()
                    request(now, wall)
                if plan is None:
                    sleep(poll_interval_s)
                    continue
            # Request cadence is measured from the last request, independent
            # of activation or the number of sends made on the current plan.
            # A missed slot causes one request, never a burst of catch-up RPCs.
            if (
                first_send is None
                and pending is None
                and retired is None
                and now >= next_request - 1e-9
                and (args.max_chunks is None or used < args.max_chunks)
            ):
                log.check()
                now, wall = clock_pair()
                request(now, wall)
                if plan is None:  # The overlap check paused the session.
                    continue
            now, wall = clock_pair()
            due = (
                planned_tick
                if last_send is None
                else max(planned_tick, last_send + interval)
            )
            if now < due - 1e-9:
                sleep(min(poll_interval_s, due - now))
                continue
            if now - planned_tick > max_schedule_slip_s + 1e-9:
                pause("schedule_slip", slip_s=now - planned_tick)
                continue
            if wall > plan.end_walltime + 1e-9:
                if args.max_chunks is not None and used >= args.max_chunks:
                    hold("completed")
                    return "completed"
                pause("rtc_plan_expired")
                continue
            # Switch only on an actual send tick, not on an earlier polling
            # iteration that will then wait for the next control interval.
            if (
                pending is not None
                and pending.activation_monotonic is not None
                and now >= pending.activation_monotonic - 1e-9
            ):
                # Recheck after any work since the polling clock above. A
                # result can never borrow the rounding-to-tick allowance.
                if pending.result is None and pending.future.done():
                    pending.result = pending.future.result()
                if pending.result is None:
                    pause("rtc_deadline_missed", request_id=pending.request_id)
                    continue
                if (
                    pending.result["completed_monotonic"]
                    > pending.deadline_monotonic + 1e-9
                ):
                    pause("rtc_late_result", request_id=pending.request_id)
                    continue
                item, pending = pending, None
                if not activate(item, now, wall):
                    continue
            now, wall = clock_pair()
            if first_send is not None:
                item, _, report = first_send
                coverage = report["prefix_hard_until_walltime"]
                if coverage is not None:
                    right = int(
                        np.searchsorted(plan.anchor + plan.offsets, wall, side="left")
                    )
                    bracket = [max(0, right - 1), right]
                    weights = item.result["prefix_weights"]
                    if (
                        wall > coverage + 1e-9
                        or right >= len(weights)
                        or not np.all(weights[bracket] == 1)
                    ):
                        pause(
                            "rtc_hard_coverage_expired",
                            request_id=item.request_id,
                            first_target_walltime=wall,
                            prefix_hard_until_walltime=coverage,
                        )
                        continue
                    report["first_target_bracket_indices"] = bracket
                    report["first_target_bracket_weights"] = weights[bracket]
            row = plan.sample(wall)
            limits.check_action(row, previous)
            sent_at = monotonic()
            receipt = sink.send_action(row, wall_time() + command_freshness_s)
            if isinstance(receipt, dict) and "send_started_monotonic" in receipt:
                submitted = float(receipt["send_started_monotonic"])
                if (
                    not math.isfinite(submitted)
                    or not sent_at <= submitted <= monotonic()
                ):
                    raise RuntimeError("Invalid RTC host submission timestamp")
                sent_at = submitted
            previous, last_send = row, sent_at
            # Rebase the NEXT send only. Tiny OS wakeup overruns must not
            # accumulate into a false 100 ms stall after minutes of operation.
            # The action plan still uses absolute timestamps, without stretching
            # its horizon or sending a burst of old points to catch up.
            planned_tick = sent_at + interval
            if first_send is not None:
                used += 1  # Count only candidates that actually sent a target.
                item, old_plan, report = first_send
                switch_delta = {}
                if old_plan is not None:
                    old_action = old_plan.sample(wall)
                    for side, k in (("left", 0), ("right", 7)):
                        switch_delta[side] = {
                            "position_m": float(
                                np.linalg.norm(row[k : k + 3] - old_action[k : k + 3])
                            ),
                            "rotation_rad": float(
                                (
                                    Rotation.from_rotvec(
                                        old_action[k + 3 : k + 6]
                                    ).inv()
                                    * Rotation.from_rotvec(row[k + 3 : k + 6])
                                ).magnitude()
                            ),
                        }
                raw_row = item.result["raw_plan"].sample(wall)
                raw_delta = (
                    {
                        side: {
                            "position_m": float(
                                np.linalg.norm(
                                    raw_row[k : k + 3]
                                    - old_plan.sample(wall)[k : k + 3]
                                )
                            ),
                            "rotation_rad": float(
                                (
                                    Rotation.from_rotvec(
                                        old_plan.sample(wall)[k + 3 : k + 6]
                                    ).inv()
                                    * Rotation.from_rotvec(raw_row[k + 3 : k + 6])
                                ).magnitude()
                            ),
                        }
                        for side, k in (("left", 0), ("right", 7))
                    }
                    if old_plan is not None
                    else {}
                )
                report.update(
                    switch_walltime=wall,
                    first_target_walltime=wall,
                    first_send_monotonic=sent_at,
                    first_send_walltime=epoch_wall + (sent_at - epoch_mono),
                    start_offset_s=plan.offset(wall),
                    switch_plan_delta=switch_delta,
                    raw_switch_plan_delta=raw_delta,
                    request_elapsed_s=sent_at - item.requested_monotonic,
                )
                log.cycle(report)
                print(
                    f"[推理] chunk {used}: RTC，预测 {report['measured_inference_elapsed_s'] * 1000:.1f} ms，"
                    f"切入 {plan.offset(wall) * 1000:.0f} ms",
                    flush=True,
                )
                first_send = None
    except BaseException:
        hold("failed")
        raise
    finally:
        # Join only AFTER hold. The caller may close camera shared memory after
        # return, so its worker must have stopped reading it. The RPC transport
        # has a finite timeout; pauses above never wait for this shutdown.
        try:
            if not held:
                hold("session_exit")
        finally:
            try:
                invalidate()
            finally:
                if own_executor:
                    executor.shutdown(wait=True, cancel_futures=True)
