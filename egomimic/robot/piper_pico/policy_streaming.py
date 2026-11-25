"""Pipelined policy execution: infer the next chunk while the current one runs.

A chunk predicted from an observation at ``t_obs`` is placed on the wall clock
like the training rows: offset ``s`` targets ``t_obs + observation_to_send_s + s``.
The first chunk after start/resume begins at offset zero at its first send,
because the robot was held while that prediction ran.

Every ``replan_steps`` sends a new observation is taken on the control thread
and handed to one inference worker. When the result arrives the command
crossfades from the old chunk to the new one over ``blend_steps`` control
intervals (linear xyz/width, SLERP rotation), so the arm never waits for
inference and never jumps between independently sampled chunks.

No device or CAN module is imported here.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from egomimic.robot.piper_pico.policy_execution import _row


class ChunkPlan:
    """One prediction placed on the wall clock; samples clamp to its endpoints."""

    def __init__(self, actions, timing, anchor_walltime):
        self.actions = np.asarray(actions, dtype=np.float64).copy()
        if (
            self.actions.shape != (timing.output_chunk_length, 14)
            or not np.isfinite(self.actions).all()
        ):
            raise ValueError("chunk actions must be finite with checkpoint shape")
        self.offsets = np.asarray(timing.output_offsets_s, dtype=np.float64)
        self.span = float(timing.effective_span_s)
        self.anchor = float(anchor_walltime)
        self._slerps = [
            Slerp(self.offsets, Rotation.from_rotvec(self.actions[:, k + 3 : k + 6]))
            for k in (0, 7)
        ]

    @property
    def end_walltime(self):
        return self.anchor + self.span

    def offset(self, walltime):
        return float(walltime) - self.anchor

    def sample(self, walltime):
        offset = min(max(self.offset(walltime), 0.0), self.span)
        row = np.empty(14)
        for k, slerp in zip((0, 7), self._slerps):
            for column in (k, k + 1, k + 2, k + 6):
                row[column] = np.interp(offset, self.offsets, self.actions[:, column])
            row[k + 3 : k + 6] = slerp([offset]).as_rotvec()[0]
        return row


def blend_rows(old, new, weight):
    """weight 0 -> old, 1 -> new; shortest-path rotation per arm."""
    weight = float(np.clip(weight, 0.0, 1.0))
    row = (1.0 - weight) * old + weight * new
    for k in (0, 7):
        a = Rotation.from_rotvec(old[k + 3 : k + 6])
        delta = (a.inv() * Rotation.from_rotvec(new[k + 3 : k + 6])).as_rotvec()
        row[k + 3 : k + 6] = (a * Rotation.from_rotvec(weight * delta)).as_rotvec()
    return row


def check_streaming_schedule(timing, control_frequency, replan_steps, blend_steps):
    """Reject settings whose replan + blend cannot fit inside one chunk span.

    A chunk is entered ``latency`` after its observation and must stay valid
    while the next one is inferred, so 2 * latency + replan + blend <= span.
    """
    interval = 1.0 / float(control_frequency)
    needed = (replan_steps + blend_steps) * interval
    budget = (timing.effective_span_s - needed) / 2
    if budget <= interval:
        raise ValueError(
            f"--execute-steps {replan_steps} + --blend-steps {blend_steps} at "
            f"{control_frequency:g} Hz needs {needed:.3f}s, but a chunk spans "
            f"{timing.effective_span_s:.3f}s; leave room for inference latency"
        )
    return {
        "replan_interval_s": replan_steps * interval,
        "blend_duration_s": blend_steps * interval,
        "max_inference_latency_s": budget,
    }


class _Prediction:
    def __init__(self, frame, future, requested_monotonic):
        self.frame, self.future = frame, future
        self.requested_monotonic = requested_monotonic


def run_streaming_loop(
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
):
    """Keyboard-controlled pipelined loop; returns stopped/completed."""
    delay = float(observation_to_send_s)
    if not math.isfinite(delay) or delay < 0:
        raise ValueError("observation_to_send_s must be finite and nonnegative")
    interval = 1.0 / float(args.control_frequency)
    replan_steps, blend_steps = int(args.execute_steps), int(args.blend_steps)
    check_streaming_schedule(
        policy.timing, args.control_frequency, replan_steps, blend_steps
    )
    blend_s = blend_steps * interval
    own_executor = executor is None
    if own_executor:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy")

    plan = previous_plan = pending = previous = None
    blend_start = next_tick = None
    steps_on_plan = used = 0
    previous_state = None
    running_sent = False
    exhausted_logged = False
    next_health = 0.0

    def drop_pending():
        nonlocal pending
        if pending is not None:
            try:
                pending.future.result()
            except Exception:
                pass
            pending = None

    def stop_running(reason):
        nonlocal plan, previous_plan, previous, running_sent
        drop_pending()
        if running_sent:
            sink.hold(reason)
        plan = previous_plan = previous = None
        running_sent = False

    def cycle_report(frame, prediction, elapsed, **fields):
        stamps = {k: float(v) for k, v in frame.source_timestamps.items()}
        report = {
            "status": "completed",
            "timing_mode": "streaming",
            "audit_request_id": prediction.get("audit_request_id"),
            "observation_timestamp": float(frame.timestamp),
            "measured_base_state": np.asarray(frame.base_state).tolist(),
            "source_timestamps": stamps,
            "source_ages_s": {k: float(frame.timestamp - v) for k, v in stamps.items()},
            "measured_inference_elapsed_s": float(elapsed),
            "policy_inference_elapsed_s": float(prediction["inference_elapsed_s"]),
            "camera_actions": np.asarray(prediction["camera_actions"]).tolist(),
            **fields,
        }
        used_now = used + 1
        print(
            f"[推理] chunk {used_now}: 推理 {elapsed * 1000:.1f} ms，"
            f"切入偏移 {fields.get('start_offset_s', 0.0) * 1000:.0f} ms",
            flush=True,
        )
        return report

    def checked_prediction(frame, prediction):
        if prediction["observation_timestamp"] != frame.timestamp:
            raise ValueError("Prediction timestamp differs from its observation")
        actions = np.asarray(prediction["base_actions"], dtype=np.float64)
        for row in actions:
            limits.check_action(row)
        return actions

    try:
        while True:
            state = keys.poll()
            if state != previous_state:
                if state == "pause" and previous_state == "run":
                    stop_running("paused")
                log.event("control", state=state)
                print(
                    {
                        "pause": "[推理] 已暂停；终端空格开始，Q 退出。",
                        "run": "[推理] 运行中（流水线推理）；终端空格/P 暂停，Q 退出。",
                        "stop": "[推理] 正在退出。",
                    }[state],
                    flush=True,
                )
                previous_state = state
            if state == "stop":
                drop_pending()
                sink.hold("stopped")
                return "stopped"
            if state == "pause":
                if monotonic() >= next_health:
                    provider.check_health()
                    next_health = monotonic() + 0.1
                sleep(0.01)
                continue

            if plan is None:
                if args.max_chunks is not None and used >= args.max_chunks:
                    sink.hold("completed")
                    return "completed"
                # Start/resume: the arm is held, so the chunk starts at offset 0.
                log.check()
                frame = provider.latest()
                started = time.perf_counter()
                prediction = policy.predict(frame)
                elapsed = time.perf_counter() - started
                actions = checked_prediction(frame, prediction)
                if keys.poll() != "run":
                    continue
                now_wall = wall_time()
                limits.check_observation(frame.timestamp, now_wall)
                limits.check_action(actions[0], frame.base_state)
                plan = ChunkPlan(actions, policy.timing, now_wall)
                log.cycle(
                    cycle_report(
                        frame,
                        prediction,
                        elapsed,
                        switch_walltime=now_wall,
                        start_offset_s=0.0,
                        blend_s=0.0,
                        start="resume",
                    )
                )
                used += 1
                previous_plan, steps_on_plan = None, 0
                next_tick = monotonic()
                previous = np.asarray(frame.base_state, dtype=np.float64)
                exhausted_logged = False

            # Wait for the next control tick, polling keys for fast pause.
            paused = False
            while True:
                now = monotonic()
                if now >= next_tick:
                    break
                if keys.poll() != "run":
                    paused = True
                    break
                sleep(min(poll_interval_s, next_tick - now))
            if paused:
                continue
            slip = now - next_tick
            if slip > max_schedule_slip_s:
                stop_running("failed")
                keys.pause()
                log.cycle(
                    {
                        "status": "paused",
                        "pause_reason": "schedule_slip",
                        "timing_mode": "streaming",
                        "slip_s": slip,
                    }
                )
                log.event("schedule_paused", slip_s=slip, limit_s=max_schedule_slip_s)
                print(
                    "[推理] 动作调度超时，双臂已保持。终端空格重新观察并推理，Q 退出。",
                    flush=True,
                )
                continue
            now_wall = wall_time()

            if pending is not None and pending.future.done():
                ready, pending = pending, None
                frame = ready.frame
                prediction = ready.future.result()
                elapsed = monotonic() - ready.requested_monotonic
                actions = checked_prediction(frame, prediction)
                candidate = ChunkPlan(actions, policy.timing, frame.timestamp + delay)
                offset = candidate.offset(now_wall)
                fresh = now_wall - frame.timestamp <= limits.max_observation_age_s
                if fresh and offset + blend_s < candidate.span:
                    previous_plan, plan = plan, candidate
                    blend_start, steps_on_plan = now_wall, 0
                    exhausted_logged = False
                    log.cycle(
                        cycle_report(
                            frame,
                            prediction,
                            elapsed,
                            switch_walltime=now_wall,
                            start_offset_s=offset,
                            blend_s=blend_s,
                            start="pipelined",
                        )
                    )
                    used += 1
                else:
                    log.event(
                        "prediction_discarded",
                        observation_timestamp=frame.timestamp,
                        start_offset_s=offset,
                        observation_age_s=now_wall - frame.timestamp,
                    )
                log.check()

            target_wall = now_wall
            row = plan.sample(target_wall)
            if previous_plan is not None:
                weight = (target_wall - blend_start + interval) / max(blend_s, 1e-9)
                if blend_s <= 0 or weight >= 1.0:
                    previous_plan = None
                else:
                    row = blend_rows(previous_plan.sample(target_wall), row, weight)
            if plan.offset(target_wall) > plan.span + 1e-6 and previous_plan is None:
                # Inference was too slow: keep the last target, never extrapolate.
                if not exhausted_logged:
                    log.event("plan_exhausted", pending=pending is not None)
                    print("[推理] 新 chunk 未及时到达，保持上一段末点。", flush=True)
                    exhausted_logged = True
                if pending is None and (
                    args.max_chunks is not None and used >= args.max_chunks
                ):
                    stop_running("completed")
                    return "completed"
            else:
                row = _row(row, "streamed action")
                limits.check_action(row, previous)
                sink.send_action(row, wall_time() + command_freshness_s)
                running_sent = True
                previous = row
            steps_on_plan += 1
            next_tick = max(next_tick + interval, now)

            if (
                pending is None
                and steps_on_plan >= replan_steps
                and (args.max_chunks is None or used < args.max_chunks)
            ):
                frame = provider.latest()
                limits.check_observation(frame.timestamp, wall_time())
                pending = _Prediction(
                    frame, executor.submit(policy.predict, frame), monotonic()
                )
    finally:
        drop_pending()
        if own_executor:
            executor.shutdown(wait=True)
