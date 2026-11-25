"""Synchronous observe -> predict -> execute orchestration without device imports.

Providers supply their latest ObservationFrame in the wall clock's domain.
Pause discards a block; restarting always obtains a newer observation and makes
a new prediction. A failed cycle is returned as an audit report after hold.
"""

from __future__ import annotations

import math
import time
from numbers import Integral

import numpy as np

from egomimic.robot.piper_pico.policy_execution import (
    BlockExecutionError,
    execute_action_block,
    resample_action_block,
)


def _hold(sink, report):
    report["hold_attempted"] = True
    try:
        sink.hold(report["status"])
    except Exception as exc:
        report["hold_error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["status"] = "failed"


def _control(callback, events, wall_time):
    def poll():
        value = callback() if callback else "run"
        value = "run" if value is None else value
        if value not in ("run", "pause", "stop"):
            raise ValueError("control_callback must return run, pause, stop or None")
        if not events or events[-1]["command"] != value:
            events.append({"command": value, "walltime": float(wall_time())})
        return value

    return poll


def run_sync_cycle(
    policy,
    provider,
    sink,
    limits,
    *,
    control_frequency=30.0,
    execute_steps=None,
    control_callback=None,
    monotonic=time.monotonic,
    wall_time=time.time,
    sleep=time.sleep,
    newer_than=None,
    send_not_before_monotonic=None,
    hold_on_completion=True,
    **execution_options,
):
    """Run one block using provider.latest(), checking all sources before/after.

    Without ``observation_to_send_s`` the executor sends the chunk from offset
    zero once prediction completes. With it, sends follow the recorded delay
    from observation to command (see ``execute_action_block``). Exceptions
    become failed reports; process interrupts propagate after attempting hold.
    A provider may raise StopIteration at recording end.
    Set ``hold_on_completion=False`` to continue to another block without a
    physical hold between predictions; the caller must hold at final shutdown.
    """
    report = {"status": "running", "control_events": [], "hold_attempted": False}
    poll = _control(control_callback, report["control_events"], wall_time)
    executor_owns_hold = False
    try:
        command = poll()
        if command != "run":
            report["status"] = "paused" if command == "pause" else "stopped"
            return report
        try:
            frame = provider.latest()
        except StopIteration:
            report["status"] = "exhausted"
            return report
        frame.check_freshness(
            wall_time(),
            limits.max_observation_age_s,
            limits.future_timestamp_tolerance_s,
        )
        if newer_than is not None and frame.timestamp <= newer_than:
            raise ValueError(
                "A new cycle requires a newer observation, never an old block"
            )
        report.update(
            observation_timestamp=float(frame.timestamp),
            measured_base_state=np.asarray(frame.base_state).tolist(),
            source_timestamps={k: float(v) for k, v in frame.source_timestamps.items()},
            source_ages_s={
                k: float(frame.timestamp - v)
                for k, v in frame.source_timestamps.items()
            },
            observation_age_before_inference_s=float(wall_time() - frame.timestamp),
        )
        if hasattr(provider, "last_record"):
            report["recording_observation"] = dict(provider.last_record)
        started = time.perf_counter()
        prediction = policy.predict(frame)
        report["audit_request_id"] = prediction.get("audit_request_id")
        report["measured_inference_elapsed_s"] = time.perf_counter() - started
        report["policy_inference_elapsed_s"] = float(prediction["inference_elapsed_s"])
        report["inference_steps"] = int(prediction["inference_steps"])
        if (
            not math.isfinite(report["policy_inference_elapsed_s"])
            or report["policy_inference_elapsed_s"] < 0
        ):
            raise ValueError(
                "Policy inference elapsed time must be finite and nonnegative"
            )
        if prediction["observation_timestamp"] != frame.timestamp:
            raise ValueError("Prediction timestamp differs from its observation")
        frame.check_freshness(
            wall_time(),
            limits.max_observation_age_s,
            limits.future_timestamp_tolerance_s,
        )
        report["observation_age_after_inference_s"] = float(
            wall_time() - frame.timestamp
        )
        camera_actions = np.asarray(prediction["camera_actions"])
        if (
            camera_actions.shape != (policy.timing.output_chunk_length, 14)
            or not np.isfinite(camera_actions).all()
        ):
            raise ValueError(
                "Camera actions must be finite with checkpoint action shape"
            )
        report["camera_actions"] = camera_actions.tolist()
        command = poll()
        if command != "run":
            report["status"] = "paused" if command == "pause" else "stopped"
            return report
        candidate = resample_action_block(
            prediction["base_actions"],
            policy.timing,
            control_frequency,
            execute_steps=execute_steps,
        )
        report["candidate_schedule"] = [
            {"index": i, "nominal_offset_s": float(offset), "action": row.tolist()}
            for i, (offset, row) in enumerate(
                zip(candidate.offsets_s, candidate.actions)
            )
        ]
        if send_not_before_monotonic is not None:
            deadline = float(send_not_before_monotonic)
            poll_interval = float(execution_options.get("poll_interval_s", 0.01))
            if (
                not math.isfinite(deadline)
                or not math.isfinite(poll_interval)
                or poll_interval <= 0
            ):
                raise ValueError(
                    "Inter-block deadline and polling interval must be finite; interval must be positive"
                )
            started_wait = float(monotonic())
            if not math.isfinite(started_wait):
                raise ValueError("Inter-block monotonic clock must be finite")
            last_clock = started_wait
            wait_report = {
                "not_before_monotonic": deadline,
                "started_monotonic": started_wait,
                "elapsed_s": 0.0,
                "status": "running",
            }
            report["inter_block_wait"] = wait_report
            while True:
                command = poll()
                now = float(monotonic())
                if not math.isfinite(now) or now < last_clock:
                    raise ValueError(
                        "Inter-block monotonic clock moved backwards or is non-finite"
                    )
                last_clock = now
                wait_report["elapsed_s"] = now - started_wait
                if command != "run":
                    report["status"] = "paused" if command == "pause" else "stopped"
                    wait_report["status"] = report["status"]
                    return report
                if now >= deadline:
                    wait_report["status"] = "ready"
                    break
                sleep(min(poll_interval, deadline - now))
            # Waiting must not turn an otherwise fresh prediction into a send
            # based on a stalled camera or arm. A pause discards this prediction.
            frame.check_freshness(
                wall_time(),
                limits.max_observation_age_s,
                limits.future_timestamp_tolerance_s,
            )
            report["observation_age_after_inter_block_wait_s"] = float(
                wall_time() - frame.timestamp
            )
        executor_owns_hold = True
        report["execution"] = execute_action_block(
            prediction["base_actions"],
            policy.timing,
            sink,
            control_frequency=control_frequency,
            initial_action=frame.base_state,
            observation_timestamp=frame.timestamp,
            limits=limits,
            execute_steps=execute_steps,
            monotonic=monotonic,
            wall_time=wall_time,
            sleep=sleep,
            control_callback=poll,
            hold_on_completion=hold_on_completion,
            **execution_options,
        )
        report["status"] = report["execution"]["status"]
        report["model_target_timestamps"] = (
            frame.timestamp + np.asarray(policy.timing.output_offsets_s)
        ).tolist()
        report["scheduling_semantics"] = (
            "block starts after prediction; no latency compensation"
            if execution_options.get("observation_to_send_s") is None
            else "send at observation + recorded observation_to_send_s; late blocks start inside the chunk"
        )
    except BlockExecutionError as exc:
        report.update(status="failed", execution=exc.report)
        report["error"] = exc.report.get("error", exc.report.get("hold_error"))
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    except BaseException:
        report["status"] = "interrupted"
        raise
    finally:
        execution = report.get("execution")
        if execution is not None:
            report["hold_attempted"] = execution["hold_attempted"]
        elif executor_owns_hold and report["status"] == "interrupted":
            # The executor always attempts hold before propagating interrupts.
            report["hold_attempted"] = True
        if not report["hold_attempted"] and (
            report["status"] != "completed" or hold_on_completion
        ):
            _hold(sink, report)
    return report


def run_sync_policy(
    policy,
    provider,
    sink,
    limits,
    *,
    max_chunks=1,
    control_callback=None,
    monotonic=time.monotonic,
    wall_time=time.time,
    sleep=time.sleep,
    pause_poll_s=0.01,
    **cycle_options,
):
    """Run bounded synchronous cycles; a pause waits for run/stop then repredicts.

    max_chunks counts cycles which consumed observations (including a paused
    block). Fresh observations are mandatory between predictions. Failed cycles
    end the run immediately; callers must inspect the returned status. Between
    blocks, prediction runs first, then only the remaining time until one actual
    resampled command interval after the previous send is waited. A slow
    prediction does not incur an additional full control interval.
    With ``hold_on_completion=False``, successful intermediate cycles do not
    hold, but the final bounded-run completion still holds once before return.
    """
    if (
        isinstance(max_chunks, bool)
        or not isinstance(max_chunks, Integral)
        or max_chunks < 1
    ):
        raise ValueError("max_chunks must be a positive integer")
    if not math.isfinite(pause_poll_s) or pause_poll_s <= 0:
        raise ValueError("pause_poll_s must be positive and finite")
    report = {"status": "running", "cycles": [], "control_events": []}
    poll = _control(control_callback, report["control_events"], wall_time)
    last_timestamp = None
    next_send_not_before = None
    used = 0
    try:
        while used < max_chunks:
            cycle = run_sync_cycle(
                policy,
                provider,
                sink,
                limits,
                control_callback=poll,
                monotonic=monotonic,
                wall_time=wall_time,
                sleep=sleep,
                newer_than=last_timestamp,
                send_not_before_monotonic=next_send_not_before,
                **cycle_options,
            )
            report["cycles"].append(cycle)
            execution = cycle.get("execution", {})
            samples = execution.get("samples", [])
            if samples:
                next_send_not_before = (
                    samples[-1]["actual_monotonic"]
                    + 1.0 / execution["actual_control_frequency"]
                )
            if "observation_timestamp" in cycle:
                used += 1
                last_timestamp = cycle["observation_timestamp"]
            status = cycle["status"]
            if status in ("failed", "stopped", "exhausted"):
                report["status"] = status
                break
            if status == "paused":
                report["status"] = "paused"
                if used >= max_chunks:
                    break
                while True:
                    command = poll()
                    if command == "stop":
                        report["status"] = "stopped"
                        return report
                    if command == "run":
                        break
                    sleep(pause_poll_s)
        else:
            report["status"] = "completed"
            if not cycle_options.get("hold_on_completion", True):
                _hold(sink, report)
    except Exception as exc:
        report.update(
            status="failed", error={"type": type(exc).__name__, "message": str(exc)}
        )
        _hold(sink, report)
    except BaseException:
        report["status"] = "interrupted"
        _hold(sink, report)
        raise
    finally:
        report["prediction_count"] = used
    return report


class OfflineReplayClock:
    """Virtual offline scheduling clock; CPU inference does not advance it.

    This tests nominal schedule geometry, not real-time inference feasibility.
    Actual inference wall duration is measured separately by run_sync_cycle.
    """

    def __init__(self, epoch=0.0):
        self.epoch = float(epoch)
        self.elapsed = 0.0
        if not math.isfinite(self.epoch):
            raise ValueError("epoch must be finite")

    def monotonic(self):
        return self.elapsed

    def wall_time(self):
        return self.epoch + self.elapsed

    def sleep(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("virtual sleep must be finite and nonnegative")
        self.elapsed += seconds
