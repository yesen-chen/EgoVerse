"""Independent RTC request-clock and first-command hard-coverage regressions.

These tests advance virtual worker/control clocks. They never touch a camera,
robot, GPU, socket, or real sleeping thread. In particular, a 20 ms SDK receipt
must not silently turn a 400 ms request cadence into a 600 ms switch cadence.
"""

from dataclasses import replace

import numpy as np
import pytest

from egomimic.robot.piper_pico.policy_rollout import OfflineReplayClock
from egomimic.robot.piper_pico.policy_rtc import (
    build_rtc_prefix,
    check_rtc_schedule,
    run_rtc_loop,
)
from egomimic.robot.piper_pico.policy_streaming import ChunkPlan
from tests.unit.test_live_policy import Provider, limits
from tests.unit.test_policy_rtc import (
    DELAY,
    TIMING,
    AsyncExecutor,
    Keys,
    Log,
    Policy,
    Sink,
    args,
)


def scenario(
    *,
    frequency=30.0,
    execute_steps=12,
    delay_ms=200.0,
    sink_lag=0.02,
    wakeup_lag=0.001,
    latencies=(0.08, 0.13),
    stop_time=3.0,
    pause_window=None,
    log_factory=Log,
    sink_factory=Sink,
    keys_factory=Keys,
    safety_limits=None,
    max_chunks=None,
    clock=None,
    control_sleep=None,
):
    clock = clock or OfflineReplayClock(epoch=1000)
    provider = Provider(clock)
    policy = Policy(clock, latencies=latencies)
    sink = sink_factory(clock, lag=sink_lag)
    log = log_factory()
    executor = AsyncExecutor(clock)
    keys = keys_factory(clock, stop_time=stop_time, pause_window=pause_window)
    status = run_rtc_loop(
        policy,
        provider,
        sink,
        keys,
        limits() if safety_limits is None else safety_limits,
        log,
        args(
            control_frequency=frequency,
            execute_steps=execute_steps,
            rtc_delay_ms=delay_ms,
            max_chunks=max_chunks,
        ),
        observation_to_send_s=DELAY,
        executor=executor,
        monotonic=clock.monotonic,
        wall_time=clock.wall_time,
        sleep=control_sleep or (lambda seconds: clock.sleep(seconds + wakeup_lag)),
    )
    return status, policy, sink, log, executor, keys


def guided_cycles(log):
    return [
        c
        for c in log.cycles
        if c.get("status") == "completed" and c.get("start") == "rtc"
    ]


def test_schedule_allows_six_hundred_ms_request_period_with_two_hundred_ms_delay():
    schedule = check_rtc_schedule(TIMING, 30, 18, 0.2)
    assert schedule["replan_interval_s"] == pytest.approx(0.6)
    assert schedule["required_span_s"] == pytest.approx(
        0.6 + 0.2 + 2 / 30 + TIMING.output_offsets_s[1]
    )
    assert schedule["required_span_s"] < TIMING.effective_span_s
    # The former activation-based cadence charged two inference delays and
    # incorrectly rejected this request-based schedule.
    assert 0.6 + 2 * 0.2 + 1 / 30 > TIMING.effective_span_s


@pytest.mark.parametrize("execute_steps", [6, 7, 8])
def test_schedule_rejects_period_at_or_below_delay_plus_two_ticks(execute_steps):
    with pytest.raises(ValueError, match="exceed delay \\+ two control intervals"):
        check_rtc_schedule(TIMING, 30, execute_steps, 0.2)


@pytest.mark.parametrize("between_grid_points,hard_count", [(False, 21), (True, 22)])
def test_prefix_covers_exact_or_between_grid_target_without_extra_hard_points(
    between_grid_points, hard_count
):
    actions = np.zeros((100, 14))
    actions[:, 0] = np.asarray(TIMING.output_offsets_s) * 0.02
    plan = ChunkPlan(actions, TIMING, 10.0)
    observation = 10.2
    grid = observation + DELAY + np.asarray(TIMING.output_offsets_s)
    latest_target = (grid[20] + grid[21]) / 2 if between_grid_points else grid[20]
    prefix = build_rtc_prefix(plan, TIMING, observation, DELAY, latest_target, 7)
    np.testing.assert_array_equal(prefix["weights"][:hard_count], 1)
    assert 0 < prefix["weights"][hard_count] < 1
    assert np.count_nonzero(prefix["weights"] == 1) == hard_count
    np.testing.assert_allclose(
        prefix["base_actions"][hard_count - 1], plan.sample(grid[hard_count - 1])
    )
    # Beyond the old plan, both values and weights remain unconstrained zeros;
    # an endpoint-clamped old action must not leak into the new free suffix.
    outside_old_plan = grid >= plan.end_walltime
    assert outside_old_plan.any()
    np.testing.assert_array_equal(prefix["weights"][outside_old_plan], 0)
    np.testing.assert_array_equal(prefix["base_actions"][outside_old_plan], 0)


def test_prefix_rejects_target_whose_right_support_passes_old_plan_end():
    plan = ChunkPlan(np.zeros((100, 14)), TIMING, 10.0)
    observation = 10.2
    grid = observation + DELAY + np.asarray(TIMING.output_offsets_s)
    last_valid = grid[grid < plan.end_walltime][-1]
    latest_target = (last_valid + plan.end_walltime) / 2
    assert last_valid < latest_target < plan.end_walltime
    assert grid[np.searchsorted(grid, latest_target)] >= plan.end_walltime
    with pytest.raises(ValueError, match="valid right interpolation point"):
        build_rtc_prefix(plan, TIMING, observation, DELAY, latest_target, 8)


@pytest.mark.parametrize(
    "frequency,steps,delay_ms",
    [(30.0, 12, 200.0), (29.0, 11, 183.0), (27.5, 11, 173.0)],
)
def test_requests_follow_request_clock_despite_receipts_and_wakeup(
    frequency, steps, delay_ms
):
    _, policy, sink, log, _, keys = scenario(
        frequency=frequency, execute_steps=steps, delay_ms=delay_ms
    )
    assert not keys.is_paused
    assert len(guided_cycles(log)) >= 4
    stamps = np.array([r[2] for r in policy.requests])
    period = steps / frequency
    # Request submission can be delayed by an in-progress SDK acknowledgement,
    # but waiting for the previous chunk's activation adds no 200 ms penalty.
    assert np.min(np.diff(stamps)) >= period - 1e-8
    assert np.max(np.diff(stamps)) <= period + 1 / frequency + 0.021
    assert np.mean(np.diff(stamps)) < period + 0.025
    sends = np.asarray([r[0] for r in sink.records])
    assert np.min(np.diff(sends)) >= 1 / frequency - 1e-9


@pytest.mark.parametrize(
    "frequency,steps,delay_ms",
    [(30.0, 12, 200.0), (29.0, 11, 183.0), (27.5, 11, 173.0)],
)
def test_first_new_target_is_bracketed_by_hard_model_grid_points(
    frequency, steps, delay_ms
):
    _, policy, sink, log, _, keys = scenario(
        frequency=frequency, execute_steps=steps, delay_ms=delay_ms
    )
    assert not keys.is_paused
    sends = np.asarray([r[0] + 1000 for r in sink.records])
    for cycle in guided_cycles(log):
        frame, prefix, _ = policy.requests[cycle["request_id"] - 1]
        first = np.searchsorted(sends, cycle["switch_walltime"] - 1e-9)
        if first == len(sends):
            continue
        new_start = frame.timestamp + DELAY
        grid = new_start + np.asarray(TIMING.output_offsets_s)
        stamp = sends[first]
        right = int(np.searchsorted(grid, stamp, side="left"))
        assert 0 < right < len(grid)
        # Floor-only coverage fails here whenever activation is between model
        # points. Both interpolation supports must carry full guidance weight.
        np.testing.assert_array_equal(prefix["weights"][[right - 1, right]], [1, 1])
        assert np.any((prefix["weights"] > 0) & (prefix["weights"] < 1))
        assert prefix["base_actions"].shape == (100, 14)


def test_timing_audit_distinguishes_deadline_tick_coverage_and_first_target():
    _, _, sink, log, _, keys = scenario()
    assert not keys.is_paused
    requested = {
        e["request_id"]: e for e in log.events if e.get("event") == "rtc_requested"
    }
    for cycle in guided_cycles(log):
        event = requested[cycle["request_id"]]
        assert cycle["deadline_monotonic"] == pytest.approx(
            cycle["requested_monotonic"] + 0.2
        )
        planned = cycle["activation_walltime"]
        deadline_wall = 1000 + cycle["deadline_monotonic"]
        assert deadline_wall - 1e-9 <= planned < deadline_wall + 1 / 30 + 1e-9
        assert cycle["hard_until_walltime"] == pytest.approx(planned + 1 / 30)
        assert (
            cycle["hard_until_walltime"] <= cycle["prefix_hard_until_walltime"] + 1e-9
        )
        assert (
            cycle["prefix_hard_until_walltime"]
            < cycle["hard_until_walltime"] + TIMING.output_offsets_s[1] + 1e-9
        )
        assert (
            planned - 1e-9
            <= cycle["first_target_walltime"]
            <= cycle["prefix_hard_until_walltime"] + 1e-9
        )
        assert cycle["switch_walltime"] == cycle["first_target_walltime"]
        assert cycle["first_send_monotonic"] + 1000 == pytest.approx(
            cycle["first_target_walltime"]
        )
        assert cycle["requested_monotonic"] == event["requested_monotonic"]
        assert event["next_request_monotonic"] == pytest.approx(
            event["requested_monotonic"] + 0.4
        )
        assert any(
            abs(t - cycle["first_send_monotonic"]) < 1e-9 for t, _, _ in sink.records
        )


def test_pause_resume_restarts_request_cadence_without_stale_activation():
    _, policy, sink, log, executor, keys = scenario(
        pause_window=(0.51, 0.78),
        stop_time=2.4,
        latencies=(0.08, 0.18, 0.08),
    )
    assert not keys.is_paused
    resume_indices = [
        i for i, (_, prefix, _) in enumerate(policy.requests) if prefix is None
    ]
    assert len(resume_indices) == 2
    resumed = resume_indices[-1]
    assert policy.requests[resumed][2] >= 0.78
    assert executor.futures[resumed].started >= executor.futures[resumed - 1].completed
    next_request = policy.requests[resumed + 1][2]
    assert 0.4 <= next_request - policy.requests[resumed][2] < 0.45
    assert not any(c.get("request_id") == 2 for c in guided_cycles(log))
    pause_time = next(t for reason, t in sink.holds if reason == "paused")
    assert not any(
        pause_time <= t < policy.requests[resumed][2] for t, _, _ in sink.records
    )


def test_late_result_before_planned_send_still_misses_request_deadline():
    # At29Hz, a183ms deadline generally falls between control ticks. A request
    # that finishes after183ms is invalid even if the next send is still ahead.
    _, _, _, probe_log, _, _ = scenario(
        frequency=29.0,
        execute_steps=11,
        delay_ms=183.0,
        latencies=(0.08, 0.01),
        stop_time=0.9,
    )
    planned = next(
        e
        for e in probe_log.events
        if e.get("event") == "rtc_requested" and e["request_id"] == 2
    )
    slack = planned["activation_monotonic"] - planned["deadline_monotonic"]
    assert slack > 0.001
    late_latency = 0.183 + slack / 2
    _, _, sink, log, executor, keys = scenario(
        frequency=29.0,
        execute_steps=11,
        delay_ms=183.0,
        latencies=(0.08, late_latency),
        stop_time=1.5,
    )
    assert keys.is_paused
    paused = [c for c in log.cycles if c.get("status") == "paused"]
    assert len(paused) == 1
    assert paused[0]["pause_reason"] in ("rtc_late_result", "rtc_deadline_missed")
    assert not guided_cycles(log)
    late = executor.futures[1]
    assert late.completed > late.started + 0.183
    assert late.completed < planned["activation_monotonic"]
    # No stale plan ever becomes active and hold precedes all later work.
    hold_time = next(
        t
        for reason, t in sink.holds
        if reason in ("rtc_late_result", "rtc_deadline_missed")
    )
    assert not any(t >= hold_time for t, _, _ in sink.records)


def delayed_first_takeover(clock, log):
    """One late wakeup that exceeds hard coverage, but not the 100 ms guard."""
    stalled = []

    def one_delayed_wakeup(seconds):
        requests = [
            e
            for e in log.events
            if e.get("event") == "rtc_requested" and e["request_id"] == 2
        ]
        if requests and not stalled:
            request = requests[0]
            # Deliberately exceed the protected model-grid endpoint, but stay
            # well below the separate100ms scheduling-slip threshold.
            if clock.wall_time() >= request["activation_walltime"] - 0.02:
                stalled.append(clock.monotonic())
                target = (
                    request["hard_until_walltime"] + TIMING.output_offsets_s[1] + 0.005
                )
                clock.sleep(max(seconds, target - clock.wall_time()))
                return
        clock.sleep(seconds + 0.001)

    return one_delayed_wakeup, stalled


def test_first_target_beyond_hard_coverage_holds_before_new_send():
    clock = OfflineReplayClock(epoch=1000)
    log = Log()
    one_delayed_wakeup, stalled = delayed_first_takeover(clock, log)
    _, _, sink, log, _, keys = scenario(
        clock=clock,
        log_factory=lambda: log,
        control_sleep=one_delayed_wakeup,
        stop_time=1.2,
    )
    assert len(stalled) == 1 and keys.is_paused
    pauses = [c for c in log.cycles if c.get("status") == "paused"]
    assert len(pauses) == 1
    assert pauses[0]["pause_reason"] == "rtc_hard_coverage_expired"
    assert not guided_cycles(log)
    hold_time = next(
        t for reason, t in sink.holds if reason == "rtc_hard_coverage_expired"
    )
    assert not any(t >= stalled[0] for t, _, _ in sink.records)
    assert hold_time - stalled[0] < 0.1


def test_receipt_stall_holds_without_catching_up():
    _, _, sink, log, _, keys = scenario(sink_lag=0.15, wakeup_lag=0.001, stop_time=1.2)
    assert keys.is_paused
    assert any(c.get("pause_reason") == "schedule_slip" for c in log.cycles)
    assert len(sink.records) == 1


def test_small_wakeup_drift_does_not_eventually_pause_or_burst():
    _, _, sink, log, _, keys = scenario(stop_time=20.0)
    assert not keys.is_paused
    assert len(sink.records) > 500
    assert len(guided_cycles(log)) > 40
    assert np.min(np.diff([row[0] for row in sink.records])) >= 1 / 30 - 1e-9


def test_initial_inference_longer_than_two_periods_sends_before_one_new_request():
    _, policy, sink, log, _, keys = scenario(
        latencies=(1.05, 0.08, 0.08, 0.08, 0.08),
        safety_limits=replace(limits(), max_observation_age_s=2.0),
        stop_time=2.3,
    )
    assert not keys.is_paused
    request_stamps = np.array([request[2] for request in policy.requests])
    first_send = sink.records[0][0]
    assert first_send >= 1.05 > 2 * 0.4
    assert len(request_stamps) >= 4
    # An expired first request slot must not submit again before the initial
    # result is used. After that first send, one fresh request resets cadence.
    assert request_stamps[1] >= first_send + 0.02 - 1e-9
    assert request_stamps[1] < first_send + 1 / 30
    assert np.count_nonzero(request_stamps < request_stamps[1] + 0.4 - 1e-9) == 2
    assert np.min(np.diff(request_stamps[1:])) >= 0.4 - 1e-9
    assert len(guided_cycles(log)) >= 2
    assert np.min(np.diff([row[0] for row in sink.records])) >= 1 / 30 - 1e-9


def test_target_sample_and_host_submission_are_distinct_audit_times():
    class DelayedSubmissionSink(Sink):
        def send_action(self, row, deadline):
            self.clock.sleep(0.005)
            submitted = self.clock.monotonic()
            super().send_action(row, deadline)
            return {"send_started_monotonic": submitted}

    _, _, sink, log, _, keys = scenario(
        sink_factory=DelayedSubmissionSink, stop_time=2.0
    )
    assert not keys.is_paused
    assert len(guided_cycles(log)) >= 2
    for cycle in guided_cycles(log):
        assert cycle["first_send_walltime"] == pytest.approx(
            cycle["first_send_monotonic"] + 1000
        )
        assert cycle["first_send_walltime"] - cycle[
            "first_target_walltime"
        ] == pytest.approx(0.005)
        assert cycle["request_elapsed_s"] == pytest.approx(
            cycle["first_send_monotonic"] - cycle["requested_monotonic"]
        )
        # Coverage is checked at the action's sample time. The separate host
        # timestamp documents submission latency; it does not retime targets.
        assert cycle["first_target_walltime"] <= cycle["prefix_hard_until_walltime"]
        np.testing.assert_array_equal(cycle["first_target_bracket_weights"], [1, 1])
        assert any(
            abs(stamp - cycle["first_send_monotonic"]) < 1e-9
            for stamp, _, _ in sink.records
        )
    assert np.min(np.diff([row[0] for row in sink.records])) >= 1 / 30 - 1e-9


def test_resume_does_not_exceed_max_chunks_already_sent():
    status, policy, sink, log, _, keys = scenario(
        max_chunks=1, pause_window=(0.3, 0.45), stop_time=2.0
    )
    assert status == "completed" and not keys.is_paused
    assert len(policy.requests) == 1
    assert len([c for c in log.cycles if c.get("status") == "completed"]) == 1
    assert any(reason == "paused" for reason, _ in sink.holds)
    assert sink.holds[-1][0] == "completed"
    assert sink.holds[-1][1] >= 0.45
    assert not any(stamp >= 0.3 for stamp, _, _ in sink.records)


def test_rejected_takeover_does_not_consume_max_chunks_before_resume():
    class ResumeAfterCoveragePause(Keys):
        def poll(self):
            if self.is_paused and self.clock.elapsed >= 0.9:
                self.is_paused = False
            return super().poll()

    clock, log = OfflineReplayClock(epoch=1000), Log()
    one_delayed_wakeup, stalled = delayed_first_takeover(clock, log)
    status, policy, sink, log, _, keys = scenario(
        clock=clock,
        log_factory=lambda: log,
        control_sleep=one_delayed_wakeup,
        keys_factory=ResumeAfterCoveragePause,
        max_chunks=2,
        stop_time=2.5,
    )
    assert status == "completed" and not keys.is_paused
    assert len(stalled) == 1
    completed = [c for c in log.cycles if c.get("status") == "completed"]
    assert [c["request_id"] for c in completed] == [1, 3]
    assert all(c["start"] == "resume" for c in completed)
    assert len(policy.requests) == 3
    assert policy.requests[1][1] is not None
    assert policy.requests[2][1] is None
    assert any(reason == "rtc_hard_coverage_expired" for reason, _ in sink.holds)
    assert sink.holds[-1][0] == "completed"
    assert not any(stalled[0] <= stamp < 0.9 for stamp, _, _ in sink.records)
