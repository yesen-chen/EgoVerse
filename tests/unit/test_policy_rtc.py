"""RTC timing and invalidation on a virtual clock, without hardware or RPC."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egomimic.rldb.action_timing import ActionTiming
from egomimic.robot.piper_pico.policy_rollout import OfflineReplayClock
from egomimic.robot.piper_pico.policy_rtc import (
    build_rtc_prefix,
    check_rtc_schedule,
    run_rtc_loop,
)
from egomimic.robot.piper_pico.policy_streaming import ChunkPlan
from tests.unit.test_live_policy import Provider, limits

DELAY = 0.007
TIMING = ActionTiming(30, 30, 1, 100)


class Log:
    def __init__(self):
        self.events, self.cycles = [], []

    def event(self, event, **fields):
        self.events.append({"event": event, **fields})

    def cycle(self, row):
        self.cycles.append(row)

    def check(self):
        pass


class Sink:
    def __init__(self, clock, lag=0.0):
        self.clock, self.lag = clock, lag
        self.records, self.holds = [], []

    def send_action(self, row, deadline):
        self.records.append((self.clock.monotonic(), np.asarray(row).copy(), deadline))
        self.clock.sleep(self.lag)

    def hold(self, reason):
        self.holds.append((reason, self.clock.monotonic()))


class Keys:
    def __init__(self, clock, *, stop_time=2.0, pause_window=None):
        self.clock, self.stop_time, self.pause_window = clock, stop_time, pause_window
        self.is_paused = False

    def poll(self):
        if self.clock.elapsed >= self.stop_time:
            return "stop"
        if self.is_paused:
            return "pause"
        if (
            self.pause_window
            and self.pause_window[0] <= self.clock.elapsed < self.pause_window[1]
        ):
            return "pause"
        return "run"

    def pause(self):
        self.is_paused = True


class AsyncExecutor:
    """Execute against a worker clock, then restore the control-thread clock.

    Worker observations, completion timestamps and results are ready at the
    corresponding future times; result() on unfinished work is a test failure.
    """

    def __init__(self, clock):
        self.clock, self.futures = clock, []

    def submit(self, fn, *args):
        start = self.clock.elapsed
        error = value = None
        try:
            value = fn(*args)
        except BaseException as exc:
            error = exc
        completed = self.clock.elapsed
        self.clock.elapsed = start
        clock = self.clock

        class Future:
            cancelled = False
            reads = 0

            def done(self):
                return clock.elapsed >= completed - 1e-9

            def result(self):
                assert self.done(), "control thread blocked on an unfinished future"
                self.reads += 1
                if error is not None:
                    raise error
                return value

            def cancel(self):
                self.cancelled = True
                return False  # Running RPC cannot be cancelled.

        future = Future()
        future.started, future.completed = start, completed
        self.futures.append(future)
        return future


class Policy:
    timing = TIMING

    def __init__(self, clock, latencies=(0.08,), bias=0.0, fail_at=None):
        self.clock, self.latencies, self.bias, self.fail_at = (
            clock,
            latencies,
            bias,
            fail_at,
        )
        self.requests = []

    def predict(self, frame, *, rtc=None):
        index = len(self.requests)
        self.requests.append((frame, rtc, self.clock.elapsed))
        self.clock.sleep(self.latencies[index % len(self.latencies)])
        if self.fail_at == index:
            raise ValueError("broken RPC")
        actions = np.repeat(frame.base_state[None], 100, axis=0)
        # The same absolute trajectory for every chunk, with a controllable
        # disagreement to show that the executor itself performs no blending.
        stamps = frame.timestamp + DELAY + np.asarray(self.timing.output_offsets_s)
        actions[:, 0] = 0.02 * (stamps - 1000) + index * self.bias
        actions[:, 3] = 0.1 * (stamps - 1000)
        if rtc is not None:
            hard = rtc["weights"] == 1
            actions[hard] = rtc["base_actions"][hard]
        return {
            "base_actions": actions,
            "camera_actions": actions.copy(),
            "observation_timestamp": frame.timestamp,
            "inference_elapsed_s": self.latencies[index % len(self.latencies)],
            "inference_steps": 10,
        }


def args(**overrides):
    return SimpleNamespace(
        **(
            {
                "control_frequency": 30.0,
                "execute_steps": 12,
                "rtc_delay_ms": 200.0,
                "max_chunks": None,
            }
            | overrides
        )
    )


def run(
    *,
    latencies=(0.08,),
    stop_time=2.0,
    pause_window=None,
    options=None,
    provider_lag=0.0,
    sink_lag=0.0,
    bias=0.0,
    fail_at=None,
    clock=None,
):
    clock = clock or OfflineReplayClock(epoch=1000)
    provider = Provider(clock)
    original_latest = provider.latest

    def latest():
        clock.sleep(provider_lag)
        return original_latest()

    provider.latest = latest
    policy = Policy(clock, latencies, bias, fail_at)
    sink, log = Sink(clock, sink_lag), Log()
    executor = AsyncExecutor(clock)
    keys = Keys(clock, stop_time=stop_time, pause_window=pause_window)
    status = run_rtc_loop(
        policy,
        provider,
        sink,
        keys,
        limits(),
        log,
        options or args(),
        observation_to_send_s=DELAY,
        executor=executor,
        monotonic=clock.monotonic,
        wall_time=clock.wall_time,
        sleep=clock.sleep,
    )
    return status, policy, sink, log, executor, keys


def test_prefix_grid_is_100_model_points_not_30_control_points():
    actions = np.zeros((100, 14))
    actions[:, 0] = np.asarray(TIMING.output_offsets_s)
    plan = ChunkPlan(actions, TIMING, 10.0)
    prefix = build_rtc_prefix(plan, TIMING, 10.2, DELAY, 10.4, "r1")
    times = 10.207 + np.asarray(TIMING.output_offsets_s)
    hard_end = np.searchsorted(times, 10.4, side="left")
    hard = np.arange(len(times)) <= hard_end
    valid = times < plan.end_walltime
    assert prefix["base_actions"].shape == (100, 14)
    assert hard.sum() == 21  # Includes the interpolation point RIGHT of takeover.
    np.testing.assert_allclose(prefix["base_actions"][valid, 0], times[valid] - 10)
    np.testing.assert_array_equal(prefix["weights"][hard], 1)
    np.testing.assert_array_equal(prefix["weights"][~valid], 0)
    assert prefix["request_id"] == "r1"


def test_prefix_uses_paper_exponential_soft_mask_and_no_repeated_tail():
    plan = ChunkPlan(np.zeros((100, 14)), TIMING, 10.0)
    prefix = build_rtc_prefix(plan, TIMING, 10.2, DELAY, 10.4, "r")
    stamps = 10.207 + np.asarray(TIMING.output_offsets_s)
    d = np.searchsorted(stamps, 10.4, side="left") + 1
    end = np.searchsorted(stamps, plan.end_walltime, side="left")
    c = (end - np.arange(d, end)) / (end - d + 1)
    np.testing.assert_allclose(prefix["weights"][d:end], c * np.expm1(c) / np.expm1(1))
    assert np.all(np.diff(prefix["weights"][d:end]) < 0)
    assert np.all(prefix["weights"][end:] == 0)
    assert np.all(prefix["base_actions"][end:] == 0)


def test_prefix_rotations_use_so3_shortest_path():
    actions = np.zeros((100, 14))
    angles = np.linspace(np.deg2rad(170), np.deg2rad(190), 100)
    actions[:, 3:6] = Rotation.from_euler("z", angles[:, None]).as_rotvec()
    plan = ChunkPlan(actions, TIMING, 10.0)
    prefix = build_rtc_prefix(plan, TIMING, 10.1, 0, 10.3, "r")
    actual = Rotation.from_rotvec(prefix["base_actions"][:50, 3:6])
    expected_angles = (
        np.deg2rad(170)
        + np.deg2rad(20)
        * (0.1 + np.asarray(TIMING.output_offsets_s)[:50])
        / TIMING.effective_span_s
    )
    expected = Rotation.from_euler("z", expected_angles[:, None])
    np.testing.assert_allclose((actual.inv() * expected).magnitude(), 0, atol=1e-12)


@pytest.mark.parametrize("observation,activation", [(10.4, 10.3), (10.1, 11.0)])
def test_invalid_activation_window_rejected(observation, activation):
    plan = ChunkPlan(np.zeros((100, 14)), TIMING, 10)
    with pytest.raises(ValueError, match="activation"):
        build_rtc_prefix(plan, TIMING, observation, DELAY, activation, "r")


@pytest.mark.parametrize("steps,delay", [(30, 0.2), (12, 0.4), (0, 0.2), (12, 0)])
def test_schedule_rejects_insufficient_overlap(steps, delay):
    with pytest.raises(ValueError):
        check_rtc_schedule(TIMING, 30, steps, delay)


def test_early_and_variable_results_wait_for_fixed_activation():
    status, policy, sink, log, _, _ = run(
        latencies=(0.06, 0.03, 0.13, 0.08), stop_time=2.5
    )
    assert status == "stopped"
    cycles = [c for c in log.cycles if c["status"] == "completed"]
    assert len(cycles) >= 4
    for c in cycles[1:]:
        assert c["switch_walltime"] == pytest.approx(
            c["activation_walltime"], abs=0.0051
        )
        assert 0.2 - DELAY - 1e-8 <= c["start_offset_s"] <= 0.2 - DELAY + 1 / 30
        assert c["switch_walltime"] <= c["prefix_hard_until_walltime"]
        assert c["prefix_hard_count"] > 15
    times = np.array([r[0] for r in sink.records])
    np.testing.assert_allclose(np.diff(times), 1 / 30, atol=1e-8)
    assert policy.requests[0][1] is None
    assert all(r[1] is not None for r in policy.requests[1:])


def test_executor_does_not_crossfade_or_shift_new_outputs():
    _, policy, sink, log, _, _ = run(bias=0.02, stop_time=1.5)
    cycle = [c for c in log.cycles if c["start"] == "rtc"][0]
    action = np.asarray(cycle["camera_actions"])
    plan = ChunkPlan(action, TIMING, cycle["observation_timestamp"] + DELAY)
    for stamp, row, _ in sink.records:
        wall = 1000 + stamp
        if cycle["switch_walltime"] <= wall < cycle["switch_walltime"] + 0.15:
            np.testing.assert_allclose(row, plan.sample(wall), atol=1e-10)
    assert len(policy.requests) > 1


def test_pause_holds_immediately_while_rpc_is_still_running():
    _, policy, sink, log, executor, _ = run(
        latencies=(0.08, 0.18, 0.08), pause_window=(0.48, 0.52), stop_time=1.4
    )
    pause = [v for v in sink.holds if v[0] == "paused"][0]
    assert pause[1] == pytest.approx(0.48, abs=0.0051)
    assert pause[1] < executor.futures[1].completed
    # Resume does not put another RPC in the queue behind the stale request.
    assert executor.futures[2].started >= executor.futures[1].completed
    assert policy.requests[2][1] is None
    assert not any(c.get("request_id") == 2 for c in log.cycles)


def test_stop_during_first_rpc_holds_without_waiting():
    status, _, sink, _, executor, _ = run(latencies=(0.4,), stop_time=0.05)
    assert status == "stopped" and not sink.records
    assert sink.holds[0][0] == "stopped"
    assert sink.holds[0][1] == pytest.approx(0.05, abs=0.0051)
    assert executor.futures[0].reads == 0


def test_late_prediction_pauses_discards_and_never_extrapolates():
    _, _, sink, log, executor, keys = run(latencies=(0.08, 0.25), stop_time=1.4)
    assert keys.is_paused
    paused = [e for e in log.events if e["event"] == "rtc_paused"]
    assert paused[0]["reason"] == "rtc_deadline_missed"
    assert len([c for c in log.cycles if c["status"] == "completed"]) == 1
    holdtime = next(t for reason, t in sink.holds if reason == "rtc_deadline_missed")
    assert holdtime < executor.futures[1].completed
    assert max(r[0] for r in sink.records) < holdtime


def test_slow_observation_copy_runs_off_control_thread():
    _, _, sink, log, _, _ = run(provider_lag=0.04, latencies=(0.08,), stop_time=1.5)
    np.testing.assert_allclose(np.diff([r[0] for r in sink.records]), 1 / 30, atol=1e-8)
    cycle = [c for c in log.cycles if c.get("start") == "rtc"][0]
    assert 0.2 - 0.04 - DELAY <= cycle["start_offset_s"] <= 0.2 - 0.04 - DELAY + 1 / 30
    assert cycle["measured_inference_elapsed_s"] == pytest.approx(0.12)


def test_max_chunks_finishes_valid_last_horizon_then_holds():
    status, policy, sink, log, _, _ = run(options=args(max_chunks=2), stop_time=5)
    assert status == "completed" and len(policy.requests) == 2
    assert sink.holds == [("completed", pytest.approx(sink.holds[0][1]))]
    cycle = log.cycles[-1]
    end = cycle["observation_timestamp"] + DELAY + TIMING.effective_span_s
    assert all(1000 + t <= end + 1e-9 for t, _, _ in sink.records)


def test_schedule_delay_never_burst_catches_up():
    _, _, sink, log, _, keys = run(sink_lag=0.15, stop_time=1.2)
    assert keys.is_paused
    assert any(c.get("pause_reason") == "schedule_slip" for c in log.cycles)
    assert len(sink.records) == 1  # stall after first submission; no catch-up send


def test_worker_error_holds_before_propagating():
    clock = OfflineReplayClock(epoch=1000)
    sink, log = Sink(clock), Log()
    with pytest.raises(ValueError, match="broken RPC"):
        run_rtc_loop(
            Policy(clock, (0.08,), fail_at=1),
            Provider(clock),
            sink,
            Keys(clock),
            limits(),
            log,
            args(),
            observation_to_send_s=DELAY,
            executor=AsyncExecutor(clock),
            monotonic=clock.monotonic,
            wall_time=clock.wall_time,
            sleep=clock.sleep,
        )
    assert sink.holds[-1][0] == "failed"


def test_wall_clock_jump_stops_instead_of_retiming_old_plan():
    clock = OfflineReplayClock(epoch=1000)
    sink = Sink(clock)

    def wall():
        return clock.wall_time() + (1 if clock.elapsed >= 0.4 else 0)

    with pytest.raises(RuntimeError, match="wall clock"):
        run_rtc_loop(
            Policy(clock),
            Provider(clock),
            sink,
            Keys(clock),
            limits(),
            Log(),
            args(),
            observation_to_send_s=DELAY,
            executor=AsyncExecutor(clock),
            monotonic=clock.monotonic,
            wall_time=wall,
            sleep=clock.sleep,
        )
    assert sink.holds[-1][0] == "failed"


def test_activation_audit_cost_cannot_cause_catch_up_sends():
    clock = OfflineReplayClock(epoch=1000)
    sink = Sink(clock)

    class SlowLog(Log):
        def cycle(self, row):
            super().cycle(row)
            clock.sleep(0.02)

    run_rtc_loop(
        Policy(clock),
        Provider(clock),
        sink,
        Keys(clock, stop_time=1.5),
        limits(),
        SlowLog(),
        args(),
        observation_to_send_s=DELAY,
        executor=AsyncExecutor(clock),
        monotonic=clock.monotonic,
        wall_time=clock.wall_time,
        sleep=clock.sleep,
    )
    assert np.min(np.diff([r[0] for r in sink.records])) >= 1 / 30 - 1e-9


def test_stale_candidate_horizon_pauses_before_switch():
    clock = OfflineReplayClock(epoch=1000)
    provider, sink, log = Provider(clock), Sink(clock), Log()
    latest = provider.latest

    def delayed_snapshot():
        frame = latest()
        return replace(frame, timestamp=frame.timestamp - 0.3)

    provider.latest = delayed_snapshot
    keys = Keys(clock, stop_time=1.2)
    run_rtc_loop(
        Policy(clock),
        provider,
        sink,
        keys,
        # Isolate the horizon check; freshness rejection is tested separately.
        replace(limits(), max_observation_age_s=0.7),
        log,
        args(),
        observation_to_send_s=DELAY,
        executor=AsyncExecutor(clock),
        monotonic=clock.monotonic,
        wall_time=clock.wall_time,
        sleep=clock.sleep,
    )
    assert keys.is_paused
    assert any(c.get("pause_reason") == "rtc_insufficient_horizon" for c in log.cycles)
    assert len([c for c in log.cycles if c["status"] == "completed"]) == 1


def test_shutdown_joins_only_after_hold_before_shared_memory_teardown(monkeypatch):
    from egomimic.robot.piper_pico import policy_rtc

    clock = OfflineReplayClock(epoch=1000)
    sink = Sink(clock)
    executor = AsyncExecutor(clock)
    shutdowns = []

    def shutdown(*, wait, cancel_futures):
        assert sink.holds and sink.holds[-1][0] == "stopped"
        assert wait is True and cancel_futures is True
        assert executor.futures[0].reads == 0
        shutdowns.append(clock.elapsed)

    executor.shutdown = shutdown
    monkeypatch.setattr(policy_rtc, "ThreadPoolExecutor", lambda **kwargs: executor)
    status = run_rtc_loop(
        Policy(clock, (0.4,)),
        Provider(clock),
        sink,
        Keys(clock, stop_time=0.05),
        limits(),
        Log(),
        args(),
        observation_to_send_s=DELAY,
        monotonic=clock.monotonic,
        wall_time=clock.wall_time,
        sleep=clock.sleep,
    )
    assert status == "stopped" and len(shutdowns) == 1
    assert shutdowns[0] < 0.4


def test_freshness_checked_after_inference_before_first_motion():
    clock = OfflineReplayClock(epoch=1000)
    sink = Sink(clock)
    with pytest.raises(ValueError, match="stale"):
        run_rtc_loop(
            Policy(clock, (0.6,)),
            Provider(clock),
            sink,
            Keys(clock, stop_time=2),
            limits(),
            Log(),
            args(),
            observation_to_send_s=DELAY,
            executor=AsyncExecutor(clock),
            monotonic=clock.monotonic,
            wall_time=clock.wall_time,
            sleep=clock.sleep,
        )
    assert not sink.records and sink.holds[-1][0] == "failed"


def test_small_wakeup_overruns_do_not_accumulate_into_false_stall():
    clock = OfflineReplayClock(epoch=1000)
    sink, log = Sink(clock), Log()
    keys = Keys(clock, stop_time=36)
    run_rtc_loop(
        Policy(clock),
        Provider(clock),
        sink,
        keys,
        limits(),
        log,
        args(),
        observation_to_send_s=DELAY,
        executor=AsyncExecutor(clock),
        monotonic=clock.monotonic,
        wall_time=clock.wall_time,
        sleep=lambda seconds: clock.sleep(seconds + 0.001),
    )
    assert len(sink.records) > 1000
    assert not keys.is_paused
    assert not any(c["status"] == "paused" for c in log.cycles)
    assert np.min(np.diff([r[0] for r in sink.records])) >= 1 / 30 - 1e-9


def test_variable_health_check_cost_keeps_host_submissions_spaced():
    from egomimic.robot.piper_pico.live_policy import LoggedSink

    clock = OfflineReplayClock(epoch=1000)
    raw, log = Sink(clock), Log()

    class Health:
        count = 0

        def check_health(self):
            clock.sleep(0.02 if self.count % 2 == 0 else 0.001)
            self.count += 1

    class Audit(Log):
        def command(self, *args, **kwargs):
            pass

    logged = LoggedSink(raw, Health(), Audit(), clock.monotonic)
    run_rtc_loop(
        Policy(clock),
        Provider(clock),
        logged,
        Keys(clock, stop_time=2),
        limits(),
        log,
        args(),
        observation_to_send_s=DELAY,
        executor=AsyncExecutor(clock),
        monotonic=clock.monotonic,
        wall_time=clock.wall_time,
        sleep=clock.sleep,
    )
    assert len(raw.records) > 20
    assert np.min(np.diff([r[0] for r in raw.records])) >= 1 / 30 - 1e-9
