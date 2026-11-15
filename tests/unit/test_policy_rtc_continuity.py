"""RTC execution conditioning contracts on a virtual clock, without devices."""

import numpy as np
import pytest

from egomimic.robot.piper_pico import policy_rtc
from egomimic.robot.piper_pico.policy_continuity import ContinuousPlan
from egomimic.robot.piper_pico.policy_rollout import OfflineReplayClock
from egomimic.robot.piper_pico.policy_streaming import ChunkPlan
from tests.unit.test_live_policy import Provider, limits
from tests.unit.test_policy_rtc import (
    DELAY,
    TIMING,
    AsyncExecutor,
    Keys,
    Log,
    Sink,
    args,
)


class NoisyPolicy:
    """Small disagreeing trajectories; RTC weights do not force exact equality."""

    timing = TIMING

    def __init__(self, clock, latencies=(0.08,)):
        self.clock, self.latencies = clock, latencies
        self.requests, self.outputs = [], []

    def actions(self, frame, rtc, index):
        rows = np.repeat(frame.base_state[None], 100, axis=0)
        alternating = np.where(np.arange(100) % 2, -1.0, 1.0)
        rows[:, 0] = 0.006 + index * 0.003 + alternating * 0.002
        rows[:, 7] = -0.004 - index * 0.002 + alternating * 0.001
        rows[:, 5] = np.deg2rad(0.5 + index * 0.2 + alternating * 0.2)
        rows[:, 6] = np.where(np.arange(100) < 40, 0.01, 0.06)
        rows[:, 13] = np.where(np.arange(100) < 60, 0.07, 0.02)
        return rows

    def predict(self, frame, *, rtc=None):
        index = len(self.requests)
        self.requests.append((frame, rtc, self.clock.elapsed))
        latency = self.latencies[index % len(self.latencies)]
        self.clock.sleep(latency)
        rows = self.actions(frame, rtc, index)
        self.outputs.append(rows.copy())
        return {
            "base_actions": rows,
            "camera_actions": rows.copy(),
            "observation_timestamp": frame.timestamp,
            "inference_elapsed_s": latency,
            "inference_steps": 10,
        }


def run_conditioned(
    *,
    policy_class=NoisyPolicy,
    latencies=(0.08,),
    stop_time=2,
    pause_window=None,
    **overrides,
):
    clock = OfflineReplayClock(epoch=1000)
    policy = policy_class(clock, latencies)
    sink, log, executor = Sink(clock), Log(), AsyncExecutor(clock)
    keys = Keys(clock, stop_time=stop_time, pause_window=pause_window)
    options = args(
        **(
            {
                "rtc_transition_ms": 200,
                "rtc_smoothing_ms": 100,
                "rtc_max_position_gap_mm": 20,
                "rtc_max_rotation_gap_deg": 5,
            }
            | overrides
        )
    )
    status = policy_rtc.run_rtc_loop(
        policy,
        Provider(clock),
        sink,
        keys,
        limits(),
        log,
        options,
        observation_to_send_s=DELAY,
        executor=executor,
        monotonic=clock.monotonic,
        wall_time=clock.wall_time,
        sleep=clock.sleep,
    )
    return status, policy, sink, log, executor, keys


def completed(log):
    return [row for row in log.cycles if row["status"] == "completed"]


def test_next_prefix_inherits_executed_transition_not_raw_prediction(monkeypatch):
    original = policy_rtc.build_rtc_prefix
    sampled = []

    def capture(plan, timing, observation, delay, activation, request_id):
        prefix = original(plan, timing, observation, delay, activation, request_id)
        sampled.append((plan, observation + delay, prefix))
        return prefix

    monkeypatch.setattr(policy_rtc, "build_rtc_prefix", capture)
    status, policy, _, log, _, keys = run_conditioned(
        stop_time=1.4, rtc_transition_ms=250
    )
    assert status == "stopped" and not keys.is_paused
    assert len(sampled) >= 2
    # At the third request the prior takeover is still transitioning. It is
    # this committed trajectory, rather than the model's raw result, that must
    # be passed back to RTC on its 100-point observation-relative time grid.
    plan, anchor, prefix = sampled[1]
    assert isinstance(plan, ContinuousPlan)
    assert plan.transition_start < anchor < plan.transition_end
    u = (anchor - plan.transition_start) / plan.duration_s
    weight = u**3 * (10 - 15 * u + 6 * u**2)
    old, new = plan.previous.sample(anchor), plan.current.sample(anchor)
    expected = old[:3] + weight * (new[:3] - old[:3])
    np.testing.assert_allclose(prefix["base_actions"][0, :3], expected, atol=1e-12)
    assert np.linalg.norm(expected - new[:3]) > 1e-6
    previous_cycle = completed(log)[1]
    raw = ChunkPlan(
        previous_cycle["base_actions"],
        TIMING,
        previous_cycle["observation_timestamp"] + DELAY,
    )
    assert np.linalg.norm(expected - raw.sample(anchor)[:3]) > 1e-4
    assert prefix["weights"][0] == 1
    np.testing.assert_array_equal(
        policy.requests[2][1]["base_actions"], prefix["base_actions"]
    )


@pytest.mark.parametrize(
    "axis,gap,metric",
    [
        (0, 0.04, "max_position_m"),
        (5, np.deg2rad(8), "max_rotation_rad"),
    ],
)
def test_rejects_future_disagreement_even_when_takeover_entry_matches(
    axis, gap, metric
):
    class DivergingPolicy(NoisyPolicy):
        def actions(self, frame, rtc, index):
            rows = np.repeat(frame.base_state[None], 100, axis=0)
            if index:
                rows[np.asarray(TIMING.output_offsets_s) >= 0.32, axis] = gap
            return rows

    _, policy, sink, log, _, keys = run_conditioned(
        policy_class=DivergingPolicy, rtc_smoothing_ms=0, stop_time=1.2
    )
    assert keys.is_paused
    assert len(completed(log)) == 1
    failure = next(
        row for row in log.cycles if row.get("pause_reason") == "rtc_plan_disagreement"
    )
    assert failure["transition_disagreement"]["left"][metric] == pytest.approx(gap)
    request = next(
        e for e in log.events if e["event"] == "rtc_requested" and e["request_id"] == 2
    )
    candidate = ChunkPlan(
        policy.outputs[1], TIMING, policy.requests[1][0].timestamp + DELAY
    )
    np.testing.assert_allclose(
        candidate.sample(request["activation_walltime"])[axis], 0
    )
    held_at = next(t for reason, t in sink.holds if reason == "rtc_plan_disagreement")
    assert all(t < held_at for t, _, _ in sink.records)


def test_raw_outputs_execution_plan_and_actual_transition_remain_distinct():
    _, policy, sink, log, _, keys = run_conditioned(stop_time=1.2)
    assert not keys.is_paused
    cycles = completed(log)
    assert len(cycles) >= 2
    first = cycles[0]
    for cycle in cycles:
        raw = policy.outputs[cycle["request_id"] - 1]
        np.testing.assert_array_equal(cycle["base_actions"], raw)
        np.testing.assert_array_equal(cycle["camera_actions"], raw)
        assert not np.allclose(cycle["execution_base_actions"][:, 0], raw[:, 0])
        np.testing.assert_array_equal(
            cycle["execution_base_actions"][:, [6, 13]], raw[:, [6, 13]]
        )
        assert cycle["execution_transition_s"] == 0.2
        assert cycle["execution_smoothing_s"] == 0.1

    # The first physical target holds feedback pose while gripper timing remains
    # the model's original command, not the held feedback's width.
    measured = first["measured_base_state"]
    first_row = sink.records[0][1]
    for k in (0, 7):
        np.testing.assert_allclose(
            first_row[k : k + 6], measured[k : k + 6], atol=1e-12
        )
    np.testing.assert_allclose(first_row[[6, 13]], policy.outputs[0][0, [6, 13]])
    assert not np.allclose(first_row[[6, 13]], measured[[6, 13]])

    # Once the initial transition ends, sends follow the logged conditioned
    # trajectory at their actual times, without replaying the 100 model points.
    plan = ChunkPlan(
        first["execution_base_actions"], TIMING, first["first_target_walltime"]
    )
    checked = 0
    for stamp, row, _ in sink.records:
        wall = 1000 + stamp
        if (
            first["first_target_walltime"] + 0.2
            < wall
            < cycles[1]["first_target_walltime"]
        ):
            np.testing.assert_allclose(row, plan.sample(wall), atol=1e-10)
            checked += 1
    assert checked >= 5
    assert cycles[1]["raw_switch_plan_delta"]["left"]["position_m"] > 0.001
    assert cycles[1]["switch_plan_delta"]["left"]["position_m"] < 1e-10


def test_pause_discards_pending_prefix_and_resumes_from_feedback_with_max_chunks():
    status, policy, sink, log, executor, _ = run_conditioned(
        latencies=(0.08, 0.18, 0.08),
        pause_window=(0.48, 0.52),
        max_chunks=2,
        stop_time=4,
    )
    assert status == "completed"
    assert len(policy.requests) == 3
    assert [row["request_id"] for row in completed(log)] == [1, 3]
    assert [row["start"] for row in completed(log)] == ["resume", "resume"]
    held_at = next(t for reason, t in sink.holds if reason == "paused")
    assert held_at < executor.futures[1].completed
    assert executor.futures[2].started >= executor.futures[1].completed
    assert policy.requests[2][1] is None
    resumed = completed(log)[1]
    first = next(
        row
        for t, row, _ in sink.records
        if abs(t - resumed["first_send_monotonic"]) < 1e-9
    )
    for k in (0, 7):
        np.testing.assert_allclose(
            first[k : k + 6], resumed["measured_base_state"][k : k + 6], atol=1e-12
        )
    assert sink.holds[-1][0] == "completed"
    end = resumed["first_target_walltime"] + TIMING.effective_span_s
    assert all(1000 + t <= end + 1e-9 for t, _, _ in sink.records)


def test_conditioning_does_not_extend_rpc_deadline_or_send_after_hold():
    _, _, sink, log, executor, keys = run_conditioned(
        latencies=(0.08, 0.25), stop_time=1.2
    )
    assert keys.is_paused
    assert len(completed(log)) == 1
    request = next(
        e for e in log.events if e["event"] == "rtc_requested" and e["request_id"] == 2
    )
    held_at = next(t for reason, t in sink.holds if reason == "rtc_deadline_missed")
    assert held_at == pytest.approx(request["deadline_monotonic"], abs=0.0051)
    assert held_at < executor.futures[1].completed
    assert all(t < held_at for t, _, _ in sink.records)
    assert np.min(np.diff([t for t, _, _ in sink.records])) >= 1 / 30 - 1e-9


def test_static_schedule_reserves_full_transition_inside_old_horizon():
    old = policy_rtc.check_rtc_schedule(TIMING, 30, 12, 0.3)
    assert old["required_span_s"] < TIMING.effective_span_s
    with pytest.raises(ValueError, match="span"):
        policy_rtc.check_rtc_schedule(TIMING, 30, 12, 0.3, transition_s=0.2)
    valid = policy_rtc.check_rtc_schedule(TIMING, 30, 12, 0.2, transition_s=0.2)
    assert valid["required_span_s"] == pytest.approx(
        0.4 + 0.2 + 2 / 30 + TIMING.output_offsets_s[1] + 0.2
    )
    with pytest.raises(ValueError, match="shorter than request period"):
        policy_rtc.check_rtc_schedule(TIMING, 30, 12, 0.2, transition_s=0.4)
