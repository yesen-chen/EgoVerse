"""Pipelined chunk execution on a virtual clock, without devices or threads."""

import json

import numpy as np
import pytest

from egomimic.robot.piper_pico import live_policy as live
from egomimic.robot.piper_pico.policy_execution import RecordingActionSink
from egomimic.robot.piper_pico.policy_rollout import OfflineReplayClock
from egomimic.robot.piper_pico.policy_streaming import (
    ChunkPlan,
    blend_rows,
    check_streaming_schedule,
    run_streaming_loop,
)
from tests.unit.test_live_policy import Policy, Provider, limits

DELAY = 0.007
VELOCITY = 0.05


class LatencyExecutor:
    """Result is computed at submit but becomes visible ``latency`` later."""

    def __init__(self, clock, latency):
        self.clock, self.latency = clock, latency
        self.submitted = []

    def submit(self, fn, *args):
        clock, ready_at, value = self.clock, self.clock.monotonic() + self.latency, fn(*args)
        self.submitted.append(clock.monotonic())

        class Future:
            def done(self):
                return clock.monotonic() >= ready_at - 1e-12

            def result(self):
                return value

        return Future()

    def shutdown(self, wait=True):
        pass


class RampPolicy(Policy):
    """Every chunk agrees on one absolute trajectory x(t) = VELOCITY * t."""

    def __init__(self, clock, bias=0.0):
        super().__init__(clock)
        self.bias = bias

    def predict(self, frame):
        result = super().predict(frame)
        start = frame.timestamp + DELAY
        offsets = np.asarray(self.timing.output_offsets_s)
        actions = result["base_actions"].copy()
        actions[:, 0] = VELOCITY * (start + offsets)
        actions[:, 0] += self.bias * (len(self.observations) % 2)
        return {**result, "base_actions": actions, "camera_actions": actions}


class Keys:
    def __init__(self, clock, sink, *, stop_after, pause_window=None, stop_time=30.0):
        self.clock, self.sink = clock, sink
        self.stop_after, self.pause_window = stop_after, pause_window
        self.stop_time = stop_time
        self.paused = False

    def poll(self):
        if len(self.sink.records) >= self.stop_after or self.clock.elapsed >= self.stop_time:
            return "stop"
        if self.pause_window is not None:
            start, end = self.pause_window
            if start <= self.clock.elapsed < end:
                return "pause"
        return "run"

    def pause(self):
        self.paused = True


def arguments(**kwargs):
    from types import SimpleNamespace

    values = dict(
        execute=False,
        control_frequency=30.0,
        execute_steps=12,
        blend_steps=5,
        max_chunks=None,
        log=True,
    )
    return SimpleNamespace(**(values | kwargs))


def run(tmp_path, policy_factory, *, latency=0.09, stop_after=120, args=None, **keys):
    clock = OfflineReplayClock(epoch=1000.0)
    provider, policy = Provider(clock), policy_factory(clock)
    raw = RecordingActionSink()
    args = args or arguments()
    log = live.SessionLog(tmp_path / "run", args)
    sink = live.LoggedSink(raw, provider, log, clock.monotonic)
    executor = LatencyExecutor(clock, latency)
    try:
        status = run_streaming_loop(
            policy,
            provider,
            sink,
            Keys(clock, raw, stop_after=stop_after, **keys),
            limits(),
            log,
            args,
            observation_to_send_s=DELAY,
            executor=executor,
            monotonic=clock.monotonic,
            wall_time=clock.wall_time,
            sleep=clock.sleep,
        )
    finally:
        log.close()
    cycles = [
        json.loads(line)
        for line in (log.directory / "cycles.jsonl").read_text().splitlines()
    ]
    return status, raw, cycles, log, executor


def test_consistent_chunks_stream_without_gap_or_jump(tmp_path):
    status, raw, cycles, log, executor = run(tmp_path, RampPolicy)
    assert status == "stopped" and raw.hold_reasons == ["stopped"]
    assert log.summary["max_action_interval_s"] == pytest.approx(1 / 30)
    assert len(cycles) >= 8
    # Pipelined switches land inside the new chunk by the latency (picked up
    # at the next control tick) minus the training observation-to-send delay.
    for cycle in cycles[1:]:
        assert cycle["start"] == "pipelined"
        assert cycle["start_offset_s"] == pytest.approx(3 / 30 - DELAY, abs=1e-6)
    x = np.array([r["action"][0] for r in raw.records])
    # The first chunk starts at offset 0 (arm was held); its first switch
    # absorbs that DELAY in the crossfade, afterwards steps are exact.
    np.testing.assert_allclose(np.diff(x[1:]), VELOCITY / 30, atol=VELOCITY * DELAY)
    np.testing.assert_allclose(np.diff(x[40:]), VELOCITY / 30, atol=1e-9)


def test_crossfade_spreads_disagreement_between_chunks(tmp_path):
    bias = 0.01

    def factory(clock):
        return RampPolicy(clock, bias=bias)

    _, raw, _, _, _ = run(tmp_path / "blend", factory)
    blended = np.diff([r["action"][0] for r in raw.records]) - VELOCITY / 30
    hard_args = arguments(blend_steps=0)
    _, raw, _, _, _ = run(tmp_path / "hard", factory, args=hard_args)
    hard = np.diff([r["action"][0] for r in raw.records]) - VELOCITY / 30
    assert np.abs(hard[40:]).max() == pytest.approx(bias, rel=1e-6)
    assert np.abs(blended[40:]).max() == pytest.approx(bias / 5, rel=1e-6)


def test_slow_inference_holds_last_target_without_extrapolating(tmp_path):
    status, raw, cycles, log, _ = run(
        tmp_path,
        RampPolicy,
        latency=0.4,
        stop_after=60,
        stop_time=5.0,
        args=arguments(execute_steps=20),
    )
    assert status == "stopped"
    events = [
        json.loads(line)
        for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()
    ]
    assert any(e["event"] == "plan_exhausted" for e in events)
    assert len(cycles) >= 2
    assert log.summary["max_action_interval_s"] > 0.05
    span = RampPolicy.timing.effective_span_s
    assert all(c["start_offset_s"] + 5 / 30 < span for c in cycles)


def test_too_late_prediction_is_discarded(tmp_path):
    status, raw, cycles, _, _ = run(
        tmp_path, RampPolicy, latency=0.9, stop_after=60, stop_time=3.0
    )
    assert status == "stopped" and len(cycles) == 1
    events = [
        json.loads(line)
        for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()
    ]
    assert any(e["event"] == "prediction_discarded" for e in events)
    assert len(raw.records) == 30


def test_pause_holds_and_resume_restarts_from_new_observation(tmp_path):
    status, raw, cycles, _, _ = run(
        tmp_path, RampPolicy, stop_after=60, pause_window=(0.5, 0.9)
    )
    assert status == "stopped"
    assert raw.hold_reasons == ["paused", "stopped"]
    resumes = [c for c in cycles if c["start"] == "resume"]
    assert len(resumes) == 2 and resumes[1]["observation_timestamp"] >= 1000.9


def test_max_chunks_finishes_last_chunk_then_holds(tmp_path):
    status, raw, cycles, _, _ = run(
        tmp_path, RampPolicy, stop_after=10_000, args=arguments(max_chunks=2)
    )
    assert status == "completed" and raw.hold_reasons == ["completed"]
    assert len(cycles) == 2


def test_schedule_must_leave_latency_budget():
    timing = Policy.timing
    with pytest.raises(ValueError, match="execute-steps 30"):
        check_streaming_schedule(timing, 30.0, 30, 5)
    with pytest.raises(ValueError):
        check_streaming_schedule(timing, 30.0, 22, 5)
    budget = check_streaming_schedule(timing, 30.0, 12, 5)
    assert budget["max_inference_latency_s"] == pytest.approx(
        (timing.effective_span_s - 17 / 30) / 2
    )


def test_parser_streaming_flags_default_off():
    base = ["--checkpoint", "m", "--training-data", "d", "--calibration", "c"]
    base += ["--camera-config", "r"]
    args = live.build_parser().parse_args(base)
    assert args.async_inference is False and args.blend_steps == 5
    args = live.build_parser().parse_args(base + ["--async-inference", "--blend-steps", "6"])
    assert args.async_inference is True and args.blend_steps == 6


def test_plan_sampling_and_blend_endpoints():
    timing = Policy.timing
    actions = np.zeros((100, 14))
    actions[:, 0] = np.linspace(0, 1, 100)
    actions[:, 3] = np.linspace(0, 0.5, 100)
    plan = ChunkPlan(actions, timing, 10.0)
    np.testing.assert_allclose(plan.sample(9.0), actions[0])
    np.testing.assert_allclose(plan.sample(20.0), actions[-1])
    old, new = actions[0], actions[-1]
    np.testing.assert_allclose(blend_rows(old, new, 0.0), old, atol=1e-12)
    np.testing.assert_allclose(blend_rows(old, new, 1.0), new, atol=1e-12)
    np.testing.assert_allclose(blend_rows(old, new, 0.5)[[0, 3]], [0.5, 0.25])
