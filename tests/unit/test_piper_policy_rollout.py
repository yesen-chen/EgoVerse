"""Offline orchestration: freshness, pause discard, replay units and audit output."""

import json
from dataclasses import replace

import numpy as np
import pytest
import zarr
from scipy.spatial.transform import Rotation

from egomimic.rldb.action_timing import ActionTiming
from egomimic.robot.piper_pico.policy_execution import RecordingActionSink, SafetyLimits
from egomimic.robot.piper_pico.policy_inference import (
    CAMERA_KEYS,
    SOURCE_NAMES,
    ObservationFrame,
)
from egomimic.robot.piper_pico.policy_rollout import (
    OfflineReplayClock,
    run_sync_cycle,
    run_sync_policy,
)
from egomimic.scripts import replay_piper_policy as replay
from tests.fixtures.piper_pico_training import write_recording


def frame(stamp=0.0):
    state = np.zeros(14)
    state[[6, 13]] = 0.04
    return ObservationFrame(
        stamp,
        {name: np.zeros((4, 6, 3), np.uint8) for name in CAMERA_KEYS},
        state,
        {name: stamp for name in SOURCE_NAMES},
    )


class Provider:
    def __init__(self, frames):
        self.frames = iter(frames)

    def latest(self):
        return next(self.frames)


class Policy:
    timing = ActionTiming(10, 3, 1, 3)

    def __init__(self, clock=None):
        self.seen = []
        self.clock = clock

    def predict(self, observation):
        self.seen.append(observation.timestamp)
        if self.clock:
            self.clock.sleep(1.0)
        actions = np.repeat(observation.base_state[None], 3, axis=0)
        return {
            "camera_actions": actions,
            "base_actions": actions,
            "observation_timestamp": observation.timestamp,
            "inference_elapsed_s": 0.001,
            "inference_steps": 10,
        }


@pytest.fixture
def limits():
    return SafetyLimits(-np.ones((2, 3)) * 2, np.ones((2, 3)) * 2)


def clock_args(clock):
    return {
        "monotonic": clock.monotonic,
        "wall_time": clock.wall_time,
        "sleep": clock.sleep,
    }


def test_one_cycle_schedule_and_measured_inference(limits):
    clock, sink, policy = OfflineReplayClock(), RecordingActionSink(), Policy()
    report = run_sync_cycle(
        policy,
        Provider([frame()]),
        sink,
        limits,
        control_frequency=10,
        **clock_args(clock),
    )
    assert report["status"] == "completed"
    assert report["execution"]["sent_count"] == 3
    assert report["execution"]["actual_sent_span_s"] == pytest.approx(0.2)
    assert report["model_target_timestamps"] == pytest.approx([0, 0.1, 0.2])
    assert report["measured_inference_elapsed_s"] >= 0
    assert len(report["candidate_schedule"]) == 3
    assert sink.hold_reasons == ["completed"]


def test_continuing_cycle_reports_no_hold_and_starts_at_zero_after_prediction(limits):
    clock, sink = OfflineReplayClock(), RecordingActionSink()
    report = run_sync_cycle(
        RampPolicy(clock, 0.13),
        Provider([frame()]),
        sink,
        limits,
        control_frequency=10,
        hold_on_completion=False,
        **clock_args(clock),
    )
    assert report["status"] == "completed"
    assert report["hold_attempted"] is False
    assert report["execution"]["hold_attempted"] is False
    assert report["execution"]["start_offset_s"] == 0
    assert [
        s["actual_monotonic"] for s in report["execution"]["samples"]
    ] == pytest.approx([0.13, 0.23, 0.33])
    assert [s["action"][0] for s in report["execution"]["samples"]] == pytest.approx(
        [0, 0.01, 0.02]
    )
    assert sink.hold_reasons == []


@pytest.mark.parametrize("command,status", [("pause", "paused"), ("stop", "stopped")])
def test_continuing_cycle_still_holds_on_control_event(command, status, limits):
    clock, sink = OfflineReplayClock(), RecordingActionSink()
    report = run_sync_cycle(
        Policy(),
        Provider([frame()]),
        sink,
        limits,
        hold_on_completion=False,
        control_callback=lambda: command,
        **clock_args(clock),
    )
    assert report["status"] == status
    assert report["hold_attempted"] is True
    assert sink.hold_reasons == [status]


@pytest.mark.parametrize("invalid_where", ["prediction", "execution"])
def test_continuing_cycle_holds_once_on_failure(invalid_where, limits):
    class BrokenPolicy(Policy):
        def predict(self, observation):
            if invalid_where == "prediction":
                raise ValueError("prediction failed")
            result = super().predict(observation)
            result["base_actions"] = result["base_actions"].copy()
            result["base_actions"][-1, 6] = -1
            return result

    clock, sink = OfflineReplayClock(), RecordingActionSink()
    report = run_sync_cycle(
        BrokenPolicy(),
        Provider([frame()]),
        sink,
        limits,
        hold_on_completion=False,
        **clock_args(clock),
    )
    assert report["status"] == "failed"
    assert report["hold_attempted"] is True
    assert not sink.records
    assert sink.hold_reasons == ["failed"]


class RampPolicy(Policy):
    """Left x moves 0.01 m per 0.1 s chunk offset; inference takes ``delay`` virtual s."""

    def __init__(self, clock, delay):
        super().__init__()
        self.virtual_clock, self.delay = clock, delay

    def predict(self, observation):
        self.virtual_clock.sleep(self.delay)
        result = super().predict(observation)
        actions = result["base_actions"].copy()
        actions[:, 0] = [0.0, 0.01, 0.02]
        return dict(result, camera_actions=actions, base_actions=actions)


def test_early_block_waits_for_recorded_observation_to_send_delay(limits):
    clock, sink = OfflineReplayClock(), RecordingActionSink()
    report = run_sync_cycle(
        RampPolicy(clock, 0.0),
        Provider([frame()]),
        sink,
        limits,
        control_frequency=10,
        observation_to_send_s=0.05,
        **clock_args(clock),
    )
    assert report["status"] == "completed"
    samples = report["execution"]["samples"]
    assert [s["observation_to_send_s"] for s in samples] == pytest.approx(
        [0.05, 0.15, 0.25]
    )
    assert [s["action"][0] for s in samples] == pytest.approx([0.0, 0.01, 0.02])
    assert report["execution"]["start_delay_s"] == pytest.approx(0.05)
    assert "recorded observation_to_send_s" in report["scheduling_semantics"]


def test_late_block_starts_inside_the_chunk_at_the_elapsed_offset(limits):
    clock, sink = OfflineReplayClock(), RecordingActionSink()
    report = run_sync_cycle(
        RampPolicy(clock, 0.13),
        Provider([frame()]),
        sink,
        limits,
        control_frequency=10,
        observation_to_send_s=0.05,
        **clock_args(clock),
    )
    assert report["status"] == "completed"
    execution = report["execution"]
    assert execution["start_offset_s"] == pytest.approx(0.08)
    samples = execution["samples"]
    assert [s["observation_to_send_s"] for s in samples] == pytest.approx([0.13, 0.23])
    assert [s["action"][0] for s in samples] == pytest.approx([0.008, 0.018])


def test_block_older_than_its_chunk_is_never_sent(limits):
    clock, sink = OfflineReplayClock(), RecordingActionSink()
    report = run_sync_cycle(
        RampPolicy(clock, 0.3),
        Provider([frame()]),
        sink,
        limits,
        control_frequency=10,
        observation_to_send_s=0.05,
        **clock_args(clock),
    )
    assert report["status"] == "failed"
    assert "beyond" in report["error"]["message"]
    assert not sink.records and sink.hold_reasons == ["failed"]


@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_each_source_rejected_before_inference(source, limits):
    clock, sink, policy = OfflineReplayClock(), RecordingActionSink(), Policy()
    observation = frame()
    times = dict(observation.source_timestamps, **{source: -1})
    report = run_sync_cycle(
        policy,
        Provider([replace(observation, source_timestamps=times)]),
        sink,
        limits,
        **clock_args(clock),
    )
    assert report["status"] == "failed"
    assert source in report["error"]["message"]
    assert not policy.seen and not sink.records
    assert sink.hold_reasons == ["failed"]


def test_post_inference_stale_frame_never_sent(limits):
    clock, sink = OfflineReplayClock(), RecordingActionSink()
    report = run_sync_cycle(
        Policy(clock), Provider([frame()]), sink, limits, **clock_args(clock)
    )
    assert report["status"] == "failed"
    assert "stale" in report["error"]["message"]
    assert not sink.records and sink.hold_reasons == ["failed"]


@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_each_source_rechecked_after_inference(source, limits):
    clock, sink = OfflineReplayClock(epoch=0.3), RecordingActionSink()
    observation = frame(0.3)
    observation = replace(
        observation,
        source_timestamps=dict(observation.source_timestamps, **{source: 0.0}),
    )

    class SlowPolicy(Policy):
        def predict(self, observation):
            clock.sleep(0.3)
            return super().predict(observation)

    policy = SlowPolicy()
    report = run_sync_cycle(
        policy, Provider([observation]), sink, limits, **clock_args(clock)
    )
    assert report["status"] == "failed"
    assert source in report["error"]["message"]
    assert policy.seen == [0.3]
    assert not sink.records and sink.hold_reasons == ["failed"]


def test_prediction_failure_holds(limits):
    class BrokenPolicy:
        def predict(self, observation):
            raise RuntimeError("prediction failed")

    clock, sink = OfflineReplayClock(), RecordingActionSink()
    report = run_sync_cycle(
        BrokenPolicy(), Provider([frame()]), sink, limits, **clock_args(clock)
    )
    assert report["error"]["message"] == "prediction failed"
    assert sink.hold_reasons == ["failed"] and not sink.records


def test_pause_discards_remaining_targets_then_predicts_new_frame(limits):
    clock, sink, policy = OfflineReplayClock(), RecordingActionSink(), Policy()
    calls = 0

    def control():
        nonlocal calls
        calls += 1
        # Initial and post-prediction polls, first send, then pause at next wait.
        return "pause" if calls == 4 else "run"

    class AdvancingProvider(Provider):
        def latest(self):
            observation = super().latest()
            clock.sleep(max(0, observation.timestamp - clock.wall_time()))
            return observation

    report = run_sync_policy(
        policy,
        AdvancingProvider([frame(), frame(0.1)]),
        sink,
        limits,
        max_chunks=2,
        control_frequency=10,
        control_callback=control,
        **clock_args(clock),
    )
    assert report["status"] == "completed"
    assert policy.seen == [0.0, 0.1]
    assert [cycle["execution"]["sent_count"] for cycle in report["cycles"]] == [1, 3]
    assert sink.hold_reasons == ["paused", "completed"]
    assert [item["command"] for item in report["control_events"]] == [
        "run",
        "pause",
        "run",
    ]


def test_reusing_observation_after_chunk_fails(limits):
    clock, sink, policy = OfflineReplayClock(), RecordingActionSink(), Policy()
    report = run_sync_policy(
        policy,
        Provider([frame(), frame()]),
        sink,
        limits,
        max_chunks=2,
        execute_steps=1,
        **clock_args(clock),
    )
    assert report["status"] == "failed"
    assert len(policy.seen) == 1
    assert len(sink.records) == 1
    assert "newer observation" in report["cycles"][-1]["error"]["message"]


class CurrentProvider:
    """A new sample arrives one millisecond after each synchronous request."""

    def __init__(self, clock):
        self.clock = clock
        self.stamps = []

    def latest(self):
        self.clock.sleep(0.001)
        stamp = self.clock.wall_time()
        self.stamps.append(stamp)
        return frame(stamp)


def test_fast_single_step_chunks_keep_actual_resampled_command_interval(limits):
    clock, sink, policy = OfflineReplayClock(), RecordingActionSink(), Policy()
    provider = CurrentProvider(clock)
    report = run_sync_policy(
        policy,
        provider,
        sink,
        limits,
        max_chunks=3,
        execute_steps=1,
        control_frequency=11,
        **clock_args(clock),
    )
    assert report["status"] == "completed"
    executions = [cycle["execution"] for cycle in report["cycles"]]
    # The 0.2-second span rounds 11 requested Hz to 10 actual Hz, so the
    # boundary interval must be 0.1 seconds, not merely 1 / requested Hz.
    assert [execution["actual_control_frequency"] for execution in executions] == [
        10
    ] * 3
    sent_times = [
        execution["samples"][0]["actual_monotonic"] for execution in executions
    ]
    assert sent_times == pytest.approx([0.001, 0.101, 0.201])
    assert np.all(np.diff(sent_times) >= 0.1 - 1e-12)
    assert policy.seen == provider.stamps == pytest.approx([0.001, 0.002, 0.102])
    assert len(set(provider.stamps)) == 3
    assert all(execution["sent_count"] == 1 for execution in executions)
    assert all(
        execution["full_span_s"] == pytest.approx(0.2) for execution in executions
    )


def test_continuing_chunks_keep_spacing_and_bounded_run_holds_only_at_exit(limits):
    clock, sink = OfflineReplayClock(), RecordingActionSink()
    report = run_sync_policy(
        Policy(),
        CurrentProvider(clock),
        sink,
        limits,
        max_chunks=3,
        execute_steps=1,
        control_frequency=10,
        hold_on_completion=False,
        **clock_args(clock),
    )
    assert report["status"] == "completed"
    assert report["hold_attempted"] is True
    assert sink.hold_reasons == ["completed"]
    assert all(cycle["hold_attempted"] is False for cycle in report["cycles"])
    times = [
        cycle["execution"]["samples"][0]["actual_monotonic"]
        for cycle in report["cycles"]
    ]
    assert times == pytest.approx([0.001, 0.101, 0.201])


def test_slow_inference_does_not_add_another_control_interval(limits):
    clock, sink = OfflineReplayClock(), RecordingActionSink()
    provider = CurrentProvider(clock)

    class SlowPolicy(Policy):
        def predict(self, observation):
            clock.sleep(0.2)
            return super().predict(observation)

    policy = SlowPolicy()
    report = run_sync_policy(
        policy,
        provider,
        sink,
        limits,
        max_chunks=3,
        execute_steps=1,
        control_frequency=11,
        **clock_args(clock),
    )
    assert report["status"] == "completed"
    sent_times = [
        cycle["execution"]["samples"][0]["actual_monotonic"]
        for cycle in report["cycles"]
    ]
    assert sent_times == pytest.approx([0.201, 0.402, 0.603])
    assert np.diff(sent_times) == pytest.approx([0.201, 0.201])
    assert policy.seen == provider.stamps == pytest.approx([0.001, 0.202, 0.403])
    assert all(
        cycle["inter_block_wait"]["elapsed_s"] == 0 for cycle in report["cycles"][1:]
    )


def test_pause_during_boundary_discards_prediction_and_fetches_new_frame(limits):
    clock, sink, policy = OfflineReplayClock(), RecordingActionSink(), Policy()
    provider = CurrentProvider(clock)

    def control():
        if len(policy.seen) == 2 and 0.03 <= clock.monotonic() < 0.07:
            return "pause"
        return "run"

    report = run_sync_policy(
        policy,
        provider,
        sink,
        limits,
        max_chunks=3,
        execute_steps=1,
        control_frequency=10,
        control_callback=control,
        **clock_args(clock),
    )
    assert report["status"] == "completed"
    assert [cycle["status"] for cycle in report["cycles"]] == [
        "completed",
        "paused",
        "completed",
    ]
    paused = report["cycles"][1]
    assert paused["inter_block_wait"]["status"] == "paused"
    assert paused["inter_block_wait"]["elapsed_s"] == pytest.approx(0.03)
    assert "execution" not in paused
    assert policy.seen == provider.stamps == pytest.approx([0.001, 0.002, 0.073])
    assert len(set(provider.stamps)) == 3
    assert len(sink.records) == 2
    assert sink.hold_reasons == ["completed", "paused", "completed"]
    sent_times = [
        cycle["execution"]["samples"][0]["actual_monotonic"]
        for cycle in report["cycles"]
        if "execution" in cycle
    ]
    assert sent_times == pytest.approx([0.001, 0.101])


def test_stop_during_boundary_holds_without_another_observation(limits):
    clock, sink, policy = OfflineReplayClock(), RecordingActionSink(), Policy()
    provider = CurrentProvider(clock)

    def control():
        return "stop" if len(policy.seen) == 2 and clock.monotonic() >= 0.03 else "run"

    report = run_sync_policy(
        policy,
        provider,
        sink,
        limits,
        max_chunks=3,
        execute_steps=1,
        control_frequency=10,
        control_callback=control,
        **clock_args(clock),
    )
    assert report["status"] == "stopped"
    assert report["cycles"][-1]["inter_block_wait"]["status"] == "stopped"
    assert policy.seen == provider.stamps == pytest.approx([0.001, 0.002])
    assert len(sink.records) == 1
    assert sink.hold_reasons == ["completed", "stopped"]


def test_boundary_wait_rechecks_freshness_before_sending(limits):
    clock, sink, policy = OfflineReplayClock(), RecordingActionSink(), Policy()
    provider = CurrentProvider(clock)
    report = run_sync_policy(
        policy,
        provider,
        sink,
        replace(limits, max_observation_age_s=0.05),
        max_chunks=2,
        execute_steps=1,
        control_frequency=10,
        **clock_args(clock),
    )
    assert report["status"] == "failed"
    assert "stale" in report["cycles"][-1]["error"]["message"]
    assert policy.seen == provider.stamps == pytest.approx([0.001, 0.002])
    assert len(sink.records) == 1
    assert sink.hold_reasons == ["completed", "failed"]


def test_stop_before_prediction(limits):
    clock, sink, policy = OfflineReplayClock(), RecordingActionSink(), Policy()
    report = run_sync_policy(
        policy,
        Provider([frame()]),
        sink,
        limits,
        control_callback=lambda: "stop",
        **clock_args(clock),
    )
    assert report["status"] == "stopped"
    assert not policy.seen and not sink.records
    assert sink.hold_reasons == ["stopped"]


def test_recording_reconstructs_measured_pose_width_and_preserves_original_time(
    tmp_path,
):
    path = write_recording(tmp_path, "piper", "units", frames=8)
    store = zarr.open_group(str(path), mode="r")
    provider = replay.RecordedObservationProvider(
        path, OfflineReplayClock(), start_frame=2, allow_synthetic_calibration=True
    )
    observation = provider.latest()
    for index, side in enumerate(("left", "right")):
        row = observation.base_state[index * 7 : index * 7 + 7]
        pose = store[f"{side}.obs_ee_pose"][2]
        np.testing.assert_allclose(row[:3], pose[:3])
        np.testing.assert_allclose(
            Rotation.from_rotvec(row[3:6]).as_matrix(),
            Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix(),
        )
        assert row[6] == pytest.approx(store[f"{side}.obs_gripper"][2].item() * 0.08)
        assert not np.allclose(row[:3], store[f"{side}.cmd_ee_pose"][2][:3])
    assert set(observation.images) == set(CAMERA_KEYS)
    assert observation.images["front_1"].shape == (32, 32, 3)
    assert observation.timestamp == 0
    assert (
        provider.last_record["original_observation_timestamp"] == store["timestamp"][2]
    )
    assert "proxy" in provider.last_record["source_timestamp_semantics"]["left"]
    with pytest.raises(ValueError, match="synthetic"):
        replay.RecordedObservationProvider(path, OfflineReplayClock())


def cli_args(path, checkpoint, output):
    return [
        "--episode",
        str(path),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output),
        "--allow-synthetic-calibration",
        "--workspace-min",
        "-2",
        "-2",
        "-2",
        "-2",
        "-2",
        "-2",
        "--workspace-max",
        "2",
        "2",
        "2",
        "2",
        "2",
        "2",
    ]


def test_cli_reports_blocked_checkpoint_and_refuses_existing_output(
    tmp_path, monkeypatch
):
    path = write_recording(tmp_path, "piper", "cli", frames=8)
    checkpoint = tmp_path / "trusted.ckpt"
    checkpoint.write_bytes(b"fixture")
    output = tmp_path / "report"

    def fail(*args, **kwargs):
        raise ValueError("Unsupported checkpoint contract")

    monkeypatch.setattr(replay.PiperPolicy, "from_checkpoint", fail)
    assert replay.main(cli_args(path, checkpoint, output)) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["sent_action_count"] == 0
    assert report["hold_reasons"] == ["failed"]
    assert "Unsupported checkpoint" in report["error"]["message"]
    original = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        replay.main(cli_args(path, checkpoint, output))
    assert (output / "report.json").read_bytes() == original


def test_cli_offline_schedule_and_sent_action_log(tmp_path, monkeypatch):
    path = write_recording(tmp_path, "piper", "cli", frames=8)
    checkpoint = tmp_path / "trusted.ckpt"
    checkpoint.write_bytes(b"fixture")
    output = tmp_path / "report"
    monkeypatch.setattr(
        replay.PiperPolicy, "from_checkpoint", lambda *a, **kw: Policy()
    )
    assert replay.main(cli_args(path, checkpoint, output)) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "completed"
    assert report["sent_action_count"] == 1
    assert "virtual" in report["clock"]
    assert "not a real-time" in report["inference_timing"]
    scheduled = json.loads((output / "scheduled_actions.jsonl").read_text())
    sent = json.loads((output / "sent_actions.jsonl").read_text())
    assert scheduled["sent"]
    assert scheduled["action"] == sent["action"]


def test_cli_refuses_output_inside_source_episode_after_resolving_symlinks(
    tmp_path, monkeypatch
):
    path = write_recording(tmp_path, "piper", "protected", frames=8)
    checkpoint = tmp_path / "trusted.ckpt"
    checkpoint.write_bytes(b"fixture")
    alias = tmp_path / "episode_alias"
    alias.symlink_to(path, target_is_directory=True)

    def unexpected_load(*args, **kwargs):
        pytest.fail("Checkpoint must not load when output is inside the episode")

    monkeypatch.setattr(replay.PiperPolicy, "from_checkpoint", unexpected_load)
    before = {
        file.relative_to(path): file.read_bytes()
        for file in path.rglob("*")
        if file.is_file()
    }
    for output in (path, path / "nested" / "report", alias / "report"):
        with pytest.raises(ValueError, match="outside the source episode"):
            replay.main(cli_args(path, checkpoint, output))
    after = {
        file.relative_to(path): file.read_bytes()
        for file in path.rglob("*")
        if file.is_file()
    }
    assert before == after
    assert not (path / "nested").exists()
    assert not (path / "report").exists()
