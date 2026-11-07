"""Virtual-clock acceptance of one-row synchronous policy execution."""

import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egomimic.rldb.action_timing import ActionTiming
from egomimic.robot.piper_pico.policy_execution import (
    BlockExecutionError,
    RecordingActionSink,
    SafetyLimits,
    SafetyViolation,
    execute_action_block,
    resample_action_block,
)


class FakeClock:
    def __init__(self):
        self.now = 20.0
        self.epoch = 1700000000.0
        self.wall_shift = 0.0
        self.sleeps = []
        self.next_oversleep = 0.0

    def monotonic(self):
        return self.now

    def wall_time(self):
        return self.epoch + self.now + self.wall_shift

    def sleep(self, seconds):
        assert seconds > 0
        self.sleeps.append(seconds)
        self.now += seconds + self.next_oversleep
        self.next_oversleep = 0.0


def limits(**overrides):
    return SafetyLimits(
        workspace_min=[[-1, -1, -1], [-1, -1, -1]],
        workspace_max=[[1, 1, 1], [1, 1, 1]],
        **overrides,
    )


def trajectory(points=100, *, horizon=31, stride=1):
    timing = ActionTiming(30, horizon, stride, points)
    fractions = np.linspace(0, 1, points)
    actions = np.zeros((points, 14))
    for start in (0, 7):
        actions[:, start] = fractions * 0.12
        actions[:, start + 2] = 0.5
        actions[:, start + 5] = fractions * 0.3
        actions[:, start + 6] = 0.02 + fractions * 0.02
    return actions, timing


def execute(actions=None, timing=None, sink=None, clock=None, **overrides):
    default_actions, default_timing = trajectory()
    actions = default_actions if actions is None else actions
    timing = default_timing if timing is None else timing
    sink = RecordingActionSink() if sink is None else sink
    clock = FakeClock() if clock is None else clock
    kwargs = dict(
        control_frequency=10,
        initial_action=actions[0],
        observation_timestamp=clock.wall_time(),
        limits=limits(),
        monotonic=clock.monotonic,
        wall_time=clock.wall_time,
        sleep=clock.sleep,
    )
    kwargs.update(overrides)
    return execute_action_block(actions, timing, sink, **kwargs), sink, clock


def test_resampling_preserves_span_independent_of_model_point_count():
    coarse, timing_coarse = trajectory(3)
    fine, timing_fine = trajectory(100)
    a = resample_action_block(coarse, timing_coarse, 30)
    b = resample_action_block(fine, timing_fine, 30)
    assert a.full_span_s == b.full_span_s == 1.0
    assert a.actual_control_frequency == b.actual_control_frequency == 30
    assert len(a.actions) == len(b.actions) == 31
    np.testing.assert_allclose(a.offsets_s, np.arange(31) / 30, atol=1e-14)
    np.testing.assert_allclose(a.actions, b.actions, atol=1e-14)
    np.testing.assert_allclose(a.actions[[0, -1]], coarse[[0, -1]], atol=1e-14)


def test_noninteger_frequency_reports_actual_and_stride_keeps_effective_span():
    actions, timing = trajectory(7, horizon=12, stride=4)
    block = resample_action_block(actions, timing, 10)
    assert timing.source_span_s == pytest.approx(11 / 30)
    assert block.full_span_s == pytest.approx(8 / 30)
    assert block.requested_control_frequency == 10
    assert block.actual_control_frequency == pytest.approx(3 / (8 / 30))
    np.testing.assert_allclose(np.diff(block.offsets_s), np.repeat(8 / 90, 3))
    assert block.offsets_s[0] == 0
    assert block.offsets_s[-1] == timing.effective_span_s
    # Even a requested rate below 1/span retains two endpoints, with an honest
    # reported frequency instead of claiming a single current pose covers it.
    low = resample_action_block(actions, timing, 0.1)
    assert low.full_point_count == 2
    assert low.actual_control_frequency == pytest.approx(30 / 8)


def test_slerp_takes_shortest_rotation_path_not_rotvec_linear_interpolation():
    actions, timing = trajectory(2)
    for start in (0, 7):
        actions[:, start + 3 : start + 6] = np.deg2rad([[0, 0, 170], [0, 0, -170]])
    block = resample_action_block(actions, timing, 2)
    for start in (0, 7):
        rotations = Rotation.from_rotvec(block.actions[:, start + 3 : start + 6])
        np.testing.assert_allclose(
            rotations[1].apply([1, 0, 0]), [-1, 0, 0], atol=1e-12
        )
        distances = (rotations[:-1].inv() * rotations[1:]).magnitude()
        np.testing.assert_allclose(distances, np.deg2rad([10, 10]), atol=1e-12)
        np.testing.assert_allclose(block.actions[:, start + 6], [0.02, 0.03, 0.04])


@pytest.mark.parametrize(
    "steps,count,span", [(None, 11, 1), (1, 1, 0), (4, 4, 0.3), (1000, 11, 1)]
)
def test_execute_steps_selects_prefix_and_never_extends(steps, count, span):
    actions, timing = trajectory()
    block = resample_action_block(actions, timing, 10, execute_steps=steps)
    assert len(block.actions) == count
    assert block.offsets_s[-1] == pytest.approx(span)
    assert block.full_point_count == 11
    assert block.full_span_s == 1


def test_exactly_one_row_each_tick_deadline_is_freshness_and_report_is_json():
    report, sink, clock = execute(command_freshness_s=0.07)
    assert report["status"] == "completed"
    assert report["sent_count"] == 11
    assert sink.hold_reasons == ["completed"]
    times = [sample["actual_monotonic"] for sample in report["samples"]]
    np.testing.assert_allclose(times, 20 + np.arange(11) / 10, atol=1e-12)
    assert report["actual_sent_span_s"] == pytest.approx(1)
    assert report["observed_send_frequency"] == pytest.approx(10)
    for record, sample in zip(sink.records, report["samples"]):
        assert np.asarray(record["action"]).shape == (14,)
        assert sample["schedule_slip_s"] == pytest.approx(0, abs=1e-12)
        assert record["deadline_walltime"] == pytest.approx(
            sample["actual_walltime"] + 0.07
        )
    assert max(clock.sleeps) <= 0.01 + 1e-12
    json.dumps(report, allow_nan=False)


def test_continuing_completed_chunk_does_not_hold_or_clear_trajectory():
    report, sink, _ = execute(hold_on_completion=False)
    assert report["status"] == "completed"
    assert report["sent_count"] == 11
    assert report["hold_attempted"] is False
    assert sink.hold_reasons == []


@pytest.mark.parametrize("command,status", [("pause", "paused"), ("stop", "stopped")])
def test_continuing_chunk_still_holds_when_operator_stops(command, status):
    clock = FakeClock()
    report, sink, _ = execute(
        clock=clock,
        hold_on_completion=False,
        control_callback=lambda: command if clock.now >= 20.15 else "run",
    )
    assert report["status"] == status
    assert report["hold_attempted"] is True
    assert report["sent_count"] == 2
    assert sink.hold_reasons == [status]


def test_continuing_chunk_still_holds_on_sink_failure():
    class BrokenSink(RecordingActionSink):
        def send_action(self, action, deadline_walltime):
            raise OSError("send failed")

    sink = BrokenSink()
    with pytest.raises(BlockExecutionError, match="send failed") as error:
        execute(sink=sink, hold_on_completion=False)
    assert error.value.report["hold_attempted"] is True
    assert sink.hold_reasons == ["failed"]


def test_continuing_chunk_still_holds_on_interrupt():
    sink = RecordingActionSink()

    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute(sink=sink, hold_on_completion=False, control_callback=interrupt)
    assert sink.hold_reasons == ["interrupted"]


def test_tolerated_stall_shifts_next_wait_instead_of_burst_catchup():
    clock = FakeClock()

    class DelayedSink(RecordingActionSink):
        def send_action(self, action, deadline_walltime):
            super().send_action(action, deadline_walltime)
            if len(self.records) == 2:
                clock.now += 0.13

    report, _, _ = execute(clock=clock, sink=DelayedSink(), max_schedule_slip_s=0.05)
    times = np.array([sample["actual_monotonic"] for sample in report["samples"]])
    assert np.diff(times).min() >= 0.1 - 1e-12
    assert report["samples"][2]["schedule_slip_s"] == pytest.approx(0.03)
    assert report["actual_sent_span_s"] == pytest.approx(1.03)
    assert report["observed_send_frequency"] < report["actual_control_frequency"]


def test_large_stall_holds_without_catchup_send():
    clock, sink = FakeClock(), RecordingActionSink()
    clock.next_oversleep = 0.3
    with pytest.raises(BlockExecutionError, match="schedule slip") as error:
        execute(clock=clock, sink=sink, max_schedule_slip_s=0.05)
    assert len(sink.records) == 1
    assert sink.hold_reasons == ["failed"]
    assert error.value.report["error"]["type"] == "ScheduleOverrun"
    assert error.value.report["hold_succeeded"] is True
    overrun = error.value.report["schedule_overrun"]
    assert overrun["kind"] == "schedule_slip" and overrun["index"] == 1
    assert overrun["slip_s"] > 0.05
    attempt = error.value.report["last_attempt"]
    assert attempt["index"] == 1 and attempt["phase"] == "schedule_check"
    assert attempt["stage_timings"]["max_sleep_overshoot_s"] == pytest.approx(0.3)
    json.dumps(error.value.report, allow_nan=False)


def test_timing_diagnostics_separate_callback_and_sink_stalls():
    clock = FakeClock()

    class Sink(RecordingActionSink):
        def send_action(self, action, deadline_walltime):
            super().send_action(action, deadline_walltime)
            clock.now += 0.015

    def callback():
        clock.now += 0.003
        return "run"

    report, _, _ = execute(
        clock=clock, sink=Sink(), control_callback=callback, execute_steps=2
    )
    for sample in report["samples"]:
        stages = sample["stage_timings"]
        assert stages["sink_elapsed_s"] == pytest.approx(0.015)
        assert stages["max_control_callback_s"] == pytest.approx(0.003)
        assert stages["max_sleep_overshoot_s"] == pytest.approx(0)
    assert report["last_attempt"]["phase"] == "sent"


@pytest.mark.parametrize("command,status", [("pause", "paused"), ("stop", "stopped")])
def test_pause_or_stop_aborts_block_and_next_block_has_a_new_schedule(command, status):
    clock, sink = FakeClock(), RecordingActionSink()
    report, _, _ = execute(
        clock=clock,
        sink=sink,
        control_callback=lambda: command if clock.now >= 20.15 else "run",
    )
    assert report["status"] == status
    assert len(sink.records) == 2
    assert sink.hold_reasons == [status]
    clock.now += 5
    next_report, next_sink, _ = execute(clock=clock, execute_steps=2)
    assert next_report["samples"][0]["actual_monotonic"] >= 25.15
    assert next_report["samples"][0]["nominal_offset_s"] == 0
    assert len(next_sink.records) == 2


def test_interrupt_always_holds_and_propagates():
    sink = RecordingActionSink()

    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute(sink=sink, control_callback=interrupt)
    assert not sink.records
    assert sink.hold_reasons == ["interrupted"]


def test_sink_failure_and_hold_failure_are_not_hidden():
    class BrokenSink(RecordingActionSink):
        def send_action(self, action, deadline_walltime):
            raise OSError("send failed")

        def hold(self, reason):
            super().hold(reason)
            raise OSError("hold failed")

    sink = BrokenSink()
    with pytest.raises(BlockExecutionError, match="send failed") as error:
        execute(sink=sink)
    assert sink.hold_reasons == ["failed"]
    assert error.value.report["error"]["message"] == "send failed"
    assert error.value.report["hold_error"]["message"] == "hold failed"
    assert error.value.report["sent_count"] == 0


@pytest.mark.parametrize("age", [0.6, -0.2, math.nan])
def test_stale_future_or_invalid_observation_prevents_any_send(age):
    clock, sink = FakeClock(), RecordingActionSink()
    with pytest.raises(BlockExecutionError) as error:
        execute(clock=clock, sink=sink, observation_timestamp=clock.wall_time() - age)
    assert not sink.records
    assert sink.hold_reasons == ["failed"]
    assert error.value.report["error"]["type"] == "SafetyViolation"


def test_wall_clock_jump_is_detected_before_next_send():
    clock, sink = FakeClock(), RecordingActionSink()

    def adjust_wall():
        if clock.now >= 20.05:
            clock.wall_shift = 1

    with pytest.raises(BlockExecutionError, match="wall clock"):
        execute(clock=clock, sink=sink, control_callback=adjust_wall)
    assert len(sink.records) == 1
    assert sink.hold_reasons == ["failed"]


@pytest.mark.parametrize(
    "column,value", [(0, 2), (7, -2), (6, -0.1), (13, 0.081), (2, math.inf)]
)
def test_selected_future_invalid_target_rejected_before_first_send(column, value):
    actions, timing = trajectory()
    actions[-1, column] = value
    sink = RecordingActionSink()
    with pytest.raises(BlockExecutionError):
        execute(actions, timing, sink=sink)
    assert not sink.records
    assert sink.hold_reasons == ["failed"]


@pytest.mark.parametrize(
    "index,value,match", [(0, 0.2, "translation"), (10, 1.0, "rotation")]
)
def test_first_command_is_limited_against_initial_measured_state(index, value, match):
    actions, timing = trajectory()
    initial = actions[0].copy()
    actions[:, index] += value
    sink = RecordingActionSink()
    with pytest.raises(BlockExecutionError, match=match):
        execute(actions, timing, sink=sink, initial_action=initial)
    assert not sink.records


def test_later_command_is_limited_against_last_sent_and_equivalent_rotations_are_ok():
    actions, timing = trajectory(2)
    actions[1, 0] = 0.8
    sink = RecordingActionSink()
    with pytest.raises(BlockExecutionError, match="translation"):
        execute(actions, timing, sink=sink, control_frequency=2)
    assert len(sink.records) == 1
    previous = actions[0].copy()
    target = previous.copy()
    previous[5], target[5] = math.pi - 0.01, -math.pi + 0.01
    limits().check_action(target, previous)
    target[5] = 0
    with pytest.raises(SafetyViolation, match="rotation"):
        limits().check_action(target, previous)


@pytest.mark.parametrize(
    "setting",
    [
        {"control_frequency": 0},
        {"control_frequency": math.inf},
        {"execute_steps": 0},
        {"execute_steps": -2},
        {"execute_steps": 1.5},
        {"max_schedule_slip_s": -1},
        {"command_freshness_s": 0},
        {"poll_interval_s": 0},
        {"hold_on_completion": "false"},
    ],
)
def test_invalid_execution_settings_hold_and_fail(setting):
    sink = RecordingActionSink()
    with pytest.raises(BlockExecutionError):
        execute(sink=sink, **setting)
    assert not sink.records
    assert sink.hold_reasons == ["failed"]


def test_workspace_is_per_arm_and_limit_arrays_cannot_change_behind_executor():
    minimum = np.array([[-1, -1, -1], [0, -1, -1.0]])
    maximum = np.ones((2, 3))
    config = SafetyLimits(minimum, maximum)
    maximum[:] = 99
    actions, _ = trajectory()
    row = actions[0].copy()
    row[0], row[7] = -0.2, -0.2
    with pytest.raises(SafetyViolation, match="right"):
        config.check_action(row)
    with pytest.raises(ValueError):
        config.workspace_max[0, 0] = 99
    row[0], row[7] = 2, 0
    with pytest.raises(SafetyViolation, match="left"):
        config.check_action(row)


def test_geometric_limits_can_be_disabled_without_disabling_data_checks():
    config = SafetyLimits(None, None, None, None)
    actions, _ = trajectory()
    previous = actions[0]
    target = previous.copy()
    target[[0, 7]] = [2, -2]
    target[3:6] = [1, 0, 0]
    target[10:13] = [0, 1, 0]
    config.check_action(target, previous)
    for column, value, message in ((0, np.nan, "finite"), (6, 0.081, "gripper")):
        bad = target.copy()
        bad[column] = value
        with pytest.raises(SafetyViolation, match=message):
            config.check_action(bad, previous)
    with pytest.raises(SafetyViolation, match="stale"):
        config.check_observation(1, 2)
    with pytest.raises(SafetyViolation, match="future"):
        config.check_observation(2, 1)
    with pytest.raises(SafetyViolation, match="finite"):
        config.check_observation(np.nan, 1)


@pytest.mark.parametrize(
    "minimum,maximum", [(None, np.ones((2, 3))), (-np.ones((2, 3)), None)]
)
def test_workspace_must_be_disabled_as_a_pair(minimum, maximum):
    with pytest.raises(ValueError, match="both be None"):
        SafetyLimits(minimum, maximum)


def test_optional_step_limits_are_independent_and_old_defaults_still_apply():
    actions, _ = trajectory()
    previous = actions[0]
    target = previous.copy()
    target[0] += 0.2
    target[5] += 1
    with pytest.raises(SafetyViolation, match="translation"):
        SafetyLimits(None, None).check_action(target, previous)
    with pytest.raises(SafetyViolation, match="rotation"):
        SafetyLimits(None, None, max_translation_step_m=None).check_action(
            target, previous
        )
    with pytest.raises(SafetyViolation, match="translation"):
        SafetyLimits(None, None, max_rotation_step_rad=None).check_action(
            target, previous
        )
    bounded = limits(max_translation_step_m=None, max_rotation_step_rad=None)
    target[0] = 2
    with pytest.raises(SafetyViolation, match="workspace"):
        bounded.check_action(target, previous)


def test_current_checkpoint_timing_restores_30_commands_from_100_predictions():
    actions, timing = trajectory(points=100, horizon=30)
    report, sink, _ = execute(
        actions,
        timing,
        limits=SafetyLimits(None, None, None, None),
        control_frequency=30,
        execute_steps=30,
        hold_on_completion=False,
    )
    assert len(sink.records) == report["sent_count"] == 30
    assert report["full_span_s"] == pytest.approx(29 / 30)
    assert report["actual_sent_span_s"] == pytest.approx(29 / 30)
    assert report["actual_control_frequency"] == pytest.approx(30)
    np.testing.assert_allclose(sink.records[0]["action"], actions[0])
    np.testing.assert_allclose(sink.records[-1]["action"], actions[-1])
