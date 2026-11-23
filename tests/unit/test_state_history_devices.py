"""History buffer and startup contracts without processes, CAN or cameras."""

import multiprocessing as mp
import multiprocessing.managers
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from test_live_policy import Policy, Provider, arguments, limits

from egomimic.robot.piper_pico import arm_process
from egomimic.robot.piper_pico import env as env_module
from egomimic.robot.piper_pico import live_policy as live
from egomimic.robot.piper_pico import live_policy_devices as devices
from egomimic.robot.piper_pico import shm as shm_module
from egomimic.robot.piper_pico.policy_rollout import OfflineReplayClock
from egomimic.robot.piper_pico.state_history import (
    HistoryCoverageError,
    sample_state_history,
)


class InMemoryManager:
    """Exercise real ring allocation/copies using owned memory, without IPC."""

    def SharedMemory(self, size):
        return SimpleNamespace(buf=bytearray(size))


@pytest.fixture
def process_context(monkeypatch):
    context = SimpleNamespace(Event=Mock(), Process=Mock())
    monkeypatch.setattr(mp, "get_context", Mock(return_value=context))
    return context


def test_long_history_survives_ring_wrap_at_feedback_frequency(
    monkeypatch, process_context
):
    now = [0.0]
    monkeypatch.setattr(
        shm_module,
        "time",
        SimpleNamespace(monotonic=lambda: now[0], sleep=Mock()),
    )
    arm = arm_process.PiperArmProcess(
        InMemoryManager(), "unused", "left", feedback_history_s=2.25
    )
    for index in range(340):
        now[0] = index / 100.0
        state = arm_process.state_examples()
        state["robot_timestamp"] = now[0]
        state["ee_pose"][0] = now[0]
        state["gripper_position"] = now[0] / 100.0
        arm.ring.put(state)
    snapshot = arm.get_all_state()
    stream = {
        "eef_pose": snapshot["ee_pose"],
        "eef_pose__timestamp": snapshot["robot_timestamp"],
        "gripper_width": snapshot["gripper_position"],
        "gripper_width__timestamp": snapshot["robot_timestamp"],
    }
    history = sample_state_history(
        [stream, stream], 3.4, (-2.0, -0.025), available_until=[3.39, 3.39]
    )
    np.testing.assert_allclose(history[:, 0], [1.4, 3.375])
    np.testing.assert_allclose(history[:, 6], [0.014, 0.03375])
    assert arm.ring.get_max_k == 225
    shm_module.time.sleep.assert_not_called()
    process_context.Process.return_value.start.assert_not_called()


def test_disabled_history_preserves_default_ring_size(process_context):
    arm = arm_process.PiperArmProcess(InMemoryManager(), "unused", "left")
    assert arm.ring.get_max_k == 128
    assert arm.ring.put_desired_frequency == 100.0
    process_context.Process.return_value.start.assert_not_called()


@pytest.mark.parametrize("seconds", [0.0, -1.0, np.nan, np.inf])
def test_invalid_history_duration_rejected_before_device_construction(
    process_context, seconds
):
    with pytest.raises(ValueError, match="feedback_history_s"):
        arm_process.PiperArmProcess(
            InMemoryManager(), "unused", "left", feedback_history_s=seconds
        )
    process_context.Process.assert_not_called()


@pytest.mark.parametrize(
    "offsets, seconds", [((), 1.28), ((-0.05, -0.025), 1.28), ((-2.1, -0.025), 2.35)]
)
def test_environment_reserves_feedback_history_at_100hz_independent_of_control(
    tmp_path, monkeypatch, offsets, seconds
):
    constructor = Mock()
    monkeypatch.setattr(env_module, "PiperArmProcess", constructor)
    monkeypatch.setattr(
        env_module,
        "ReplayBuffer",
        lambda *args: SimpleNamespace(root=SimpleNamespace(attrs={})),
    )
    forbidden = Mock(side_effect=AssertionError("hardware must not be opened"))
    monkeypatch.setattr(env_module, "CameraProcess", forbidden)
    monkeypatch.setattr(env_module, "PicoProcess", forbidden)
    env = env_module.PiperPicoEnv(
        object(),
        tmp_path,
        simulate=True,
        enable_pico=False,
        frequency=30,
        state_history_offsets_s=offsets,
    )
    assert env.state_history_offsets_s == offsets
    assert constructor.call_count == 2
    for call in constructor.call_args_list:
        assert call.kwargs["feedback_history_s"] == pytest.approx(seconds)
        # Control is 30 Hz; the feedback worker retains its native 100 Hz.
        assert "frequency" not in call.kwargs
    forbidden.assert_not_called()


@pytest.fixture
def startup(tmp_path, monkeypatch):
    clock = OfflineReplayClock()
    keys = SimpleNamespace(poll=Mock(return_value="pause"), pause=Mock())
    log = SimpleNamespace(directory=tmp_path, summary={}, event=Mock())
    env = SimpleNamespace(start=Mock(), stop=Mock(), cameras=[])
    constructor = Mock(return_value=env)
    provider = Provider(clock)
    original_latest = provider.latest
    provider.latest = Mock(side_effect=original_latest)
    sink = SimpleNamespace(hold=Mock(), home=Mock())
    policy = Policy(clock)
    policy.adapter = SimpleNamespace(gripper_open_m=0.08)
    loop = Mock(return_value="stopped")
    monkeypatch.setattr(live, "KeyboardControl", lambda: nullcontext(keys))
    monkeypatch.setattr(
        multiprocessing.managers,
        "SharedMemoryManager",
        lambda **kwargs: nullcontext(object()),
    )
    monkeypatch.setattr(env_module, "PiperPicoEnv", constructor)
    monkeypatch.setattr(devices, "LiveObservationProvider", lambda *a, **k: provider)
    monkeypatch.setattr(devices, "LiveActionSink", lambda *a, **k: sink)
    monkeypatch.setattr(live, "run_live_loop", loop)
    monkeypatch.setattr(
        live, "time", SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep)
    )
    args = arguments(
        left_can="unused-left",
        right_can="unused-right",
        move_spd_rate=50,
        max_observation_age=0.5,
        preview=False,
    )

    def run():
        return live._run_device_session(
            args, log, {"camera_specs": []}, policy, limits()
        )

    return SimpleNamespace(
        run=run,
        policy=policy,
        provider=provider,
        clock=clock,
        keys=keys,
        sink=sink,
        env=env,
        constructor=constructor,
        loop=loop,
        original_latest=original_latest,
    )


def test_waits_for_feedback_coverage_before_first_prediction(startup):
    startup.policy.state_history_offsets_s = (-2.0, -0.025)
    attempt = [0]

    def latest():
        attempt[0] += 1
        if attempt[0] < 4:
            assert not startup.policy.observations
            raise HistoryCoverageError("feedback has not covered the oldest offset")
        return startup.original_latest()

    startup.provider.latest.side_effect = latest
    assert startup.run() == "stopped"
    assert startup.clock.elapsed == pytest.approx(0.03)
    assert len(startup.policy.observations) == 1
    assert startup.constructor.call_args.kwargs["state_history_offsets_s"] == (
        -2.0,
        -0.025,
    )
    startup.loop.assert_called_once()
    startup.sink.home.assert_not_called()
    startup.sink.hold.assert_called_once_with("session_exit")
    startup.env.stop.assert_called_once()


def test_coverage_wait_times_out_without_fake_history_or_prediction(startup):
    startup.policy.state_history_offsets_s = (-0.075, -0.025)
    startup.provider.latest.side_effect = HistoryCoverageError("missing")
    with pytest.raises(RuntimeError, match="反馈历史覆盖不足"):
        startup.run()
    assert 3.075 <= startup.clock.elapsed < 3.1
    assert not startup.policy.observations
    startup.loop.assert_not_called()
    startup.sink.hold.assert_called_once_with("session_exit")
    startup.env.stop.assert_called_once()


def test_stop_interrupts_history_wait_without_prediction(startup):
    startup.policy.state_history_offsets_s = (-0.075,)
    startup.provider.latest.side_effect = HistoryCoverageError("missing")
    startup.keys.poll.side_effect = ["pause", "stop"]
    assert startup.run() == "stopped"
    assert startup.clock.elapsed == pytest.approx(0.01)
    assert not startup.policy.observations
    startup.loop.assert_not_called()
    startup.sink.hold.assert_called_once_with("session_exit")
    startup.env.stop.assert_called_once()


def test_health_failure_during_history_wait_is_not_retried(startup):
    startup.policy.state_history_offsets_s = (-0.075,)
    startup.provider.latest.side_effect = RuntimeError("stale CAN feedback")
    with pytest.raises(RuntimeError, match="stale CAN"):
        startup.run()
    assert startup.clock.elapsed == 0
    startup.provider.latest.assert_called_once()
    assert not startup.policy.observations
    startup.sink.hold.assert_called_once_with("session_exit")
    startup.env.stop.assert_called_once()


def test_old_checkpoint_skips_history_wait_and_keeps_device_constructor(startup):
    assert startup.run() == "stopped"
    assert "state_history_offsets_s" not in startup.constructor.call_args.kwargs
    assert startup.clock.elapsed == 0
    startup.provider.latest.assert_called_once()
    assert len(startup.policy.observations) == 1
    startup.loop.assert_called_once()
    startup.env.stop.assert_called_once()
