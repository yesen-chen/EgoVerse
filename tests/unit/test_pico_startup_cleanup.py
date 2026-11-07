"""Startup rollback must work without launching a process or opening hardware."""

from types import SimpleNamespace

import pytest

from egomimic.robot.piper_pico.env import PiperPicoEnv
from egomimic.scripts import sim_piper_pico as sim


@pytest.mark.parametrize("failure_stage", ["camera0", "camera1", "arm0", "arm1", "pico", "ready", "seed"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_environment_rolls_back_every_device_when_enter_fails(failure_stage, error_type):
    calls = []
    failure = error_type("injected startup failure")

    def step(name):
        calls.append(name)
        if name == failure_stage:
            raise failure

    def device(name):
        return SimpleNamespace(
            start=lambda: step(name),
            stop=lambda: calls.append(f"stop:{name}"),
        )

    # Bypass the constructor so no shared memory, SDK, CAN, or camera is opened.
    env = PiperPicoEnv.__new__(PiperPicoEnv)
    env.cameras = [device("camera0"), device("camera1")]
    env.arms = [device("arm0"), device("arm1")]
    env.pico = device("pico")
    env._recording = False
    env.is_ready = False
    env._wait_ready = lambda timeout: step("ready")
    env.seed_clutches_from_measured = lambda: step("seed")

    with pytest.raises(error_type) as caught:
        with env:
            pytest.fail("failed startup must not enter the body")

    assert caught.value is failure
    assert calls[-5:] == ["stop:pico", "stop:arm0", "stop:arm1", "stop:camera0", "stop:camera1"]
    assert not env.is_ready


def test_environment_preserves_startup_error_when_cleanup_also_fails(capsys):
    failure = RuntimeError("original startup failure")
    env = PiperPicoEnv.__new__(PiperPicoEnv)
    env.cameras = []
    env.arms = []

    def fail_start():
        raise failure

    def fail_stop():
        raise RuntimeError("cleanup failure")

    env.pico = SimpleNamespace(start=fail_start)
    env.stop = fail_stop
    with pytest.raises(RuntimeError) as caught:
        env.start()
    assert caught.value is failure
    assert "cleanup failure" in capsys.readouterr().out


@pytest.mark.parametrize("failure_stage", ["camera_start", "camera_ready", "env_construct", "env_enter"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_sim_stops_camera_before_shared_memory_on_startup_failure(monkeypatch, failure_stage, error_type):
    calls = []
    failure = error_type("injected startup failure")

    def step(name):
        calls.append(name)
        if name == failure_stage:
            raise failure

    class MemoryManager:
        def __enter__(self):
            calls.append("shm_enter")
            return self

        def __exit__(self, *args):
            calls.append("shm_exit")

    class Stereo:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            step("camera_start")

        def stop(self):
            calls.append("camera_stop")

    class Env:
        horizon = 2
        frequency = 30

        def __init__(self, *args, **kwargs):
            step("env_construct")

        def __enter__(self):
            step("env_enter")
            return self

        def __exit__(self, *args):
            calls.append("env_exit")

    recorder = SimpleNamespace(
        cmd_jumps=0,
        held_rows=0,
        reanchor_rows=0,
        close=lambda: calls.append("recorder_close"),
    )
    monkeypatch.setattr(sim, "SharedMemoryManager", MemoryManager)
    monkeypatch.setattr(sim, "PicoStereoProcess", Stereo)
    monkeypatch.setattr(sim, "_wait_until_ready", lambda stereo: step("camera_ready"))
    monkeypatch.setattr(sim, "PiperPicoEnv", Env)
    monkeypatch.setattr(sim, "TeleopRecorder", lambda *args, **kwargs: recorder)
    monkeypatch.setattr(sim, "run_teleop", lambda *args, **kwargs: pytest.fail("startup should fail first"))
    monkeypatch.setattr(sim, "SimBimanualView", lambda: SimpleNamespace(close=lambda: calls.append("view_close")))

    with pytest.raises(error_type) as caught:
        sim.main(["--headset-ip", "192.0.2.1"])

    assert caught.value is failure
    assert calls.count("camera_stop") == 1
    assert calls.index("camera_stop") < calls.index("shm_exit")
    assert calls[-2:] == ["recorder_close", "view_close"]
