"""Control timing with fake clocks/devices: no GUI, network, sleeps, or CAN."""

import json
from types import SimpleNamespace

import pytest

from egomimic.robot.piper_pico import collect
from egomimic.robot.piper_pico.loop_profile import LoopProfiler
from egomimic.scripts import sim_piper_pico as sim


class Clock:
    mono = 10.0
    cpu = 0.0

    def advance(self, wall, cpu=0.0):
        self.mono += wall
        self.cpu += cpu

    def profiler(self, **kwargs):
        return LoopProfiler(monotonic=lambda: self.mono, wall=lambda: 1700000000 + self.mono,
                            cpu=lambda: self.cpu, **kwargs)


def test_nested_stage_stall_and_uncovered_delay_are_separate(tmp_path):
    clock = Clock()
    profile = clock.profiler()
    with profile.tick(0, 10.033):
        with profile.span("on_tick"):
            with profile.span("sim.recorder_write"):
                clock.advance(0.2, 0.002)
        clock.advance(0.07, 0.001)  # Delay outside any labelled stage.
    tick = next(e for e in profile.events if e["kind"] == "slow_tick")
    assert tick["unattributed_wall_s"] == pytest.approx(0.07)
    assert tick["wall_s"] == pytest.approx(0.27)
    assert tick["thread_cpu_s"] == pytest.approx(0.003)
    assert [s["depth"] for s in tick["spans"]] == [1, 0]
    assert profile.stats["sim.recorder_write"]["wall_max_s"] == pytest.approx(0.2)
    # Profiling does not open/write a file until explicitly requested at exit.
    assert list(tmp_path.iterdir()) == []
    path = tmp_path / "timing.json"
    profile.write_json(path)
    saved = json.loads(path.read_text())
    assert "local_time" in saved["events"][0]
    assert saved["events"][0]["tick"] == 0


def test_wait_deadline_and_delay_between_ticks_are_observable():
    clock = Clock()
    profile = clock.profiler()
    with profile.tick(0, 10.033):
        with profile.span("precise_wait", deadline_monotonic_s=10.033):
            clock.advance(0.033)
    assert profile.events == []  # The ordinary 30 Hz wait is not a slow stage.
    clock.advance(0.6)
    with profile.tick(1, 10.066):
        with profile.span("precise_wait", deadline_monotonic_s=10.066):
            pass  # Behind schedule: precise_wait returns immediately.
    interval = next(e for e in profile.events if e["kind"] == "slow_tick_interval")
    assert interval["between_ticks_s"] == pytest.approx(0.6)
    assert interval["wall_s"] == pytest.approx(0.633)
    assert interval["previous_tick"]["spans"][0]["stage"] == "precise_wait"
    wait = next(e for e in profile.events if e.get("stage") == "precise_wait")
    assert wait["wall_s"] == 0
    assert wait["deadline_lateness_s"] == pytest.approx(0.567)


def test_event_storage_is_bounded_and_exception_is_preserved():
    clock = Clock()
    profile = clock.profiler(max_events=2)
    failure = RuntimeError("injected")
    with pytest.raises(RuntimeError) as caught:
        with profile.tick(0, 10.033):
            for i in range(10):
                with profile.span("work"):
                    clock.advance(0.1, 0.01)
            with profile.span("failure"):
                clock.advance(0.1)
                raise failure
    assert caught.value is failure
    assert len(profile.events) == 2
    assert profile.dropped_events == 10  # 11 stages plus the completed slow tick.
    assert profile.stats["work"]["count"] == 10
    assert profile.stats["work"]["wall_total_s"] == pytest.approx(1)
    assert profile._active is None


def test_collect_profiles_action_and_restarts_deadline_after_overrun(monkeypatch):
    clock = Clock()
    profile = clock.profiler()
    calls = []
    polls = iter([False, False, True])

    class Keys:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def poll(self):
            return set()

    def sample():
        return dict(button_a=next(polls), button_b=0, button_x=0, button_y=0)

    def action(*args):
        clock.advance(0.2, 0.001)
        return [0]

    def wait(deadline):
        calls.append(deadline)
        clock.advance(max(0.0, deadline - clock.mono))

    env = SimpleNamespace(cameras=[], _recording=False, get_pico=sample,
                          teleop_action=action, exec_actions=lambda *a: [0])
    monkeypatch.setattr(collect, "KeyPoller", Keys)
    monkeypatch.setattr(collect.signal, "signal", lambda *a: None)
    monkeypatch.setattr(collect.time, "monotonic", lambda: clock.mono)
    monkeypatch.setattr(collect.time, "time", lambda: 1700000000 + clock.mono)
    monkeypatch.setattr(collect, "precise_wait", wait)
    collect.run_teleop(env, 30, 3, on_tick=lambda *a: True, loop_profiler=profile)
    # Each 0.2 s action overruns; the next deadline is one period after the
    # late tick, never a burst of catch-up ticks.
    assert calls == pytest.approx([10 + 1/30, 10.2 + 1/30])
    assert profile.stats["teleop_action"]["count"] == 2
    assert profile.stats["teleop_action"]["wall_max_s"] == pytest.approx(0.2)
    assert profile.stats["get_pico"]["count"] == 3
    assert profile.stats["precise_wait"]["wall_max_s"] == 0


@pytest.mark.parametrize("startup_failure", [False, True])
def test_sim_writes_sidecar_after_cleanup_including_startup_failure(monkeypatch, tmp_path, startup_failure):
    calls = []
    path = tmp_path / "run.csv"
    sidecar = tmp_path / "run_timing.json"

    class Context:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            calls.append("context_exit")

    class Env(Context):
        frequency = 30
        horizon = 2
        def __init__(self, *args, **kwargs):
            assert kwargs["simulate"]
        def __enter__(self):
            if startup_failure:
                raise RuntimeError("startup")
            calls.append("env_started")
            return self

    class Recorder:
        cmd_jumps = held_rows = reanchor_rows = 0
        def __init__(self, csv_path, *, profiler):
            assert csv_path == path
            assert isinstance(profiler, LoopProfiler)
        def start(self):
            assert calls[-1] == "env_started"
            calls.append("recorder_start")
        def write(self, *args, **kwargs):
            assert not sidecar.exists()
        def close(self):
            assert not sidecar.exists()
            calls.append("recorder_close")
        def timing_report(self):
            assert calls[-1] == "recorder_close"
            return {"writer_alive": False}

    def run(env, frequency, home_seconds, on_tick, *, loop_profiler):
        with loop_profiler.tick(0, 1e20):
            assert on_tick(env, {}) is True

    monkeypatch.setattr(sim, "default_log_path", lambda prefix: path)
    monkeypatch.setattr(sim, "SharedMemoryManager", Context)
    monkeypatch.setattr(sim, "PiperPicoEnv", Env)
    monkeypatch.setattr(sim, "TeleopRecorder", Recorder)
    monkeypatch.setattr(sim, "RateMonitor", lambda **kw: SimpleNamespace(observe=lambda *a: None, report=lambda *a, **kw: "done"))
    monkeypatch.setattr(sim, "run_teleop", run)
    argv = ["--profile-loop", "--no-view", "--camera-fps", "60", "--camera-mode", "right"]
    if startup_failure:
        with pytest.raises(RuntimeError, match="startup"):
            sim.main(argv)
    else:
        sim.main(argv)
    assert calls[-1] == "recorder_close"
    data = json.loads(sidecar.read_text())
    assert data["recorder"] == {"writer_alive": False}
    assert data["metadata"]["csv_path"] == str(path)
    assert data["metadata"]["frequency_hz"] == 30
    assert data["metadata"]["camera_fps"] == 60
    assert data["metadata"]["camera_mode"] == "right"
    assert data["metadata"]["view_enabled"] is False
    assert data["metadata"]["error_type"] == ("RuntimeError" if startup_failure else None)
    if not startup_failure:
        assert data["stats"]["sim.recorder_write"]["count"] == 1
    assert not sim.build_parser().parse_args([]).profile_loop


@pytest.mark.parametrize("control_failure", [False, True])
def test_writer_close_failure_preserves_cleanup_profile_and_prior_error(monkeypatch, tmp_path, control_failure):
    calls = []
    path = tmp_path / "writer_error.csv"
    writer_error = RuntimeError("background file flush failed")
    original_error = ValueError("control failure")

    class Context:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            calls.append("context_exit")

    class Env(Context):
        frequency = 30
        horizon = 2
        def __init__(self, *args, **kwargs):
            pass

    class Recorder:
        cmd_jumps = held_rows = reanchor_rows = 0
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            calls.append("recorder_start")
        def close(self):
            calls.append("recorder_close")
            raise writer_error
        def timing_report(self):
            assert calls[-1] == "view_close"
            return {"writer_alive": False, "writer_error": str(writer_error)}

    def run(*args, **kwargs):
        if control_failure:
            raise original_error

    monkeypatch.setattr(sim, "default_log_path", lambda prefix: path)
    monkeypatch.setattr(sim, "SharedMemoryManager", Context)
    monkeypatch.setattr(sim, "PiperPicoEnv", Env)
    monkeypatch.setattr(sim, "TeleopRecorder", Recorder)
    monkeypatch.setattr(sim, "SimBimanualView", lambda: SimpleNamespace(close=lambda: calls.append("view_close")))
    monkeypatch.setattr(sim, "RateMonitor", lambda **kw: SimpleNamespace(report=lambda *a, **kw: "done"))
    monkeypatch.setattr(sim, "run_teleop", run)
    expected = original_error if control_failure else writer_error
    with pytest.raises(type(expected)) as caught:
        sim.main(["--profile-loop"])
    assert caught.value is expected
    assert calls[-2:] == ["recorder_close", "view_close"]
    saved = json.loads((tmp_path / "writer_error_timing.json").read_text())
    assert saved["metadata"]["error_type"] == type(expected).__name__
    assert saved["recorder"]["writer_error"] == str(writer_error)
