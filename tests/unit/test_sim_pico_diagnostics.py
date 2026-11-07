"""Exercise simulation diagnostics without a headset, CAN, or a GUI."""
from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.scripts import sim_piper_pico as sim


class FakeEye:
    def __init__(self, count, stamp):
        self.ring = SimpleNamespace(count=count)
        self.stamp = stamp
        self.requested_k = None

    def get_last(self, k):
        self.requested_k = k
        return {"camera_receive_timestamp": np.array([self.stamp])}


def test_eye_reader_preserves_producer_counts_and_labels_read_phase(monkeypatch):
    stereo = SimpleNamespace(left=FakeEye(100, 10.0), right=FakeEye(101, 10.03),
                             mode="stereo", enabled_eyes=("left", "right"))
    monkeypatch.setattr(sim.time, "time", lambda: 10.04)
    data = sim._read_eyes(stereo, 6)
    assert data["camera_left_count"] == 100
    assert data["camera_right_count"] == 101
    assert data["camera_left_age_s"] == pytest.approx(0.04)
    assert data["camera_right_age_s"] == pytest.approx(0.01)
    assert data["camera_read_phase_skew_s"] == pytest.approx(0.03)
    assert stereo.left.requested_k == stereo.right.requested_k == 6
    assert data["camera_left_count_mono"] <= data["camera_right_count_mono"]


@pytest.mark.parametrize("mode", ["left", "right"])
def test_mono_eye_reader_omits_inactive_eye_and_stereo_skew(monkeypatch, mode):
    eye = FakeEye(42, 10.0)
    camera = SimpleNamespace(mode=mode, enabled_eyes=(mode,), **{mode: eye})
    monkeypatch.setattr(sim.time, "time", lambda: 10.02)
    data = sim._read_eyes(camera, 6)
    inactive = "right" if mode == "left" else "left"
    assert data["camera_mode"] == mode
    assert data[f"camera_{mode}_count"] == 42
    assert data[f"camera_{mode}_age_s"] == pytest.approx(0.02)
    assert eye.requested_k == 6
    assert not any(key.startswith(f"camera_{inactive}_") for key in data)
    assert "camera_read_phase_skew_s" not in data


def test_no_camera_sim_still_observes_source_and_prints_diagnostics(monkeypatch, capsys):
    class Context:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    class FakeEnv(Context):
        horizon = 2
        frequency = 30
        def __init__(self, *args, **kwargs):
            pass

    records = []
    class Recorder:
        cmd_jumps = 0
        held_rows = 0
        reanchor_rows = 0
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def write(self, env, sample, diagnostics=None):
            records.append(diagnostics)
        def close(self):
            pass

    def run(env, frequency, home_seconds, on_tick):
        on_tick(env, {"receive_timestamp": 1000})

    monkeypatch.setattr(sim, "SharedMemoryManager", Context)
    monkeypatch.setattr(sim, "PiperPicoEnv", FakeEnv)
    monkeypatch.setattr(sim, "TeleopRecorder", Recorder)
    monkeypatch.setattr(sim, "SimBimanualView", lambda: SimpleNamespace(update=lambda env: True, close=lambda: None))
    monkeypatch.setattr(sim, "run_teleop", run)
    sim.main([])
    assert records == [None]
    output = capsys.readouterr().out
    assert "运行诊断" in output
    assert "姿态诊断不完整" in output
    assert "解码发布" not in output


@pytest.mark.parametrize("frequency", [30, 60, 90])
def test_control_frequency_reaches_env_loop_monitor_and_history_without_a_view(monkeypatch, frequency):
    seen = {}

    class Context:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    class Env(Context):
        horizon = 2
        def __init__(self, *args, **kwargs):
            assert kwargs["simulate"] is True
            self.frequency = kwargs["frequency"]
            seen["env_hz"] = self.frequency

    class Recorder:
        cmd_jumps = 2
        held_rows = 3
        reanchor_rows = 4
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def write(self, env, sample, diagnostics=None):
            seen["recorded"] = True
        def close(self):
            pass

    def monitor_factory(**kwargs):
        seen["monitor_hz"] = kwargs["control_hz"]
        def report(cmd_jumps, *, held_rows, reanchor_rows):
            seen["report"] = (cmd_jumps, held_rows, reanchor_rows)
            return "done"
        return SimpleNamespace(observe=lambda *a: None, report=report)

    def history(horizon, hz):
        seen["history_hz"] = hz
        return 4

    def run(env, hz, home_seconds, on_tick):
        seen["loop_hz"] = hz
        assert on_tick(env, {}) is True

    def forbidden_view():
        raise AssertionError("--no-view must not open a GUI")

    monkeypatch.setattr(sim, "SharedMemoryManager", Context)
    monkeypatch.setattr(sim, "PiperPicoEnv", Env)
    monkeypatch.setattr(sim, "TeleopRecorder", Recorder)
    monkeypatch.setattr(sim, "RateMonitor", monitor_factory)
    monkeypatch.setattr(sim, "observation_k", history)
    monkeypatch.setattr(sim, "SimBimanualView", forbidden_view)
    monkeypatch.setattr(sim, "run_teleop", run)
    sim.main(["--frequency", str(frequency), "--no-view"])
    assert seen == {"env_hz": frequency, "monitor_hz": frequency,
                    "loop_hz": frequency, "history_hz": frequency, "recorded": True,
                    "report": (2, 3, 4)}


def test_frequency_defaults_to_30_and_rejects_unsupported_values():
    assert sim.build_parser().parse_args([]).frequency == 30
    for value in ("0", "-1", "nan", "inf", "120"):
        with pytest.raises(SystemExit):
            sim.build_parser().parse_args(["--frequency", value])


def test_camera_mode_default_choices_and_validation():
    parser = sim.build_parser()
    assert parser.parse_args([]).camera_mode == "stereo"
    for mode in ("stereo", "left", "right"):
        assert parser.parse_args(["--camera-mode", mode]).camera_mode == mode
    with pytest.raises(SystemExit):
        parser.parse_args(["--camera-mode", "both"])


@pytest.mark.parametrize("mode", ["stereo", "left", "right"])
def test_camera_mode_reaches_headset_reader_monitor_and_csv(monkeypatch, mode):
    seen = {}
    sides = ("left", "right") if mode == "stereo" else (mode,)

    class Context:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    class Camera:
        def __init__(self, *args, **kwargs):
            seen["camera_mode"] = self.mode = kwargs["mode"]
            self.enabled_eyes = sides
            for side in sides:
                setattr(self, side, FakeEye(42, 10.0))
        def start(self):
            seen["started"] = True
        def stop(self):
            seen["stopped"] = True

    class Env(Context):
        horizon = 2
        frequency = 30
        def __init__(self, *args, **kwargs):
            assert kwargs["simulate"] is True

    class Recorder:
        cmd_jumps = held_rows = reanchor_rows = 0
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def write(self, env, sample, diagnostics=None):
            seen["recorded"] = diagnostics
        def close(self):
            pass

    def monitor(**kwargs):
        seen["monitor_mode"] = kwargs["camera_mode"]
        return SimpleNamespace(observe=lambda *a: None, report=lambda *a, **kw: "done")

    monkeypatch.setattr(sim, "SharedMemoryManager", Context)
    monkeypatch.setattr(sim, "PicoStereoProcess", Camera)
    monkeypatch.setattr(sim, "PiperPicoEnv", Env)
    monkeypatch.setattr(sim, "TeleopRecorder", Recorder)
    monkeypatch.setattr(sim, "RateMonitor", monitor)
    monkeypatch.setattr(sim, "_wait_until_ready", lambda camera: None)
    monkeypatch.setattr(sim, "run_teleop", lambda env, hz, home, on_tick: on_tick(env, {}))
    sim.main(["--headset-ip", "192.0.2.1", "--no-view", "--camera-mode", mode])
    assert seen["camera_mode"] == seen["monitor_mode"] == mode
    assert seen["started"] and seen["stopped"]
    assert seen["recorded"]["camera_mode"] == mode
    for side in ("left", "right"):
        assert (f"camera_{side}_count" in seen["recorded"]) == (side in sides)
