"""Keyboard and live-session orchestration without devices or GPU."""

from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.rldb.action_timing import ActionTiming
from egomimic.robot.piper_pico import live_policy as live
from egomimic.robot.piper_pico.policy_execution import RecordingActionSink, SafetyLimits
from egomimic.robot.piper_pico.policy_inference import (
    CAMERA_KEYS,
    SOURCE_NAMES,
    ObservationFrame,
)
from egomimic.robot.piper_pico.policy_rollout import OfflineReplayClock


def arguments(**kwargs):
    values = dict(
        execute=False,
        control_frequency=30.0,
        execute_steps=3,
        max_chunks=2,
        home_on_start=False,
    )
    return SimpleNamespace(**(values | kwargs))


class Provider:
    def __init__(self, clock):
        self.clock = clock
        self.health_checks = 0

    def check_health(self):
        self.health_checks += 1

    def latest(self):
        stamp = self.clock.wall_time()
        row = np.zeros(14)
        row[[6, 13]] = 0.04
        return ObservationFrame(
            stamp,
            {name: np.zeros((4, 6, 3), np.uint8) for name in CAMERA_KEYS},
            row,
            {name: stamp for name in SOURCE_NAMES},
        )


class Policy:
    timing = ActionTiming(30, 30, 1, 100)

    def __init__(self, clock, elapsed=0.0):
        self.clock, self.elapsed = clock, elapsed
        self.observations = []

    def predict(self, frame):
        self.observations.append(frame.timestamp)
        self.clock.sleep(self.elapsed)
        actions = np.repeat(frame.base_state[None], 100, axis=0)
        return dict(
            base_actions=actions,
            camera_actions=actions,
            observation_timestamp=frame.timestamp,
            inference_elapsed_s=self.elapsed,
            inference_steps=10,
        )


def limits():
    return SafetyLimits(
        None, None, max_translation_step_m=None, max_rotation_step_rad=None
    )


def test_keyboard_starts_paused_and_stop_is_sticky():
    keys = live.KeyboardControl()
    assert keys.poll() == "pause"
    keys.feed(" ")
    assert keys.poll() == "run"
    keys.feed("p")
    assert keys.poll() == "pause"
    keys.feed(" ")
    keys.feed("q")
    keys.feed(" ")
    keys.pause()
    assert keys.poll() == "stop"


@pytest.mark.parametrize("key", ["q", "Q", "\x1b", "\x03", ""])
def test_terminal_quit_and_eof(key):
    keys = live.KeyboardControl()
    keys.feed(key)
    assert keys.poll() == "stop"


def test_parser_requires_explicit_calibration_and_parses_false():
    parser = live.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--checkpoint", "model.ckpt"])
    args = parser.parse_args(
        [
            "--checkpoint",
            "model.ckpt",
            "--training-data",
            "data",
            "--calibration",
            "calibration.json",
            "--camera-config",
            "rig.json",
            "--home-on-start",
            "false",
        ]
    )
    assert args.home_on_start is False and args.execute is False
    assert args.execute_steps == 30 and args.control_frequency == 30
    assert args.inference_steps == 10 and args.max_chunks is None
    assert args.log is True
    assert not any("workspace" in key for key in vars(args))


def run_loop(tmp_path, args, *, inference_elapsed=0.0, key_factory=None):
    clock = OfflineReplayClock()
    provider, policy = Provider(clock), Policy(clock, inference_elapsed)
    raw_sink = RecordingActionSink()
    keys = live.KeyboardControl()
    keys.feed(" ")
    if key_factory:
        keys = key_factory(clock, raw_sink)
    log = live.SessionLog(tmp_path / "run", args)
    sink = live.LoggedSink(raw_sink, provider, log, clock.monotonic)
    try:
        status = live.run_live_loop(
            policy,
            provider,
            sink,
            keys,
            limits(),
            log,
            args,
            monotonic=clock.monotonic,
            wall_time=clock.wall_time,
            sleep=clock.sleep,
        )
        log.summary["status"] = status
    finally:
        log.close()
    return policy, provider, raw_sink, log


@pytest.mark.parametrize("logging", [True, False])
def test_sync_chunks_do_not_hold_at_boundary_and_keep_cadence(tmp_path, logging):
    policy, provider, sink, log = run_loop(tmp_path, arguments(log=logging))
    assert len(policy.observations) == 2
    assert policy.observations[1] > policy.observations[0]
    assert len(sink.records) == 6
    assert provider.health_checks == 6
    # Only the bounded run ending holds, never an intermediate chunk.
    assert sink.hold_reasons == ["completed"]
    assert log.summary["max_action_interval_s"] == pytest.approx(1 / 30)
    if not logging:
        assert not list(log.directory.glob("*.jsonl"))
        assert log.summary["action_count"] == 6
        return
    import json

    reports = [
        json.loads(line)
        for line in (log.directory / "cycles.jsonl").read_text().splitlines()
    ]
    assert all(report["execution"]["full_point_count"] == 30 for report in reports)
    assert all(report["execution"]["start_offset_s"] == 0 for report in reports)
    assert all(not report["hold_attempted"] for report in reports)


def test_sync_inference_pause_is_logged_not_compensated(tmp_path):
    policy, _, sink, log = run_loop(tmp_path, arguments(), inference_elapsed=0.12)
    assert len(sink.records) == 6
    assert log.summary["max_action_interval_s"] == pytest.approx(0.12)
    assert policy.observations[1] == pytest.approx(0.12 + 2 / 30)


def test_pause_discards_chunk_and_resume_observes_again(tmp_path):
    def key_factory(clock, sink):
        class Keys:
            paused_once = False

            def poll(self):
                if len(sink.records) >= 2 and not self.paused_once:
                    if clock.elapsed < 0.4:
                        return "pause"
                    self.paused_once = True
                return "run"

        return Keys()

    policy, _, sink, log = run_loop(tmp_path, arguments(), key_factory=key_factory)
    assert len(policy.observations) == 2
    assert policy.observations[1] >= 0.4
    assert len(sink.records) == 5  # two before pause, three from a NEW prediction
    assert sink.hold_reasons == ["paused", "completed"]
    assert log.summary["prediction_count"] == 2


def test_pause_between_completed_chunks_still_holds_once(tmp_path):
    def key_factory(clock, sink):
        class Keys:
            def poll(self):
                return (
                    "pause" if len(sink.records) == 3 and clock.elapsed < 0.4 else "run"
                )

        return Keys()

    policy, _, sink, _ = run_loop(tmp_path, arguments(), key_factory=key_factory)
    assert len(policy.observations) == 2 and policy.observations[1] >= 0.4
    assert len(sink.records) == 6
    assert sink.hold_reasons == ["paused", "completed"]


@pytest.mark.parametrize("resume", [True, False])
def test_schedule_slip_holds_pauses_and_requires_keyboard_to_repredict(
    tmp_path, resume
):
    import json

    clock = OfflineReplayClock()
    provider = Provider(clock)

    class ChangingPolicy(Policy):
        def predict(self, frame):
            result = super().predict(frame)
            result["base_actions"][:, 0] = 0.01 * len(self.observations)
            return result

    class StallOnce(RecordingActionSink):
        def send_action(self, action, deadline_walltime):
            super().send_action(action, deadline_walltime)
            if len(self.records) == 2:
                clock.sleep(0.2)

    class Keys(live.KeyboardControl):
        def __init__(self):
            super().__init__()
            self.paused_at = None
            self.manual_input = False
            self.feed(" ")

        def pause(self):
            self.paused_at = clock.wall_time()
            super().pause()

        def poll(self):
            if (
                self.paused_at is not None
                and not self.manual_input
                and clock.wall_time() >= self.paused_at + 0.25
            ):
                assert len(policy.observations) == 1
                assert len(raw.records) == 2
                self.feed(" " if resume else "q")
                self.manual_input = True
            return super().poll()

    policy, raw, keys = ChangingPolicy(clock), StallOnce(), Keys()
    args = arguments()
    log = live.SessionLog(tmp_path / "run", args)
    sink = live.LoggedSink(raw, provider, log, clock.monotonic)
    try:
        status = live.run_live_loop(
            policy,
            provider,
            sink,
            keys,
            limits(),
            log,
            args,
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
    assert (
        cycles[0]["status"] == "paused" and cycles[0]["pause_reason"] == "schedule_slip"
    )
    assert cycles[0]["execution"]["sent_count"] == 2
    assert cycles[0]["execution"]["hold_succeeded"] is True
    assert log.summary["schedule_pause_count"] == 1
    assert keys.manual_input
    assert raw.hold_reasons == ["failed", "completed" if resume else "stopped"]
    assert status == ("completed" if resume else "stopped")
    if resume:
        assert len(policy.observations) == 2
        assert policy.observations[1] >= keys.paused_at + 0.25
        np.testing.assert_allclose(
            [r["action"][0] for r in raw.records], [0.01, 0.01, 0.02, 0.02, 0.02]
        )
        assert cycles[1]["execution"]["samples"][0]["nominal_offset_s"] == 0
        assert "measured_base_state" in cycles[1]
    else:
        assert len(policy.observations) == 1 and len(raw.records) == 2


@pytest.mark.parametrize(
    "kind,hold_ok,error_type",
    [
        ("schedule_slip", False, "ScheduleOverrun"),
        ("clock_fault", True, "ScheduleOverrun"),
        (None, True, "TimeoutError"),
        (None, True, "ValueError"),
    ],
)
def test_only_confirmed_held_schedule_slip_is_recoverable(
    tmp_path, monkeypatch, kind, hold_ok, error_type
):
    report = {
        "status": "failed",
        "hold_attempted": True,
        "error": {"type": error_type, "message": "fatal test failure"},
        "execution": {
            "hold_attempted": True,
            "hold_succeeded": hold_ok,
            "schedule_overrun": {"kind": kind},
        },
    }
    if not hold_ok:
        report["execution"]["hold_error"] = {"message": "hold failed"}
    monkeypatch.setattr(live, "run_sync_cycle", lambda *a, **k: report)
    with pytest.raises(RuntimeError, match="fatal test failure"):
        run_loop(tmp_path, arguments())
    import json

    recorded = json.loads((tmp_path / "run" / "cycles.jsonl").read_text())
    assert recorded["status"] == "failed" and "pause_reason" not in recorded


def test_hold_happens_even_when_logging_fails():
    class BrokenLog:
        def event(self, *args, **kwargs):
            raise OSError("disk full")

    raw = RecordingActionSink()
    sink = live.LoggedSink(raw, None, BrokenLog())
    with pytest.raises(OSError, match="disk full"):
        sink.hold("paused")
    assert raw.hold_reasons == ["paused"]


def test_gripper_bounding_preserves_raw_prediction_and_all_pose_values(tmp_path):
    log = live.SessionLog(tmp_path / "run", arguments())
    clock = OfflineReplayClock()
    frame = Provider(clock).latest()
    source = Policy(clock).predict(frame)
    source["base_actions"][:, 6] = -0.001
    source["base_actions"][:, 13] = 0.083
    policy = SimpleNamespace(timing=Policy.timing, predict=lambda frame: source)
    try:
        prediction = live.ActuatorPolicy(policy, 0.08, log).predict(frame)
        expected = source["base_actions"].copy()
        expected[:, 6], expected[:, 13] = 0.0, 0.08
        np.testing.assert_array_equal(prediction["base_actions"], expected)
        assert source["base_actions"][0, 6] == -0.001  # no mutation
        assert prediction["camera_actions"] is source["camera_actions"]
        source["base_actions"][0, 6] = np.nan
        with pytest.raises(ValueError, match="finite"):
            live.ActuatorPolicy(policy, 0.08, log).predict(frame)
    finally:
        log.close()
    assert "gripper_clamped" in (log.directory / "events.jsonl").read_text()


def test_health_failure_prevents_send(tmp_path):
    args = arguments()
    log = live.SessionLog(tmp_path / "run", args)
    raw = RecordingActionSink()

    class Broken:
        def check_health(self):
            raise RuntimeError("front camera stopped")

    sink = live.LoggedSink(raw, Broken(), log)
    try:
        with pytest.raises(RuntimeError, match="front camera stopped"):
            sink.send_action(np.zeros(14), 100.0)
        assert not raw.records and log.summary["action_count"] == 0
    finally:
        log.close()


def test_noninteractive_main_fails_before_model_or_devices(tmp_path, monkeypatch):
    monkeypatch.setattr(live.sys, "stdin", SimpleNamespace(isatty=lambda: False))

    def forbidden(*args):
        pytest.fail("No policy/device initialization without keyboard controls")

    monkeypatch.setattr(live, "load_policy_contract", forbidden)
    result = live.main(
        [
            "--checkpoint",
            "missing.ckpt",
            "--training-data",
            "missing",
            "--calibration",
            "missing.json",
            "--camera-config",
            "missing.json",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    assert result == 1
    import json

    assert (
        json.loads((tmp_path / "run" / "summary.json").read_text())["status"]
        == "failed"
    )


def test_existing_output_is_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        live.SessionLog(tmp_path, arguments())


def test_home_is_rejected_in_readonly_mode(tmp_path):
    args = arguments(home_on_start=True)
    with pytest.raises(ValueError, match="--execute"):
        live.load_policy_contract(args, None)


@pytest.mark.parametrize("execute,home", [(False, False), (True, False), (True, True)])
def test_session_wiring_home_and_no_pico(tmp_path, monkeypatch, execute, home):
    args, log, events, keys = mock_session(
        tmp_path, monkeypatch, execute=execute, home=home
    )
    try:
        assert live.run_session(args, log) == "stopped"
    finally:
        log.close()
    options = next(value for event, value in events if event == "env_options")
    assert options["enable_pico"] is False
    assert options["enable_motion"] is execute
    assert options["preserve_gripper_on_enable"] is True
    assert any(event == "home" for event, _ in events) is home
    assert events[-1][0] == "env_stop"
    assert ("hold", "session_exit") in events
    assert keys.state == "pause"


def mock_session(
    tmp_path,
    monkeypatch,
    *,
    execute=False,
    home=False,
    fail_prediction=False,
    fail_hold=False,
    cancel_home=False,
):
    import multiprocessing.managers

    from egomimic.robot.piper_pico import env as env_module
    from egomimic.robot.piper_pico import live_policy_devices as devices

    options = [
        "--checkpoint",
        "model.ckpt",
        "--calibration",
        "calibration.json",
        "--training-data",
        "data",
        "--camera-config",
        "rig.json",
        "--no-preview",
    ]
    if execute:
        options += ["--execute"]
    if home:
        options += ["--home-on-start", "true"]
    args = live.build_parser().parse_args(options)
    log = live.SessionLog(tmp_path / "run", args)
    events = []
    clock = OfflineReplayClock()
    policy = Policy(clock)
    policy.adapter = SimpleNamespace(gripper_open_m=0.08)
    if fail_prediction:

        def predict(frame):
            raise RuntimeError("prediction broke")

        policy.predict = predict
    contract = {"camera_specs": []}
    monkeypatch.setattr(
        live, "load_policy_contract", lambda *unused: (contract, policy, limits())
    )
    monkeypatch.setattr(live.sys, "stdin", SimpleNamespace(isatty=lambda: True))

    class Keys(live.KeyboardControl):
        def __enter__(self):
            return self

        def __exit__(self, *unused):
            pass

    keys = Keys()
    monkeypatch.setattr(live, "KeyboardControl", lambda: keys)

    class Shm:
        def __init__(self, **kwargs):
            assert kwargs["ctx"].get_start_method() == "spawn"

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            pass

    monkeypatch.setattr(multiprocessing.managers, "SharedMemoryManager", Shm)

    class Env:
        cameras = []

        def __init__(self, *unused, **kwargs):
            events.append(("env_options", kwargs))

        def start(self):
            events.append(("env_start", None))

        def stop(self):
            events.append(("env_stop", None))

    monkeypatch.setattr(env_module, "PiperPicoEnv", Env)
    monkeypatch.setattr(
        devices, "LiveObservationProvider", lambda *a, **kw: Provider(clock)
    )

    class Sink:
        def __init__(self, *unused, **kwargs):
            assert kwargs["execute"] is execute
            assert kwargs["max_source_age_s"] == args.max_observation_age

        def home(self, **kwargs):
            events.append(("home", kwargs))
            if cancel_home:
                keys.feed("q")
                assert kwargs["should_stop"]()
                raise InterruptedError("home cancelled")

        def hold(self, reason):
            events.append(("hold", reason))
            if fail_hold:
                raise RuntimeError("hold broke")

    monkeypatch.setattr(devices, "LiveActionSink", Sink)

    def loop(*values):
        assert values[3].state == "pause"
        events.append(("loop", None))
        return "stopped"

    monkeypatch.setattr(live, "run_live_loop", loop)
    return args, log, events, keys


def test_home_cancel_is_normal_stop_and_cleans_devices(tmp_path, monkeypatch):
    args, log, events, _ = mock_session(
        tmp_path, monkeypatch, execute=True, home=True, cancel_home=True
    )
    try:
        assert live.run_session(args, log) == "stopped"
    finally:
        log.close()
    assert ("hold", "session_exit") in events
    assert not any(event == "loop" for event, _ in events)
    assert events[-1][0] == "env_stop"


def test_cleanup_error_keeps_primary_prediction_failure(tmp_path, monkeypatch):
    args, log, events, _ = mock_session(
        tmp_path, monkeypatch, execute=True, fail_prediction=True, fail_hold=True
    )
    try:
        with pytest.raises(RuntimeError, match="prediction broke"):
            live.run_session(args, log)
        assert log.summary["cleanup_errors"][0]["message"] == "hold broke"
    finally:
        log.close()
    assert events[-1][0] == "env_stop"
