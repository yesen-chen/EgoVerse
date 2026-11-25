"""Recording CLI and lifecycle, without camera, CAN or model connections."""

import json
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.robot.piper_pico import live_policy as live
from tests.unit.test_live_policy import mock_session


def test_spawn_entry_does_not_import_model_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import egomimic.scripts.infer_piper_policy; assert 'torch' not in sys.modules; assert 'egomimic.robot.piper_pico.live_policy' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def parse(*flags):
    return live.build_parser().parse_args(
        [
            "--calibration",
            "cal.json",
            "--training-data",
            "data",
            "--camera-config",
            "rig.json",
            *flags,
        ]
    )


def test_recording_is_opt_in_and_all_modes_are_configurable():
    args = parse()
    assert args.record_images == "off" and not args.record_feedback
    assert not live.validate_recording_args(args)
    args = parse(
        "--record-images",
        "all",
        "--record-feedback",
        "--record-video-format",
        "lossless",
        "--record-queue-capacity",
        "3",
    )
    assert live.validate_recording_args(args)
    assert args.record_video_format == "lossless" and args.record_queue_capacity == 3


@pytest.mark.parametrize(
    "flags",
    [
        ("--no-log", "--record-images", "requests"),
        ("--no-log", "--record-feedback"),
        ("--offline-eval", "--record-images", "all"),
    ],
)
def test_incompatible_modes_fail_before_devices(flags):
    with pytest.raises(ValueError):
        live.validate_recording_args(parse(*flags))


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--record-queue-capacity", "0"),
        ("--record-cache-frames", "-1"),
        ("--record-segment-seconds", "nan"),
    ],
)
def test_invalid_buffer_settings_rejected(flag, value):
    with pytest.raises(SystemExit):
        parse(flag, value)


def test_contended_logging_never_waits_for_inference_thread(tmp_path):
    log = live.SessionLog(tmp_path / "run", SimpleNamespace(log=True, execute=False))
    log._request_publish_lock.acquire()
    try:
        start = time.monotonic()
        assert log.request({"request_id": 1}) is False
        log.command(np.zeros(14), 1, 0, 0.01)
        assert time.monotonic() - start < 0.2
    finally:
        log._request_publish_lock.release()
        log.close()
    assert log.summary["request_logging"]["dropped"]["requests"] == 1
    assert log.summary["logging"]["written"]["commands"] == 1
    assert log.summary["request_logging"]["incomplete"]
    assert not log.summary["logging"]["incomplete"]


def test_partial_submission_failure_retains_original_error_and_target():
    events = []

    class Sink:
        def send_action(self, *unused):
            raise TimeoutError("right SDK acknowledgement missing")

    log = SimpleNamespace(event=lambda name, **fields: events.append((name, fields)))
    sink = live.LoggedSink(Sink(), SimpleNamespace(check_health=lambda: None), log)
    with pytest.raises(TimeoutError, match="right SDK"):
        sink.send_action(np.arange(14), time.time() + 1)
    name, record = events[0]
    assert name == "command_failed"
    np.testing.assert_array_equal(record["action_base"], np.arange(14))
    assert "unconfirmed" in record["receipt_status"]


@pytest.mark.parametrize("startup_failure", [False, True])
def test_recording_lifecycle_and_warmup_request_are_wired(
    tmp_path, monkeypatch, startup_failure
):
    from egomimic.robot.piper_pico import env as env_module
    from egomimic.robot.piper_pico import policy_recording

    args, log, events, _ = mock_session(tmp_path, monkeypatch)
    args.record_images, args.record_feedback = "all", True
    env_class = env_module.PiperPicoEnv
    env_class.arms = []

    class Recorder:
        def __init__(self, directory, cameras, arms, **options):
            assert options["image_mode"] == "all" and options["record_feedback"]
            assert cameras == arms == {}
            self.stats = {"incomplete": False, "drained": False}

        def start(self):
            events.append(("recorder_start", None))
            if startup_failure:
                raise OSError("recorder startup failed")

        def request(self, request_id, refs):
            events.append(("request", request_id))
            assert set(refs) == {"front_1", "left_wrist", "right_wrist"}
            return {}

        def poll(self):
            pass

        def close(self):
            events.append(("recorder_close", None))
            self.stats["drained"] = True

    monkeypatch.setattr(policy_recording, "PolicyRecorder", Recorder)
    try:
        if startup_failure:
            with pytest.raises(OSError, match="startup failed"):
                live.run_session(args, log)
        else:
            assert live.run_session(args, log) == "stopped"
    finally:
        log.close()
    names = [name for name, _ in events]
    assert names.index("recorder_close") < names.index("env_stop")
    assert log.summary["recording"]["drained"]
    if not startup_failure:
        records = [
            json.loads(row)
            for row in (log.directory / "requests.jsonl").read_text().splitlines()
        ]
        assert [row["phase"] for row in records] == ["started", "completed"]
        assert all(row["policy_phase"] == "warmup" for row in records)
        assert records[0]["request_id"] == records[1]["request_id"]
        assert names.index("hold") < names.index("recorder_close")


def test_request_snapshot_is_owned_and_output_background_encoded(tmp_path):
    log = live.SessionLog(tmp_path / "run", SimpleNamespace(log=True, execute=False))
    state = np.arange(14, dtype=float)
    log.request({"request_id": 7, "base_state": state})
    state[:] = -1
    log.close()
    record = json.loads((log.directory / "requests.jsonl").read_text())
    assert record["base_state"] == list(range(14))
    assert log.summary["request_logging"]["written"]["requests"] == 1


def test_rtc_cycle_links_to_audited_request_including_warmup():
    from egomimic.robot.piper_pico.policy_audit import RecordedPolicy
    from egomimic.robot.piper_pico.policy_rollout import OfflineReplayClock
    from egomimic.robot.piper_pico.policy_rtc import run_rtc_loop
    from tests.unit.test_live_policy import Provider, limits
    from tests.unit.test_policy_rtc import (
        DELAY,
        AsyncExecutor,
        Keys,
        Log,
        Policy,
        Sink,
        args,
    )

    clock = OfflineReplayClock(epoch=1000)
    provider, log = Provider(clock), Log()
    requests = []
    log.request = lambda row: requests.append(row)
    policy = RecordedPolicy(Policy(clock, (0.08,)), log)
    policy.phase = "warmup"
    policy.predict(provider.latest())
    policy.phase = "policy"
    run_rtc_loop(
        policy,
        provider,
        Sink(clock),
        Keys(clock, stop_time=1.5),
        limits(),
        log,
        args(),
        observation_to_send_s=DELAY,
        executor=AsyncExecutor(clock),
        monotonic=clock.monotonic,
        wall_time=clock.wall_time,
        sleep=clock.sleep,
    )
    completed = {
        row["request_id"]: row for row in requests if row["phase"] == "completed"
    }
    assert completed[1]["policy_phase"] == "warmup"
    cycles = [row for row in log.cycles if row.get("status") == "completed"]
    assert len(cycles) >= 2
    for cycle in cycles:
        request = completed[cycle["audit_request_id"]]
        assert request["policy_phase"] == "policy"
        assert request["observation_timestamp"] == cycle["observation_timestamp"]
        np.testing.assert_array_equal(
            request["prediction"]["base_actions"], cycle["base_actions"]
        )
    assert cycles[0]["audit_request_id"] != cycles[0]["request_id"]


def test_synchronous_cycle_links_to_audited_request():
    from egomimic.robot.piper_pico.policy_audit import RecordedPolicy
    from egomimic.robot.piper_pico.policy_execution import RecordingActionSink
    from egomimic.robot.piper_pico.policy_rollout import (
        OfflineReplayClock,
        run_sync_cycle,
    )
    from tests.unit.test_live_policy import Policy, Provider, limits

    clock = OfflineReplayClock()
    records = []
    log = SimpleNamespace(
        request=lambda row: records.append(row), event=lambda *a, **k: None
    )
    policy = RecordedPolicy(Policy(clock), log)
    report = run_sync_cycle(
        policy,
        Provider(clock),
        RecordingActionSink(),
        limits(),
        execute_steps=3,
        monotonic=clock.monotonic,
        wall_time=clock.wall_time,
        sleep=clock.sleep,
    )
    assert report["status"] == "completed"
    assert report["audit_request_id"] == records[-1]["request_id"]
    assert records[-1]["observation_timestamp"] == report["observation_timestamp"]
