"""Real spawned logging workers, fake actions, no cameras/CAN/GPU."""

import json
import os
import signal
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.rldb.action_timing import ActionTiming
from egomimic.robot.piper_pico import live_policy as live
from egomimic.robot.piper_pico.policy_execution import (
    RecordingActionSink,
    SafetyLimits,
    execute_action_block,
)
from egomimic.robot.piper_pico.policy_logging import AsyncLogWriter


@pytest.mark.parametrize("kinds", [("commands", "cycles", "events"), ("requests",)])
def test_background_log_ignores_terminal_interrupt_until_parent_drains(tmp_path, kinds):
    writer = AsyncLogWriter(tmp_path, kinds=kinds)
    try:
        os.kill(writer._process.pid, signal.SIGINT)
        assert writer.submit(kinds[0], {"sample": 7})
    finally:
        writer.close()
    assert writer.stats["drained"] and not writer.stats["incomplete"]
    assert writer.stats["written"][kinds[0]] == 1


def _never_consume(directory, records, status):
    status.send(("ready", {"pid": os.getpid()}))
    time.sleep(30)


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("flag,enabled", [("--log", True), ("--no-log", False)])
def test_cli_switch(flag, enabled):
    args = live.build_parser().parse_args(
        [
            "--checkpoint",
            "model.ckpt",
            "--calibration",
            "cal.json",
            "--training-data",
            "data",
            "--camera-config",
            "rig.json",
            flag,
        ]
    )
    assert args.log is enabled


def test_disabled_has_no_worker_no_stream_serialization_and_keeps_summary(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("Disabled logging must not start a worker or encode records")

    monkeypatch.setattr(live, "AsyncLogWriter", forbidden)
    log = live.SessionLog(tmp_path / "off", SimpleNamespace(execute=False, log=False))
    log.write_manifest({"checkpoint": "test.ckpt"})
    with monkeypatch.context() as patch:
        patch.setattr(live, "_json", forbidden)
        log.event("not_written", unsupported=object())
        log.command(np.zeros(14), 1.0, 0.0, 0.01)
        log.cycle({"observation_timestamp": 1.0, "measured_inference_elapsed_s": 0.03})
        log.check()
    log.close()
    log.close()
    assert not list(log.directory.glob("*.jsonl"))
    summary = json.loads((log.directory / "summary.json").read_text())
    assert summary["action_count"] == summary["prediction_count"] == 1
    assert summary["logging"]["enabled"] is False
    assert summary["request_logging"]["enabled"] is False
    assert (log.directory / "arguments.json").exists()
    assert (log.directory / "manifest.json").exists()


def test_worker_encodes_and_drains_all_kinds(tmp_path, monkeypatch):
    log = live.SessionLog(tmp_path / "on", SimpleNamespace(execute=True, log=True))
    pid = log.summary["logging"]["writer_pid"]
    assert pid != os.getpid()
    assert log.summary["request_logging"]["writer_pid"] not in (pid, os.getpid())
    try:
        with monkeypatch.context() as patch:

            def forbidden(*args, **kwargs):
                pytest.fail("Producer must not encode JSONL")

            patch.setattr(live, "_json", forbidden)
            for i in range(100):
                log.command(
                    np.full(14, i, dtype=float),
                    2.0,
                    i / 30,
                    i / 30 + 0.01,
                    {"receipts": [{"command_id": np.int64(i)}]},
                )
            log.event("ready", value=np.array([1, 2]))
            log.cycle(
                {"observation_timestamp": 1.0, "measured_inference_elapsed_s": 0.03}
            )
    finally:
        log.close()
    rows = _records(log.directory / "commands.jsonl")
    assert [row["index"] for row in rows] == list(range(100))
    assert rows[-1]["action_base"] == [99.0] * 14
    assert rows[-1]["device_receipt"]["receipts"][0]["command_id"] == 99
    assert _records(log.directory / "events.jsonl")[0]["value"] == [1, 2]
    stats = log.summary["logging"]
    assert (
        stats["written"]
        == stats["submitted"]
        == {"commands": 100, "cycles": 1, "events": 1}
    )
    assert stats["drained"] and not stats["incomplete"]
    assert stats["writer_exitcode"] == 0
    assert log.summary["request_logging"]["written"] == {"requests": 0}
    assert log.summary["request_logging"]["drained"]


def test_queued_snapshots_do_not_change_with_callers(tmp_path, monkeypatch):
    class Capture:
        def __init__(self, directory, **kwargs):
            self.stats = {}
            self.rows = []

        def submit(self, kind, value):
            self.rows.append((kind, value))

        def poll(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(live, "AsyncLogWriter", Capture)
    log = live.SessionLog(tmp_path / "snapshots", SimpleNamespace(execute=False))
    row, receipt = np.zeros(14), {"receipts": [{"command_id": 1}]}
    report = {"observation_timestamp": 1, "execution": {"samples": [1]}}
    log.command(row, 2, 1, 1.01, receipt)
    log.cycle(report)
    log.event("test", widths=row)
    request = {
        "base_state": row,
        "base_state_history": np.zeros((2, 14)),
        "rtc": {"prefix": np.ones((100, 14)), "weights": np.ones(100)},
        "prediction": {"actions": np.full((100, 14), 2.0)},
    }
    log.request(request)
    row[:] = 8
    receipt["receipts"][0]["command_id"] = 99
    report["execution"]["samples"].append(2)
    request["base_state_history"][:] = 9
    request["rtc"]["prefix"][:] = 9
    request["rtc"]["weights"][:] = 9
    request["prediction"]["actions"][:] = 9
    saved = dict(log._writer.rows)
    assert saved["commands"]["action_base"].tolist() == [0] * 14
    assert saved["commands"]["device_receipt"]["receipts"][0]["command_id"] == 1
    assert saved["cycles"]["execution"]["samples"] == [1]
    assert saved["events"]["widths"].tolist() == [0] * 14
    assert set(saved) == {"commands", "cycles", "events"}
    [(kind, saved_request)] = log._request_writer.rows
    assert kind == "requests"
    np.testing.assert_array_equal(saved_request["base_state"], np.zeros(14))
    np.testing.assert_array_equal(
        saved_request["base_state_history"], np.zeros((2, 14))
    )
    np.testing.assert_array_equal(saved_request["rtc"]["prefix"], np.ones((100, 14)))
    np.testing.assert_array_equal(saved_request["rtc"]["weights"], np.ones(100))
    np.testing.assert_array_equal(
        saved_request["prediction"]["actions"], np.full((100, 14), 2.0)
    )


def test_full_queue_drops_new_records_without_waiting_and_close_is_bounded(tmp_path):
    writer = AsyncLogWriter(
        tmp_path, capacity=2, close_timeout=0.1, _worker_target=_never_consume
    )
    started = time.monotonic()
    try:
        accepted = [writer.submit("commands", {"index": i}) for i in range(100)]
        # The worker sleeps 30s; no producer call may wait for it/queue space.
        assert time.monotonic() - started < 1.0
        assert sum(accepted) == 2
        assert writer.stats["dropped"]["commands"] == 98
    finally:
        before_close = time.monotonic()
        writer.close()
    assert time.monotonic() - before_close < 3.0
    assert writer.stats["incomplete"] and not writer.stats["drained"]
    assert writer.stats["unwritten"]["commands"] == 2
    writer.close()


def test_encoding_failure_is_reported_and_future_records_are_dropped(tmp_path):
    writer = AsyncLogWriter(tmp_path)
    try:
        assert writer.submit("events", {"value": float("nan")})
        writer._process.join(timeout=5)
        writer.poll()
        assert writer.stats["incomplete"]
        assert any("JSON compliant" in error for error in writer.stats["errors"])
        assert not writer.submit("commands", {"index": 0})
    finally:
        writer.close()
    assert writer.stats["dropped"]["commands"] == 1
    assert writer.stats["unwritten"]["events"] == 1


def test_dead_worker_does_not_block_producer_or_shutdown(tmp_path):
    writer = AsyncLogWriter(tmp_path)
    try:
        writer._process.terminate()
        writer._process.join(timeout=2)
        writer.poll()
        assert not writer.submit("commands", {"index": 0})
    finally:
        writer.close()
    assert writer.stats["incomplete"]
    assert writer.stats["writer_exitcode"] != 0


def test_worker_startup_failure_preserves_existing_file(tmp_path):
    (tmp_path / "events.jsonl").write_text("original")
    with pytest.raises(RuntimeError, match="FileExistsError"):
        AsyncLogWriter(tmp_path)
    assert (tmp_path / "events.jsonl").read_text() == "original"


def test_stalled_writer_does_not_stall_real_clock_action_block(tmp_path, monkeypatch):
    monkeypatch.setattr(
        live,
        "AsyncLogWriter",
        lambda directory, **kwargs: AsyncLogWriter(
            directory,
            capacity=1,
            close_timeout=0.1,
            _worker_target=_never_consume,
            **kwargs,
        ),
    )
    log = live.SessionLog(tmp_path / "stalled", SimpleNamespace(execute=False))
    raw = RecordingActionSink()
    health_calls = []
    provider = SimpleNamespace(check_health=lambda: health_calls.append(True))
    sink = live.LoggedSink(raw, provider, log)
    actions = np.zeros((100, 14))
    try:
        report = execute_action_block(
            actions,
            ActionTiming(30, 30, 1, 100),
            sink,
            control_frequency=30,
            execute_steps=4,
            initial_action=actions[0],
            observation_timestamp=time.time(),
            limits=SafetyLimits(
                None, None, max_translation_step_m=None, max_rotation_step_rad=None
            ),
        )
    finally:
        log.close()
    assert report["status"] == "completed" and report["sent_count"] == 4
    assert len(raw.records) == len(health_calls) == 4
    assert raw.hold_reasons == ["completed"]
    assert log.summary["logging"]["dropped"]["commands"] == 3


def test_worker_failure_warns_once_and_preserves_actions_hold_and_summary(
    tmp_path, capsys
):
    log = live.SessionLog(tmp_path / "failed", SimpleNamespace(execute=False))
    raw = RecordingActionSink()
    health_calls = []
    sink = live.LoggedSink(
        raw, SimpleNamespace(check_health=lambda: health_calls.append(True)), log
    )
    try:
        log._writer._process.terminate()
        log._writer._process.join(timeout=2)
        log.check()
        log.check()
        sink.send_action(np.zeros(14), time.time() + 0.1)
        sink.hold("paused")
    finally:
        log.close()
    assert len(raw.records) == len(health_calls) == 1
    assert raw.hold_reasons == ["paused"]
    assert capsys.readouterr().err.count("日志不完整") == 1
    summary = json.loads((log.directory / "summary.json").read_text())
    assert summary["action_count"] == 1
    assert summary["logging"]["incomplete"]
    assert summary["logging"]["dropped"] == {
        "commands": 1,
        "cycles": 0,
        "events": 2,
    }


def test_writer_only_creates_and_accepts_its_owned_streams(tmp_path):
    writer = AsyncLogWriter(tmp_path, kinds=("requests",))
    try:
        with pytest.raises(ValueError, match="not owned"):
            writer.submit("commands", {"index": 0})
        assert writer.submit("requests", {"request_id": 3, "state": np.arange(14)})
    finally:
        writer.close()
    assert {path.name for path in tmp_path.glob("*.jsonl")} == {"requests.jsonl"}
    assert writer.stats["written"] == {"requests": 1}
    assert writer.stats["unwritten"] == {"requests": 0}
    assert _records(tmp_path / "requests.jsonl")[0]["state"] == list(range(14))


def test_request_snapshot_copy_and_lock_do_not_drop_control_records(
    tmp_path, monkeypatch
):
    log = live.SessionLog(tmp_path / "isolation", SimpleNamespace(execute=False))
    entered, release = threading.Event(), threading.Event()
    payload = {"request_id": 1, "prediction": {"actions": np.ones((100, 14))}}
    deepcopy = live.copy.deepcopy

    def delayed_copy(value, *args, **kwargs):
        if value is payload:
            entered.set()
            assert release.wait(3)
        return deepcopy(value, *args, **kwargs)

    monkeypatch.setattr(live.copy, "deepcopy", delayed_copy)
    worker = threading.Thread(target=log.request, args=(payload,))
    try:
        worker.start()
        assert entered.wait(2)
        assert log._request_publish_lock.locked()
        before = time.monotonic()
        log.command(np.zeros(14), 1, 0, 0.001)
        log.event("during_request_copy")
        log.cycle({"observation_timestamp": 1})
        assert time.monotonic() - before < 0.5
        assert log._writer.stats["submitted"] == {
            "commands": 1,
            "events": 1,
            "cycles": 1,
        }
        # A second request may drop, but cannot occupy/control the other lock.
        assert log.request({"request_id": 2}) is False
        assert log._request_writer.stats["dropped"] == {"requests": 1}
        assert not log._writer.stats["incomplete"]
    finally:
        release.set()
        worker.join(3)
        log.close()
    assert not worker.is_alive()
    assert log.summary["logging"]["drained"]
    assert not log.summary["logging"]["incomplete"]
    assert log.summary["request_logging"]["written"] == {"requests": 1}
    assert (
        _records(log.directory / "requests.jsonl")[0]["prediction"]["actions"]
        == [[1.0] * 14] * 100
    )


def test_stalled_request_queue_cannot_consume_control_queue_slots(
    tmp_path, monkeypatch
):
    def factory(directory, **kwargs):
        if kwargs.get("kinds") == ("requests",):
            return AsyncLogWriter(
                directory,
                capacity=2,
                close_timeout=0.1,
                _worker_target=_never_consume,
                **kwargs,
            )
        return AsyncLogWriter(directory, **kwargs)

    monkeypatch.setattr(live, "AsyncLogWriter", factory)
    log = live.SessionLog(tmp_path / "request_stall", SimpleNamespace(execute=False))
    try:
        before = time.monotonic()
        accepted = [log.request({"request_id": i}) for i in range(100)]
        for i in range(100):
            log.command(np.full(14, i), 2, i / 30, i / 30 + 0.001)
        assert time.monotonic() - before < 1
        assert sum(accepted) == 2
    finally:
        log.close()
    assert log.summary["logging"]["written"]["commands"] == 100
    assert not log.summary["logging"]["incomplete"]
    assert log.summary["request_logging"]["dropped"]["requests"] == 98
    assert log.summary["request_logging"]["unwritten"]["requests"] == 2
    assert log.summary["request_logging"]["incomplete"]


def test_request_startup_failure_closes_already_started_control_writer(
    tmp_path, monkeypatch
):
    created = []

    def factory(directory, **kwargs):
        if kwargs.get("kinds") == ("requests",):
            raise RuntimeError("request worker failed to start")
        writer = AsyncLogWriter(directory, **kwargs)
        created.append(writer)
        return writer

    monkeypatch.setattr(live, "AsyncLogWriter", factory)
    with pytest.raises(RuntimeError, match="failed to start"):
        live.SessionLog(tmp_path / "partial", SimpleNamespace(execute=False))
    assert len(created) == 1
    assert created[0]._closed and created[0].stats["drained"]
    assert created[0].stats["writer_exitcode"] == 0
