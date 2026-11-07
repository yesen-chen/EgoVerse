"""Real spawned logging workers, fake actions, no cameras/CAN/GPU."""

import json
import os
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
    assert (log.directory / "arguments.json").exists()
    assert (log.directory / "manifest.json").exists()


def test_worker_encodes_and_drains_all_kinds(tmp_path, monkeypatch):
    log = live.SessionLog(tmp_path / "on", SimpleNamespace(execute=True, log=True))
    pid = log.summary["logging"]["writer_pid"]
    assert pid != os.getpid()
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


def test_queued_snapshots_do_not_change_with_callers(tmp_path, monkeypatch):
    class Capture:
        def __init__(self, directory):
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
    row[:] = 8
    receipt["receipts"][0]["command_id"] = 99
    report["execution"]["samples"].append(2)
    saved = dict(log._writer.rows)
    assert saved["commands"]["action_base"].tolist() == [0] * 14
    assert saved["commands"]["device_receipt"]["receipts"][0]["command_id"] == 1
    assert saved["cycles"]["execution"]["samples"] == [1]
    assert saved["events"]["widths"].tolist() == [0] * 14


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
        lambda directory: AsyncLogWriter(
            directory, capacity=1, close_timeout=0.1, _worker_target=_never_consume
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
    assert summary["logging"]["dropped"] == {"commands": 1, "cycles": 0, "events": 2}
