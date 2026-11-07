"""Bounded background CSV writer checks; no hardware or timing thresholds."""

from __future__ import annotations

import csv
import io
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.robot.piper_pico.loop_profile import LoopProfiler
from egomimic.robot.piper_pico.teleop_log import TeleopRecorder


class RecordingFile:
    """CSV sink with event-controlled I/O and observable thread ownership."""

    def __init__(self, *, block_data=False, failure=None):
        self.buffer = io.StringIO()
        self.calls = []
        self.block_data = block_data
        self.failure = failure
        self.data_started = threading.Event()
        self.release_data = threading.Event()
        self.closed_event = threading.Event()
        self.closed = False
        self.saw_data = False

    def write(self, value):
        self.calls.append(("write", threading.get_ident()))
        if not value.startswith("t_wall,"):
            self.saw_data = True
            self.data_started.set()
            if self.block_data:
                assert self.release_data.wait(5), "test did not release CSV write"
            if self.failure == "writerow":
                raise OSError("injected writerow failure")
        return self.buffer.write(value)

    def flush(self):
        self.calls.append(("flush", threading.get_ident()))
        if self.saw_data and self.failure == "flush":
            raise OSError("injected flush failure")

    def close(self):
        self.calls.append(("close", threading.get_ident()))
        self.closed = True
        self.closed_event.set()

    def rows(self):
        return list(csv.DictReader(io.StringIO(self.buffer.getvalue())))


def install_sink(monkeypatch, path, **kwargs):
    sink = RecordingFile(**kwargs)
    original_open = Path.open

    def open_file(self, *args, **kw):
        if self != path:
            return original_open(self, *args, **kw)
        sink.calls.append(("open", threading.get_ident()))
        return sink

    monkeypatch.setattr(Path, "open", open_file)
    return sink


def env_and_sample(index=0):
    env = SimpleNamespace(
        left_clutch=SimpleNamespace(last_event="accepted", enabled=True),
        right_clutch=SimpleNamespace(last_event="accepted", enabled=True),
        last_action=np.array([0.2, 0.0, 0.3, 0, 0, 0, 0.5] * 2),
        measured_pose=lambda index: (np.array([0.2, 0.0, 0.3]), None),
    )
    sample = {
        "receive_timestamp": 100.0 + index / 30,
        "source_timestamp_ns": np.array(1_760_000_000_123_456_789 + index, dtype=np.int64),
        "source_age_s": np.array(0.01),
        "source_stall_count": np.array(0),
        "source_max_gap_s": np.array(0.04),
        "head_pos": np.array([0.1, 1.5, -0.1]),
        "head_quat_xyzw": np.array([0, 0, 0, 1]),
    }
    for side, sign in (("left", 1), ("right", -1)):
        sample.update({
            f"{side}_pos": np.array([sign * 0.2, 0.3, -0.4]),
            f"{side}_quat_xyzw": np.array([0, 0, 0, 1]),
            f"{side}_grip": 0.8,
            f"{side}_trigger": 0.2,
        })
    return env, sample


def test_constructor_does_no_io_and_writer_owns_file_lifetime(monkeypatch, tmp_path):
    path = tmp_path / "async.csv"
    sink = install_sink(monkeypatch, path)
    recorder = TeleopRecorder(path)
    assert sink.calls == []
    with pytest.raises(RuntimeError):
        recorder.write(*env_and_sample())
    recorder.start()
    recorder.write(*env_and_sample())
    recorder.close()
    recorder.close()  # Normal close is idempotent.
    with pytest.raises(RuntimeError):
        recorder.write(*env_and_sample())
    owners = {ident for _, ident in sink.calls}
    assert len(owners) == 1
    assert threading.get_ident() not in owners
    assert sink.calls[0][0] == "open"
    assert [operation for operation, _ in sink.calls].count("close") == 1
    assert len(sink.rows()) == recorder.rows == recorder.written_rows == 1
    report = recorder.timing_report()
    assert report["queued_rows"] == report["written_rows"] == 1
    assert report["rejected_rows"] == 0
    assert report["writer_alive"] is False
    assert report["writer_error"] is None


def test_blocked_file_write_does_not_block_enqueue_and_close_drains_in_order(monkeypatch, tmp_path):
    path = tmp_path / "blocked.csv"
    sink = install_sink(monkeypatch, path, block_data=True)
    recorder = TeleopRecorder(path, queue_capacity=4)
    recorder.start()
    try:
        recorder.write(*env_and_sample(0))
        assert sink.data_started.wait(5)
        recorder.write(*env_and_sample(1))
        recorder.write(*env_and_sample(2))
        # File I/O is still gated, while all three calls have returned.
        assert not sink.release_data.is_set()
        assert recorder.rows == 3
        assert recorder.written_rows == 0
    finally:
        sink.release_data.set()
        recorder.close()
    rows = sink.rows()
    assert [row["t_wall"] for row in rows] == [f"{100 + i / 30:.6f}" for i in range(3)]
    assert recorder.written_rows == 3
    report = recorder.timing_report()
    assert report["queue_capacity"] == 4
    assert 2 <= report["max_queue_depth"] <= 4
    assert report["queue_wait_total_s"] >= report["queue_wait_max_s"] >= 0


def test_queued_rows_snapshot_arrays_and_diagnostics_before_caller_mutates(monkeypatch, tmp_path):
    path = tmp_path / "snapshot.csv"
    sink = install_sink(monkeypatch, path, block_data=True)
    recorder = TeleopRecorder(path)
    recorder.start()
    try:
        recorder.write(*env_and_sample(0))
        assert sink.data_started.wait(5)
        env, sample = env_and_sample(1)
        camera = {"camera_mode": "left", "camera_left_count": np.array(42),
                  "camera_left_receive_timestamp": np.array(100.0)}
        recorder.write(env, sample, diagnostics=camera)
        sample["source_timestamp_ns"][...] = 0
        sample["source_age_s"][...] = 99
        sample["head_pos"][...] = 99
        sample["left_pos"][...] = 99
        env.last_action[...] = 99
        env.left_clutch.last_event = "held"
        camera["camera_left_count"][...] = 99
        camera["camera_left_receive_timestamp"][...] = 99
    finally:
        sink.release_data.set()
        recorder.close()
    row = sink.rows()[1]
    assert row["source_timestamp_ns"] == "1760000000123456790"
    assert float(row["source_age_s"]) == pytest.approx(0.01)
    assert float(row["head_raw_y"]) == pytest.approx(1.5)
    assert float(row["left_raw_x"]) == pytest.approx(0.2)
    assert float(row["left_target_x"]) == pytest.approx(0.2)
    assert row["left_event"] == "accepted"
    assert row["camera_left_count"] == "42"
    assert float(row["camera_left_receive_timestamp"]) == 100.0


def test_full_queue_fails_explicitly_without_discarding_accepted_rows(monkeypatch, tmp_path):
    path = tmp_path / "full.csv"
    sink = install_sink(monkeypatch, path, block_data=True)
    recorder = TeleopRecorder(path, queue_capacity=1)
    recorder.start()
    try:
        recorder.write(*env_and_sample(0))
        assert sink.data_started.wait(5)
        recorder.write(*env_and_sample(1))
        with pytest.raises(RuntimeError, match="[Qq]ueue|[Ff]ull|[Oo]verflow|队列已满"):
            recorder.write(*env_and_sample(2))
        assert not sink.release_data.is_set()
        assert recorder.rows == 2
    finally:
        sink.release_data.set()
        recorder.close()
    assert len(sink.rows()) == recorder.written_rows == 2
    report = recorder.timing_report()
    assert report["queued_rows"] == 2
    assert report["rejected_rows"] == 1
    assert report["max_queue_depth"] == 1


@pytest.mark.parametrize("stage", ["writerow", "flush"])
def test_background_io_failure_surfaces_on_write_and_close(monkeypatch, tmp_path, stage):
    path = tmp_path / "error.csv"
    sink = install_sink(monkeypatch, path, block_data=True, failure=stage)
    recorder = TeleopRecorder(path)
    recorder.start()
    recorder.write(*env_and_sample(0))
    assert sink.data_started.wait(5)
    sink.release_data.set()
    assert sink.closed_event.wait(5)
    with pytest.raises(RuntimeError):
        recorder.write(*env_and_sample(1))
    with pytest.raises(RuntimeError):
        recorder.close()
    assert recorder.rows == 1
    assert recorder.written_rows == 0
    report = recorder.timing_report()
    assert report["writer_alive"] is False
    assert stage in report["writer_error"]


def test_close_timeout_is_explicit_and_retry_drains_after_io_recovers(monkeypatch, tmp_path):
    path = tmp_path / "timeout.csv"
    sink = install_sink(monkeypatch, path, block_data=True)
    recorder = TeleopRecorder(path, close_timeout=0.01)
    recorder.start()
    try:
        recorder.write(*env_and_sample())
        assert sink.data_started.wait(5)
        with pytest.raises(TimeoutError):
            recorder.close()
        with pytest.raises(RuntimeError):
            recorder.write(*env_and_sample(1))
    finally:
        sink.release_data.set()
        assert sink.closed_event.wait(5)
        recorder.close()
    assert recorder.written_rows == 1
    assert recorder.timing_report()["writer_alive"] is False


def test_io_profile_is_separate_from_main_thread_profile(monkeypatch, tmp_path):
    path = tmp_path / "profile.csv"
    sink = install_sink(monkeypatch, path)
    profile = LoopProfiler(slow_s=0.0)
    recorder = TeleopRecorder(path, profiler=profile)
    recorder.start()
    with profile.tick(0, 1e20):
        recorder.write(*env_and_sample())
    recorder.close()
    assert profile.stats["sim.recorder_prepare"]["count"] == 1
    assert profile.stats["sim.recorder_enqueue"]["count"] == 1
    assert not any(name.startswith("csv.") for name in profile.stats)
    io_profile = recorder.timing_report()["io_profile"]
    assert {"csv.open", "csv.header", "csv.writerow", "csv.flush", "csv.close"} <= io_profile["stats"].keys()
    assert "sim.recorder_prepare" not in io_profile["stats"]
    assert io_profile["stats"]["csv.writerow"]["count"] == 1
    row_event = next(event for event in io_profile["events"] if event.get("stage") == "csv.writerow")
    assert "row_index" in row_event
    assert row_event["sample_timestamp_s"] == 100.0
    assert len(sink.rows()) == 1
