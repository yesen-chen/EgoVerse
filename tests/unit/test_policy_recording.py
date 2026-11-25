"""Recording exercises with fake sources and real spawned workers; no hardware."""

import hashlib
import json
import os
import threading
import time
from multiprocessing.managers import SharedMemoryManager

import cv2
import numpy as np
import pytest

from egomimic.robot.piper_pico.policy_recording import (
    PolicyRecorder,
    _CameraSink,
    _RingReader,
    _stats,
    _WritePump,
)
from egomimic.robot.piper_pico.shm import SharedMemoryRingBuffer


@pytest.fixture
def manager():
    with SharedMemoryManager() as manager:
        yield manager


def _camera_ring(manager, k=32):
    return SharedMemoryRingBuffer.create_from_examples(
        manager,
        dict(
            frame_id=np.int64(0),
            color=np.zeros((12, 16, 3), np.uint8),
            timestamp=np.float64(0),
            camera_receive_timestamp=np.float64(0),
        ),
        get_max_k=k,
        get_time_budget=0.2,
        put_desired_frequency=30,
    )


def _put(ring, value=None):
    frame_id = ring.count
    row = dict(
        frame_id=np.int64(frame_id),
        color=np.full((12, 16, 3), frame_id if value is None else value, np.uint8),
        timestamp=np.float64(100 + frame_id / 30),
        camera_receive_timestamp=np.float64(100.08 + frame_id / 30),
    )
    ring.put(row)
    return row


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for recording")
        time.sleep(0.01)


def _sleeping_camera(directory, ring, commands, status, config):
    status.send(("ready", {"pid": os.getpid()}))
    time.sleep(30)


def _crashing_camera(directory, ring, commands, status, config):
    status.send(("ready", {"pid": os.getpid()}))
    os._exit(4)


def _false_success_camera(directory, ring, commands, status, config):
    status.send(("ready", {"pid": os.getpid()}))
    while commands.get() is not None:
        pass
    stats = _stats()
    stats["drained"] = True
    status.send(("closed", stats))
    status.close()
    os._exit(1)


def test_disabled_has_no_worker_and_no_files(tmp_path):
    recorder = PolicyRecorder(tmp_path / "off", {}, {})
    recorder.start()
    assert recorder.request("x", {"front": {"frame_id": 0, "timestamp": 1}}) == {}
    recorder.close()
    recorder.close()
    assert not recorder._workers and not (tmp_path / "off").exists()
    assert recorder.stats["drained"] and not recorder.stats["incomplete"]


def test_precise_request_png_not_latest_and_repeated_request_reuses_frame(
    tmp_path, manager
):
    ring = _camera_ring(manager)
    rows = [_put(ring) for _ in range(5)]
    recorder = PolicyRecorder(tmp_path, {"front": ring}, {}, image_mode="requests")
    recorder.start()
    assert recorder.stats["workers"]["camera:front"]["pid"] != os.getpid()
    try:
        for request in ("request/1", "request/2"):
            assert recorder.request(
                request, {"front": {"frame_id": 1, "timestamp": rows[1]["timestamp"]}}
            )["front"]
    finally:
        recorder.close()
    path = tmp_path / "cameras/front"
    saved = _jsonl(path / "requests.jsonl")
    assert [row["request_id"] for row in saved] == ["request/1", "request/2"]
    assert all(row["frame_id"] == 1 and row["status"] == "saved" for row in saved)
    assert len(list((path / "requests").glob("*.png"))) == 1
    png = cv2.cvtColor(cv2.imread(str(path / saved[0]["file"])), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(png, rows[1]["color"])
    assert saved[0]["sha256_rgb"] == hashlib.sha256(png.tobytes()).hexdigest()
    assert not _jsonl(path / "frames.jsonl")
    assert recorder.stats["drained"] and not recorder.stats["incomplete"]


def test_pending_request_waits_for_exact_future_frame(tmp_path, manager):
    ring = _camera_ring(manager)
    recorder = PolicyRecorder(tmp_path, {"front": ring}, {}, image_mode="requests")
    recorder.start()
    try:
        recorder.request(0, {"front": {"frame_id": 2, "timestamp": 100 + 2 / 30}})
        time.sleep(0.03)
        for _ in range(3):
            _put(ring)
    finally:
        recorder.close()
    saved = _jsonl(tmp_path / "cameras/front/requests.jsonl")
    assert saved[0]["frame_id"] == 2 and saved[0]["status"] == "saved"


def test_shutdown_resolves_request_after_multiple_reader_batches(tmp_path, manager):
    ring = _camera_ring(manager)
    recorder = PolicyRecorder(tmp_path, {"front": ring}, {}, image_mode="requests")
    recorder.start()
    recorder.request("last", {"front": {"frame_id": 23, "timestamp": 100 + 23 / 30}})
    for _ in range(24):
        _put(ring)
    recorder.close()
    saved = _jsonl(tmp_path / "cameras/front/requests.jsonl")
    assert saved[0]["frame_id"] == 23 and saved[0]["status"] == "saved"
    assert recorder.stats["drained"]


def test_cache_miss_timestamp_mismatch_and_invalid_ref_are_explicit(tmp_path, manager):
    ring = _camera_ring(manager)
    for _ in range(5):
        _put(ring)
    recorder = PolicyRecorder(
        tmp_path, {"front": ring}, {}, image_mode="requests", cache_frames=2
    )
    recorder.start()
    try:
        recorder.request("gone", {"front": {"frame_id": 0, "timestamp": 100}})
        recorder.request("wrong", {"front": {"frame_id": 4, "timestamp": 100}})
        assert not recorder.request(
            "invalid", {"front": {"frame_id": None, "timestamp": 100}}
        )["front"]
    finally:
        recorder.close()
    saved = _jsonl(tmp_path / "cameras/front/requests.jsonl")
    assert [row["status"] for row in saved] == [
        "frame_unavailable",
        "timestamp_mismatch",
    ]
    assert recorder.stats["incomplete"] and recorder.stats["drained"]


def test_lossless_all_frames_and_feedback_drain(tmp_path, manager):
    camera = _camera_ring(manager)
    arm = SharedMemoryRingBuffer.create_from_examples(
        manager,
        dict(
            feedback_id=np.int64(0),
            ee_pose=np.zeros(6),
            robot_receive_timestamp=np.float64(0),
            robot_pose_source_timestamp=np.float64(np.nan),
            last_command_id=np.int64(-1),
        ),
        get_max_k=32,
        get_time_budget=0.1,
        put_desired_frequency=100,
    )
    recorder = PolicyRecorder(
        tmp_path,
        {"front": camera},
        {"left": arm},
        image_mode="all",
        video_format="lossless",
        record_feedback=True,
    )
    recorder.start()
    try:
        for i in range(6):
            _put(camera)
            arm.put(
                dict(
                    feedback_id=np.int64(i),
                    ee_pose=np.full(6, i),
                    robot_receive_timestamp=np.float64(i / 100),
                    robot_pose_source_timestamp=np.float64(np.nan),
                    last_command_id=np.int64(i - 1),
                )
            )
    finally:
        recorder.close()
    frames = _jsonl(tmp_path / "cameras/front/frames.jsonl")
    assert [row["frame_id"] for row in frames] == list(range(6))
    assert all(row["lossless"] for row in frames)
    feedback = _jsonl(tmp_path / "feedback/left.jsonl")
    assert [row["feedback_id"] for row in feedback] == list(range(6))
    assert feedback[5]["ee_pose"] == [5] * 6
    assert feedback[5]["last_command_id"] == 4
    assert feedback[0]["robot_pose_source_timestamp"] is None
    assert recorder.stats["drained"] and not recorder.stats["incomplete"]
    assert recorder.stats["workers"]["camera:front"]["written"] == 6
    assert recorder.stats["workers"]["feedback"]["written"] == 6
    assert recorder.stats["workers"]["feedback"]["arms"]["left"]["drained"]


def test_mp4_segments_index_and_decode(tmp_path):
    import av

    stats = _stats()
    sink = _CameraSink(tmp_path, "mp4", 30, 0.06, stats)
    try:
        for i in range(5):
            sink(
                (
                    "frame",
                    dict(
                        frame_id=i,
                        color=np.full((16, 16, 3), 50 + i, np.uint8),
                        timestamp=i / 30,
                        camera_receive_timestamp=i / 30,
                    ),
                    None,
                )
            )
    finally:
        sink.close()
    index = _jsonl(tmp_path / "frames.jsonl")
    assert [row["video_frame"] for row in index] == [0, 1, 0, 1, 0]
    assert all(not row["lossless"] for row in index)
    decoded = 0
    for path in sorted((tmp_path / "frames").glob("*.mp4")):
        with av.open(path) as container:
            decoded += sum(1 for _ in container.decode(video=0))
    assert decoded == 5


def test_ring_wrap_skips_missing_and_copies_owned_values(manager):
    ring = _camera_ring(manager, k=3)
    stats = _stats()
    reader = _RingReader(ring, "frame_id", stats, start_count=0)
    for _ in range(7):
        _put(ring)
    rows = reader.read()
    assert [int(row["frame_id"]) for row in rows] == [4, 5, 6]
    assert stats["source_missing"] == 4
    ring.shared_arrays["color"].get()[:] = 255
    assert np.all(rows[0]["color"] == 4)
    assert not rows[0]["color"].flags.writeable
    assert stats["missing_ranges"] == [{"first": 0, "end": 4, "kind": "source_missing"}]


def test_copy_budget_or_wrong_publication_rejected(manager, monkeypatch):
    ring = _camera_ring(manager)
    _put(ring)
    stats = _stats()
    reader = _RingReader(ring, "frame_id", stats, start_count=0)
    values = iter([0, 1])
    with monkeypatch.context() as patch:
        patch.setattr(
            "egomimic.robot.piper_pico.policy_recording.time.monotonic",
            lambda: next(values),
        )
        assert reader.read() == []
    assert stats["read_timeouts"] == stats["unsafe_read_dropped"] == 1
    ring.shared_arrays["frame_id"].get()[0] = 99
    reader.cursor = 0
    assert reader.read() == []
    assert stats["unsafe_read_dropped"] == 2


def test_shutdown_missing_ranges_do_not_extend_past_stop_count(manager):
    ring = _camera_ring(manager, k=32)
    # The source kept producing while the recorder was stalled on scheduling.
    ring.counter.store(100)
    stats = _stats()
    reader = _RingReader(ring, "frame_id", stats, start_count=50)
    assert reader.read(limit=60) == []
    assert reader.cursor == 60
    assert stats["source_missing"] == 10
    assert stats["missing_ranges"] == [
        {"first": 50, "end": 60, "kind": "source_missing"}
    ]


def test_slow_writer_bounded_drop_priority_and_owned_snapshot():
    entered, release = threading.Event(), threading.Event()
    consumed = []

    def write(item):
        if item == "block":
            entered.set()
            assert release.wait(3)
        consumed.append(item)

    stats = _stats()
    pump = _WritePump(write, lambda: None, stats, 2)
    assert pump.submit("block")
    assert entered.wait(1)
    started = time.monotonic()
    accepted = [pump.submit(i) for i in range(1000)]
    assert time.monotonic() - started < 0.5
    assert sum(accepted) == 2 and stats["queue_dropped"] == 998
    assert pump.submit("priority", priority=True)
    release.set()
    pump.close(2)
    assert consumed == ["block", "priority", 0, 1]
    assert stats["drained"] and stats["incomplete"]


def test_writer_failure_is_reported_and_submit_stays_nonblocking():
    def broken(item):
        raise OSError("disk full")

    stats = _stats()
    pump = _WritePump(broken, lambda: None, stats, 2)
    pump.submit("x")
    assert pump.failed.wait(2)
    assert not pump.submit("y")
    pump.close(1)
    assert not stats["drained"]
    assert "disk full" in stats["errors"][0]


def test_failed_png_write_propagates_to_worker_summary(tmp_path, monkeypatch):
    stats = _stats()
    sink = _CameraSink(tmp_path, "lossless", 30, 60, stats)
    monkeypatch.setattr(sink.cv2, "imwrite", lambda *args: False)
    pump = _WritePump(sink, sink.close, stats, 2)
    pump.submit(
        (
            "request",
            dict(frame_id=0, color=np.zeros((12, 16, 3), np.uint8), timestamp=0),
            {"request_id": "x"},
        ),
        priority=True,
    )
    assert pump.failed.wait(2)
    pump.close(1)
    sink.close()
    assert stats["requests_written"] == 0
    assert stats["incomplete"] and "failed to write" in stats["errors"][0]


def test_blocked_sink_close_reports_pending_without_waiting_for_disk():
    entered, release = threading.Event(), threading.Event()

    def blocked(item):
        entered.set()
        release.wait(2)

    stats = _stats()
    pump = _WritePump(blocked, lambda: None, stats, 2)
    pump.submit(1)
    assert entered.wait(1)
    pump.submit(2)
    started = time.monotonic()
    pump.close(0.02)
    assert time.monotonic() - started < 0.5
    assert stats["pending_normal"] == 1 and stats["writer_busy_on_close"]
    assert not stats["drained"] and stats["incomplete"]
    release.set()
    pump.thread.join(2)


def test_parent_descriptor_overflow_and_shutdown_are_bounded(tmp_path, manager):
    ring = _camera_ring(manager)
    recorder = PolicyRecorder(
        tmp_path,
        {"front": ring},
        {},
        image_mode="requests",
        queue_capacity=2,
        close_timeout=0.1,
        _camera_target=_sleeping_camera,
    )
    recorder.start()
    before = time.monotonic()
    result = [
        recorder.request(i, {"front": {"frame_id": i, "timestamp": 1}})["front"]
        for i in range(100)
    ]
    assert sum(result) == 2
    assert time.monotonic() - before < 0.5
    recorder.close()
    assert time.monotonic() - before < 2
    assert recorder.stats["requests_dropped"] == 98
    assert recorder.stats["incomplete"] and not recorder.stats["drained"]


def test_worker_crash_is_visible_without_device_access(tmp_path, manager):
    recorder = PolicyRecorder(
        tmp_path,
        {"front": _camera_ring(manager)},
        {},
        image_mode="requests",
        _camera_target=_crashing_camera,
    )
    try:
        try:
            recorder.start()
        except RuntimeError:
            pass  # The crash can race readiness and be caught at startup.
        else:
            _wait_for(
                lambda: not recorder._workers["camera:front"]["process"].is_alive()
            )
            recorder.poll()
        assert recorder.stats["incomplete"]
    finally:
        recorder.close()


def test_nonzero_exit_overrides_success_summary(tmp_path, manager):
    recorder = PolicyRecorder(
        tmp_path,
        {"front": _camera_ring(manager)},
        {},
        image_mode="requests",
        _camera_target=_false_success_camera,
    )
    recorder.start()
    recorder.close()
    assert recorder.stats["workers"]["camera:front"]["exitcode"] == 1
    assert not recorder.stats["drained"] and recorder.stats["incomplete"]


def test_old_schema_rejected_explicitly(manager):
    ring = SharedMemoryRingBuffer.create_from_examples(
        manager, {"color": np.zeros((2, 2, 3), np.uint8)}
    )
    with pytest.raises(ValueError, match="frame_id"):
        _RingReader(ring, "frame_id", _stats())
