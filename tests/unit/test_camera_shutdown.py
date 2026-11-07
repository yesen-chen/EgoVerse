"""Camera shutdown must release USB before flushing and expose forced exits."""

from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from egomimic.robot.piper_pico import camera_process as camera
from egomimic.robot.piper_pico import pico_camera


def capture_stub():
    cap = Mock()
    cap.isOpened.return_value = True
    cap.get.side_effect = {
        cv2.CAP_PROP_FRAME_WIDTH: 640,
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
        cv2.CAP_PROP_FPS: 30,
        cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
    }.__getitem__
    return cap


def test_queued_stop_never_enters_another_potentially_blocking_read(monkeypatch):
    cap = capture_stub()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: cap)
    commands = Queue()
    commands.put({"cmd": "stop"})
    camera.camera_process_loop(
        "/dev/fake", 640, 480, 30, 0, "MJPG", Mock(),
        commands, Queue(), Event(), Event(),
    )
    cap.read.assert_not_called()
    cap.release.assert_called_once()


@pytest.mark.parametrize("fail_flush", [False, True])
def test_usb_released_before_active_recording_flush_even_on_failure(monkeypatch, tmp_path, fail_flush):
    cap = capture_stub()
    stop = Event()
    calls = []
    cap.release.side_effect = lambda: calls.append("release")

    def read():
        stop.set()
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    cap.read.side_effect = read
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: cap)
    container = Mock()
    stream = Mock()

    def encode(frame=None):
        if frame is None:
            calls.append("flush")
            if fail_flush:
                raise OSError("flush failed")
        return []

    stream.encode.side_effect = encode
    monkeypatch.setattr(camera, "_open_writer", lambda *a: (container, stream))
    commands = Queue()
    commands.put({"cmd": "start", "path": str(tmp_path / "episode.mp4")})
    args = ("/dev/fake", 640, 480, 30, 0, "MJPG", Mock(),
            commands, Queue(), stop, Event())
    if fail_flush:
        with pytest.raises(OSError, match="flush failed"):
            camera.camera_process_loop(*args)
    else:
        camera.camera_process_loop(*args)
        assert len(np.load(tmp_path / "episode.npy")) == 1
    assert calls == ["release", "flush"]
    container.close.assert_called_once()


@pytest.mark.parametrize("alive_after_terminate", [False, True])
def test_forced_or_stuck_shutdown_is_never_reported_as_success(alive_after_terminate):
    process = Mock(pid=123)
    process.is_alive.side_effect = [True, alive_after_terminate]
    with pytest.raises(TimeoutError, match="仍未退出" if alive_after_terminate else "强制终止"):
        camera.stop_camera_process(process, Event(), 15, "test_camera")
    process.terminate.assert_called_once()


def test_abnormal_exit_is_reported():
    process = Mock(pid=123, exitcode=1)
    process.is_alive.return_value = False
    with pytest.raises(RuntimeError, match="异常退出"):
        camera.stop_camera_process(process, Event(), 15, "test_camera")
    process.terminate.assert_not_called()


@pytest.mark.parametrize("kind", ["uvc", "pico"])
def test_clean_stop_is_idempotent_and_closes_ipc_without_forcing_child(kind):
    cls = camera.CameraProcess if kind == "uvc" else pico_camera.PicoStereoProcess
    device = cls.__new__(cls)
    device.name, device.stop_event = "camera", Event()
    device.process = Mock(pid=123, exitcode=0)
    device.process.is_alive.return_value = False
    queues = (Mock(), Mock())
    if kind == "uvc":
        device.commands, device.acks = queues
    else:
        device.recording_commands, device.recording_acks = queues
    device.stop()
    device.stop()
    assert device.stop_event.is_set()
    device.process.terminate.assert_not_called()
    assert all(q.close.call_count == 2 for q in queues)


def test_unstarted_process_does_not_join():
    process = SimpleNamespace(pid=None, join=Mock())
    stop = Event()
    camera.stop_camera_process(process, stop, 15, "camera")
    assert stop.is_set()
    process.join.assert_not_called()


def _synthetic_camera_worker(*args):
    """Runs in the spawned child; no real camera or driver is opened."""
    import multiprocessing as mp
    import time

    assert mp.get_start_method() == "spawn"
    cap = capture_stub()

    def read():
        time.sleep(1 / 30)
        return True, np.full((480, 640, 3), (20, 60, 180), dtype=np.uint8)

    cap.read.side_effect = read
    cv2.VideoCapture = lambda *a: cap
    camera.camera_process_loop(*args)


def test_spawn_transfers_shared_memory_and_records_after_parent_video_use(monkeypatch, tmp_path):
    """Exercise real IPC and encoding over repeated worker creation."""
    import time
    from multiprocessing.managers import SharedMemoryManager

    av = pytest.importorskip("av")
    monkeypatch.setattr(camera, "camera_process_loop", _synthetic_camera_worker)
    # The child imports the original module; this override only selects its
    # entry point. Neither capture mocks nor native thread state are inherited.
    cv2.cvtColor(np.zeros((480, 640, 3), dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    with SharedMemoryManager() as manager:
        for cycle in range(2):
            device = camera.CameraProcess(manager, "/dev/fake", "synthetic")
            path = tmp_path / f"{cycle}.mp4"
            try:
                device.start()
                assert device.ready_event.wait(10)
                device.start_recording(str(path))
                initial = device.ring.count
                deadline = time.monotonic() + 5
                while device.ring.count < initial + 6 and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert device.ring.count >= initial + 6
                device.stop_recording()
            finally:
                device.stop()
            assert device.process.exitcode == 0
            assert not device.process.is_alive()
            stamps = np.load(path.with_suffix(".npy"))
            receive_stamps = np.load(path.with_suffix(".receive_timestamps.npy"))
            with av.open(str(path)) as video:
                frames = list(video.decode(video=0))
            assert len(frames) == len(stamps) >= 6
            assert len(receive_stamps) == len(stamps)
            np.testing.assert_allclose(receive_stamps, stamps)
            assert np.all(np.diff(stamps) > 0)
