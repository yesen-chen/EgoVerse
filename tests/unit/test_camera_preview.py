"""Preview is an optional, read-only process, never a control-loop callback."""

from contextlib import contextmanager
from multiprocessing.managers import SharedMemoryManager
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from egomimic.robot.piper_pico import camera_preview as preview
from egomimic.robot.piper_pico import collect
from egomimic.robot.piper_pico.camera_process import camera_examples
from egomimic.robot.piper_pico.shm import SharedMemoryRingBuffer


def make_ring(manager, height=48, width=64):
    return SharedMemoryRingBuffer.create_from_examples(
        manager,
        camera_examples(height, width),
        get_time_budget=0.2,
        put_desired_frequency=30,
    )


def publish(ring, value=180, stamp=100.0):
    rgb = np.zeros(ring.shared_arrays["color"].shape[1:], np.uint8)
    rgb[:, :, 0] = value
    ring.put(
        {"color": rgb, "timestamp": stamp - 0.066, "camera_receive_timestamp": stamp}
    )


def test_readers_skip_to_latest_without_consuming_frames_or_mutating_rgb():
    with SharedMemoryManager() as manager:
        ring = make_ring(manager)
        a, b = preview.LatestFrame("front_1", ring), preview.LatestFrame("other", ring)
        a.read()  # Startup before any image must not wait.
        assert a.image is None
        publish(ring, value=100)
        a.read()
        out = a.out
        publish(ring, value=150)
        publish(ring, value=200)
        a.read()
        b.read()
        assert ring.count == 3
        assert a.out is out  # One reusable full-size copy per camera.
        assert a.received == b.received == 100.0
        np.testing.assert_array_equal(a.image, b.image)
        assert tuple(a.image[120, 160]) == (0, 0, 200)
        canvas = preview.compose_preview([a, b], 15, 15, 100.1)
        assert canvas.shape == (304, 640, 3)
        assert ring.count == 3
        assert tuple(ring.get()["color"][0, 0]) == (200, 0, 0)
        # A stopped camera retains its last image and becomes visibly stale.
        stale = preview.compose_preview([a, b], 15, 15, 102.0)
        assert np.any(canvas[:40] != stale[:40])
        np.testing.assert_array_equal(canvas[40:280], stale[40:280])


def test_non_four_by_three_frame_is_letterboxed_not_stretched():
    with SharedMemoryManager() as manager:
        ring = make_ring(manager, height=30, width=60)
        publish(ring)
        reader = preview.LatestFrame("wide", ring)
        reader.read()
        assert reader.image.shape == (160, 320, 3)
        canvas = preview.compose_preview([reader], None, 15, 100.0)
        assert np.all(canvas[40:80] == 24)
        assert tuple(canvas[80, 0]) == (0, 0, 180)


def test_suspended_copy_is_discarded_without_retrying(monkeypatch):
    with SharedMemoryManager() as manager:
        ring = make_ring(manager)
        publish(ring)
        reader = preview.LatestFrame("front", ring)
        ticks = iter([1.0, 1.0, 1.3])
        with monkeypatch.context() as patch:
            patch.setattr(preview.time, "monotonic", lambda: next(ticks))
            reader.read()
        assert reader.image is None
        assert reader.last_count == -1
        assert ring.count == 1


def test_camera_rate_counts_skipped_publications_and_reports_stall():
    rate = preview.FrameRate()
    assert rate.update(10, 0.0) is None
    assert rate.update(25, 0.5) is None
    assert rate.update(40, 1.0) == 30.0
    assert rate.update(40, 2.0) == 0.0
    assert rate.update(1, 3.0) is None


@pytest.mark.parametrize("value", ["0", "nan", "inf", "31", "-1", "invalid"])
def test_bad_preview_rate_fails_before_starting_hardware(value, monkeypatch):
    manager = Mock()
    monkeypatch.setattr(collect, "SharedMemoryManager", manager)
    with pytest.raises(SystemExit):
        collect.main(["--camera", "front_1=0", "--preview-fps", value])
    manager.assert_not_called()


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("fails", [False, True])
def test_collect_preview_lifecycle_does_not_change_control_loop(
    enabled, fails, monkeypatch
):
    calls = []
    camera = SimpleNamespace(name="front_1", ring=object())

    class FakeEnv:
        cameras = [camera]

        def __enter__(self):
            calls.append("env start")
            return self

        def __exit__(self, *args):
            calls.append("env stop")

    env = FakeEnv()

    @contextmanager
    def resource(name, obj):
        calls.append(name + " start")
        try:
            yield obj
        finally:
            calls.append(name + " stop")

    monkeypatch.setattr(
        collect, "SharedMemoryManager", lambda: resource("shm", object())
    )
    monkeypatch.setattr(collect, "PiperPicoEnv", lambda *a, **kw: env)

    def viewer(cameras, fps):
        assert cameras == [camera] and fps == 15
        return resource("preview", object())

    monkeypatch.setattr(collect, "CameraPreview", viewer)

    def run(actual_env, frequency, home_seconds):
        assert actual_env is env and frequency == 30 and home_seconds == 3
        calls.append("control loop")
        if fails:
            raise RuntimeError("control error")

    monkeypatch.setattr(collect, "run_teleop", run)
    args = ["--camera", "front_1=0"] + ([] if enabled else ["--no-preview"])
    if fails:
        with pytest.raises(RuntimeError, match="control error"):
            collect.main(args)
    else:
        collect.main(args)
    expected = ["shm start", "env start"]
    if enabled:
        expected.append("preview start")
    expected.append("control loop")
    if enabled:
        expected.append("preview stop")
    assert calls == expected + ["env stop", "shm stop"]


def test_headless_start_skips_gui_without_allocating_process(monkeypatch):
    monkeypatch.setattr(preview.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    context = Mock(side_effect=AssertionError("must not create a process"))
    monkeypatch.setattr(preview.mp, "get_context", context)
    with preview.CameraPreview(
        [SimpleNamespace(name="front", ring=object())]
    ) as viewer:
        assert viewer.process is None
    context.assert_not_called()


def test_gui_failure_stays_local(monkeypatch):
    monkeypatch.setattr(preview.signal, "signal", lambda *args: None)
    monkeypatch.setattr(preview.os, "nice", lambda *args: None)
    monkeypatch.setattr(
        cv2, "namedWindow", Mock(side_effect=RuntimeError("no GUI backend"))
    )
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    stop, ready = Event(), Event()
    preview.preview_loop([("front", object())], stop, ready, 15)
    assert not stop.is_set() and not ready.is_set()


@pytest.mark.parametrize("key,visible", [(27, 1), (ord("q"), 1), (-1, 0)])
def test_window_close_does_not_signal_collection_or_camera_stop(
    monkeypatch, key, visible
):
    monkeypatch.setattr(preview.signal, "signal", lambda *args: None)
    monkeypatch.setattr(preview.os, "nice", lambda *args: None)
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *args: key)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: visible)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    ring = SimpleNamespace(count=0)
    stop, ready = Event(), Event()
    preview.preview_loop([("front", ring)], stop, ready, 15)
    assert not stop.is_set()
    assert ring.count == 0


def _fake_window_worker(*args):
    """Real spawn, SHM, copying and rendering; replace only native window calls."""

    def forbidden(*args, **kwargs):
        raise AssertionError("preview must never reopen cameras")

    cv2.VideoCapture = forbidden
    cv2.namedWindow = lambda *args: None
    cv2.imshow = lambda *args: None
    cv2.waitKey = lambda *args: -1
    cv2.getWindowProperty = lambda *args: 1.0
    cv2.destroyAllWindows = lambda: None
    preview.preview_loop(*args)


def test_spawned_preview_reads_existing_ring_and_cleans_up(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":fake-window")
    monkeypatch.setattr(preview, "preview_loop", _fake_window_worker)
    with SharedMemoryManager() as manager:
        ring = make_ring(manager)
        publish(ring)
        cameras = [
            SimpleNamespace(name=name, ring=ring)
            for name in ("front_1", "right_wrist", "left_wrist")
        ]
        viewer = preview.CameraPreview(cameras)
        assert [name for name, _ in viewer.sources] == [
            "left_wrist",
            "front_1",
            "right_wrist",
        ]
        try:
            viewer.start()
            assert viewer.process._start_method == "spawn"
            assert viewer.ready_event.wait(10)
            assert ring.count == 1
            publish(ring, value=220)
            assert tuple(ring.get()["color"][0, 0]) == (220, 0, 0)
        finally:
            viewer.stop()
        viewer.stop()
        assert viewer.process.exitcode == 0
        assert not viewer.process.is_alive()


def test_stuck_preview_cleanup_is_bounded():
    viewer = preview.CameraPreview([])
    viewer.stop_event = Event()
    viewer.process = Mock(pid=123)
    viewer.process.is_alive.side_effect = [True, True, False]
    viewer.stop()
    assert viewer.stop_event.is_set()
    assert [call.kwargs["timeout"] for call in viewer.process.join.call_args_list] == [
        1.0,
        0.5,
        0.5,
    ]
    viewer.process.terminate.assert_called_once()
    viewer.process.kill.assert_called_once()
