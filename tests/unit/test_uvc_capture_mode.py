"""Opening /dev/videoN must use V4L2 and reject an unfulfilled mode request."""

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock
from queue import Queue

import cv2
import numpy as np
import pytest

from egomimic.robot.piper_pico import camera_process as camera


@pytest.mark.parametrize("device,source", [("/dev/video0", "/dev/video0"), ("0", 0)])
def test_uvc_path_and_index_use_v4l2_and_publish_at_received_resolution(monkeypatch, device, source):
    stop, ready = Event(), Event()
    props = {
        cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480,
        cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
    }
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.side_effect = props.__getitem__
    capture.set.return_value = True

    def read():
        stop.set()
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    capture.read.side_effect = read
    opener = Mock(return_value=capture)
    monkeypatch.setattr(cv2, "VideoCapture", opener)
    published = []
    camera.camera_process_loop(
        device, 640, 480, 30, 0.08, "MJPG",
        SimpleNamespace(put=published.append), Queue(), Queue(), stop, ready,
    )
    opener.assert_called_once_with(source, cv2.CAP_V4L2)
    capture.release.assert_called_once()
    assert ready.is_set()
    assert len(published) == 1
    assert published[0]["color"].shape == (480, 640, 3)
    assert published[0]["camera_receive_timestamp"] - published[0]["timestamp"] == pytest.approx(0.08)


@pytest.mark.parametrize("wrong_prop,value", [
    (cv2.CAP_PROP_FRAME_WIDTH, 1920),
    (cv2.CAP_PROP_FPS, 5),
    (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV")),
])
def test_wrong_driver_mode_fails_before_any_frame_is_accepted(monkeypatch, wrong_prop, value):
    props = {
        cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480,
        cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
        wrong_prop: value,
    }
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.side_effect = props.__getitem__
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: capture)
    with pytest.raises(RuntimeError, match="模式未生效"):
        camera.camera_process_loop(
            "/dev/video0", 640, 480, 30, 0, "MJPG",
            SimpleNamespace(put=Mock()), Queue(), Queue(), Event(), Event(),
        )
    capture.read.assert_not_called()
    capture.release.assert_called_once()


def test_flush_failure_is_not_acknowledged_as_success(tmp_path):
    container = Mock()
    stream = Mock()
    stream.encode.side_effect = OSError("disk failure")
    video = tmp_path / "x.mp4"
    with pytest.raises(OSError, match="disk failure"):
        camera._close_writer(container, stream, video, [1.0])
    container.close.assert_called_once()
    assert not video.with_suffix(".npy").exists()


def test_empty_recording_is_not_considered_complete(tmp_path):
    container = Mock()
    stream = SimpleNamespace(encode=lambda: [])
    with pytest.raises(RuntimeError, match="没有收到图像"):
        camera._close_writer(container, stream, tmp_path / "empty.mp4", [])
    container.close.assert_called_once()
