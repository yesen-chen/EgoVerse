"""Worker validation and recording boundaries without opening a real camera."""

import json
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from egomimic.robot.piper_pico import camera_process as camera
from egomimic.robot.piper_pico import uvc_controls


def fake_capture(monkeypatch):
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.side_effect = {
        cv2.CAP_PROP_FRAME_WIDTH: 640,
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
        cv2.CAP_PROP_FPS: 30,
        cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
    }.__getitem__
    opener = Mock(return_value=capture)
    monkeypatch.setattr(cv2, "VideoCapture", opener)
    return capture, opener


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((240, 320, 3), dtype=np.uint8),
        np.zeros((480, 640, 4), dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.float32),
    ],
)
def test_unexpected_native_frame_is_rejected_before_publish_or_ready(
    monkeypatch, frame
):
    capture, _ = fake_capture(monkeypatch)
    capture.read.return_value = True, frame
    ring, ready = Mock(), Event()
    with pytest.raises(RuntimeError, match="拒绝缩放"):
        camera.camera_process_loop(
            "/dev/fake",
            640,
            480,
            30,
            0,
            "MJPG",
            ring,
            Queue(),
            Queue(),
            Event(),
            ready,
        )
    capture.release.assert_called_once()
    ring.put.assert_not_called()
    assert not ready.is_set()


def test_identity_failure_happens_before_capture_is_opened(monkeypatch):
    _, opener = fake_capture(monkeypatch)
    monkeypatch.setattr(
        camera, "verify_identity", Mock(side_effect=RuntimeError("identity mismatch"))
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        camera.camera_process_loop(
            "/dev/fake",
            640,
            480,
            30,
            0,
            "MJPG",
            Mock(),
            Queue(),
            Queue(),
            Event(),
            Event(),
            expected_identity={"ID_SERIAL": "expected"},
        )
    opener.assert_not_called()


def test_calibrated_wrong_serial_is_rejected_without_explicit_identity_before_open(
    monkeypatch,
):
    _, opener = fake_capture(monkeypatch)
    monkeypatch.setattr(uvc_controls.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        uvc_controls.subprocess,
        "run",
        Mock(
            return_value=SimpleNamespace(
                returncode=0,
                stdout="ID_SERIAL=wrong-camera\n",
                stderr="",
            )
        ),
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        camera.camera_process_loop(
            "/dev/fake",
            640,
            480,
            30,
            0,
            "MJPG",
            Mock(),
            Queue(),
            Queue(),
            Event(),
            Event(),
            calibration={"camera": {"ID_SERIAL": "calibrated-camera"}},
        )
    opener.assert_not_called()


def test_control_failure_happens_before_capture_is_opened(monkeypatch):
    _, opener = fake_capture(monkeypatch)
    monkeypatch.setattr(
        camera, "verify_controls", Mock(side_effect=RuntimeError("focus mismatch"))
    )
    ring, ready = Mock(), Event()
    with pytest.raises(RuntimeError, match="focus mismatch"):
        camera.camera_process_loop(
            "/dev/fake",
            640,
            480,
            30,
            0,
            "MJPG",
            ring,
            Queue(),
            Queue(),
            Event(),
            ready,
            controls={"focus_absolute": 120},
        )
    opener.assert_not_called()
    ring.put.assert_not_called()
    assert not ready.is_set()


@pytest.mark.parametrize("end_drift", [False, True])
def test_recording_has_paired_timestamps_metadata_and_checked_end_ack(
    monkeypatch, tmp_path, end_drift
):
    pytest.importorskip("av")
    capture, _ = fake_capture(monkeypatch)
    commands, acks = Queue(), Queue()
    path = tmp_path / "wrist.mp4"
    commands.put({"cmd": "start", "path": str(path)})
    frames = []

    def read():
        commands.put({"cmd": "stop_record"})
        commands.put({"cmd": "stop"})
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    capture.read.side_effect = read
    container = Mock()
    stream = SimpleNamespace(encode=lambda *args: [])
    monkeypatch.setattr(camera, "_open_writer", lambda *args: (container, stream))
    verification = {"status": "verified", "readback": {"focus_absolute": 120}}
    verify = Mock(
        side_effect=[
            verification,
            verification,
            RuntimeError("focus drift") if end_drift else verification,
        ]
    )
    monkeypatch.setattr(camera, "verify_controls", verify)
    args = (
        "/dev/fake",
        640,
        480,
        30,
        0.08,
        "MJPG",
        SimpleNamespace(put=frames.append),
        commands,
        acks,
        Event(),
        Event(),
    )
    if end_drift:
        with pytest.raises(RuntimeError, match="focus drift"):
            camera.camera_process_loop(
                *args, controls={"focus_absolute": 120}, name="left_wrist"
            )
    else:
        camera.camera_process_loop(
            *args, controls={"focus_absolute": 120}, name="left_wrist"
        )
    capture.release.assert_called_once()
    container.close.assert_called_once()
    assert verify.call_count == 3
    assert acks.get_nowait() == "started"
    if end_drift:
        assert acks.empty()
    else:
        assert acks.get_nowait() == "stopped"
    stamps = np.load(path.with_suffix(".npy"))
    received = np.load(path.with_suffix(".receive_timestamps.npy"))
    assert len(stamps) == len(received) == len(frames) == 1
    np.testing.assert_array_equal(stamps, [frames[0]["timestamp"]])
    np.testing.assert_array_equal(received, [frames[0]["camera_receive_timestamp"]])
    np.testing.assert_allclose(received - stamps, [0.08], atol=1e-6)
    metadata = json.loads(path.with_suffix(".stream_metadata.json").read_text())
    assert metadata["name"] == "left_wrist"
    assert metadata["frame_count"] == 1
    assert metadata["native_frame_shape"] == [480, 640, 3]
    assert metadata["latency_status"] == "unmeasured"
    assert metadata["preprocessing"]["resize"] is None
    assert ("end_verification_error" in metadata) == end_drift


def test_inconsistent_receive_timestamp_count_never_writes_success_sidecar(tmp_path):
    container = Mock()
    path = tmp_path / "broken.mp4"
    with pytest.raises(RuntimeError, match="counts differ"):
        camera._close_writer(
            container, SimpleNamespace(encode=lambda: []), path, [1.0], []
        )
    container.close.assert_called_once()
    assert not path.with_suffix(".npy").exists()
    assert not path.with_suffix(".receive_timestamps.npy").exists()
