"""Camera eye selection is handled before transport and shared-memory writes."""

import multiprocessing as mp
import struct
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from egomimic.robot.piper_pico import pico_camera as camera


@pytest.mark.parametrize("mode,render_mode,width", [("stereo", 2, 2160), ("left", 0, 1080), ("right", 1, 1080)])
def test_camera_request_serializes_selected_mode(mode, render_mode, width):
    request = camera.camera_request(width, 810, 60, 5_000_000, "192.0.2.2", 12345, mode)
    assert request[:3] == b"\xca\xfe\x01"
    assert struct.unpack("<7i", request[3:31]) == (width, 810, 60, 5_000_000, 0, render_mode, 12345)
    assert request[31:] == b"\x02VR\x09192.0.2.2"


def test_camera_request_defaults_to_stereo():
    args = (2160, 810, 30, 5_000_000, "192.0.2.2", 12345)
    assert camera.camera_request(*args) == camera.camera_request(*args, mode="stereo")


class FakeEvent:
    def __init__(self):
        self.value = False

    def is_set(self):
        return self.value

    def set(self):
        self.value = True


@pytest.fixture
def fake_process_resources(monkeypatch):
    rings = []

    def create_ring(manager, examples, **kwargs):
        ring = SimpleNamespace(examples=examples, options=kwargs)
        rings.append(ring)
        return ring

    monkeypatch.setattr(camera.SharedMemoryRingBuffer, "create_from_examples", create_ring)
    monkeypatch.setattr(mp, "Event", FakeEvent)
    monkeypatch.setattr(mp, "Process", lambda **kwargs: SimpleNamespace(**kwargs))
    return rings


@pytest.mark.parametrize("mode,enabled,width", [("stereo", ("left", "right"), 2160), ("left", ("left",), 1080), ("right", ("right",), 1080)])
def test_only_enabled_eyes_allocate_shared_memory(fake_process_resources, mode, enabled, width):
    process = camera.PicoStereoProcess(object(), "192.0.2.1", mode=mode)
    assert process.mode == mode
    assert process.enabled_eyes == enabled
    assert process.stream_width == width
    assert process.stereo_width == 2160
    assert process.eye_width == 1080
    assert len(fake_process_resources) == len(enabled)
    assert process.process.target is camera.pico_camera_loop
    assert process.process.kwargs["mode"] == mode
    for eye in ("left", "right"):
        if eye in enabled:
            view = process.eye(eye)
            assert view.name == f"pico_{eye}"
            assert view.ring.examples["color"].shape == (810, 1080, 3)
            assert process.process.kwargs[f"{eye}_ring"] is view.ring
        else:
            assert getattr(process, eye) is None
            assert process.process.kwargs[f"{eye}_ring"] is None
            with pytest.raises(ValueError, match="disabled"):
                process.eye(eye)
    with pytest.raises(ValueError, match="left or right"):
        process.eye("both")


def test_process_defaults_to_stereo(fake_process_resources):
    process = camera.PicoStereoProcess(object(), "192.0.2.1")
    assert process.mode == "stereo"
    assert process.enabled_eyes == ("left", "right")
    assert process.stream_width == 2160


@pytest.mark.parametrize("mode", ["both", "LEFT", "", None])
def test_invalid_mode_is_rejected_before_allocating(fake_process_resources, mode):
    with pytest.raises(ValueError, match="camera mode"):
        camera.PicoStereoProcess(object(), "192.0.2.1", mode=mode)
    with pytest.raises(ValueError, match="camera mode"):
        camera.camera_request(1080, 810, 60, 5_000_000, "192.0.2.2", 12345, mode)
    assert not fake_process_resources


@pytest.mark.parametrize("mode,width,height", [("stereo", 2159, 810), ("left", 2162, 810), ("right", 2162, 810), ("left", 2160, 809), ("right", 0, 810), ("stereo", 2160, -2)])
def test_invalid_h264_dimensions_rejected_before_allocating(fake_process_resources, mode, width, height):
    with pytest.raises(ValueError, match="H.264"):
        camera.PicoStereoProcess(object(), "192.0.2.1", stereo_width=width, stereo_height=height, mode=mode)
    assert not fake_process_resources


def _frame(image):
    def to_ndarray(format):
        assert format == "bgr24"
        return image

    return SimpleNamespace(to_ndarray=to_ndarray)


@pytest.mark.parametrize("mode", ["left", "right"])
def test_mono_publishes_whole_frame_only_to_selected_eye(monkeypatch, mode):
    bgr = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    received = []
    selected = SimpleNamespace(put=received.append)
    ready = FakeEvent()
    monkeypatch.setattr(camera.time, "time", lambda: 100.0)
    monkeypatch.setattr(camera, "split_eyes", Mock(side_effect=AssertionError("mono must not be split")))
    camera._publish_frame(
        _frame(bgr), ready, 4, 2, 0.1, 0.2,
        selected if mode == "left" else None,
        selected if mode == "right" else None,
        mode,
    )
    assert ready.is_set()
    assert len(received) == 1
    np.testing.assert_array_equal(received[0]["color"], bgr[:, :, ::-1])
    assert received[0]["camera_receive_timestamp"] == 100.0
    assert received[0]["timestamp"] == 100.0 - (0.1 if mode == "left" else 0.2)


def test_default_stereo_splits_pixels_and_shares_receive_timestamp(monkeypatch):
    bgr = np.arange(2 * 8 * 3, dtype=np.uint8).reshape(2, 8, 3)
    left, right = [], []
    ready = FakeEvent()
    monkeypatch.setattr(camera.time, "time", lambda: 100.0)
    camera._publish_frame(_frame(bgr), ready, 4, 2, 0.1, 0.2, SimpleNamespace(put=left.append), SimpleNamespace(put=right.append))
    np.testing.assert_array_equal(left[0]["color"], bgr[:, :4, ::-1])
    np.testing.assert_array_equal(right[0]["color"], bgr[:, 4:, ::-1])
    assert left[0]["camera_receive_timestamp"] == right[0]["camera_receive_timestamp"] == 100.0
    assert left[0]["timestamp"] == 99.9
    assert right[0]["timestamp"] == 99.8
    assert ready.is_set()


@pytest.mark.parametrize("mode,shape", [("left", (2, 8, 3)), ("right", (2, 8, 3)), ("left", (4, 4, 3)), ("right", (2, 2, 3)), ("stereo", (2, 7, 3)), ("stereo", (2, 8)), ("left", (0, 4, 3))])
def test_bad_decoded_shape_fails_without_publishing_or_marking_ready(mode, shape):
    ring = SimpleNamespace(put=Mock())
    ready = FakeEvent()
    with pytest.raises(ValueError):
        camera._publish_frame(_frame(np.zeros(shape, dtype=np.uint8)), ready, 4, 2, 0.0, 0.0, ring, ring, mode)
    ring.put.assert_not_called()
    assert not ready.is_set()


@pytest.mark.parametrize("mode,width", [("stereo", 2160), ("left", 1080), ("right", 1080)])
def test_loop_requests_actual_stream_width_and_forwards_mode(monkeypatch, mode, width):
    server, control = Mock(), Mock()
    server.getsockname.return_value = ("0.0.0.0", 12345)
    monkeypatch.setattr(camera.socket, "socket", Mock(return_value=server))
    monkeypatch.setattr(camera.socket, "gethostbyname", lambda host: host)
    monkeypatch.setattr(camera.socket, "create_connection", Mock(return_value=control))
    monkeypatch.setattr(camera, "release_pico_camera", Mock())
    receive = Mock()
    monkeypatch.setattr(camera, "_receive", receive)
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace())
    camera.pico_camera_loop(
        "192.0.2.1", "192.0.2.2", 2160, 810, 60, 5_000_000,
        0.0, 0.0, None, None, FakeEvent(), FakeEvent(), 25.0,
        mode=mode,
    )
    assert control.sendall.call_args_list[0].args == (
        camera.control_packet("OPEN_CAMERA", camera.camera_request(width, 810, 60, 5_000_000, "192.0.2.2", 12345, mode)),
    )
    assert receive.call_args.args[-1] == mode
    assert receive.call_args.args[5:7] == (1080, 810)
    assert control.sendall.call_args_list[-1].args == (camera.control_packet("CLOSE_CAMERA"),)
    control.close.assert_called_once()
    server.close.assert_called_once()


def test_packet_forwards_mode_to_each_decoded_frame(monkeypatch):
    frames = [object(), object()]
    codec = SimpleNamespace(decode=lambda packet: frames)
    publish = Mock()
    monkeypatch.setattr(camera, "_publish_frame", publish)
    camera._publish_packet(object(), codec, FakeEvent(), 4, 2, 0.0, 0.0, None, object(), "right")
    assert publish.call_count == 2
    assert [call.args[0] for call in publish.call_args_list] == frames
    assert all(call.args[-1] == "right" for call in publish.call_args_list)
