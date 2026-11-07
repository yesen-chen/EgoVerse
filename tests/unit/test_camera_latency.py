"""QR latency pairing, without a camera."""

import time

import cv2
import numpy as np

from egomimic.robot.piper_pico.camera_latency import (
    decode_qr,
    latency_offsets,
    make_timestamp_qr,
    pair_first_receive,
    summarize_offsets,
)
from egomimic.robot.piper_pico.pico_camera import control_packet, split_eyes


def test_detector_crash_skips_the_frame_or_retries():
    image, _t_sample, text = make_timestamp_qr(480)

    class Broken:
        def detectAndDecodeCurved(self, _image):
            raise cv2.error("Seed point is outside of image")

        def detectAndDecode(self, _image):
            raise cv2.error("Seed point is outside of image")

    code, corners = decode_qr(Broken(), image)
    assert code == text
    assert corners is not None
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    assert decode_qr(Broken(), blank) == ("", None)


def test_split_eyes_and_control_packet():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, :3] = 1
    image[:, 3:] = 2
    left, right = split_eyes(image)
    assert left.shape == (4, 3, 3)
    assert int(left[0, 0, 0]) == 1
    assert int(right[0, 0, 0]) == 2
    packet = control_packet("CLOSE_CAMERA")
    assert packet[4:20] == b"\x0c\x00\x00\x00CLOSE_CAMERA"


def test_qr_roundtrip_keeps_the_timestamp():
    image, t_sample, text = make_timestamp_qr(720)
    code, corners = decode_qr(cv2.QRCodeDetector(), image)
    assert code == text
    assert corners is not None
    assert abs(float(code) - t_sample) < 1e-6


def test_first_receive_and_display_latency():
    early, _t0, early_text = make_timestamp_qr(480)
    time.sleep(0.002)
    late, _t1, late_text = make_timestamp_qr(480)
    detector = cv2.QRCodeDetector()
    decoded = []
    for received, image in ((1.20, early), (1.50, early), (2.00, late)):
        code, _corners = decode_qr(detector, image)
        decoded.append((received, code))
    pairs = pair_first_receive(decoded)
    assert pairs[early_text] == 1.20
    assert pairs[late_text] == 2.00
    offsets = latency_offsets(pairs, display_latency=0.05)
    expected = [
        1.20 - float(early_text) - 0.05,
        2.00 - float(late_text) - 0.05,
    ]
    assert np.allclose(offsets, expected)


def test_startup_timeout_closes_the_capture_thread():
    from egomimic.robot.piper_pico.camera_latency import FrameBuffer, _CaptureThread

    class Silent(_CaptureThread):
        def _run(self) -> None:
            self.stop.wait(5)

    capture = Silent(FrameBuffer(2), startup_timeout=0.4)
    try:
        capture.start()
    except TimeoutError as exc:
        assert "等待图像超时" in str(exc)
    else:
        raise AssertionError("startup should time out")
    assert capture._thread is not None
    assert not capture._thread.is_alive()


def test_summary_reports_median_and_p95():
    offsets = np.array([0.01, 0.02, 0.03, 0.04, 0.10], dtype=np.float64)
    summary = summarize_offsets(offsets)
    assert summary["count"] == 5
    assert summary["median_sec"] == 0.03
    assert summary["mean_sec"] == 0.04
    assert summary["p95_sec"] == np.percentile(offsets, 95)
