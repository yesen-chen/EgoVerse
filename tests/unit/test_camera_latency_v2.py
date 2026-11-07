"""Latency calibration acceptance checks without cameras, sockets, or a display."""

import csv
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from egomimic.robot.piper_pico import camera_latency as latency

PICO = ["pico", "--headset-ip", "192.0.2.10"]


@pytest.mark.parametrize("size", [480, 720, 721])
def test_timestamp_qr_roundtrips_deterministic_payloads_at_display_sizes(
    monkeypatch, size
):
    """The first payload regresses a real OpenCV empty decode at 720 pixels."""
    detector = cv2.QRCodeDetector()
    encoder = latency._qr_encoder()
    for index in range(100):
        timestamp = 1792827928.3378956 + index * 0.103713
        monkeypatch.setattr(latency.time, "time", lambda t=timestamp: t)

        image, generated, text = latency.make_timestamp_qr(size, encoder)
        decoded, corners = latency.decode_qr(detector, image)

        assert image.shape == (size, size, 3)
        assert generated == timestamp
        assert text == str(timestamp)
        assert decoded == text, (size, index, text)
        assert corners is not None
        assert np.isfinite(corners).all()
        assert np.all(corners >= 0)
        assert np.all(corners < size)


def test_half_size_decode_maps_corners_to_original_rectangular_frame(monkeypatch):
    shapes = []
    small_corners = np.array([[[10, 5], [40, 5], [40, 30], [10, 30]]], dtype=np.float32)

    class Empty:
        def detectAndDecodeCurved(self, image):
            return "", None, None

        def detectAndDecode(self, image):
            return "", None, None

    class HalfSize(Empty):
        def detectAndDecode(self, image):
            shapes.append(image.shape)
            return "1792827928.3378956", small_corners.copy(), None

    monkeypatch.setattr(latency.cv2, "QRCodeDetector", HalfSize)
    image = np.zeros((101, 203, 3), dtype=np.uint8)

    decoded, corners = latency.decode_qr(Empty(), image)

    assert decoded == "1792827928.3378956"
    assert shapes == [(50, 101, 3)]
    np.testing.assert_allclose(corners, small_corners * [203 / 101, 101 / 50])


def test_cli_defaults_match_the_collection_stream():
    args = latency.build_parser().parse_args(PICO)

    assert args.mode == "left"
    assert args.view is None
    assert args.width is None
    assert args.height == 810
    assert args.fps == 60
    assert args.n_frames == 240
    assert args.qr_hz == 10.0
    assert args.min_samples == 30
    assert args.refresh_hz == 120.0
    assert args.display_latency_ms == 0.0
    assert args.record_video is False


@pytest.mark.parametrize("refresh_hz", [60, 120, 144])
@pytest.mark.parametrize("correction_ms", [0.0, 7.5])
def test_refresh_rate_is_metadata_and_only_explicit_correction_is_forwarded(
    monkeypatch, tmp_path, refresh_hz, correction_ms
):
    measurements = []
    monkeypatch.setattr(latency, "UvcFeed", lambda **_: object())
    monkeypatch.setattr(
        latency,
        "run_measurement",
        lambda capture, qr_size, n_frames, output, meta: measurements.append(meta),
    )

    latency.main(
        [
            f"--refresh-hz={refresh_hz}",
            f"--display-latency-ms={correction_ms}",
            "--record-video",
            "--output",
            str(tmp_path / "measurement"),
            "uvc",
        ]
    )

    assert len(measurements) == 1
    assert measurements[0]["display_refresh_hz"] == refresh_hz
    assert measurements[0]["display_latency_sec"] == correction_ms / 1000
    assert measurements[0]["record_video"] is True


@pytest.mark.parametrize(
    "argv",
    [
        [f"--{option}={value}", "uvc"]
        for option in ("refresh-hz", "qr-hz")
        for value in ("nan", "inf", "-inf", "0", "-1")
    ]
    + [
        [f"--display-latency-ms={value}", "uvc"]
        for value in ("nan", "inf", "-inf", "-1")
    ]
    + [
        ["--n-frames=1", "uvc"],
        ["--min-samples=0", "uvc"],
        ["--qr-size=63", "uvc"],
    ]
    + [
        ["uvc", f"--{option}={value}"]
        for option in ("fps", "startup-timeout")
        for value in ("nan", "inf", "-inf", "0", "-1")
    ]
    + [
        ["uvc", f"--{option}={value}"]
        for option in ("width", "height")
        for value in ("0", "-1")
    ]
    + [
        [*PICO, f"--{option}={value}"]
        for option in ("width", "height", "fps", "bitrate")
        for value in ("0", "-1")
    ]
    + [
        [*PICO, f"--startup-timeout={value}"]
        for value in ("nan", "inf", "-inf", "0", "-1")
    ]
    + [
        [*PICO, "--width=1079"],
        [*PICO, "--height=809"],
        [*PICO, "--mode=left", "--view=right"],
        [*PICO, "--mode=right", "--view=left"],
        ["--record-video", *PICO, "--mode=stereo"],
    ],
)
def test_invalid_cli_is_rejected_before_any_camera_allocation(monkeypatch, argv):
    def unexpected_hardware(*args, **kwargs):
        pytest.fail("invalid input reached camera construction")

    monkeypatch.setattr(latency, "UvcFeed", unexpected_hardware)
    monkeypatch.setattr(latency, "PicoFeed", unexpected_hardware)
    monkeypatch.setattr(latency, "run_measurement", unexpected_hardware)

    with pytest.raises(SystemExit) as exc:
        latency.main(argv)

    assert exc.value.code != 0


def _mock_process_constructors(monkeypatch):
    calls = []

    class Process:
        def __init__(self, kind, *args, **kwargs):
            calls.append((kind, args, kwargs))
            self.eyes = {
                "left": SimpleNamespace(ring=object()),
                "right": SimpleNamespace(ring=object()),
            }
            self.ring = self.eyes[kwargs.get("eye", "left")].ring
            self.started = False

        def eye(self, name):
            return self.eyes[name]

        def start(self):
            self.started = True

    for kind in ("CameraProcess", "PicoCameraProcess", "PicoStereoProcess"):
        monkeypatch.setattr(
            latency,
            kind,
            lambda *args, _kind=kind, **kwargs: Process(_kind, *args, **kwargs),
        )
    return calls


@pytest.mark.parametrize(
    ("argv", "kind", "width", "height", "fps", "view"),
    [
        (["uvc"], "CameraProcess", 640, 480, 30, None),
        (PICO, "PicoCameraProcess", 1080, 810, 60, "left"),
        ([*PICO, "--mode=right"], "PicoCameraProcess", 1080, 810, 60, "right"),
        ([*PICO, "--mode=stereo"], "PicoStereoProcess", 2160, 810, 60, "left"),
        (
            [*PICO, "--mode=stereo", "--view=right"],
            "PicoStereoProcess",
            2160,
            810,
            60,
            "right",
        ),
        (
            [*PICO, "--mode=right", "--width=960", "--height=720", "--fps=30"],
            "PicoCameraProcess",
            960,
            720,
            30,
            "right",
        ),
    ],
)
def test_cli_uses_the_selected_production_capture_path(
    monkeypatch, tmp_path, argv, kind, width, height, fps, view
):
    calls = _mock_process_constructors(monkeypatch)
    measurement = {}
    destination = tmp_path / "new_measurement"

    def fake_measurement(capture, qr_size, n_frames, output, meta, **kwargs):
        capture._shm = object()
        capture._open()
        assert capture._process.started
        measurement.update(capture=capture, meta=meta, output=output, **kwargs)

    monkeypatch.setattr(latency, "run_measurement", fake_measurement)
    latency.main(["--output", str(destination), *argv])

    assert len(calls) == 1
    actual_kind, _args, settings = calls[0]
    assert actual_kind == kind
    assert settings["fps"] == fps
    assert settings["get_max_k"] == 240
    if kind == "PicoStereoProcess":
        assert settings["stereo_width"] == width
        assert settings["stereo_height"] == height
        assert settings.get("latency_left", 0.0) == 0.0
        assert settings.get("latency_right", 0.0) == 0.0
        assert (
            measurement["capture"]._ring
            is measurement["capture"]._process.eyes[view].ring
        )
    else:
        assert settings["width"] == width
        assert settings["height"] == height
        assert settings["latency"] == 0.0
        assert measurement["capture"]._ring is measurement["capture"]._process.ring
        if kind == "PicoCameraProcess":
            assert settings["eye"] == view

    meta = measurement["meta"]
    assert meta["source"] == argv[0]
    assert meta["requested_width"] == width
    assert meta["requested_height"] == height
    assert meta["requested_fps"] == fps
    assert meta["reader"].endswith(kind)
    assert measurement["output"] == destination
    if view is not None:
        assert meta["view"] == view


@pytest.mark.parametrize("source", ["uvc", "pico"])
def test_measurement_preserves_per_qr_correction_in_csv_json_and_output(
    monkeypatch, tmp_path, capsys, source
):
    """Exercise capture lifecycle, offline decoding, pairing and persisted results."""
    frames = []
    for tick in range(999, 1024):
        received = tick / 10
        code = 0 if tick < 1002 else 1 if tick < 1013 else 2 if tick < 1023 else 3
        frames.append((received, np.full((2, 2, 3), code, dtype=np.uint8)))

    class Capture:
        actual = {"stored_width": 2, "stored_height": 2}
        error = None
        frame_count = len(frames)

        def __init__(self):
            self.events = []
            self.closed = False

        def start(self):
            self.events.append("start")

        def start_recording(self, path):
            self.events.append(("record", path))

        def latest(self):
            return None

        def snapshot(self):
            return frames

        def close(self):
            self.closed = True
            self.events.append("close")

    capture = Capture()
    generation = iter((100.0, 101.0, 102.0))
    submission = iter((100.02, 101.06, 102.04))
    monotonic = iter((0.0, 0.0, 1.0, 1.0, 2.0, 2.0))
    keys = iter((-1, -1, ord("c")))

    def make_qr(*args):
        generated = next(generation)
        return np.zeros((2, 2, 3), dtype=np.uint8), generated, str(generated)

    def decode(_detector, image):
        assert capture.closed, "QR recognition must run after capture stops"
        index = int(image[0, 0, 0])
        return (str(99.0 + index) if index else ""), None

    monkeypatch.setattr(latency, "_qr_encoder", lambda: object())
    monkeypatch.setattr(latency, "make_timestamp_qr", make_qr)
    monkeypatch.setattr(latency, "decode_qr", decode)
    monkeypatch.setattr(
        latency,
        "time",
        SimpleNamespace(
            time=lambda: next(submission), monotonic=lambda: next(monotonic)
        ),
    )
    monkeypatch.setattr(latency.cv2, "setNumThreads", lambda _: None)
    monkeypatch.setattr(latency.cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(latency.cv2, "waitKey", lambda _: next(keys))
    monkeypatch.setattr(latency.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(latency, "_save_plot", lambda *_: None)
    meta = {
        "source": source,
        "qr_hz": 10.0,
        "record_video": True,
        "display_latency_sec": 0.01,
        "display_refresh_hz": 120.0,
        "min_samples": 3,
        "requested_fps": 10.0,
    }

    report = latency.run_measurement(capture, 720, 240, tmp_path, meta)

    assert capture.events == ["start", ("record", tmp_path / "load_test.mp4"), "close"]
    assert report["measured_fps"] == pytest.approx(10.0)
    assert report["recommendation_allowed"] is True
    assert report["valid_sample_count"] == 3
    expected = [
        (100.0, 100.02, 100.2, 0.02, 0.20, 0.18, 0.17),
        (101.0, 101.06, 101.3, 0.06, 0.30, 0.24, 0.23),
        (102.0, 102.04, 102.3, 0.04, 0.30, 0.26, 0.25),
    ]
    for sample, values in zip(report["samples"], expected, strict=True):
        assert [
            sample[key]
            for key in (
                "generated_unix",
                "submitted_unix",
                "receive_unix",
                "software_sec",
                "raw_sec",
                "submitted_to_receive_sec",
                "corrected_sec",
            )
        ] == pytest.approx(values)
    assert report["latency"]["mean_sec"] == pytest.approx((0.17 + 0.23 + 0.25) / 3)
    assert json.loads((tmp_path / "latency_report.json").read_text()) == report
    with (tmp_path / "samples.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(report["samples"])
    for row, sample in zip(rows, report["samples"], strict=True):
        assert row["qr_text"] == sample["qr_text"]
        assert row["valid"] == str(sample["valid"])
        for key in row.keys() - {"qr_text", "valid"}:
            assert float(row[key]) == sample[key]

    text = capsys.readouterr().out
    expected_flag = "--pico-camera-latency" if source == "pico" else "--camera-latency"
    wrong_flag = "--camera-latency" if source == "pico" else "--pico-camera-latency"
    assert expected_flag in text
    assert wrong_flag not in text

    report_path = tmp_path / "latency_report.json"
    original = report_path.read_bytes()
    with pytest.raises(FileExistsError):
        latency.save_report(tmp_path, {**report, "source": "modified"})
    assert report_path.read_bytes() == original


@pytest.mark.parametrize("source", ["uvc", "pico"])
def test_failed_quality_check_never_prints_a_usable_compensation(
    tmp_path, capsys, source
):
    report = latency.analyze_decoded(
        [(100.2, "100.0")],
        {"100.0": {"generated_unix": 100.0, "submitted_unix": 100.02}},
        min_samples=30,
    )
    report.update(source=source, measured_fps=30.0)

    latency._print_summary(report, tmp_path)

    assert report["recommendation_allowed"] is False
    output = capsys.readouterr().out
    assert "--camera-latency" not in output
    assert "--pico-camera-latency" not in output


@pytest.mark.parametrize("stop_at", ["startup", "recording", "quit", "snapshot"])
def test_measurement_closes_camera_and_windows_on_early_exit(
    monkeypatch, tmp_path, stop_at
):
    events = []

    class Capture:
        error = None
        actual = {}
        frame_count = 0

        def start(self):
            events.append("start")
            if stop_at == "startup":
                raise RuntimeError("startup failed")

        def start_recording(self, path):
            events.append("record")
            if stop_at == "recording":
                raise RuntimeError("recording failed")

        def latest(self):
            return None

        def snapshot(self):
            raise RuntimeError("snapshot failed")

        def close(self):
            events.append("close")

    monkeypatch.setattr(latency, "_qr_encoder", lambda: object())
    monkeypatch.setattr(
        latency,
        "make_timestamp_qr",
        lambda *_: (np.zeros((2, 2, 3), dtype=np.uint8), 100.0, "100.0"),
    )
    monkeypatch.setattr(latency.cv2, "setNumThreads", lambda _: None)
    monkeypatch.setattr(latency.cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(
        latency.cv2, "waitKey", lambda _: ord("q" if stop_at == "quit" else "c")
    )
    monkeypatch.setattr(
        latency.cv2, "destroyAllWindows", lambda: events.append("destroy_windows")
    )
    monkeypatch.setattr(
        latency, "save_report", lambda *_: pytest.fail("failed run wrote a report")
    )
    expected_exception = SystemExit if stop_at == "quit" else RuntimeError
    with pytest.raises(expected_exception) as error:
        latency.run_measurement(
            Capture(), 720, 240, tmp_path, {"qr_hz": 10.0, "record_video": True}
        )

    assert events[-2:] == ["close", "destroy_windows"]
    if stop_at == "quit":
        assert error.value.code == 0
    else:
        assert f"{stop_at} failed" in str(error.value)
    assert not (tmp_path / "latency_report.json").exists()


def test_recording_shutdown_failure_still_stops_process_and_shared_memory():
    events = []

    def fail_stop_recording():
        events.append("stop_recording")
        raise RuntimeError("writer failed")

    capture = latency._RingFeed(startup_timeout=1)
    capture._process = SimpleNamespace(
        stop_recording=fail_stop_recording,
        stop=lambda: events.append("stop_process"),
    )
    capture._shm = SimpleNamespace(shutdown=lambda: events.append("shutdown_shm"))
    capture._recording = True

    with pytest.raises(RuntimeError, match="writer failed"):
        capture.close()

    assert events == ["stop_recording", "stop_process", "shutdown_shm"]
    assert capture._process is None
    assert capture._shm is None
    assert capture._recording is False
