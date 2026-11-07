"""No camera access: verify native capture data, focus checks and cleanup."""

import json

import cv2
import numpy as np
import pytest

from egomimic.scripts import calibrate_uvc_intrinsics as cli


class FakeCapture:
    def __init__(self, width=640, height=480, fps=30, fourcc="MJPG"):
        self.values = {
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*fourcc),
        }
        self.calls = []
        self.releases = 0

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.calls.append((prop, value))
        return True

    def get(self, prop):
        return self.values[prop]

    def release(self):
        self.releases += 1


def test_open_requires_native_mode_and_releases_on_mismatch(monkeypatch):
    cap = FakeCapture(width=320)
    monkeypatch.setattr(cli.cv2, "VideoCapture", lambda *args: cap)
    with pytest.raises(RuntimeError, match="模式未生效"):
        cli.open_camera("/dev/fake", 640, 480, 30, "MJPG")
    assert cap.releases == 1
    assert cap.calls[0][0] == cv2.CAP_PROP_FOURCC


def test_open_matching_mode_hands_ownership_to_caller(monkeypatch):
    cap = FakeCapture()
    monkeypatch.setattr(cli.cv2, "VideoCapture", lambda *args: cap)
    returned, actual = cli.open_camera("/dev/fake", 640, 480, 30, "MJPG")
    assert returned is cap and cap.releases == 0
    assert actual["fourcc"] == "MJPG"


def test_save_original_pixels_rejects_duplicates_and_size_changes(tmp_path):
    frame = np.random.default_rng(12).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    session = {"camera": {"image_size": [64, 48]}, "samples": []}
    detection = {
        "accepted": True,
        "corner_count": 25,
        "coverage_fraction": 0.3,
    }
    assert cli.save_sample(tmp_path, session, frame, "train", detection, {})
    original = cv2.imread(str(tmp_path / session["samples"][0]["file"]))
    np.testing.assert_array_equal(original, frame)
    assert not cli.save_sample(tmp_path, session, frame, "validation", detection, {})
    with pytest.raises(ValueError, match="尺寸"):
        cli.save_sample(tmp_path, session, frame[:40], "train", detection, {})
    assert len(json.loads((tmp_path / "session.json").read_text())["samples"]) == 1


def test_focus_auto_and_mid_session_focus_change_are_rejected():
    controls = {"available": True, "values": {"focus_auto": 1, "focus_absolute": 30}}
    with pytest.raises(ValueError, match="自动对焦"):
        cli.validate_focus(controls)
    controls["values"]["focus_auto"] = 0
    baseline = cli.focus_signature(controls)
    cli.validate_focus(controls, baseline)
    controls["values"]["focus_absolute"] = 35
    with pytest.raises(ValueError, match="变化"):
        cli.validate_focus(controls, baseline)
    with pytest.raises(ValueError, match="无法读回"):
        cli.validate_focus({"available": False, "values": {}}, baseline)


def test_read_controls_preserves_driver_units(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/v4l2-ctl")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=(
                "focus_absolute 0x009a090a (int) : min=0 max=1023 step=1 default=0 value=153\n"
                "focus_automatic_continuous 0x009a090c (bool) : default=1 value=0\n"
            ),
        ),
    )
    assert cli.read_controls("/dev/fake")["values"] == {
        "focus_absolute": 153,
        "focus_automatic_continuous": 0,
    }


def test_capture_init_failure_closes_device_without_writing_session(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DISPLAY", ":fake")
    cap = FakeCapture()
    monkeypatch.setattr(cli, "open_camera", lambda *args: (cap, {}))
    monkeypatch.setattr(
        cli,
        "read_controls",
        lambda *args: {
            "available": True,
            "values": {"focus_auto": 1},
        },
    )
    monkeypatch.setattr(cli.cv2, "destroyAllWindows", lambda: None)
    output = tmp_path / "new_session"
    args = cli.build_parser().parse_args(
        [
            "capture",
            "--device",
            "/dev/fake",
            "--output",
            str(output),
        ]
    )
    with pytest.raises(ValueError, match="自动对焦"):
        cli.capture_session(args)
    assert cap.releases == 1
    assert not output.exists()


def test_output_creation_race_does_not_overwrite_existing_session(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DISPLAY", ":fake")
    cap = FakeCapture()
    output = tmp_path / "new_session"

    def concurrent_creation(*args):
        output.mkdir()
        (output / "session.json").write_text("existing session")
        return cap, {}

    monkeypatch.setattr(cli, "open_camera", concurrent_creation)
    monkeypatch.setattr(
        cli, "read_controls", lambda *args: {"available": True, "values": {}}
    )
    monkeypatch.setattr(cli, "camera_identity", lambda *args: {})
    monkeypatch.setattr(cli.cv2, "destroyAllWindows", lambda: None)
    args = cli.build_parser().parse_args(
        [
            "capture",
            "--device",
            "/dev/fake",
            "--output",
            str(output),
        ]
    )
    with pytest.raises(FileExistsError):
        cli.capture_session(args)
    assert (output / "session.json").read_text() == "existing session"
    assert cap.releases == 1


@pytest.mark.parametrize("exit_by_close", [False, True])
def test_preview_save_quit_preserves_pixels_and_releases_camera(
    monkeypatch, tmp_path, exit_by_close
):
    monkeypatch.setenv("DISPLAY", ":fake")
    detector = cli.CharucoCorners(cli.default_board_config())
    gray = np.full((480, 640), 255, np.uint8)
    gray[40:440, 120:520] = detector.board.generateImage((400, 400))
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    original = frame.copy()
    cap = FakeCapture()
    cap.read = lambda: (True, frame)
    actual = {"width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}
    monkeypatch.setattr(cli, "open_camera", lambda *a: (cap, actual))
    monkeypatch.setattr(cli, "camera_identity", lambda *a: {"device": "/dev/fake"})
    monkeypatch.setattr(
        cli,
        "read_controls",
        lambda *a: {
            "available": True,
            "values": {"focus_auto": 0, "focus_absolute": 30},
        },
    )
    keys = iter([32, 255 if exit_by_close else ord("q")])
    visible = iter([1, 0])
    previews, closed = [], []
    monkeypatch.setattr(cli.cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(cli.cv2, "resizeWindow", lambda *a: None)
    monkeypatch.setattr(cli.cv2, "imshow", lambda name, im: previews.append(im.copy()))
    monkeypatch.setattr(cli.cv2, "waitKey", lambda *a: next(keys))
    monkeypatch.setattr(cli.cv2, "getWindowProperty", lambda *a: next(visible))
    monkeypatch.setattr(cli.cv2, "destroyAllWindows", lambda: closed.append(True))
    output = tmp_path / "session"
    args = cli.build_parser().parse_args(
        [
            "capture",
            "--device",
            "/dev/fake",
            "--output",
            str(output),
            "--width",
            "640",
            "--height",
            "480",
        ]
    )
    cli.capture_session(args)
    manifest = json.loads((output / "session.json").read_text())
    assert manifest["closed_cleanly"]
    assert len(manifest["samples"]) == 1
    saved = cv2.imread(str(output / manifest["samples"][0]["file"]))
    np.testing.assert_array_equal(saved, original)
    np.testing.assert_array_equal(frame, original)
    assert not np.array_equal(previews[0], original)
    assert cap.releases == 1 and closed == [True]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_framerates_rejected_before_hardware(value):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "capture",
                "--device",
                "/dev/fake",
                "--output",
                "/tmp/unused",
                "--fps",
                value,
            ]
        )
