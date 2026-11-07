"""Offline commands and invalid collection arguments never open devices."""

import json
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from egomimic.robot.piper_pico.calibration import build_parser, main
from egomimic.robot.piper_pico.calibration_capture import CaptureLimits


@pytest.mark.parametrize("arm", ["left", "right"])
def test_collect_persists_real_environment_camera_config_without_hardware(
    monkeypatch, tmp_path, arm
):
    from egomimic.robot.piper_pico import calibration
    from egomimic.robot.piper_pico import env as env_module

    # Keep the actual environment constructor and JSON writes. Replace every
    # device constructor and background entry point before invoking the CLI.
    def device(*args, **kwargs):
        return SimpleNamespace(config=kwargs, stop=Mock())

    for name in (
        "CameraProcess",
        "PicoCameraProcess",
        "PiperArmProcess",
        "PicoProcess",
    ):
        monkeypatch.setattr(env_module, name, device)
    monkeypatch.setattr(
        env_module,
        "ReplayBuffer",
        lambda path: SimpleNamespace(root=SimpleNamespace(attrs={})),
    )
    monkeypatch.setattr(env_module.PiperPicoEnv, "start", lambda self: None)
    monkeypatch.setattr(
        calibration, "SharedMemoryManager", lambda: nullcontext(object())
    )
    worker = SimpleNamespace(
        start=Mock(), stop_event=Event(), cancel=Mock(), close=Mock()
    )
    worker_factory = Mock(return_value=worker)
    run_loop = Mock()
    monkeypatch.setattr(calibration, "CaptureWorker", worker_factory)
    monkeypatch.setattr(calibration, "run_calibration", run_loop)
    output = tmp_path / "session"

    main(
        [
            "collect",
            "--arm",
            arm,
            "--headset-ip",
            "unused",
            "--no-preview",
            "--output",
            str(output),
        ]
    )

    camera_config = json.loads(
        (output / "runtime" / "videos" / "camera_config.json").read_text()
    )
    session = json.loads((output / "session.json").read_text())
    assert len(camera_config) == 1
    assert camera_config[0]["eye"] == "left"
    assert camera_config[0]["width"] == 1080
    assert camera_config[0]["height"] == 810
    assert type(camera_config[0]["width"]) is int
    assert type(camera_config[0]["height"]) is int
    assert session["camera_stream"]["width"] == camera_config[0]["width"]
    assert session["camera_stream"]["height"] == camera_config[0]["height"]
    assert session["arm"] == arm
    run_loop.assert_called_once()
    constructed_env = run_loop.call_args.args[0]
    assert type(constructed_env) is env_module.PiperPicoEnv
    assert worker_factory.call_args.args[0] is constructed_env.cameras[0]
    assert (
        worker_factory.call_args.args[1]
        is constructed_env.arms[0 if arm == "left" else 1]
    )
    worker.start.assert_called_once_with()
    assert worker.stop_event.is_set()
    worker.close.assert_called_once_with()


@pytest.mark.parametrize(
    "extra",
    [
        ["--frequency", "nan"],
        ["--left-can", "can0", "--right-can", "can0"],
        ["--pico-camera-latency", "1"],
        ["--stable-seconds", "0.9", "--post-stable-seconds", "0.3"],
    ],
)
def test_invalid_collection_arguments_fail_before_device_construction(
    monkeypatch, tmp_path, extra
):
    def forbidden(*args, **kwargs):
        pytest.fail("No hardware should be constructed during validation")

    monkeypatch.setattr("egomimic.robot.piper_pico.env.PiperPicoEnv", forbidden)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "collect",
                "--arm",
                "left",
                "--headset-ip",
                "unused",
                "--no-preview",
                "--output",
                str(tmp_path / "session"),
                *extra,
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "session").exists()


def test_check_image_writes_diagnostic_without_hardware(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline command must not construct a robot environment")

    monkeypatch.setattr("egomimic.robot.piper_pico.env.PiperPicoEnv", forbidden)
    image = tmp_path / "blank.png"
    cv2.imwrite(str(image), np.full((810, 1080, 3), 255, np.uint8))
    output = tmp_path / "check"
    main(["check-image", "--image", str(image), "--output", str(output)])
    report = json.loads((output / "detection.json").read_text())
    assert report["detection"]["reason"] == "no_board"
    assert not report["detection"]["accepted"]
    assert report["board"]["square_length_m"] == 0.019
    assert (output / "overlay.jpg").is_file()


def test_automatic_output_and_left_optical_collection_defaults():
    args = build_parser().parse_args(
        ["collect", "--arm", "right", "--headset-ip", "unused"]
    )
    assert args.output is None
    assert args.frequency == 30
    assert args.pico_camera_latency == 0.1
    CaptureLimits().validate()
