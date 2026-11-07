"""Complete per-camera modes and raw intrinsics, with no USB/CAN/network access."""

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from egomimic.robot.piper_pico import collect
from egomimic.robot.piper_pico.camera_config import (
    load_camera_config,
    validate_camera_specs,
)


def uvc(name="front_1", device="/dev/fake-front", width=1920, height=1080):
    return {
        "name": name,
        "type": "uvc",
        "device": device,
        "width": width,
        "height": height,
        "fps": 30,
        "fourcc": "MJPG",
        "latency": 0,
        "latency_status": "unmeasured",
    }


@pytest.fixture
def raw_calibration():
    """The current solve_session standalone schema, with synthetic geometry only."""
    return {
        "schema": "uvc_intrinsics_calibration_v1",
        "camera_model": "opencv_pinhole_brown5",
        "type": "uvc",
        "camera_name": "front_1",
        "image_size": [1920, 1080],
        "K": [[1100.0, 0, 960.0], [0, 1090.0, 540.0], [0, 0, 1]],
        "D": [-0.23, 0.07, 0.001, -0.002, -0.01],
        "distortion_parameter_order": ["k1", "k2", "p1", "p2", "k3"],
        "camera": {
            "name": "front_1",
            "image_size": [1920, 1080],
            "device": "/dev/fake-front",
            "ID_SERIAL": "synthetic-fixture",
            "requested_mode": {
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "fourcc": "MJPG",
            },
            "actual_mode": {"width": 1920, "height": 1080, "fps": 30, "fourcc": "MJPG"},
            "controls": {
                "available": True,
                "values": {"focus_automatic_continuous": 0, "focus_absolute": 0},
            },
        },
        "preprocessing": {"crop": None, "resize": None, "mirror": False},
        "production_approved": False,
        "quality_warnings": ["Synthetic fixture: not a physical calibration"],
        "rectified_camera_file": "rectified_camera.json",
        "session_json_sha256": "1" * 64,
    }


@pytest.fixture
def rig_file(tmp_path, raw_calibration):
    intrinsic = tmp_path / "intrinsics" / "calibration.json"
    intrinsic.parent.mkdir()
    intrinsic.write_text(json.dumps(raw_calibration))
    cameras = [
        {
            **uvc(),
            "calibration_file": "intrinsics/calibration.json",
            "controls": {"focus_automatic_continuous": 0, "focus_absolute": 0},
            "expected_identity": {"ID_SERIAL": "synthetic-fixture"},
        },
        uvc("left_wrist", "/dev/fake-left", 640, 480),
        uvc("right_wrist", "/dev/fake-right", 640, 480),
    ]
    path = tmp_path / "rig.json"
    path.write_text(json.dumps({"schema": "camera_rig_v1", "cameras": cameras}))
    return path


def test_mixed_native_sizes_snapshot_current_raw_intrinsics(rig_file, raw_calibration):
    specs = load_camera_config(rig_file)
    assert [(x["width"], x["height"]) for x in specs] == [
        (1920, 1080),
        (640, 480),
        (640, 480),
    ]
    calibration = specs[0]["calibration"]
    for key, value in raw_calibration.items():
        assert calibration[key] == value
    source = rig_file.parent / "intrinsics/calibration.json"
    assert specs[0]["calibration_file"] == calibration["source_path"] == str(source)
    assert (
        calibration["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert calibration["D"][0] != 0  # Native pixels retain nonzero distortion.
    assert "calibration" not in specs[1]
    source.write_text("changed after loading")
    assert validate_camera_specs(specs) == specs  # Snapshot needs no later file read.
    json.dumps(specs, allow_nan=False)


def test_validation_copies_input_and_does_not_claim_measured_latency():
    spec = uvc()
    del spec["latency_status"]
    validated = validate_camera_specs([spec])
    assert "latency_status" not in spec
    assert validated[0]["latency_status"] == "unmeasured"
    validated[0]["width"] = 1
    assert spec["width"] == 1920


def test_rig_defaults_all_three_brightness_and_preserves_per_camera_overrides(rig_file):
    assert [x["controls"]["brightness"] for x in load_camera_config(rig_file)] == [48] * 3
    raw = json.loads(rig_file.read_text())
    raw["cameras"][0]["controls"]["brightness"] = 0
    raw["cameras"][1]["controls"] = {"brightness": 40, "exposure_absolute": 100}
    rig_file.write_text(json.dumps(raw))
    specs = load_camera_config(rig_file)
    assert [x["controls"]["brightness"] for x in specs] == [0, 40, 48]
    assert specs[0]["controls"]["focus_absolute"] == 0
    assert specs[1]["controls"]["exposure_absolute"] == 100
    assert json.loads(rig_file.read_text()) == raw  # Defaults do not rewrite user config.


@pytest.mark.parametrize("value", [48.5, True, "48", None])
def test_brightness_requires_driver_integer(value):
    with pytest.raises(ValueError, match="brightness.*integer"):
        validate_camera_specs([{**uvc(), "controls": {"brightness": value}}])


@pytest.mark.parametrize("option,expected", [([], 48), (["--camera-brightness", "0"], 0),
                                          (["--camera-brightness", "40"], 40)])
def test_legacy_brightness_applies_to_all_uvc_but_never_pico(option, expected):
    args = collect.build_parser().parse_args([
        "--headset-ip", "192.0.2.1", "--camera", "left_wrist=0",
        "--camera", "right_wrist=2", *option,
    ])
    pico, *wrists = collect.camera_specs_from_args(args)
    assert "controls" not in pico
    assert [spec["controls"]["brightness"] for spec in wrists] == [expected] * 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "../front"),
        ("name", ""),
        ("name", "front/name"),
        ("type", "realsense"),
        ("width", 0),
        ("height", 10.5),
        ("width", True),
        ("fps", 0),
        ("fps", float("nan")),
        ("fps", "30"),
        ("latency", -0.01),
        ("latency", float("inf")),
        ("latency_status", "estimated"),
        ("device", ""),
        ("device", -1),
        ("device", False),
        ("fourcc", "MJPEG"),
        ("fourcc", "\nJPG"),
        ("controls", {"focus_absolute": 0.5}),
        ("controls", {"focus_absolute": -1}),
        ("controls", {"focus_automatic_continuous": 2}),
        ("controls", {"focus_auto": 1, "focus_automatic_continuous": 0}),
        ("controls", {"focus;anything": 0}),
        ("expected_identity", {"device": "foo"}),
        ("expected_identity", {"ID_SERIAL": ""}),
    ],
)
def test_invalid_modes_and_controls_fail_without_hardware(field, value):
    with pytest.raises(ValueError):
        validate_camera_specs([{**uvc(), field: value}])


def test_duplicate_names_devices_indices_and_symlinks(tmp_path):
    with pytest.raises(ValueError, match="names must be unique"):
        validate_camera_specs([uvc(), uvc(device="/dev/another")])
    with pytest.raises(ValueError, match="only be opened once"):
        validate_camera_specs([uvc(device="0"), uvc("wrist", "/dev/video0")])
    target = tmp_path / "fake-device"
    target.touch()
    alias = tmp_path / "alias"
    alias.symlink_to(target)
    with pytest.raises(ValueError, match="only be opened once"):
        validate_camera_specs([uvc(device=str(target)), uvc("wrist", str(alias))])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("K", [[0, 0, 0], [0, 1, 0], [0, 0, 1]], "K"),
        ("K", [[1, 0, 0], [0, 1, 0], [0, 1, 1]], "K"),
        ("K", [[1, 0, 0], [0, 1, 0]], "K"),
        ("K", [["100", 0, 0], [0, 1, 0], [0, 0, 1]], "K/D"),
        ("D", [0, 0, 0, 0], "D"),
        ("D", [0, 0, 0, 0, float("nan")], "D"),
        ("D", [False, 0, 0, 0, 0], "K/D"),
        ("image_size", [640, 480], "image_size"),
        ("camera_model", "fisheye", "camera_model"),
        ("schema", "uvc_rectified_camera_v1", "raw"),
        ("distortion_parameter_order", ["k1", "k2", "k3", "p1", "p2"], "order"),
        ("K_new", [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "raw"),
    ],
)
def test_intrinsic_mismatch_is_rejected(raw_calibration, field, value, message):
    raw_calibration[field] = value
    with pytest.raises(ValueError, match=message):
        validate_camera_specs([{**uvc(), "calibration": raw_calibration}])


@pytest.mark.parametrize(
    "change",
    [
        {"crop": [0, 0, 640, 480]},
        {"resize": [1920, 1080]},
        {"mirror": True},
        {"undistort": True},
        {"rotation": 180},
        {"flip_horizontal": True},
    ],
)
def test_preprocessing_cannot_silently_reinterpret_raw_K(raw_calibration, change):
    preprocessing = {**raw_calibration["preprocessing"], **change}
    raw_calibration["preprocessing"] = preprocessing
    with pytest.raises(ValueError, match="preprocessing"):
        validate_camera_specs([{**uvc(), "calibration": raw_calibration}])
    with pytest.raises(ValueError, match="preprocessing"):
        validate_camera_specs([{**uvc(), "preprocessing": preprocessing}])


def test_known_calibrated_controls_and_identity_must_match(raw_calibration):
    spec = {**uvc(), "calibration": raw_calibration, "controls": {"focus_auto": 0}}
    assert validate_camera_specs([spec])[0]["controls"] == {"focus_auto": 0, "brightness": 48}
    spec["controls"]["focus_auto"] = 1
    with pytest.raises(ValueError, match="autofocus"):
        validate_camera_specs([spec])
    spec["controls"] = {"focus_absolute": 10}
    with pytest.raises(ValueError, match="control.*differs"):
        validate_camera_specs([spec])
    spec["controls"] = {}
    spec["expected_identity"] = {"ID_SERIAL": "different-device"}
    with pytest.raises(ValueError, match="identity.*differs"):
        validate_camera_specs([spec])


def test_explicit_usb_port_move_preserves_calibration_history(rig_file):
    intrinsic = rig_file.parent / "intrinsics/calibration.json"
    calibration = json.loads(intrinsic.read_text())
    calibration["camera"]["ID_PATH"] = "old-port"
    intrinsic.write_text(json.dumps(calibration))
    original_bytes = intrinsic.read_bytes()
    rig = json.loads(rig_file.read_text())
    rig["cameras"][0]["device"] = "/dev/new-port"
    rig["cameras"][0]["expected_identity"]["ID_PATH"] = "new-port"
    rig_file.write_text(json.dumps(rig))

    front = load_camera_config(rig_file)[0]
    assert front["expected_identity"]["ID_PATH"] == "new-port"
    assert front["calibration"]["camera"]["ID_PATH"] == "old-port"
    assert intrinsic.read_bytes() == original_bytes


def test_usb_port_move_without_recorded_serial_is_rejected(raw_calibration):
    del raw_calibration["camera"]["ID_SERIAL"]
    raw_calibration["camera"]["ID_PATH"] = "old-port"
    spec = {
        **uvc(),
        "calibration": raw_calibration,
        "expected_identity": {"ID_PATH": "new-port"},
    }
    with pytest.raises(ValueError, match="identity ID_PATH differs"):
        validate_camera_specs([spec])


def test_exposure_changes_do_not_invalidate_geometry(raw_calibration):
    raw_calibration["camera"]["controls"]["values"]["exposure_absolute"] = 100
    spec = {
        **uvc(),
        "calibration": raw_calibration,
        "controls": {"exposure_absolute": 200},
    }
    assert validate_camera_specs([spec])[0]["controls"]["exposure_absolute"] == 200


def test_calibration_cannot_claim_fixed_intrinsics_with_autofocus(raw_calibration):
    raw_calibration["camera"]["controls"]["values"]["focus_automatic_continuous"] = 1
    with pytest.raises(ValueError, match="autofocus"):
        validate_camera_specs([{**uvc(), "calibration": raw_calibration}])


def test_calibration_native_mode_metadata_cannot_contradict_raw_size(raw_calibration):
    raw_calibration["camera"]["actual_mode"]["width"] = 640
    with pytest.raises(ValueError, match="contradicts"):
        validate_camera_specs([{**uvc(), "calibration": raw_calibration}])


def test_relative_calibration_path_missing_and_invalid_files(rig_file):
    path = rig_file.parent / "intrinsics/calibration.json"
    path.unlink()
    with pytest.raises(ValueError, match="cannot read calibration_file"):
        load_camera_config(rig_file)
    path.write_text("not json")
    with pytest.raises(ValueError, match="cannot read calibration_file"):
        load_camera_config(rig_file)
    with pytest.raises(ValueError, match="cannot read camera config"):
        load_camera_config(rig_file.parent / "missing.json")


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"schema": "camera_rig_v2", "cameras": []},
        {"schema": "camera_rig_v1", "cameras": []},
        {"schema": "camera_rig_v1", "cameras": [{}]},
        {"schema": "camera_rig_v1", "cameras": [{**uvc(), "calibration": {}}]},
    ],
)
def test_incomplete_or_unsupported_config_fails(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_camera_config(path)


def test_config_uses_each_mode_without_legacy_scalar_override(rig_file):
    args = collect.build_parser().parse_args(["--camera-config", str(rig_file)])
    specs = collect.camera_specs_from_args(args)
    assert specs[0]["width"] == 1920 and specs[1]["width"] == 640
    assert [x["latency"] for x in specs] == [0, 0, 0]
    assert args.width == 640 and args.camera_latency == 0.08  # Not applied to config.


@pytest.mark.parametrize(
    "legacy",
    [
        ["--camera", "front=/dev/fake"],
        ["--headset-ip", "192.0.2.1"],
        ["--width", "1920"],
        ["--height", "1080"],
        ["--camera-fps", "60"],
        ["--camera-latency", "0"],
        ["--camera-brightness", "40"],
        ["--fourcc", "YUYV"],
        ["--pc-ip", "192.0.2.2"],
        ["--pico-camera-name", "pico_front"],
        ["--pico-camera-latency", "0"],
    ],
)
def test_cli_config_conflicts_fail_before_allocating_hardware(
    rig_file, legacy, monkeypatch
):
    manager = Mock(side_effect=AssertionError("must not allocate hardware"))
    monkeypatch.setattr(collect, "SharedMemoryManager", manager)
    with pytest.raises(SystemExit):
        collect.main(["--camera-config", str(rig_file), *legacy])
    manager.assert_not_called()


def test_legacy_cli_keeps_original_mode_and_latency_defaults():
    args = collect.build_parser().parse_args(
        [
            "--headset-ip",
            "192.0.2.1",
            "--camera",
            "left_wrist=0",
        ]
    )
    pico, wrist = collect.camera_specs_from_args(args)
    assert (pico["width"], pico["height"], pico["fps"], pico["latency"]) == (
        1080,
        810,
        60,
        0.1,
    )
    assert (wrist["width"], wrist["height"], wrist["fps"], wrist["latency"]) == (
        640,
        480,
        30,
        0.08,
    )
    assert wrist["device"] == "0"
    assert pico["latency_status"] == wrist["latency_status"] == "unmeasured"


def test_config_supports_pico_and_optional_uncalibrated_wrists(tmp_path):
    pico = {
        "type": "pico",
        "name": "front_1",
        "headset_ip": "192.0.2.1",
        "width": 1080,
        "height": 810,
        "fps": 60,
        "latency": 0,
    }
    path = tmp_path / "pico.json"
    path.write_text(
        json.dumps(
            {"schema": "camera_rig_v1", "cameras": [pico, uvc("wrist", "1", 640, 480)]}
        )
    )
    assert [spec["type"] for spec in load_camera_config(path)] == ["pico", "uvc"]
    other = copy.deepcopy(pico)
    other["name"] = "another_pico"
    with pytest.raises(ValueError, match="one PICO"):
        validate_camera_specs([pico, other])


def test_example_never_claims_a_latency_measurement():
    path = Path(collect.__file__).parent / "configs/uvc_three_camera.example.json"
    assert all(
        spec["latency_status"] == "unmeasured" for spec in load_camera_config(path)
    )
