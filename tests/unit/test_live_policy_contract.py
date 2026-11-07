"""Hermetic live preflight checks using metadata-only synthetic Zarr stores."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from egomimic.robot.piper_pico.camera_config import load_camera_config
from egomimic.robot.piper_pico.image_geometry import camera_image_geometry
from egomimic.robot.piper_pico.live_policy_contract import prepare_live_contract


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_episode(path, metadata, *, version=3):
    filename = "zarr.json" if version == 3 else ".zattrs"
    data = (
        {"node_type": "group", "zarr_format": 3, "attributes": metadata}
        if version == 3
        else metadata
    )
    write_json(path / filename, data)


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "training"
    camera_path = tmp_path / "camera.json"
    calibration_path = tmp_path / "calibration.json"
    K = [[25.0, 0, 16], [0, 25.0, 16], [0, 0, 1.0]]
    raw = {
        "schema": "uvc_intrinsics_calibration_v1",
        "camera_model": "opencv_pinhole_brown5",
        "image_size": [32, 32],
        "K": K,
        "D": [-0.05, 0.01, 0, 0, 0],
        "preprocessing": {"crop": None, "resize": None, "mirror": False},
        "camera": {"ID_SERIAL": "front-unit", "ID_MODEL_ID": "0001"},
    }
    write_json(tmp_path / "raw.json", raw)
    specs = []
    for index, name in enumerate(("front_1", "left_wrist", "right_wrist")):
        specs.append(
            {
                "name": name,
                "type": "uvc",
                "device": f"/dev/test-front-{index}",
                "width": 32,
                "height": 32,
                "fps": 30,
                "latency": 0.066,
                "fourcc": "MJPG",
                "controls": {"brightness": 48, "auto_exposure": 1},
                "expected_identity": {
                    "ID_SERIAL": "front-unit" if index == 0 else f"wrist-unit-{index}",
                    "ID_MODEL_ID": "0001",
                    "ID_PATH": f"usb-port-{index}",
                },
            }
        )
    specs[0]["calibration_file"] = "raw.json"
    config = {"schema": "camera_rig_v1", "cameras": specs}
    write_json(camera_path, config)
    calibration = {
        "calibration_id": "test-rig",
        "reference_camera": "front_1",
        "extrinsics_convention": "T_base_camera",
        "translation_unit": "metre",
        "extrinsics": {side: np.eye(4).tolist() for side in ("left", "right")},
        "intrinsics": {"front_1": K},
        "image_sizes": {"front_1": [32, 32]},
        "distortion_coefficients": {"front_1": raw["D"]},
        "camera_models": {"front_1": raw["camera_model"]},
        "raw_camera_calibration": raw,
    }
    write_json(calibration_path, calibration)
    loaded = load_camera_config(camera_path)
    processing = {
        spec["name"]: camera_image_geometry(
            spec["name"], spec, calibration, [32, 32], [16, 16], alpha=0
        )
        for spec in loaded
    }
    metadata = {
        "schema": "piper_base_v3",
        "embodiment": "piper_bimanual",
        "fps": 30,
        "pose_representation": "xyz_m+quat_wxyz",
        "pose_coordinates": {side: f"piper_{side}_base" for side in ("left", "right")},
        "reference_camera": "front_1",
        "extrinsics_convention": "T_base_camera",
        "extrinsics": copy.deepcopy(calibration["extrinsics"]),
        "calibration_status": "provided",
        "calibration": {"data": copy.deepcopy(calibration), "sha256": "old-file-hash"},
        "intrinsics": {"front_1": processing["front_1"]["K_output"]},
        "image_processing": processing,
        "image_space": "processed",
        "distortion_coefficients": {"front_1": [0.0] * 5},
        "source_capture_config": {"cameras": loaded},
        "time_alignment": {"observation_to_send_s": 0.004},
        "gripper_encoding": {"unit": "normalized_width", "closed_m": 0, "open_m": 0.08},
        "features": {
            f"images.{name}": {"shape": [16, 16, 3], "dtype": "jpeg"}
            for name in processing
        },
    }
    episode = root / "recording" / "episode_000000.zarr"
    write_episode(episode, metadata)
    return {
        "root": root,
        "episode": episode,
        "metadata": metadata,
        "calibration": calibration,
        "calibration_path": calibration_path,
        "config": config,
        "camera_path": camera_path,
    }


def prepare(fixture, root=None):
    return prepare_live_contract(
        root or fixture["root"], fixture["calibration_path"], fixture["camera_path"]
    )


def test_reads_only_metadata_preserves_pixels_and_aggregates_episode_delays(fixture):
    second = copy.deepcopy(fixture["metadata"])
    second["time_alignment"]["observation_to_send_s"] = 0.008
    second["conversion"] = {"source_dataset": "independent-recording"}
    write_episode(
        fixture["root"] / "recording" / "episode_000001.zarr", second, version=2
    )
    write_episode(
        fixture["root"] / "ego" / "episode_000000.zarr",
        {"schema": "pico_ego_world_v1", "embodiment": "human_bimanual"},
    )
    result = prepare(fixture)
    assert result["audit"]["episode_count"] == 2
    assert result["audit"]["ignored_non_piper_episode_count"] == 1
    assert result["audit"]["observation_to_send_s"]["median"] == pytest.approx(0.006)
    assert result["metadata"]["time_alignment"][
        "observation_to_send_s"
    ] == pytest.approx(0.006)
    assert (
        result["metadata"]["image_processing"]
        == fixture["metadata"]["image_processing"]
    )
    assert result["metadata"]["image_processing"]["left_wrist"]["K_raw"] is None
    assert result["audit"]["calibration"]["sha256"] != "old-file-hash"
    assert "does not embed" in result["audit"]["checkpoint_calibration_binding"]
    json.dumps(result["audit"], allow_nan=False)


@pytest.mark.parametrize("identity", [True, False])
def test_identity_and_measured_transform_use_same_contract(fixture, identity):
    if not identity:
        for side, x in (("left", 0.4), ("right", -0.7)):
            fixture["calibration"]["extrinsics"][side][0][3] = x
        fixture["metadata"]["extrinsics"] = copy.deepcopy(
            fixture["calibration"]["extrinsics"]
        )
        fixture["metadata"]["calibration"]["data"] = copy.deepcopy(
            fixture["calibration"]
        )
        write_json(fixture["calibration_path"], fixture["calibration"])
        write_episode(fixture["episode"], fixture["metadata"])
    result = prepare(fixture, fixture["episode"])
    assert result["metadata"]["extrinsics"] == fixture["calibration"]["extrinsics"]


@pytest.mark.parametrize(
    "change", ["extrinsics", "intrinsics", "distortion", "direction", "units"]
)
def test_rejects_different_requested_calibration(fixture, change):
    calibration = fixture["calibration"]
    if change == "extrinsics":
        calibration["extrinsics"]["left"][0][3] = 0.1
    elif change == "intrinsics":
        calibration["intrinsics"]["front_1"][0][0] += 1
    elif change == "distortion":
        calibration["distortion_coefficients"]["front_1"][0] += 0.1
    elif change == "direction":
        calibration["extrinsics_convention"] = "T_camera_base"
    else:
        calibration["translation_unit"] = "millimetre"
    write_json(fixture["calibration_path"], calibration)
    with pytest.raises(ValueError):
        prepare(fixture)


def test_multiple_robot_rigs_require_explicit_selection(fixture):
    second = copy.deepcopy(fixture["metadata"])
    second["extrinsics"]["left"][0][3] = 0.1
    second["calibration"]["data"]["extrinsics"] = copy.deepcopy(second["extrinsics"])
    write_episode(fixture["root"] / "other-rig" / "episode_000000.zarr", second)
    with pytest.raises(
        ValueError, match="multiple geometry/preprocessing.*--training-data"
    ):
        prepare(fixture)
    assert prepare(fixture, fixture["episode"])["audit"]["episode_count"] == 1


@pytest.mark.parametrize(
    "change", ["brightness", "latency", "serial", "remove_identity"]
)
def test_live_capture_changes_cannot_silently_pass(fixture, change):
    spec = fixture["config"]["cameras"][1]
    if change == "brightness":
        spec["controls"]["brightness"] = 32
    elif change == "latency":
        spec["latency"] += 0.05
    elif change == "serial":
        spec["expected_identity"]["ID_SERIAL"] = "another-camera"
    else:
        spec.pop("expected_identity")
    write_json(fixture["camera_path"], fixture["config"])
    with pytest.raises(ValueError, match="left_wrist"):
        prepare(fixture)


def test_port_and_report_changes_do_not_change_geometry(fixture):
    spec = fixture["config"]["cameras"][0]
    spec["device"] = "/dev/different-test-usb-port"
    spec["expected_identity"]["ID_PATH"] = "usb-new-port"
    write_json(fixture["camera_path"], fixture["config"])
    fixture["calibration"]["notes"] = "A report edit does not change geometry"
    fixture["calibration"]["calibration_id"] = "equivalent-report"
    write_json(fixture["calibration_path"], fixture["calibration"])
    result = prepare(fixture)
    assert result["camera_specs"][0]["device"] == spec["device"]
    assert result["audit"]["calibration"]["calibration_id"] == "equivalent-report"


def test_no_piper_and_broken_metadata_fail_clearly(fixture, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No Piper"):
        prepare(fixture, empty)
    (fixture["episode"] / "zarr.json").write_text("not-json")
    with pytest.raises(ValueError, match="Cannot read JSON"):
        prepare(fixture)
