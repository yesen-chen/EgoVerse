"""Shared native UVC processing, replay image-space, and deployment rig checks."""

import copy
from dataclasses import replace

import cv2
import numpy as np
import pytest
import simplejpeg
from fixtures.piper_pico_training import write_recording

from egomimic.robot.piper_pico.image_geometry import (
    RgbImageProcessor,
    make_image_geometry,
)
from egomimic.robot.piper_pico.policy_inference import (
    CAMERA_KEYS,
    SOURCE_NAMES,
    ObservationFrame,
    PiperObservationAdapter,
    check_capture_config,
)
from egomimic.robot.piper_pico.policy_rollout import OfflineReplayClock
from egomimic.scripts.replay_piper_policy import RecordedObservationProvider


@pytest.fixture
def metadata():
    extrinsics = {side: np.eye(4).tolist() for side in ("left", "right")}
    value = {
        "schema": "piper_base_v3",
        "embodiment": "piper_bimanual",
        "pose_representation": "xyz_m+quat_wxyz",
        "reference_camera": "front_1",
        "extrinsics_convention": "T_base_camera",
        "extrinsics": extrinsics,
        "pose_coordinates": {side: f"piper_{side}_base" for side in ("left", "right")},
        "gripper_encoding": {"unit": "normalized_width", "open_m": 0.08},
        "time_alignment": {"observation_to_send_s": 0.004},
        "features": {
            f"images.{name}": {"shape": [32, 32, 3], "dtype": "jpeg"}
            for name in CAMERA_KEYS
        },
        "calibration_status": "synthetic",
        "calibration": {
            "path": "synthetic-fixture-only",
            "data": {
                "calibration_id": "synthetic-policy-geometry",
                "synthetic": True,
                "reference_camera": "front_1",
                "extrinsics_convention": "T_base_camera",
                "extrinsics": copy.deepcopy(extrinsics),
            },
        },
    }
    raw = {
        "schema": "uvc_intrinsics_calibration_v1",
        "type": "uvc",
        "camera_name": "front_1",
        "camera_model": "opencv_pinhole_brown5",
        "K": [[45.0, 0, 31.5], [0, 46.0, 23.5], [0, 0, 1]],
        "D": [-0.35, 0.09, 0.001, -0.002, -0.01],
        "image_size": [64, 48],
        "preprocessing": {"crop": None, "resize": None, "mirror": False},
        "source_sha256": "a" * 64,
        "source_path": "/original/calibration.json",
        "camera": {
            "ID_SERIAL": "synthetic-front",
            "ID_PATH": "old-usb-port",
            "controls": {
                "available": True,
                "values": {"focus_auto": 0, "focus_absolute": 0},
            },
        },
    }
    specs = [
        {
            "name": name,
            "type": "uvc",
            "device": f"/dev/fake-{name}",
            "width": 64,
            "height": 48,
            "fps": 30,
            "fourcc": "MJPG",
            "latency": 0,
        }
        for name in CAMERA_KEYS
    ]
    specs[0].update(
        calibration=raw,
        controls={"focus_automatic_continuous": 0, "focus_absolute": 0},
        expected_identity={"ID_SERIAL": "synthetic-front", "ID_PATH": "old-usb-port"},
    )
    value["source_capture_config"] = {"cameras": specs}
    value["image_space"] = "processed"
    value["distortion_coefficients"] = {"front_1": [0.0] * 5}
    value["image_processing"] = {
        name: make_image_geometry(
            [64, 48],
            [32, 32],
            K=raw["K"] if name == "front_1" else None,
            D=raw["D"] if name == "front_1" else None,
        )
        for name in CAMERA_KEYS
    }
    value["intrinsics"] = {
        "front_1": np.column_stack(
            [value["image_processing"]["front_1"]["K_output"], np.zeros(3)]
        ).tolist()
    }
    value["calibration"]["data"].update(
        intrinsics={"front_1": raw["K"]},
        image_sizes={"front_1": [64, 48]},
        distortion_coefficients={"front_1": raw["D"]},
        camera_models={"front_1": raw["camera_model"]},
        raw_camera_calibration=raw,
        image_geometry="raw_distorted",
    )
    return value


def native_frame():
    yy, xx = np.indices((48, 64))
    images = {
        name: np.stack(
            [(xx * 3 + index * 10) % 256, (yy * 4) % 256, ((xx + yy) * 2) % 256],
            axis=-1,
        ).astype(np.uint8)
        for index, name in enumerate(CAMERA_KEYS)
    }
    return ObservationFrame(
        1.0, images, np.zeros(14), {name: 1.0 for name in SOURCE_NAMES}
    )


def test_raw_policy_images_equal_shared_conversion_core(metadata):
    adapter = PiperObservationAdapter(metadata, allow_synthetic_calibration=True)
    frame = native_frame()
    batch = adapter.prepare(frame)[7]
    for name, key in CAMERA_KEYS.items():
        expected = RgbImageProcessor(metadata["image_processing"][name]).apply(
            frame.images[name]
        )
        np.testing.assert_array_equal(
            batch[key][0].numpy().transpose(1, 2, 0), expected
        )
    plain_resize = cv2.resize(
        frame.images["front_1"], (32, 32), interpolation=cv2.INTER_AREA
    )
    assert not np.array_equal(
        batch[CAMERA_KEYS["front_1"]][0].numpy().transpose(1, 2, 0), plain_resize
    )


def test_processed_replay_is_not_remapped_and_only_jpeg_error_remains(
    metadata, monkeypatch
):
    frame = native_frame()
    processed = {
        name: RgbImageProcessor(metadata["image_processing"][name]).apply(image)
        for name, image in frame.images.items()
    }
    decoded = {
        name: simplejpeg.decode_jpeg(
            simplejpeg.encode_jpeg(
                image, quality=95, colorspace="RGB", colorsubsampling="444"
            ),
            colorspace="RGB",
        )
        for name, image in processed.items()
    }
    adapter = PiperObservationAdapter(metadata, allow_synthetic_calibration=True)
    monkeypatch.setattr(
        cv2,
        "remap",
        lambda *args, **kwargs: pytest.fail("processed replay must not remap"),
    )
    batch = adapter.prepare(replace(frame, images=decoded, image_space="processed"))[7]
    for name, key in CAMERA_KEYS.items():
        actual = batch[key][0].numpy().transpose(1, 2, 0)
        np.testing.assert_array_equal(actual, decoded[name])
        assert np.abs(actual.astype(float) - processed[name]).mean() < 1.5


@pytest.mark.parametrize("image_space", ["raw", "processed"])
def test_wrong_dimensions_are_not_silently_resized(metadata, image_space):
    adapter = PiperObservationAdapter(metadata, allow_synthetic_calibration=True)
    frame = replace(native_frame(), image_space=image_space)
    wrong = {name: np.zeros((33, 32, 3), np.uint8) for name in CAMERA_KEYS}
    with pytest.raises(ValueError, match="image must be RGB uint8 at size"):
        adapter.prepare(replace(frame, images=wrong))


def test_adapter_rejects_model_shape_drift_and_partial_processing(metadata):
    shapes = {key: (3, 16, 16) for key in CAMERA_KEYS.values()}
    with pytest.raises(ValueError, match="image shape differs"):
        PiperObservationAdapter(
            metadata, image_shapes=shapes, allow_synthetic_calibration=True
        )
    del metadata["image_processing"]["left_wrist"]
    with pytest.raises(ValueError):
        PiperObservationAdapter(metadata, allow_synthetic_calibration=True)


def test_image_space_is_explicit_and_legacy_defaults_remain_raw():
    frame = native_frame()
    assert frame.image_space == "raw"
    with pytest.raises(ValueError, match="image_space"):
        replace(frame, image_space="guess").validate()


def test_live_capture_tolerates_paths_ports_report_hash_and_control_aliases(metadata):
    live = copy.deepcopy(metadata["source_capture_config"]["cameras"])
    live[0]["device"] = "/dev/video99"
    live[0]["calibration_file"] = "/new-location/calibration.json"
    live[0]["calibration"]["source_path"] = "/new-location/calibration.json"
    live[0]["calibration"]["source_sha256"] = "b" * 64
    live[0]["calibration"]["K"][0][0] += 1e-10
    live[0]["calibration"]["camera"]["ID_PATH"] = "new-usb-port"
    live[0]["controls"]["focus_auto"] = live[0]["controls"].pop(
        "focus_automatic_continuous"
    )
    live[0]["expected_identity"]["ID_PATH"] = "new-usb-port"
    live[0]["expected_identity"]["ID_MODEL"] = "extra-verification"
    check_capture_config(metadata, live)


@pytest.mark.parametrize(
    "problem",
    [
        "K",
        "D",
        "size",
        "focus",
        "calibrated_focus",
        "fourcc",
        "identity",
        "missing_calibration",
        "duplicate",
    ],
)
def test_live_capture_rejects_geometry_or_appearance_changes(metadata, problem):
    live = copy.deepcopy(metadata["source_capture_config"]["cameras"])
    if problem == "K":
        live[0]["calibration"]["K"][0][0] += 1
    elif problem == "D":
        live[0]["calibration"]["D"][0] += 0.01
    elif problem == "size":
        live[0]["width"] = 32
    elif problem == "focus":
        live[0]["controls"]["focus_absolute"] = 1
    elif problem == "calibrated_focus":
        live[0]["calibration"]["camera"]["controls"]["values"]["focus_absolute"] = 1
    elif problem == "fourcc":
        live[0]["fourcc"] = "YUYV"
    elif problem == "identity":
        live[0]["expected_identity"]["ID_SERIAL"] = "other"
    elif problem == "missing_calibration":
        del live[0]["calibration"]
    elif problem == "duplicate":
        live.append(copy.deepcopy(live[0]))
    with pytest.raises(ValueError):
        check_capture_config(metadata, live)


def test_live_front_intrinsics_must_match_processing_even_without_recorded_snapshot(
    metadata,
):
    live = copy.deepcopy(metadata["source_capture_config"]["cameras"])
    del metadata["source_capture_config"]["cameras"][0]["calibration"]
    check_capture_config(metadata, live)
    live[0]["calibration"]["D"][0] += 0.1
    with pytest.raises(ValueError, match="image processing contract"):
        check_capture_config(metadata, live)


def test_recorded_provider_marks_images_processed(tmp_path):
    path = write_recording(tmp_path, "piper", "replay")
    provider = RecordedObservationProvider(
        path, OfflineReplayClock(), allow_synthetic_calibration=True
    )
    assert provider.latest().image_space == "processed"
