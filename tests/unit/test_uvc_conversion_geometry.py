"""Rectified export keeps raw provenance, timing, and base-frame commands intact."""

import copy
import hashlib
import json

import av
import cv2
import numpy as np
import pytest
import simplejpeg
import zarr
from test_piper_pico_to_zarr import (
    SELECTED,
    TICKS,
    calibration_file,
)
from test_piper_pico_to_zarr import (
    capture as capture_fixture,
)
from test_uvc_image_geometry import independent_brown_warp

from egomimic.rldb.zarr.piper_dataset import validate_piper_metadata
from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.robot.piper_pico.image_geometry import validate_dataset_image_geometry
from egomimic.scripts.piper_pico_to_zarr import Options, convert_dataset

NAMES = ["front_1", "left_wrist", "right_wrist"]
# Register the shared raw capture fixture without duplicating its recording setup.
capture = capture_fixture
SIZES = [(96, 72), (80, 60), (64, 48)]
DISTORTION = [-0.29, 0.075, 0.006, -0.004, 0.008]


def write_patterned_video(path, size):
    width, height = size
    yy, xx = np.mgrid[:height, :width]
    base = np.stack(
        [
            ((xx // 5 + yy // 7) % 2) * 210 + 20,
            (xx * 4 + yy * 2) % 256,
            (xx * 7 - yy * 5) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    with av.open(str(path), mode="w") as output:
        stream = output.add_stream("mpeg4", rate=40)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for index in range(8):
            frame = av.VideoFrame.from_ndarray(
                np.roll(base, index, axis=1), format="rgb24"
            )
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def prepare_uvc_capture(source, tmp_path):
    specs = []
    for index, (name, size) in enumerate(zip(NAMES, SIZES)):
        width, height = size
        K = [
            [width * 0.82, 0, width * 0.48],
            [0, height * 1.07, height * 0.47],
            [0, 0, 1],
        ]
        raw = {
            "schema": "uvc_intrinsics_calibration_v1",
            "camera_model": "opencv_pinhole_brown5",
            "K": K,
            "D": DISTORTION,
            "image_size": list(size),
            "preprocessing": {"crop": None, "resize": None, "mirror": False},
            "camera": {"name": name, "ID_SERIAL": f"fixture-{name}"},
        }
        raw["source_path"] = f"/synthetic-fixture/{name}.json"
        raw["source_sha256"] = hashlib.sha256(
            json.dumps(raw, sort_keys=True).encode()
        ).hexdigest()
        specs.append(
            {
                "name": name,
                "type": "uvc",
                "device": str(index),
                "width": width,
                "height": height,
                "fps": 30,
                "fourcc": "MJPG",
                "latency": 0.01,
                "latency_status": "unmeasured",
                "calibration": raw,
            }
        )
        write_patterned_video(source / f"videos/0/{index}.mp4", size)
    (source / "videos/camera_config.json").write_text(json.dumps(specs))
    path, data = calibration_file(tmp_path / "calibration.json")
    data["intrinsics"]["front_1"] = specs[0]["calibration"]["K"]
    data["image_sizes"]["front_1"] = list(SIZES[0])
    data["distortion_coefficients"] = {"front_1": DISTORTION}
    data["camera_models"] = {"front_1": "opencv_pinhole_brown5"}
    data["image_geometry"] = "raw_distorted"
    data["raw_camera_calibration"] = copy.deepcopy(specs[0]["calibration"])
    path.write_text(json.dumps(data))
    return path, data, specs


def decode_video(path):
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def jpeg_roundtrip(image):
    encoded = simplejpeg.encode_jpeg(
        np.ascontiguousarray(image), quality=ZarrWriter.JPEG_QUALITY, colorspace="RGB"
    )
    return simplejpeg.decode_jpeg(encoded)


def test_nonuniform_uvc_views_are_rectified_and_base_actions_timing_remain_identical(
    capture, tmp_path
):
    calibration_path, raw_data, specs = prepare_uvc_capture(capture, tmp_path)
    video_paths = [capture / f"videos/0/{index}.mp4" for index in range(3)]
    video_hashes = [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in video_paths
    ]
    options = Options(width=48, height=32)
    output = convert_dataset(
        capture,
        tmp_path / "rectified",
        calibration_path=calibration_path,
        options=options,
    )[0]
    store = zarr.open_group(str(output), mode="r")
    metadata = dict(store.attrs)
    validate_piper_metadata(metadata)
    assert metadata["image_space"] == "processed"
    assert metadata["source_capture_config"]["cameras"] == specs
    assert (
        metadata["calibration"]["data"]["raw_camera_calibration"]
        == raw_data["raw_camera_calibration"]
    )
    assert (
        metadata["calibration"]["data"]["distortion_coefficients"]["front_1"]
        == DISTORTION
    )
    for index, spec in enumerate(specs):
        name, size = spec["name"], SIZES[index]
        contract = metadata["image_processing"][name]
        assert contract["source_size"] == list(size)
        assert contract["output_size"] == [48, 32]
        assert contract["operation"] == "undistort_resize"
        assert contract["D_raw"] == DISTORTION
        assert metadata["distortion_coefficients"][name] == [0.0] * 5
        K = np.asarray(spec["calibration"]["K"])
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            K, np.asarray(DISTORTION), size, 0.0, size
        )
        expected_K = np.diag([48 / size[0], 32 / size[1], 1.0]) @ new_K
        np.testing.assert_allclose(
            metadata["intrinsics"][name],
            np.column_stack([expected_K, np.zeros(3)]),
            atol=1e-10,
        )
        raw_frames = decode_video(video_paths[index])
        np.testing.assert_array_equal(store[f"source.{name}.frame_index"][:6], SELECTED)
        for row, frame_index in enumerate(SELECTED):
            raw = raw_frames[frame_index]
            rectified = independent_brown_warp(raw, K, np.asarray(DISTORTION), new_K)
            expected = jpeg_roundtrip(
                cv2.resize(rectified, (48, 32), interpolation=cv2.INTER_AREA)
            )
            actual = simplejpeg.decode_jpeg(store[f"images.{name}"][row : row + 1][0])
            assert actual.shape == (32, 48, 3)
            assert np.abs(actual.astype(float) - expected).mean() < 0.25
            resize_only = jpeg_roundtrip(
                cv2.resize(raw, (48, 32), interpolation=cv2.INTER_AREA)
            )
            assert np.abs(actual.astype(float) - resize_only).mean() > 4
    assert [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in video_paths
    ] == video_hashes

    # Export the same observations with only legacy resize geometry; all non-image
    # arrays must be bit-for-bit identical, including NaNs for unsent commands.
    legacy_data = copy.deepcopy(raw_data)
    for key in (
        "distortion_coefficients",
        "camera_models",
        "image_geometry",
        "raw_camera_calibration",
    ):
        legacy_data.pop(key)
    legacy_path = tmp_path / "legacy-calibration.json"
    legacy_path.write_text(json.dumps(legacy_data))
    legacy_specs = [
        {key: value for key, value in spec.items() if key != "calibration"}
        for spec in specs
    ]
    (capture / "videos/camera_config.json").write_text(json.dumps(legacy_specs))
    legacy_output = convert_dataset(
        capture, tmp_path / "legacy", calibration_path=legacy_path, options=options
    )[0]
    legacy = zarr.open_group(str(legacy_output), mode="r")
    for key, values in store.arrays():
        if not key.startswith("images."):
            np.testing.assert_array_equal(values[:6], legacy[key][:6], err_msg=key)
    assert metadata["extrinsics"] == dict(legacy.attrs)["extrinsics"]
    assert metadata["time_alignment"] == dict(legacy.attrs)["time_alignment"]
    np.testing.assert_array_equal(store["timestamp"][:6], TICKS)
    np.testing.assert_array_equal(store["left.cmd_ee_pose"][:6, 0], 10 + np.arange(6))
    np.testing.assert_allclose(store["left.obs_ee_pose"][:6, 0], TICKS - 1000)


def test_dataset_validator_rejects_processed_intrinsics_mismatch_and_missing_contracts(
    capture, tmp_path
):
    calibration_path, _, _ = prepare_uvc_capture(capture, tmp_path)
    output = convert_dataset(
        capture,
        tmp_path / "rectified",
        calibration_path=calibration_path,
        options=Options(width=48, height=32),
    )[0]
    metadata = dict(zarr.open_group(str(output), mode="r").attrs)
    cases = []
    wrong_K = copy.deepcopy(metadata)
    wrong_K["intrinsics"]["front_1"][0][0] *= 1.02
    cases.append((wrong_K, "K_output"))
    missing_contract = copy.deepcopy(metadata)
    missing_contract.pop("image_processing")
    cases.append((missing_contract, "requires image_processing"))
    snapshot_only = copy.deepcopy(missing_contract)
    snapshot_only["source_capture_config"]["cameras"] = None
    snapshot_only["calibration"]["data"].pop("distortion_coefficients")
    snapshot_only["calibration"]["data"].pop("image_geometry")
    cases.append((snapshot_only, "requires image_processing"))
    wrong_D = copy.deepcopy(metadata)
    wrong_D["distortion_coefficients"]["left_wrist"] = DISTORTION
    cases.append((wrong_D, "zero output distortion"))
    missing_view = copy.deepcopy(metadata)
    missing_view["image_processing"].pop("right_wrist")
    cases.append((missing_view, "exactly the stored camera views"))
    for malformed, message in cases:
        with pytest.raises(ValueError, match=message):
            validate_piper_metadata(malformed)


def test_historical_metadata_without_measured_distortion_remains_supported(
    capture, tmp_path
):
    calibration_path, _ = calibration_file(tmp_path / "legacy.json")
    output = convert_dataset(
        capture,
        tmp_path / "legacy",
        calibration_path=calibration_path,
        options=Options(width=8, height=6),
    )[0]
    metadata = dict(zarr.open_group(str(output), mode="r").attrs)
    assert all(
        value["operation"] == "resize"
        for value in metadata["image_processing"].values()
    )
    assert metadata["distortion_coefficients"] == {}
    for key in ("image_processing", "image_space", "distortion_coefficients"):
        metadata.pop(key)
    validate_dataset_image_geometry(metadata)
    validate_piper_metadata(metadata)
