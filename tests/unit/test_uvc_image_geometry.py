"""Independent Brown5 projection and OpenCV image oracles; no hardware access."""

import copy
import hashlib
import json

import cv2
import numpy as np
import pytest

from egomimic.robot.piper_pico.image_geometry import (
    RgbImageProcessor,
    camera_image_geometry,
    make_image_geometry,
    validate_image_geometry,
)

K = np.array([[355.0, 0, 319.3], [0, 377.0, 237.8], [0, 0, 1]])
D = np.array([-0.19, 0.05, 0.001, -0.002, 0.009])


def rehash(contract):
    """Re-sign deliberately inconsistent metadata to exercise semantic checks."""
    data = {key: value for key, value in contract.items() if key != "sha256"}
    contract["sha256"] = hashlib.sha256(
        json.dumps(data, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return contract


def raw_calibration(size=(640, 480)):
    return {
        "schema": "uvc_intrinsics_calibration_v1",
        "camera_model": "opencv_pinhole_brown5",
        "image_size": list(size),
        "K": K.tolist(),
        "D": D.tolist(),
        "preprocessing": {"crop": None, "resize": None, "mirror": False},
        "camera": {"name": "front_1", "ID_SERIAL": "fixture-front"},
    }


@pytest.mark.parametrize("output_size", [(320, 240), (320, 180), (800, 600)])
def test_brown5_raw_points_rectify_to_scaled_pinhole_without_changing_optical_frame(
    output_size,
):
    contract = make_image_geometry((640, 480), output_size, K=K, D=D)
    # Brown-Conrady is evaluated independently rather than calling projectPoints.
    xy = np.array(
        [
            [-0.32, -0.24],
            [0.3, -0.22],
            [-0.28, 0.27],
            [0.29, 0.26],
            [0, 0],
            [0.16, -0.04],
        ]
    )
    x, y = xy.T
    radius2 = x * x + y * y
    k1, k2, p1, p2, k3 = D
    radial = 1 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
    distorted = (
        np.column_stack(
            [
                x * radial + 2 * p1 * x * y + p2 * (radius2 + 2 * x * x),
                y * radial + p1 * (radius2 + 2 * y * y) + 2 * p2 * x * y,
                np.ones(len(x)),
            ]
        )
        @ K.T
    )
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (640, 480), 0.0, (640, 480))
    np.testing.assert_allclose(contract["K_rectified"], new_K, atol=1e-12, rtol=0)
    rectified = cv2.undistortPoints(
        distorted[:, :2].reshape(-1, 1, 2),
        K,
        D,
        P=new_K,
    ).reshape(-1, 2)
    rectified *= np.asarray(output_size) / [640, 480]
    pinhole = (
        np.column_stack([xy, np.ones(len(x))]) @ np.asarray(contract["K_output"]).T
    )
    np.testing.assert_allclose(rectified, pinhole[:, :2], atol=2e-6, rtol=0)
    assert contract["valid_roi"] == list(roi)
    assert contract["roi_cropped"] is False
    assert contract["rectification_rotation"] == "identity_optical_frame_unchanged"
    assert contract["D_output"] == [0.0] * 5


def patterned_rgb(width, height):
    yy, xx = np.mgrid[:height, :width]
    return np.stack(
        [
            ((xx // 6 + yy // 8) % 2) * 210 + 20,
            (xx * 3 + yy * 2) % 256,
            (xx * 7 - yy * 5) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)


def independent_brown_warp(raw, intrinsic, distortion, new_K):
    """Build inverse image coordinates from the Brown equations, without map APIs."""
    yy, xx = np.mgrid[: raw.shape[0], : raw.shape[1]]
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    rays = pixels @ np.linalg.inv(new_K).T
    x, y = rays[..., 0], rays[..., 1]
    radius2 = x * x + y * y
    k1, k2, p1, p2, k3 = distortion
    radial = 1 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
    distorted = (
        np.stack(
            [
                x * radial + 2 * p1 * x * y + p2 * (radius2 + 2 * x * x),
                y * radial + p1 * (radius2 + 2 * y * y) + 2 * p2 * x * y,
                np.ones_like(x),
            ],
            axis=-1,
        )
        @ intrinsic.T
    )
    return cv2.remap(
        raw,
        distorted[..., 0].astype(np.float32),
        distorted[..., 1].astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def test_pixels_match_independent_opencv_undistort_then_area_resize_and_never_double_correct():
    source, target = (128, 96), (80, 54)
    intrinsic = np.array([[95.0, 0, 58.4], [0, 103.0, 45.2], [0, 0, 1]])
    distortion = np.array([-0.26, 0.09, 0.003, -0.002, 0.0])
    raw = patterned_rgb(*source)
    contract = make_image_geometry(source, target, K=intrinsic, D=distortion)
    processor = RgbImageProcessor(contract)
    # undistort is an independent API from the processor's float remap maps.
    oracle = cv2.undistort(
        raw, intrinsic, distortion, None, np.asarray(contract["K_rectified"])
    )
    oracle = cv2.resize(oracle, target, interpolation=cv2.INTER_AREA)
    actual = processor.apply(raw)
    # OpenCV's undistort uses fixed-point interpolation maps; table rounding may
    # differ by up to one 1/32-pixel interpolation bin on sharp synthetic edges.
    np.testing.assert_allclose(actual, oracle, atol=8, rtol=0)
    assert np.abs(actual.astype(float) - oracle).mean() < 1
    analytic = independent_brown_warp(
        raw, intrinsic, distortion, np.asarray(contract["K_rectified"])
    )
    analytic = cv2.resize(analytic, target, interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(actual, analytic)
    legacy = cv2.resize(raw, target, interpolation=cv2.INTER_AREA)
    assert np.abs(actual.astype(float) - legacy).mean() > 8
    saved = actual.copy()
    # A noncontiguous processed input must also bypass all geometry operations.
    processed = processor.apply(saved[:, ::-1], image_space="processed")
    np.testing.assert_array_equal(processed, saved[:, ::-1])
    assert processed.flags.c_contiguous
    np.testing.assert_array_equal(raw, patterned_rgb(*source))


def test_saved_rectification_matrix_is_used_instead_of_recomputed_from_opencv_defaults():
    source = (128, 96)
    intrinsic = np.array([[95.0, 0, 58.4], [0, 103.0, 45.2], [0, 0, 1]])
    contract = make_image_geometry(source, source, K=intrinsic, D=D)
    contract["K_rectified"][0][2] += 3.25
    contract["K_output"] = copy.deepcopy(contract["K_rectified"])
    rehash(contract)
    raw = patterned_rgb(*source)
    actual = RgbImageProcessor(contract).apply(raw)
    oracle = cv2.undistort(raw, intrinsic, D, None, np.asarray(contract["K_rectified"]))
    np.testing.assert_allclose(actual, oracle, atol=8, rtol=0)
    assert np.abs(actual.astype(float) - oracle).mean() < 1
    analytic = independent_brown_warp(
        raw, intrinsic, D, np.asarray(contract["K_rectified"])
    )
    np.testing.assert_array_equal(actual, analytic)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_size": (0, 480)},
        {"source_size": (640.0, 480)},
        {"output_size": (320, True)},
        {"K": [[1, 2], [3, 4]]},
        {"K": [[355, 1, 320], [0, 377, 240], [0, 0, 1]]},
        {"D": [0, 0, 0, 0]},
        {"D": [0, 0, float("nan"), 0, 0]},
        {"D": [0, 0, float("inf"), 0, 0]},
        {"alpha": float("nan")},
        {"alpha": 1.1},
        {"alpha": True},
        {"K": None},
    ],
)
def test_invalid_geometry_inputs_fail_before_processing(kwargs):
    arguments = {"source_size": (640, 480), "output_size": (320, 240), "K": K, "D": D}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        make_image_geometry(**arguments)


def test_mutated_contract_hash_and_semantically_wrong_scaled_matrix_are_rejected():
    contract = make_image_geometry((640, 480), (320, 240), K=K, D=D)
    contract["K_output"][0][2] += 0.5
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_image_geometry(contract)
    rehash(contract)
    with pytest.raises(ValueError, match="K_output"):
        RgbImageProcessor(contract)


@pytest.mark.parametrize(
    "field,value",
    [
        ("roi_cropped", True),
        ("resize_interpolation", "INTER_LINEAR"),
        ("D_output", D.tolist()),
        ("source_size", [640, -480]),
    ],
)
def test_resigned_but_unsupported_geometry_contract_is_rejected(field, value):
    contract = make_image_geometry((640, 480), (320, 240), K=K, D=D)
    contract[field] = value
    rehash(contract)
    with pytest.raises(ValueError):
        validate_image_geometry(contract)


@pytest.mark.parametrize(
    "input_space,frame",
    [
        ("raw", np.zeros((240, 320, 3), dtype=np.uint8)),
        ("processed", np.zeros((480, 640, 3), dtype=np.uint8)),
        ("raw", np.zeros((480, 640, 3), dtype=np.float32)),
        ("unknown", np.zeros((480, 640, 3), dtype=np.uint8)),
    ],
)
def test_image_space_and_native_dimensions_are_explicit(input_space, frame):
    processor = RgbImageProcessor(make_image_geometry((640, 480), (320, 240), K=K, D=D))
    with pytest.raises(ValueError):
        processor.apply(frame, image_space=input_space)


def test_legacy_resize_does_not_invent_zero_distortion_or_intrinsics():
    contract = make_image_geometry((128, 96), (64, 48))
    raw = patterned_rgb(128, 96)
    np.testing.assert_array_equal(
        RgbImageProcessor(contract).apply(raw),
        cv2.resize(raw, (64, 48), interpolation=cv2.INTER_AREA),
    )
    assert contract["operation"] == "resize"
    assert contract["D_raw"] is contract["D_output"] is contract["K_output"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("K", [[356, 0, 319.3], [0, 377, 237.8], [0, 0, 1]]),
        ("D", [-0.2, 0.05, 0.001, -0.002, 0.009]),
    ],
)
def test_capture_and_batch_raw_calibration_snapshots_cannot_disagree(field, value):
    captured = raw_calibration()
    batch = copy.deepcopy(captured)
    batch[field] = value
    with pytest.raises(ValueError, match="disagree|differs|mismatch"):
        camera_image_geometry(
            "front_1",
            {"calibration": captured},
            {"reference_camera": "front_1", "raw_camera_calibration": batch},
            (640, 480),
            (320, 240),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("camera", {"name": "front_1", "ID_SERIAL": "different-physical-camera"}),
        ("preprocessing", {"crop": None, "resize": None, "mirror": True}),
    ],
)
def test_batch_raw_snapshot_cannot_hide_wrong_camera_or_nonraw_pixels(field, value):
    captured = raw_calibration()
    batch = copy.deepcopy(captured)
    batch[field] = value
    with pytest.raises(ValueError):
        camera_image_geometry(
            "front_1",
            {"calibration": captured},
            {"reference_camera": "front_1", "raw_camera_calibration": batch},
            (640, 480),
            (320, 240),
        )


@pytest.mark.parametrize(
    "capture_spec",
    [
        {"expected_identity": {"ID_SERIAL": "different-physical-camera"}},
        {"controls": {"focus_auto": 0, "focus_absolute": 200}},
    ],
    ids=["wrong-known-serial", "wrong-known-focus"],
)
def test_batch_calibration_fallback_rejects_known_capture_identity_or_focus_mismatch(
    capture_spec,
):
    batch = raw_calibration()
    batch["camera"]["controls"] = {
        "available": True,
        "values": {"focus_automatic_continuous": 0, "focus_absolute": 120},
    }
    # No intrinsic snapshot was attached to the capture itself; the converter
    # must still compare its known settings with the later batch calibration.
    assert "calibration" not in capture_spec
    with pytest.raises(ValueError):
        camera_image_geometry(
            "front_1",
            capture_spec,
            {"reference_camera": "front_1", "raw_camera_calibration": batch},
            (640, 480),
            (320, 240),
        )


def test_batch_calibration_fallback_accepts_matching_known_serial_and_focus_alias():
    batch = raw_calibration()
    batch["camera"]["controls"] = {
        "available": True,
        "values": {
            "focus_automatic_continuous": 0,
            "focus_absolute": 120,
            "zoom_absolute": 100,
            "exposure_absolute": 200,
        },
    }
    spec = {
        "expected_identity": {"ID_SERIAL": "fixture-front"},
        "controls": {
            "focus_auto": 0,
            "focus_absolute": 120,
            "zoom_absolute": 100,
            "exposure_absolute": 300,
        },
    }
    result = camera_image_geometry(
        "front_1",
        spec,
        {"reference_camera": "front_1", "raw_camera_calibration": batch},
        (640, 480),
        (320, 240),
    )
    assert result["operation"] == "undistort_resize"
    np.testing.assert_array_equal(result["K_raw"], K)
    np.testing.assert_array_equal(result["D_raw"], D)
