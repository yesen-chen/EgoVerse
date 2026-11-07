"""Robot-derived SDK frame overlay, projection direction and visibility checks."""

import json

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_pico.calibration_overlay import render_verification


def transform(rpy=(0, 0, 0), xyz=(0, 0, 0)):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    value[:3, 3] = xyz
    return value


def fixture():
    camera_end = transform((0.26, -0.34, 0.61), (0.04, -0.025, 0.52))
    end_board = transform((-0.18, 0.1, -0.23), (-0.045, -0.025, 0.065))
    points = np.array([[0, 0, 0], [0.019, 0, 0], [0, 0.019, 0], [0.019, 0.019, 0]])
    K = np.array([[510.0, 0, 319.5], [0, 530.0, 239.5], [0, 0, 1]])
    D = np.array([0.13, -0.07, 0.003, -0.002, 0.01])
    camera_board = camera_end @ end_board
    observed = cv2.projectPoints(
        points,
        cv2.Rodrigues(camera_board[:3, :3])[0],
        camera_board[:3, 3],
        K,
        D,
    )[0].reshape(-1, 2)
    return {
        "rgb": np.full((480, 640, 3), 180, dtype=np.uint8),
        "T_camera_end": camera_end,
        "T_end_board": end_board,
        "object_points": points,
        "corner_ids": [0, 1, 2, 3],
        "corners_px": observed,
        "K": K,
        "D": D,
        "sample_id": "000012",
        "arm": "left",
        "split": "validation",
    }


def test_sdk_origin_has_no_offset_and_axes_follow_end_rotation():
    data = fixture()
    original = data["rgb"].copy()
    axes, comparison, metrics = render_verification(**data)
    T = data["T_camera_end"]
    expected_camera = np.vstack([T[:3, 3], T[:3, 3] + 0.04 * T[:3, :3].T])
    np.testing.assert_allclose(metrics["axes_camera_xyz"], expected_camera, atol=1e-14)
    expected_pixels = cv2.projectPoints(
        expected_camera, np.zeros(3), np.zeros(3), data["K"], data["D"]
    )[0].reshape(-1, 2)
    np.testing.assert_allclose(metrics["axes_pixels_xy"], expected_pixels, atol=1e-12)
    assert metrics["frame"] == "SDK_END"
    assert metrics["tcp_offset_applied"] is False
    assert metrics["reprojection_rms_px"] < 1e-10
    assert metrics["split"] == "validation"
    assert axes.shape == comparison.shape == (612, 640, 3)
    np.testing.assert_array_equal(data["rgb"], original)
    np.testing.assert_array_equal(axes[:40], original[:40])
    assert not np.array_equal(axes, comparison)
    json.dumps(metrics, allow_nan=False)


def test_nontrivial_camera_base_inverse_gives_correct_sdk_projection():
    data = fixture()
    base_camera = transform((0.7, -0.6, 1.0), (0.4, -0.25, 0.7))
    base_end = base_camera @ data["T_camera_end"]
    data["T_camera_end"] = np.linalg.inv(base_camera) @ base_end
    _, _, correct = render_verification(**data)
    np.testing.assert_allclose(
        correct["T_camera_end"], fixture()["T_camera_end"], atol=1e-14
    )
    assert correct["reprojection_rms_px"] < 1e-10
    data["T_camera_end"] = base_camera @ base_end
    _, _, wrong = render_verification(**data)
    assert wrong["reprojection_rms_px"] is None or wrong["reprojection_rms_px"] > 30


def test_prediction_uses_robot_transform_and_distortion_not_image_pnp():
    data = fixture()
    _, _, correct = render_verification(**data)
    data["T_camera_end"] = data["T_camera_end"].copy()
    data["T_camera_end"][0, 3] += 0.025
    _, _, perturbed = render_verification(**data)
    assert perturbed["reprojection_rms_px"] > 20
    np.testing.assert_array_equal(perturbed["detected_corners_px"], data["corners_px"])
    data = fixture()
    data["D"] = np.zeros(5)
    _, _, without_distortion = render_verification(**data)
    assert (
        without_distortion["reprojection_rms_px"]
        > correct["reprojection_rms_px"] + 0.01
    )
    assert (
        np.max(
            np.abs(
                np.asarray(without_distortion["axes_pixels_xy"])
                - correct["axes_pixels_xy"]
            )
        )
        > 0.01
    )


def test_corner_ids_select_correct_object_points_and_observation_order():
    data = fixture()
    data["corner_ids"] = [3, 1]
    data["corners_px"] = data["corners_px"][[3, 1]]
    _, _, metrics = render_verification(**data)
    assert metrics["corner_ids"] == [3, 1]
    assert len(metrics["predicted_corners_px"]) == 2
    assert metrics["reprojection_rms_px"] < 1e-10


def test_behind_camera_predictions_are_null_and_never_mirrored():
    data = fixture()
    data["T_camera_end"] = transform(xyz=(0, 0, -0.5))
    axes, comparison, metrics = render_verification(**data)
    assert metrics["origin_status"] == "behind_camera"
    assert metrics["axes_pixels_xy"] == [None] * 4
    assert metrics["predicted_corners_px"] == [None] * 4
    assert metrics["reprojection_rms_px"] is None
    assert metrics["axes_segments_drawn"] == [False] * 3
    np.testing.assert_array_equal(axes[:480], data["rgb"])
    # Observed corners remain visible even when the prediction is impossible.
    assert not np.array_equal(comparison[:480], data["rgb"])
    json.dumps(metrics, allow_nan=False)


def test_partial_invalid_board_does_not_report_subset_rms():
    data = fixture()
    data["T_camera_end"] = transform(xyz=(0, 0, 0.01))
    data["T_end_board"] = transform((0, np.pi / 2, 0))
    _, _, metrics = render_verification(**data)
    assert metrics["predicted_corners_px"][0] is not None
    assert metrics["predicted_corners_px"][1] is None
    assert metrics["reprojection_rms_px"] is None
    assert not metrics["all_corner_projections_valid"]


def test_offscreen_origin_is_not_replaced_by_fake_boundary_dot():
    data = fixture()
    data["T_camera_end"] = transform(xyz=(2, 0, 0.5))
    axes, _, metrics = render_verification(**data)
    assert metrics["origin_status"] == "outside_image"
    assert metrics["axes_pixels_xy"][0][0] > data["rgb"].shape[1]
    np.testing.assert_array_equal(axes[:480], data["rgb"])


def test_crossing_axis_is_clipped_without_moving_origin():
    data = fixture()
    # At fx=510 and z=.5, origin x=-5 px; +X end is inside the image.
    data["T_camera_end"] = transform(xyz=((-5 - 319.5) * 0.5 / 510, 0, 0.5))
    data["D"] = np.zeros(5)
    axes, _, metrics = render_verification(**data)
    assert metrics["axes_pixels_xy"][0][0] == pytest.approx(-5)
    assert metrics["origin_status"] == "outside_image"
    assert metrics["axes_segments_drawn"][0]
    assert np.any(axes[239:242, :35, 0] != data["rgb"][239:242, :35, 0])
    assert not metrics["axes_points_in_image"][0]


def test_huge_projections_do_not_overflow_drawing_or_json():
    data = fixture()
    data["T_camera_end"] = transform(xyz=(1e100, 0, 1))
    data["D"] = np.zeros(5)
    axes, _, metrics = render_verification(**data)
    assert metrics["origin_status"] in ("outside_image", "unprojectable")
    np.testing.assert_array_equal(axes[:480], data["rgb"])
    json.dumps(metrics, allow_nan=False)


def test_zero_corner_overlay_still_visualizes_sdk_frame():
    data = fixture()
    data["corner_ids"] = []
    data["corners_px"] = []
    _, _, metrics = render_verification(**data)
    assert metrics["reprojection_rms_px"] is None
    assert metrics["corner_ids"] == []
    assert metrics["origin_status"] == "in_image"
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("key", ["T_camera_end", "T_end_board", "K", "D"])
def test_nonfinite_camera_and_transform_inputs_fail_clearly(key):
    data = fixture()
    data[key] = np.array(data[key], copy=True)
    data[key].flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        render_verification(**data)


@pytest.mark.parametrize("axis_length", [0, -0.04, float("nan"), float("inf")])
def test_invalid_axis_length_is_rejected(axis_length):
    with pytest.raises(ValueError, match="axis_length_m"):
        render_verification(**fixture(), axis_length_m=axis_length)


@pytest.mark.parametrize("ids", [[-1], [4], [0, 0], [0.5], [float("nan")]])
def test_invalid_corner_ids_are_rejected(ids):
    data = fixture()
    data["corner_ids"] = ids
    with pytest.raises(ValueError, match="corner_ids"):
        render_verification(**data)
