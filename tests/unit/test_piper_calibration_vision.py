"""Metric directions, validation independence, and native ChArUco detection."""

import json

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_pico.calibration_vision import (
    CharucoEstimator,
    default_board_config,
    default_camera_config,
    solve_session,
)


def transform(rpy, xyz):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    value[:3, 3] = xyz
    return value


def make_session(tmp_path, *, degenerate=False, noise=0.0):
    camera, board = default_camera_config(), default_board_config()
    (tmp_path / "session.json").write_text(
        json.dumps({"camera": camera, "board": board, "arm": "left"})
    )
    estimator = CharucoEstimator(camera, board)
    X = transform([0.75, -0.43, 1.1], [0.34, -0.21, 0.57])
    Y = transform([-0.31, 0.66, -0.45], [0.018, -0.035, 0.091])
    rng = np.random.default_rng(73)
    for i in range(21):
        rpy = [0, 0, i * 0.065] if degenerate else rng.uniform(-0.65, 0.65, 3)
        V = transform(rpy, rng.uniform([-0.09, -0.09, 0.45], [0.02, 0.02, 0.72]))
        G = X @ V @ np.linalg.inv(Y)
        if noise:
            V = V @ transform(rng.normal(0, noise * 2, 3), rng.normal(0, noise, 3))
        points = cv2.projectPoints(
            estimator.object_points,
            cv2.Rodrigues(V[:3, :3])[0],
            V[:3, 3],
            np.asarray(camera["K"]),
            np.asarray(camera["D"]),
        )[0].reshape(-1, 2)
        sample = {
            "arm": "left",
            "split": "train" if i < 16 else "validation",
            "T_base_end": G.tolist(),
            "detection": {
                "accepted": True,
                "T_camera_board": V.tolist(),
                "corner_ids": list(range(25)),
                "corners_px": points.tolist(),
            },
        }
        directory = tmp_path / "samples" / f"{i:06d}"
        directory.mkdir(parents=True)
        (directory / "sample.json").write_text(json.dumps(sample))
    return X, Y


def test_actual_print_and_native_intrinsic_defaults():
    camera, board = default_camera_config(), default_board_config()
    assert camera["image_size"] == [1080, 810]
    assert board["square_length_m"] == pytest.approx(0.019)
    assert board["marker_length_m"] / board["square_length_m"] == pytest.approx(0.7)
    assert board["squares_x"] * board["square_length_m"] == pytest.approx(0.114)
    assert not board["legacy_pattern"]


def test_native_board_detection_and_annotation():
    estimator = CharucoEstimator(default_camera_config(), default_board_config())
    gray = np.full((810, 1080), 255, dtype=np.uint8)
    gray[100:700, 240:840] = estimator.board.generateImage((600, 600))
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    original = rgb.copy()
    result = estimator.detect(rgb)
    assert result["accepted"], result
    assert result["marker_count"] == 18
    assert result["corner_count"] == 25
    assert result["reprojection_rms_px"] < 0.1
    assert np.asarray(result["T_camera_board"])[2, 3] > 0
    json.dumps(result, allow_nan=False)
    annotated = estimator.annotate(rgb, result)
    assert annotated.shape == rgb.shape
    assert not np.array_equal(annotated, original)
    np.testing.assert_array_equal(rgb, original)


def test_detector_rejects_resized_or_missing_board_without_crashing():
    estimator = CharucoEstimator(default_camera_config(), default_board_config())
    assert (
        estimator.detect(np.zeros((405, 540, 3), dtype=np.uint8))["reason"]
        == "image_size_or_format_mismatch"
    )
    assert (
        estimator.detect(np.zeros((810, 1080, 3), dtype=np.uint8))["reason"]
        == "no_board"
    )


def test_print_scale_changes_metric_translation_not_reprojection():
    camera, measured_board = default_camera_config(), default_board_config()
    correct = CharucoEstimator(camera, measured_board)
    nominal_board = {
        **measured_board,
        "square_length_m": 0.020,
        "marker_length_m": 0.014,
    }
    wrong = CharucoEstimator(camera, nominal_board)
    gray = np.full((810, 1080), 255, dtype=np.uint8)
    gray[100:700, 240:840] = correct.board.generateImage((600, 600))
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    measured, nominal = correct.detect(rgb), wrong.detect(rgb)
    assert measured["accepted"] and nominal["accepted"]
    measured_T, nominal_T = (
        np.asarray(measured["T_camera_board"]),
        np.asarray(nominal["T_camera_board"]),
    )
    np.testing.assert_allclose(nominal_T[:3, 3], measured_T[:3, 3] * 20 / 19, atol=1e-7)
    assert nominal["reprojection_rms_px"] == pytest.approx(
        measured["reprojection_rms_px"], abs=2e-5
    )


@pytest.mark.parametrize(
    "change",
    [
        {"eye": "right"},
        {"D": [float("nan")] * 5},
        {"image_size": [1080.5, 810]},
        {"K": np.zeros((3, 3)).tolist()},
    ],
)
def test_invalid_camera_configuration_fails_before_capture(change):
    camera = default_camera_config()
    camera.update(change)
    with pytest.raises(ValueError):
        CharucoEstimator(camera, default_board_config())


def test_solver_recovers_nontrivial_camera_to_base_and_board_offset(tmp_path):
    X, Y = make_session(tmp_path)
    result = solve_session(tmp_path)
    np.testing.assert_allclose(result["T_base_camera"], X, atol=1e-8)
    np.testing.assert_allclose(result["T_camera_base"], np.linalg.inv(X), atol=1e-8)
    np.testing.assert_allclose(result["T_end_board"], Y, atol=1e-8)
    assert result["training"]["translation_error_mm"]["max"] < 1e-6
    assert result["validation"]["reprojection_rms_px"]["max"] < 1e-6
    assert result["validation_available"]
    assert not result["production_approved"]
    assert (tmp_path / "calibration_result.json").is_file()


def test_validation_samples_never_change_fitted_transforms(tmp_path):
    make_session(tmp_path)
    first = solve_session(tmp_path)
    for path in (tmp_path / "samples").glob("*/sample.json"):
        sample = json.loads(path.read_text())
        if sample["split"] == "validation":
            sample["detection"]["T_camera_board"][0][3] += 0.04
            path.write_text(json.dumps(sample))
    second = solve_session(tmp_path)
    np.testing.assert_array_equal(first["T_base_camera"], second["T_base_camera"])
    np.testing.assert_array_equal(first["T_end_board"], second["T_end_board"])
    assert second["validation"]["translation_error_mm"]["median"] > 39


def test_solver_handles_small_measurement_noise(tmp_path):
    X, Y = make_session(tmp_path, noise=0.00015)
    result = solve_session(tmp_path)
    np.testing.assert_allclose(result["T_base_camera"], X, atol=0.0015)
    np.testing.assert_allclose(result["T_end_board"], Y, atol=0.0015)
    assert result["validation"]["translation_error_mm"]["p95"] < 1.0


def test_solver_rejects_single_axis_rotations(tmp_path):
    make_session(tmp_path, degenerate=True)
    with pytest.raises(ValueError, match="Degenerate rotation axes"):
        solve_session(tmp_path)
    assert not (tmp_path / "calibration_result.json").exists()


def test_invalidated_sample_does_not_pollute_fit(tmp_path):
    X, _ = make_session(tmp_path)
    sample_dir = tmp_path / "samples" / "000001"
    data = json.loads((sample_dir / "sample.json").read_text())
    data["T_base_end"][0][3] += 1
    (sample_dir / "sample.json").write_text(json.dumps(data))
    (sample_dir / "invalidated.json").write_text('{"reason":"operator undo"}')
    result = solve_session(tmp_path)
    np.testing.assert_allclose(result["T_base_camera"], X, atol=1e-8)
    assert result["training"]["count"] == 15
    assert result["excluded_samples"] == [
        {"sample_id": "000001", "reason": "invalidated"}
    ]


def test_solver_ignores_uncommitted_pending_sample(tmp_path):
    X, _ = make_session(tmp_path)
    pending = tmp_path / "samples" / ".000022.pending"
    pending.mkdir()
    # A failure between writing metadata and the final directory rename must
    # not make even malformed or incomplete metadata visible to the solver.
    (pending / "sample.json").write_text('{"T_base_end":')
    result = solve_session(tmp_path)
    assert result["training"]["count"] == 16
    assert result["validation"]["count"] == 5
    np.testing.assert_allclose(result["T_base_camera"], X, atol=1e-8)


def test_solver_rejects_bad_transforms_and_wrong_arm(tmp_path):
    make_session(tmp_path)
    sample_path = tmp_path / "samples" / "000000" / "sample.json"
    sample = json.loads(sample_path.read_text())
    sample["arm"] = "right"
    sample_path.write_text(json.dumps(sample))
    with pytest.raises(ValueError, match="arm differs"):
        solve_session(tmp_path)
    sample["arm"] = "left"
    sample["T_base_end"][0][0] += 0.2
    sample_path.write_text(json.dumps(sample))
    with pytest.raises(ValueError, match="orthonormal"):
        solve_session(tmp_path)


def test_no_validation_is_explicitly_incomplete(tmp_path):
    make_session(tmp_path)
    for path in (tmp_path / "samples").glob("*/sample.json"):
        sample = json.loads(path.read_text())
        sample["split"] = "train"
        path.write_text(json.dumps(sample))
    result = solve_session(tmp_path)
    assert result["validation"]["count"] == 0
    assert not result["validation_available"]
    assert not result["production_approved"]
