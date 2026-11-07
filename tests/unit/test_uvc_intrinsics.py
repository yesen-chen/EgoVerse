"""CPU-only UVC geometry, source-image auditing, and real ChArUco detection."""

import copy
import json

import cv2
import numpy as np
import pytest

from egomimic.robot.piper_pico import uvc_intrinsics as intrinsics

SIZE = (1280, 720)
KNOWN_K = np.array([[890.0, 0, 631.0], [0, 870.0, 357.0], [0, 0, 1.0]])
KNOWN_D = np.array([-0.22, 0.11, 0.0015, -0.002, -0.035])


@pytest.fixture(scope="module")
def projections():
    """Use broad known poses and nonzero radial and tangential distortion."""
    board = intrinsics.default_board_config()
    detector = intrinsics.CharucoCorners(board)
    obj = detector.object_points
    rng = np.random.default_rng(271)
    views = []
    while len(views) < 30:
        rvec = rng.uniform([-0.65, -0.65, -0.5], [0.65, 0.65, 0.5])
        R = cv2.Rodrigues(rvec)[0]
        depth = rng.uniform(0.20, 0.43)
        center_px = rng.uniform([160, 130], [1120, 590])
        center = np.array(
            [
                (center_px[0] - KNOWN_K[0, 2]) * depth / KNOWN_K[0, 0],
                (center_px[1] - KNOWN_K[1, 2]) * depth / KNOWN_K[1, 1],
                depth,
            ]
        )
        tvec = center - R @ obj.mean(axis=0)
        points = cv2.projectPoints(obj, rvec, tvec, KNOWN_K, KNOWN_D)[0].reshape(-1, 2)
        if np.any(points < 5) or np.any(points >= np.array(SIZE) - 5):
            continue
        views.append(
            {
                "ids": list(range(len(obj))),
                "corners_px": points.tolist(),
                "file": f"{len(views):06d}.png",
            }
        )
    return board, views[:24], views[24:]


def test_nonzero_brown5_and_intrinsics_recovered_from_training(projections):
    board, train, val = projections
    result = intrinsics.fit_observations(train, val, board, SIZE)
    np.testing.assert_allclose(result["K"], KNOWN_K, atol=0.01, rtol=0)
    np.testing.assert_allclose(result["D"], KNOWN_D, atol=0.0002, rtol=0)
    assert result["train"]["rms_px"] < 0.0001
    assert result["validation"]["rms_px"] < 0.0001
    assert result["distortion_parameter_order"] == ["k1", "k2", "p1", "p2", "k3"]
    assert len(result["parameter_std"]) == 9
    assert result["training_pose_diversity"]["max_board_normal_separation_deg"] > 30
    assert not result["production_approved"]
    json.dumps(result, allow_nan=False)


def test_validation_changes_never_change_training_intrinsics(projections):
    board, train, val = projections
    first = intrinsics.fit_observations(train, val, board, SIZE)
    changed = copy.deepcopy(val)
    rng = np.random.default_rng(222)
    for view in changed:
        points = np.asarray(view["corners_px"])
        view["corners_px"] = (points + rng.normal(0, 1.2, points.shape)).tolist()
    second = intrinsics.fit_observations(train, changed, board, SIZE)
    # OpenCV's parallel linear algebra may differ at machine precision.
    np.testing.assert_allclose(first["K"], second["K"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(first["D"], second["D"], rtol=0, atol=1e-8)
    assert first["train"]["rms_px"] == pytest.approx(
        second["train"]["rms_px"], abs=1e-8
    )
    assert second["validation"]["rms_px"] > 1
    assert not second["validation"]["rms_within_threshold"]
    assert any("validation" in warning for warning in second["quality_warnings"])
    assert (
        "not independent geometric ground truth"
        in second["validation"]["interpretation"]
    )


def test_actual_charuco_detection_without_camera_intrinsics():
    detector = intrinsics.CharucoCorners(intrinsics.default_board_config())
    gray = np.full((720, 1280), 255, dtype=np.uint8)
    gray[60:660, 340:940] = detector.board.generateImage((600, 600))
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    original = bgr.copy()
    result = detector.detect(bgr)
    assert result["accepted"], result
    assert result["corner_count"] == 25
    assert sorted(result["ids"]) == list(range(25))
    assert result["coverage_fraction"] == pytest.approx(
        400 * 400 / (1280 * 720), abs=0.001
    )
    assert len(result["corners_px"]) == 25
    np.testing.assert_array_equal(original, bgr)


def test_missing_board_and_wrong_format_are_rejected():
    detector = intrinsics.CharucoCorners(intrinsics.default_board_config())
    assert detector.detect(np.zeros((720, 1280, 3), np.uint8))["reason"] == "no_board"
    assert (
        detector.detect(np.zeros((720, 1280, 3), float))["reason"]
        == "image_format_mismatch"
    )


def test_insufficient_training_or_validation_views_are_rejected(projections):
    board, train, val = projections
    for training, validation in ((train[:14], val), (train, val[:4])):
        with pytest.raises(ValueError, match="Insufficient accepted samples"):
            intrinsics.fit_observations(training, validation, board, SIZE)


def test_weak_pose_diversity_is_flagged_even_with_low_rms():
    board = intrinsics.default_board_config()
    obj = intrinsics.CharucoCorners(board).object_points
    observations = []
    for index in range(20):
        rvec = np.array([0.08 + index * 0.001, -0.08 + index * 0.002, index * 0.001])
        tvec = np.array([-0.056 + index * 0.0002, -0.056, 0.35 + index * 0.0003])
        points = cv2.projectPoints(obj, rvec, tvec, KNOWN_K, KNOWN_D)[0].reshape(-1, 2)
        observations.append({"ids": list(range(25)), "corners_px": points.tolist()})
    result = intrinsics.fit_observations(
        observations[:15], observations[15:], board, SIZE
    )
    assert result["train"]["rms_px"] < 0.01
    assert any("tilts lack diversity" in w for w in result["quality_warnings"])
    assert any("distances lack diversity" in w for w in result["quality_warnings"])
    assert not result["production_approved"]


@pytest.mark.parametrize("count", [3, 8])
def test_too_few_or_collinear_corners_rejected(count):
    detector = intrinsics.CharucoCorners(intrinsics.default_board_config())

    class StubDetection:
        def detectBoard(self, gray):
            corners = np.column_stack([np.arange(count) * 10 + 20, np.full(count, 50)])
            return (
                corners.astype(np.float32).reshape(-1, 1, 2),
                np.arange(count).reshape(-1, 1),
                None,
                None,
            )

    detector.detector = StubDetection()
    result = detector.detect(np.zeros((720, 1280, 3), np.uint8))
    assert not result["accepted"]
    assert result["reason"] == (
        "insufficient_corners" if count < 8 else "collinear_corners"
    )


def make_session(tmp_path, entries):
    """Write native PNGs and a minimal capture manifest, without opening a device."""
    tmp_path.mkdir(exist_ok=True)
    samples = []
    for index, (split, image) in enumerate(entries):
        relative = f"{split}/{index:06d}.png"
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        assert cv2.imwrite(str(path), image)
        samples.append(
            {
                "file": relative,
                "split": split,
                "detection": {"accepted": True, "corners_px": [[-1, -1]]},
            }
        )
    metadata = {
        "schema": "uvc_intrinsics_capture_v1",
        "board": intrinsics.default_board_config(),
        "camera": {
            "name": "front_1",
            "device": "/dev/test-camera-never-opened",
            "image_size": list(SIZE),
            "requested_mode": {"width": SIZE[0], "height": SIZE[1], "fps": 30},
            "actual_mode": {"width": SIZE[0], "height": SIZE[1], "fps": 30},
            "focus_note": "test fixture",
            "controls": {"focus_auto": 0},
        },
        "preprocessing": {"crop": None, "resize": None, "mirror": False},
        "samples": samples,
    }
    (tmp_path / "session.json").write_text(json.dumps(metadata))
    return metadata


def test_mixed_image_dimensions_abort_without_writing(tmp_path):
    make_session(tmp_path, [("train", np.zeros((360, 640, 3), np.uint8))])
    with pytest.raises(ValueError, match="Image size mismatch"):
        intrinsics.solve_session(tmp_path, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_duplicate_images_across_splits_abort_even_when_no_board(tmp_path):
    image = np.full((720, 1280, 3), 255, np.uint8)
    make_session(tmp_path, [("train", image), ("validation", image)])
    # Different PNG encoding still represents the same image and must not leak.
    assert cv2.imwrite(
        str(tmp_path / "validation/000001.png"), image, [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )
    assert (tmp_path / "train/000000.png").read_bytes() != (
        tmp_path / "validation/000001.png"
    ).read_bytes()
    with pytest.raises(ValueError, match="Duplicate image across train/validation"):
        intrinsics.solve_session(tmp_path, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_insufficient_valid_samples_cannot_use_cached_corners(tmp_path):
    make_session(tmp_path, [("train", np.full((720, 1280, 3), 255, np.uint8))])
    with pytest.raises(ValueError, match="Insufficient accepted samples: train=0"):
        intrinsics.solve_session(tmp_path, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "calibration.json"
    sentinel.write_text("keep this")
    with pytest.raises(FileExistsError, match="Output already exists"):
        intrinsics.solve_session(tmp_path, output)
    assert sentinel.read_text() == "keep this"


def test_full_session_writes_separate_raw_and_rectified_models(
    tmp_path, monkeypatch, projections
):
    board, train, val = projections
    views = train + val
    entries = []
    for index, _ in enumerate(views):
        # Encode an index in a PNG; only this test's detector is substituted.
        image = np.zeros((720, 1280, 3), np.uint8)
        image[0, 0, 0] = index + 1
        entries.append(("train" if index < len(train) else "validation", image))
    entries.extend(
        [
            ("train", entries[0][1]),
            ("validation", np.full((720, 1280, 3), 255, np.uint8)),
        ]
    )
    metadata = make_session(tmp_path, entries)
    reads = []

    def detect(self, bgr):
        index = int(bgr[0, 0, 0]) - 1
        reads.append(index)
        if index == 254:
            return {"accepted": False, "reason": "no_board"}
        return {**views[index], "accepted": True, "reason": "accepted"}

    monkeypatch.setattr(intrinsics.CharucoCorners, "detect", detect)
    output = tmp_path / "result"
    result = intrinsics.solve_session(tmp_path, output, alpha=0.5)
    assert len(reads) == 31  # every unique PNG, including rejected no-board image
    assert result["train"]["count"] == 24
    assert result["validation"]["count"] == 6
    assert result["camera"] == metadata["camera"]
    assert len(result["input_files"]) == 32
    assert all(len(item["sha256"]) == 64 for item in result["input_files"])
    assert {s["reason"] for s in result["rejected_samples"]} == {
        "duplicate_image_within_split",
        "no_board",
    }
    np.testing.assert_allclose(result["K"], KNOWN_K, atol=0.01, rtol=0)
    saved = json.loads((output / "calibration.json").read_text())
    rectified = json.loads((output / "rectified_camera.json").read_text())
    assert saved == result
    assert saved["schema"] == "uvc_intrinsics_calibration_v1"
    assert saved["camera_name"] == "front_1"
    assert saved["opencv_version"] == cv2.__version__
    assert any("controls are unavailable" in w for w in saved["quality_warnings"])
    assert any("closed cleanly" in w for w in saved["quality_warnings"])
    expected_K, expected_roi = cv2.getOptimalNewCameraMatrix(
        np.asarray(saved["K"]), np.asarray(saved["D"]), SIZE, 0.5, SIZE
    )
    np.testing.assert_array_equal(rectified["K"], expected_K)
    assert rectified["K"] == rectified["K_new"]
    assert rectified["D"] == [0] * 5
    assert rectified["valid_roi"] == list(expected_roi)
    assert rectified["image_size"] == list(SIZE)
    assert rectified["roi_cropped"] is False
    assert not rectified["production_approved"]
    assert len(list((output / "comparison").glob("*.jpg"))) == 10
    assert cv2.imread(str(output / "coverage.png")).shape == (752, 1280, 3)


@pytest.mark.parametrize(
    "preprocessing",
    [{"mirror": True}, {"resize": [640, 360]}, {"crop": [0, 0, 640, 360]}],
)
def test_preprocessed_sessions_are_rejected(tmp_path, preprocessing):
    metadata = make_session(tmp_path, [])
    metadata["preprocessing"].update(preprocessing)
    (tmp_path / "session.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="original images"):
        intrinsics.solve_session(tmp_path, tmp_path / "result")


def test_enabled_autofocus_is_rejected_offline(tmp_path):
    metadata = make_session(tmp_path, [])
    metadata["camera"]["controls"] = {
        "available": True,
        "values": {"focus_automatic_continuous": 1, "focus_absolute": 20},
    }
    (tmp_path / "session.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Autofocus was enabled"):
        intrinsics.solve_session(tmp_path, tmp_path / "result")


def test_changed_lens_setting_is_rejected_offline(tmp_path):
    metadata = make_session(tmp_path, [("train", np.zeros((720, 1280, 3), np.uint8))])
    metadata["camera"]["controls"] = {
        "available": True,
        "values": {"focus_auto": 0, "focus_absolute": 20},
    }
    metadata["samples"][0]["controls"] = {
        "available": True,
        "values": {"focus_auto": 0, "focus_absolute": 30},
    }
    (tmp_path / "session.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Focus/zoom controls changed"):
        intrinsics.solve_session(tmp_path, tmp_path / "result")
