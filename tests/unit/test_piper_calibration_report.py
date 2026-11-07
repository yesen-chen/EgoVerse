"""Offline SDK-end overlays validate robot-derived projection and retain sources."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_pico.calibration import main
from egomimic.robot.piper_pico.calibration_vision import (
    CharucoEstimator,
    default_board_config,
    solve_session,
)


def _transform(rpy, xyz):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    value[:3, 3] = xyz
    return value


@pytest.fixture
def image_session(tmp_path):
    camera = {
        "eye": "left",
        "image_size": [320, 240],
        "K": [[260.0, 0.0, 160.0], [0.0, 255.0, 120.0], [0.0, 0.0, 1.0]],
        "D": [0.008, -0.002, 0.0002, -0.0001, 0.0],
    }
    board = default_board_config()
    (tmp_path / "session.json").write_text(
        json.dumps({"camera": camera, "board": board, "arm": "left"})
    )
    estimator = CharucoEstimator(camera, board)
    X = _transform([0.75, -0.43, 1.1], [0.34, -0.21, 0.57])
    Y = _transform([-0.31, 0.66, -0.45], [0.018, -0.035, 0.091])
    rng = np.random.default_rng(731)
    hashes = {}
    samples = {}
    for i in range(13):
        V = _transform(
            rng.uniform(-0.65, 0.65, 3),
            rng.uniform([-0.08, -0.07, 0.45], [0.01, 0.01, 0.72]),
        )
        G = X @ V @ np.linalg.inv(Y)
        points = cv2.projectPoints(
            estimator.object_points,
            cv2.Rodrigues(V[:3, :3])[0],
            V[:3, 3],
            np.asarray(camera["K"]),
            np.asarray(camera["D"]),
        )[0].reshape(-1, 2)
        sample_id = f"{i:06d}"
        sample = {
            "arm": "left",
            "split": "train" if i < 10 else "validation",
            "T_base_end": G.tolist(),
            "detection": {
                "accepted": True,
                "T_camera_board": V.tolist(),
                "corner_ids": list(range(25)),
                "corners_px": points.tolist(),
            },
        }
        directory = tmp_path / "samples" / sample_id
        directory.mkdir(parents=True)
        # The first sample exercises the stored filename; all others use the
        # historical raw.png default. A textured source exposes accidental edits.
        filename = "captured_left.png" if i == 0 else "raw.png"
        if i == 0:
            sample["image"] = filename
        image = np.full((240, 320, 3), 48 + i, dtype=np.uint8)
        image[170:200, 25:85, :] = [180, 120, 70]
        image_path = directory / filename
        assert cv2.imwrite(str(image_path), image)
        hashes[image_path] = hashlib.sha256(image_path.read_bytes()).hexdigest()
        (directory / "sample.json").write_text(json.dumps(sample))
        samples[sample_id] = sample
    return {
        "directory": tmp_path,
        "X": X,
        "Y": Y,
        "camera": camera,
        "samples": samples,
        "hashes": hashes,
    }


def _write_sample(session, sample_id, data):
    path = session["directory"] / "samples" / sample_id / "sample.json"
    path.write_text(json.dumps(data))


def _report(result):
    info = result["visualization"]
    directory = Path(info["directory"])
    manifest = json.loads((directory / "summary.json").read_text())
    return info, directory, manifest


def test_solve_exports_every_sample_and_preserves_raw_images(image_session):
    result = solve_session(image_session["directory"])
    info, directory, manifest = _report(result)
    np.testing.assert_allclose(result["T_base_camera"], image_session["X"], atol=1e-8)
    assert info["status"] == "complete"
    assert info["sample_count"] == 13
    assert info["image_count"] == 26
    assert not info["skipped_samples"]
    assert manifest["display_frame"] == "SDK_END"
    assert not manifest["tcp_offset_applied"]
    assert manifest["native_image_size"] == [320, 240]
    assert {row["sample_id"] for row in manifest["samples"]} == set(
        image_session["samples"]
    )
    assert directory.parent == image_session["directory"] / "verification"
    snapshot = json.loads((directory / "calibration_used.json").read_text())
    np.testing.assert_allclose(snapshot["T_base_camera"], image_session["X"], atol=1e-8)
    for sample_id, sample in image_session["samples"].items():
        source_path = (
            image_session["directory"]
            / "samples"
            / sample_id
            / sample.get("image", "raw.png")
        )
        source = cv2.imread(str(source_path))
        for suffix in ("sdk_end", "board_check"):
            path = directory / sample["split"] / f"{sample_id}_{suffix}.png"
            assert path.is_file()
            overlay = cv2.imread(str(path))
            assert overlay is not None
            assert overlay.shape[1:] == source.shape[1:]
            assert overlay.shape[0] > source.shape[0]
            assert not np.array_equal(overlay[: source.shape[0]], source)
    for row in manifest["samples"]:
        source_path = Path(row["source_image"])
        assert row["source_image_sha256"] == image_session["hashes"][source_path]
        metrics = row["metrics"]
        assert metrics["reprojection_rms_px"] < 1e-6
        assert not metrics["tcp_offset_applied"]
    for path, digest in image_session["hashes"].items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    persisted = json.loads(
        (image_session["directory"] / "calibration_result.json").read_text()
    )
    assert persisted["visualization"] == info


def test_repeated_solves_keep_separate_complete_reports(image_session):
    first = solve_session(image_session["directory"])
    _, first_dir, _ = _report(first)
    first_hashes = {
        path.relative_to(first_dir): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in first_dir.rglob("*")
        if path.is_file()
    }
    second = solve_session(image_session["directory"])
    _, second_dir, _ = _report(second)
    assert first_dir != second_dir
    assert first_dir.is_dir() and second_dir.is_dir()
    for path, digest in first_hashes.items():
        assert hashlib.sha256((first_dir / path).read_bytes()).hexdigest() == digest
    np.testing.assert_array_equal(first["T_base_camera"], second["T_base_camera"])


def test_invalidated_rejected_and_pending_samples_are_not_visualized(image_session):
    root = image_session["directory"]
    (root / "samples" / "000000" / "invalidated.json").write_text("{}")
    rejected = image_session["samples"]["000012"]
    rejected["detection"]["accepted"] = False
    _write_sample(image_session, "000012", rejected)
    pending = root / "samples" / ".000999.pending"
    pending.mkdir()
    (pending / "sample.json").write_text('{"T_base_end":')
    result = solve_session(root)
    info, directory, _ = _report(result)
    assert info["sample_count"] == 11
    assert info["image_count"] == 22
    assert not list(directory.rglob("000000_*.png"))
    assert not list(directory.rglob("000012_*.png"))
    assert not list(directory.rglob("000999_*.png"))
    np.testing.assert_allclose(result["T_base_camera"], image_session["X"], atol=1e-8)


def test_bad_source_images_leave_auditable_partial_report_and_valid_fit(image_session):
    root = image_session["directory"]
    (root / "samples" / "000001" / "raw.png").unlink()
    (root / "samples" / "000002" / "raw.png").write_bytes(b"not a png")
    assert cv2.imwrite(
        str(root / "samples" / "000003" / "raw.png"),
        np.zeros((120, 160, 3), dtype=np.uint8),
    )
    result = solve_session(root)
    info, directory, _ = _report(result)
    assert info["status"] == "partial"
    assert info["sample_count"] == 10
    assert info["attempted_sample_count"] == 13
    assert info["image_count"] == 20
    assert {row["sample_id"] for row in info["skipped_samples"]} == {
        "000001",
        "000002",
        "000003",
    }
    assert all(row["reason"] for row in info["skipped_samples"])
    assert (root / "calibration_result.json").is_file()
    assert (directory / "calibration_used.json").is_file()
    np.testing.assert_allclose(result["T_base_camera"], image_session["X"], atol=1e-8)


def test_explicit_report_directory_collision_is_rejected_before_fitting(
    image_session, monkeypatch
):
    root = image_session["directory"]
    existing = root / "my_report"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("previous report")

    def forbidden(*args, **kwargs):
        pytest.fail("Report output collision should be caught before fitting")

    monkeypatch.setattr(
        "egomimic.robot.piper_pico.calibration_vision._initial_handeye", forbidden
    )
    with pytest.raises((FileExistsError, ValueError)):
        solve_session(root, visualization_dir=existing)
    assert sentinel.read_text() == "previous report"
    assert not (root / "calibration_result.json").exists()


def test_heldout_robot_motion_changes_projection_without_changing_the_fit(
    image_session,
):
    root = image_session["directory"]
    first = solve_session(root)
    _, _, before = _report(first)
    sample = image_session["samples"]["000010"]
    original_pnp = np.asarray(sample["detection"]["T_camera_board"]).copy()
    G = np.asarray(sample["T_base_end"])
    # Move the feedback 25 mm along camera x while leaving the image and its
    # measured board PnP untouched. An overlay wrongly derived from PnP will
    # stay perfect; the required robot-derived overlay must show this mismatch.
    G[:3, 3] += image_session["X"][:3, :3] @ np.array([0.025, 0.0, 0.0])
    sample["T_base_end"] = G.tolist()
    _write_sample(image_session, "000010", sample)
    second = solve_session(root, axis_length_m=0.06)
    info, _, after = _report(second)
    np.testing.assert_array_equal(first["T_base_camera"], second["T_base_camera"])
    np.testing.assert_array_equal(first["T_end_board"], second["T_end_board"])
    assert info["axis_length_m"] == 0.06
    metrics = next(
        row["metrics"] for row in after["samples"] if row["sample_id"] == "000010"
    )
    before_metrics = next(
        row["metrics"] for row in before["samples"] if row["sample_id"] == "000010"
    )
    expected_end = np.linalg.inv(image_session["X"]) @ G
    expected_board = expected_end @ image_session["Y"]
    np.testing.assert_allclose(metrics["T_camera_end"], expected_end, atol=1e-8)
    np.testing.assert_allclose(
        metrics["T_camera_board_predicted"], expected_board, atol=1e-8
    )
    assert not np.allclose(metrics["T_camera_board_predicted"], original_pnp)
    assert metrics["reprojection_rms_px"] > 8.0
    assert before_metrics["reprojection_rms_px"] < 1e-6
    assert not np.allclose(
        metrics["axes_pixels_xy"][0], before_metrics["axes_pixels_xy"][0]
    )
    points = np.asarray(metrics["detected_corners_px"])
    predicted = np.asarray(metrics["predicted_corners_px"])
    expected_rms = np.sqrt(np.mean(np.sum((predicted - points) ** 2, axis=1)))
    assert metrics["reprojection_rms_px"] == pytest.approx(expected_rms)
    assert metrics["reprojection_rms_px"] == pytest.approx(
        next(
            row["reprojection_rms_px"]
            for row in second["validation"]["samples"]
            if row["sample_id"] == "000010"
        )
    )


def test_explicit_output_directory_is_used_and_report_manifest_matches(image_session):
    root = image_session["directory"]
    destination = root / "review_images"
    result_path = root / "results" / "left.json"
    result = solve_session(root, result_path, visualization_dir=destination)
    info, directory, _ = _report(result)
    assert directory == destination
    assert Path(info["manifest"]) == directory / "summary.json"
    assert result_path.is_file()
    assert not (root / "calibration_result.json").exists()
    assert not (root / "verification").exists()


@pytest.mark.parametrize("custom_output", [False, True])
def test_solve_cli_exports_images_with_requested_axis_length(
    image_session, monkeypatch, capsys, custom_output
):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline verification must not run collection or connect devices")

    monkeypatch.setattr(
        "egomimic.robot.piper_pico.calibration.collect_session", forbidden
    )
    monkeypatch.setattr("egomimic.robot.piper_pico.env.PiperPicoEnv", forbidden)
    root = image_session["directory"]
    args = ["solve", "--session", str(root)]
    result_path = root / "calibration_result.json"
    if custom_output:
        args.extend(
            ["--visualization-dir", str(root / "cli_images"), "--axis-length-mm", "60"]
        )
    else:
        result_path = root.parent / f"{root.name}_external_result.json"
        args.extend(["--output", str(result_path), "--axis-length-mm", "50"])
    main(args)
    result = json.loads(result_path.read_text())
    info, directory, manifest = _report(result)
    expected_axis_length = 0.06 if custom_output else 0.05
    assert info["status"] == "complete"
    assert info["axis_length_m"] == expected_axis_length
    assert all(
        row["metrics"]["axis_length_m"] == expected_axis_length
        for row in manifest["samples"]
    )
    if custom_output:
        assert directory == root / "cli_images"
    else:
        assert directory.parent == root / "verification"
        assert not (root / "calibration_result.json").exists()
    output = capsys.readouterr().out
    assert str(directory) in output
    assert str(result_path) in output
    assert "validation/" in output


def test_image_write_failure_is_reported_without_losing_numeric_calibration(
    image_session, monkeypatch
):
    def fail_write(*args, **kwargs):
        raise OSError("synthetic image write failure")

    monkeypatch.setattr(
        "egomimic.robot.piper_pico.calibration_report._write_rgb", fail_write
    )
    root = image_session["directory"]
    result = solve_session(root)
    info, directory, _ = _report(result)
    assert info["status"] == "failed"
    assert info["attempted_sample_count"] == 13
    assert info["sample_count"] == 0
    assert info["image_count"] == 0
    assert len(info["skipped_samples"]) == 13
    assert all(
        "synthetic image write failure" in row["reason"]
        for row in info["skipped_samples"]
    )
    persisted = json.loads((root / "calibration_result.json").read_text())
    np.testing.assert_allclose(
        persisted["T_base_camera"], image_session["X"], atol=1e-8
    )
    assert persisted["visualization"]["status"] == "failed"
    assert (directory / "calibration_used.json").is_file()
