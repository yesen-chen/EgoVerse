"""Native UVC eye-to-hand routes, distorted PnP and checked dual-arm export."""

import copy
import json
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_pico import calibration
from egomimic.robot.piper_pico import calibration_vision as vision
from egomimic.robot.piper_pico.camera_config import load_camera_config


@pytest.fixture
def rig(tmp_path):
    raw = {
        "schema": "uvc_intrinsics_calibration_v1",
        "type": "uvc",
        "camera_name": "front_1",
        "camera_model": "opencv_pinhole_brown5",
        "image_size": [640, 480],
        "K": [[510.0, 0, 320], [0, 515.0, 240], [0, 0, 1]],
        "D": [-0.2, 0.08, 0.001, -0.002, -0.01],
        "preprocessing": {"crop": None, "resize": None, "mirror": False},
        "camera": {
            "ID_SERIAL": "synthetic-only",
            "controls": {
                "available": True,
                "values": {"focus_automatic_continuous": 0, "focus_absolute": 0},
            },
        },
        "quality_warnings": ["Synthetic fixture only"],
        "production_approved": False,
    }
    intrinsic = tmp_path / "intrinsics.json"
    intrinsic.write_text(json.dumps(raw))
    spec = {
        "name": "front_1",
        "type": "uvc",
        "device": "/dev/fake-front",
        "width": 640,
        "height": 480,
        "fps": 30,
        "fourcc": "MJPG",
        "latency": 0,
        "latency_status": "unmeasured",
        "calibration_file": "intrinsics.json",
        "controls": {"focus_automatic_continuous": 0, "focus_absolute": 0},
        "expected_identity": {"ID_SERIAL": "synthetic-only"},
    }
    path = tmp_path / "rig.json"
    path.write_text(json.dumps({"schema": "camera_rig_v1", "cameras": [spec]}))
    return path


def matrix(rpy, xyz):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def synthetic_session(path, spec, arm):
    path.mkdir()
    camera = {**copy.deepcopy(spec["calibration"]), "name": "front_1"}
    board = vision.default_board_config()
    (path / "session.json").write_text(
        json.dumps(
            {
                "camera": camera,
                "camera_stream": spec,
                "board": board,
                "arm": arm,
            }
        )
    )
    estimator = vision.CharucoEstimator(camera, board)
    X = matrix([0.5, -0.25, 0.8], [0.3 if arm == "left" else -0.3, -0.2, 0.5])
    Y = matrix([-0.2, 0.4, 0.1], [0.02, -0.03, 0.09])
    rng = np.random.default_rng(935)
    for index in range(16):
        V = matrix(
            rng.uniform(-0.7, 0.7, 3),
            rng.uniform([-0.07, -0.07, 0.4], [0.01, 0.01, 0.6]),
        )
        G = X @ V @ np.linalg.inv(Y)
        points = cv2.projectPoints(
            estimator.object_points,
            cv2.Rodrigues(V[:3, :3])[0],
            V[:3, 3],
            estimator.K,
            estimator.D,
        )[0].reshape(-1, 2)
        sample = {
            "arm": arm,
            "split": "train" if index < 12 else "validation",
            "T_base_end": G.tolist(),
            "detection": {
                "accepted": True,
                "T_camera_board": V.tolist(),
                "corner_ids": list(range(25)),
                "corners_px": points.tolist(),
            },
        }
        directory = path / "samples" / f"{index:06d}"
        directory.mkdir(parents=True)
        (directory / "sample.json").write_text(json.dumps(sample))
    return X, Y


@pytest.fixture
def solved_pair(rig, tmp_path):
    spec = load_camera_config(rig)[0]
    paths = {}
    for arm in ("left", "right"):
        session = tmp_path / arm
        X, Y = synthetic_session(session, spec, arm)
        result = vision.solve_session(session)
        np.testing.assert_allclose(result["T_base_camera"], X, atol=1e-8)
        np.testing.assert_allclose(result["T_end_board"], Y, atol=1e-8)
        assert result["camera_type"] == "uvc" and result["camera_name"] == "front_1"
        assert "camera_eye" not in result
        assert result["camera_stream"] == spec
        assert result["validation"]["reprojection_rms_px"]["max"] < 1e-6
        paths[arm] = session / "calibration_result.json"
    return paths


def export_args(rig, solved_pair, output):
    return calibration.build_parser().parse_args(
        [
            "export",
            "--camera-rig",
            str(rig),
            "--left-result",
            str(solved_pair["left"]),
            "--right-result",
            str(solved_pair["right"]),
            "--calibration-id",
            "synthetic-test",
            "--output",
            str(output),
        ]
    )


def test_raw_distorted_pnp_recovers_known_pose(rig):
    spec = load_camera_config(rig)[0]
    estimator = vision.CharucoEstimator(
        spec["calibration"], vision.default_board_config()
    )
    pose = matrix([0.25, -0.3, 0.1], [-0.04, -0.04, 0.4])
    points = cv2.projectPoints(
        estimator.object_points,
        cv2.Rodrigues(pose[:3, :3])[0],
        pose[:3, 3],
        estimator.K,
        estimator.D,
    )[0]
    estimator.detector = SimpleNamespace(
        detectBoard=lambda gray: (
            points,
            np.arange(25).reshape(-1, 1),
            None,
            None,
        )
    )
    detection = estimator.detect(np.zeros((480, 640, 3), np.uint8))
    assert detection["accepted"], detection
    np.testing.assert_allclose(detection["T_camera_board"], pose, atol=1e-7)
    assert detection["reprojection_rms_px"] < 1e-7


def test_dual_arm_solve_export_preserves_raw_geometry_and_quality(
    rig, solved_pair, tmp_path
):
    output = tmp_path / "export.json"
    result = calibration.export_calibration(export_args(rig, solved_pair, output))
    raw = load_camera_config(rig)[0]["calibration"]
    assert result["extrinsics_convention"] == "T_base_camera"
    assert result["image_geometry"] == "raw_distorted"
    assert result["raw_camera_calibration"] == raw
    assert result["intrinsics"]["front_1"] == raw["K"]
    assert result["distortion_coefficients"]["front_1"] == raw["D"]
    assert not result["production_approved"]
    assert result["quality"]["intrinsics"]["quality_warnings"] == [
        "Synthetic fixture only"
    ]
    assert result["quality"]["extrinsics"]["left"]["warnings"]
    assert all(
        len(value["sha256"]) == 64 for value in result["source_results"].values()
    )
    assert json.loads(output.read_text()) == result
    # Ordinary converter loading preserves the raw distortion fields for stage 3.
    from egomimic.scripts.piper_pico_to_zarr import Options, load_calibration

    loaded = load_calibration(output, Options())
    assert loaded["data"]["distortion_coefficients"]["front_1"] == raw["D"]
    with pytest.raises(FileExistsError):
        calibration.export_calibration(export_args(rig, solved_pair, output))


@pytest.mark.parametrize(
    "problem",
    [
        "role",
        "geometry",
        "identity",
        "mode",
        "control",
        "status",
        "validation",
        "direction",
        "transform",
        "eye",
    ],
)
def test_export_rejects_conflicting_results_before_writing(
    rig, solved_pair, tmp_path, problem
):
    path = solved_pair["right"]
    result = json.loads(path.read_text())
    if problem == "role":
        result["arm"] = "left"
    elif problem == "geometry":
        result["camera"]["D"][0] += 0.1
    elif problem == "identity":
        result["camera"]["camera"]["ID_SERIAL"] = "other"
    elif problem == "mode":
        result["camera_stream"]["fps"] = 60
    elif problem == "control":
        result["camera_stream"]["controls"]["focus_absolute"] = 1
    elif problem == "status":
        result["optimization"]["success"] = False
    elif problem == "validation":
        result["validation_available"] = False
    elif problem == "direction":
        result["extrinsics_convention"] = "T_camera_base"
    elif problem == "transform":
        result["T_base_camera"][0][0] = 5
    elif problem == "eye":
        result["camera_eye"] = "left"
    path.write_text(json.dumps(result))
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError):
        calibration.export_calibration(export_args(rig, solved_pair, output))
    assert not output.exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--headset-ip", "unused"],
        ["--camera-config", "unused.json"],
        ["--pico-camera-latency", "0.2"],
        ["--camera-name", "missing"],
    ],
)
def test_uvc_cli_conflicts_fail_before_hardware(rig, tmp_path, monkeypatch, extra):
    forbidden = Mock(side_effect=AssertionError("must not allocate hardware"))
    monkeypatch.setattr(calibration, "SharedMemoryManager", forbidden)
    with pytest.raises(SystemExit):
        calibration.main(
            [
                "collect",
                "--arm",
                "left",
                "--camera-rig",
                str(rig),
                "--no-preview",
                "--output",
                str(tmp_path / "session"),
                *extra,
            ]
        )
    forbidden.assert_not_called()
    assert not (tmp_path / "session").exists()


@pytest.mark.parametrize("problem", ["missing_intrinsics", "wrong_size", "latency"])
def test_bad_selected_spec_fails_before_hardware(rig, tmp_path, monkeypatch, problem):
    config = json.loads(rig.read_text())
    if problem == "missing_intrinsics":
        del config["cameras"][0]["calibration_file"]
    elif problem == "wrong_size":
        config["cameras"][0]["width"] = 320
    else:
        config["cameras"][0]["latency"] = 10
    rig.write_text(json.dumps(config))
    forbidden = Mock(side_effect=AssertionError("must not allocate hardware"))
    monkeypatch.setattr(calibration, "SharedMemoryManager", forbidden)
    with pytest.raises(SystemExit):
        calibration.main(
            [
                "collect",
                "--arm",
                "left",
                "--camera-rig",
                str(rig),
                "--no-preview",
                "--output",
                str(tmp_path / "session"),
            ]
        )
    forbidden.assert_not_called()


@pytest.mark.parametrize("odd_width", [False, True])
def test_uvc_collect_uses_existing_environment_and_pico_teleop_without_headset_images(
    rig, tmp_path, monkeypatch, odd_width
):
    from egomimic.robot.piper_pico import env as env_module

    if odd_width:
        config = json.loads(rig.read_text())
        config["cameras"][0]["width"] = 641
        rig.write_text(json.dumps(config))
        intrinsic = rig.parent / "intrinsics.json"
        raw = json.loads(intrinsic.read_text())
        raw["image_size"][0] = 641
        intrinsic.write_text(json.dumps(raw))
    constructors = {}
    for name in (
        "CameraProcess",
        "PicoCameraProcess",
        "PiperArmProcess",
        "PicoProcess",
    ):
        constructors[name] = Mock(
            side_effect=lambda *a, **kw: SimpleNamespace(config=kw, stop=Mock())
        )
        monkeypatch.setattr(env_module, name, constructors[name])
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
    monkeypatch.setattr(calibration, "CaptureWorker", Mock(return_value=worker))
    run_loop = Mock()
    monkeypatch.setattr(calibration, "run_calibration", run_loop)
    output = tmp_path / "capture"
    calibration.main(
        [
            "collect",
            "--arm",
            "right",
            "--camera-rig",
            str(rig),
            "--output",
            str(output),
            "--no-preview",
        ]
    )
    assert constructors["CameraProcess"].call_count == 1
    assert constructors["PicoCameraProcess"].call_count == 0
    assert constructors["PicoProcess"].call_count == 1
    session = json.loads((output / "session.json").read_text())
    assert session["camera_stream"] == load_camera_config(rig)[0]
    assert session["camera"]["D"][0] == -0.2
    assert "uvc" in session["frames"]["camera"]
    assert run_loop.call_args.args[2] == "right"


def test_collect_requires_an_image_source_before_hardware(tmp_path, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("must not allocate hardware"))
    monkeypatch.setattr(calibration, "SharedMemoryManager", forbidden)
    with pytest.raises(SystemExit):
        calibration.main(
            [
                "collect",
                "--arm",
                "left",
                "--no-preview",
                "--output",
                str(tmp_path / "session"),
            ]
        )
    forbidden.assert_not_called()
    assert not (tmp_path / "session").exists()


def test_uvc_offline_check_image_uses_raw_config_and_native_size(
    rig, tmp_path, monkeypatch
):
    forbidden = Mock(side_effect=AssertionError("offline must not allocate hardware"))
    monkeypatch.setattr(calibration, "SharedMemoryManager", forbidden)
    image = tmp_path / "blank.png"
    cv2.imwrite(str(image), np.zeros((480, 640, 3), np.uint8))
    output = tmp_path / "check"
    calibration.main(
        [
            "check-image",
            "--camera-rig",
            str(rig),
            "--image",
            str(image),
            "--output",
            str(output),
        ]
    )
    report = json.loads((output / "detection.json").read_text())
    assert report["camera"]["D"][0] == -0.2
    assert report["camera_stream"]["name"] == "front_1"
    assert report["detection"]["reason"] == "no_board"
    forbidden.assert_not_called()
