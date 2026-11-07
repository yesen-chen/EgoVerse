"""ChArUco observation and offline, fixed-camera PiPER calibration.

All transforms use ``T_A_B @ p_B = p_A`` and translations in metres.
This module never connects to a camera or robot.  Its solver works with OpenCV
builds that do not expose ``calibrateHandEye``.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def default_camera_config() -> dict:
    return {
        "eye": "left",
        "image_size": [1080, 810],
        "K": [
            [686.8340695362958, 0.0, 539.75],
            [0.0, 686.8680648828482, 404.5],
            [0.0, 0.0, 1.0],
        ],
        "D": [0.0] * 5,
        "note": "Previously checked PICO left output-mode intrinsics; D=0 is an output-image assumption, not measured raw lens distortion.",
    }


def default_board_config() -> dict:
    return {
        "squares_x": 6,
        "squares_y": 6,
        "square_length_m": 0.019,
        "marker_length_m": 0.0133,
        "dictionary": "DICT_5X5_100",
        "legacy_pattern": False,
        "note": "Measured print: approximately 114 x 114 mm; marker size assumes uniform 95% scaling.",
    }


def _camera_parameters(config: dict) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    K = np.asarray(config["K"], dtype=np.float64)
    D = np.asarray(config.get("D", [0.0] * 5), dtype=np.float64).reshape(-1)
    size = np.asarray(config["image_size"], dtype=float)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(
            "Camera K must be a finite 3x3 intrinsic matrix with positive focal lengths"
        )
    if not np.allclose(K[2], [0, 0, 1]) or not np.allclose([K[0, 1], K[1, 0]], 0):
        raise ValueError("OpenCV camera K must have zero skew and last row [0, 0, 1]")
    if D.size not in (4, 5, 8, 12, 14) or not np.all(np.isfinite(D)):
        raise ValueError("Camera D must contain 4, 5, 8, 12 or 14 finite coefficients")
    if (
        size.shape != (2,)
        or not np.all(np.isfinite(size))
        or np.any(size < 1)
        or np.any(size != size.astype(int))
    ):
        raise ValueError("image_size must be [width, height] in integer pixels")
    source = config.get("type", "pico")
    if source not in ("pico", "uvc"):
        raise ValueError("Camera type must be pico or uvc")
    if source == "pico" and config.get("eye", "left") != "left":
        raise ValueError("Both arm sessions must use the PICO left camera")
    if source == "uvc" and config.get("eye") is not None:
        raise ValueError("UVC camera must not be labeled as a PICO eye")
    # Dimensions also enter the camera stream's JSON configuration.
    return K, D, tuple(int(value) for value in size)


def _make_board(config: dict):
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
        raise RuntimeError(
            "This calibration requires OpenCV with aruco.CharucoDetector support"
        )
    nx, ny = int(config["squares_x"]), int(config["squares_y"])
    square, marker = float(config["square_length_m"]), float(config["marker_length_m"])
    if (
        nx != config["squares_x"]
        or ny != config["squares_y"]
        or nx < 3
        or ny < 3
        or not np.isfinite([square, marker]).all()
        or not 0 < marker < square
    ):
        raise ValueError(
            "Board needs >=3 squares per axis and 0 < marker_length_m < square_length_m"
        )
    dictionary_name = config["dictionary"]
    if (
        not isinstance(dictionary_name, str)
        or not dictionary_name.startswith("DICT_")
        or not hasattr(cv2.aruco, dictionary_name)
    ):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name!r}")
    if not isinstance(config.get("legacy_pattern", False), bool):
        raise ValueError("legacy_pattern must be a boolean")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    board = cv2.aruco.CharucoBoard((nx, ny), square, marker, dictionary)
    if not hasattr(board, "setLegacyPattern"):
        raise RuntimeError(
            "This calibration requires OpenCV CharucoBoard.setLegacyPattern"
        )
    board.setLegacyPattern(config.get("legacy_pattern", False))
    return board


def _transform(rotation, translation) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(translation).reshape(3)
    return T


def _check_transform(value, name: str) -> np.ndarray:
    T = np.asarray(value, dtype=float)
    if (
        T.shape != (4, 4)
        or not np.isfinite(T).all()
        or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-7)
    ):
        raise ValueError(f"{name} must be a finite 4x4 homogeneous transform")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(R), 1.0, atol=1e-5
    ):
        raise ValueError(f"{name} rotation must be orthonormal with determinant +1")
    return T


class CharucoEstimator:
    """Detect at the native resolution, with model-free corner interpolation."""

    def __init__(
        self,
        camera_config: dict,
        board_config: dict,
        min_corners: int = 16,
        max_rms_px: float = 0.8,
    ):
        self.K, self.D, self.image_size = _camera_parameters(camera_config)
        self.board = _make_board(board_config)
        self.object_points = np.asarray(
            self.board.getChessboardCorners(), dtype=np.float64
        ).reshape(-1, 3)
        self.min_corners = int(min_corners)
        self.max_rms_px = float(max_rms_px)
        if (
            not 4 <= self.min_corners <= len(self.object_points)
            or not np.isfinite(max_rms_px)
            or max_rms_px <= 0
        ):
            raise ValueError("Invalid min_corners or max_rms_px")
        for name in ("solvePnPGeneric", "SOLVEPNP_IPPE", "solvePnPRefineLM"):
            if not hasattr(cv2, name):
                raise RuntimeError(f"OpenCV is missing {name}")
        # Deliberately omit K/D here: a candidate intrinsic matrix must not pull
        # the measured corners toward its own predictions.
        params = cv2.aruco.CharucoParameters()
        self.detector = cv2.aruco.CharucoDetector(self.board, params)

    def detect(self, rgb: np.ndarray) -> dict:
        result = {
            "accepted": False,
            "reason": "no_board",
            "marker_count": 0,
            "corner_count": 0,
            "corner_ids": [],
            "corners_px": [],
            "reprojection_rms_px": None,
            "T_camera_board": None,
        }
        if (
            not isinstance(rgb, np.ndarray)
            or rgb.dtype != np.uint8
            or rgb.shape != (self.image_size[1], self.image_size[0], 3)
        ):
            result["reason"] = "image_size_or_format_mismatch"
            return result
        try:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            corners, ids, _, marker_ids = self.detector.detectBoard(gray)
            result["marker_count"] = (
                0 if marker_ids is None else int(np.asarray(marker_ids).size)
            )
            if corners is None or ids is None or not np.asarray(ids).size:
                return result
            points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
            ids = np.asarray(ids, dtype=int).reshape(-1)
            result.update(
                corner_count=len(ids),
                corner_ids=ids.tolist(),
                corners_px=points.tolist(),
            )
            if len(ids) < self.min_corners:
                result["reason"] = "insufficient_corners"
                return result
            if (
                not np.isfinite(points).all()
                or len(set(ids.tolist())) != len(ids)
                or np.any(ids < 0)
                or np.any(ids >= len(self.object_points))
            ):
                result["reason"] = "invalid_corners"
                return result
            obj = self.object_points[ids]
            spread = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
            object_spread = np.linalg.svd(
                obj[:, :2] - obj[:, :2].mean(axis=0), compute_uv=False
            )
            area = float(cv2.contourArea(cv2.convexHull(points.astype(np.float32))))
            result["corner_hull_area_px2"] = area
            if (
                area < 1500
                or spread[1] < 0.05 * spread[0]
                or object_spread[1] < 0.1 * object_spread[0]
            ):
                result["reason"] = "insufficient_spatial_spread"
                return result
            candidates = cv2.solvePnPGeneric(
                obj, points, self.K, self.D, flags=cv2.SOLVEPNP_IPPE
            )
            if not candidates[0]:
                result["reason"] = "pnp_failed"
                return result
            valid = []
            for rvec, tvec in zip(candidates[1], candidates[2]):
                R = cv2.Rodrigues(rvec)[0]
                if (
                    np.min(
                        (self.object_points @ R.T + np.asarray(tvec).reshape(3))[:, 2]
                    )
                    <= 0
                ):
                    continue
                projected = cv2.projectPoints(obj, rvec, tvec, self.K, self.D)[
                    0
                ].reshape(-1, 2)
                rms = float(np.sqrt(np.mean(np.sum((projected - points) ** 2, axis=1))))
                if np.isfinite(rms):
                    valid.append((rms, rvec, tvec))
            if not valid:
                result["reason"] = "pnp_no_positive_depth"
                return result
            valid.sort(key=lambda candidate: candidate[0])
            result["pnp_candidate_rms_px"] = [item[0] for item in valid]
            _, rvec, tvec = valid[0]
            rvec, tvec = cv2.solvePnPRefineLM(
                obj, points, self.K, self.D, rvec.copy(), tvec.copy()
            )
            R = cv2.Rodrigues(rvec)[0]
            projected = cv2.projectPoints(obj, rvec, tvec, self.K, self.D)[0].reshape(
                -1, 2
            )
            rms = float(np.sqrt(np.mean(np.sum((projected - points) ** 2, axis=1))))
            T = _transform(R, tvec)
            if (
                not np.isfinite(T).all()
                or not np.isfinite(rms)
                or np.min((self.object_points @ R.T + T[:3, 3])[:, 2]) <= 0
            ):
                result["reason"] = "pnp_invalid_refinement"
                return result
            result.update(reprojection_rms_px=rms, T_camera_board=T.tolist())
            result["accepted"] = rms <= self.max_rms_px
            result["reason"] = (
                "accepted" if result["accepted"] else "high_reprojection_error"
            )
            return result
        except cv2.error as exc:
            result["reason"] = "opencv_detection_error"
            result["detail"] = str(exc).splitlines()[-1][:250]
            return result

    def annotate(self, rgb: np.ndarray, result: dict) -> np.ndarray:
        annotated = rgb.copy()
        color = (0, 220, 70) if result.get("accepted") else (255, 160, 0)
        for point in result.get("corners_px", []):
            if np.isfinite(point).all():
                cv2.circle(annotated, tuple(np.rint(point).astype(int)), 3, color, -1)
        T = result.get("T_camera_board")
        if T is not None:
            T = np.asarray(T)
            # OpenCV's axis helper assumes BGR; convert so this API remains RGB.
            bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.drawFrameAxes(
                bgr, self.K, self.D, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], 0.04
            )
            annotated = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rms = result.get("reprojection_rms_px")
        text = f"{result.get('reason', '')}: {result.get('corner_count', 0)} corners"
        if rms is not None:
            text += f"  RMS {rms:.3f} px"
        cv2.putText(
            annotated,
            text,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        return annotated


def _initial_handeye(
    G: np.ndarray, V: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    # A Y = Y B, with A = G_i^-1 G_j and B = V_i^-1 V_j.
    As, Bs, va, vb = [], [], [], []
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            A, B = np.linalg.inv(G[i]) @ G[j], np.linalg.inv(V[i]) @ V[j]
            a, b = (
                Rotation.from_matrix(A[:3, :3]).as_rotvec(),
                Rotation.from_matrix(B[:3, :3]).as_rotvec(),
            )
            angle_a, angle_b = np.linalg.norm(a), np.linalg.norm(b)
            # Tiny rotations add little information; near pi has ambiguous log sign.
            if min(angle_a, angle_b) >= np.deg2rad(5) and max(
                angle_a, angle_b
            ) < np.deg2rad(170):
                As.append(A)
                Bs.append(B)
                va.append(a)
                vb.append(b)
    if len(As) < 6:
        raise ValueError(
            "Insufficient rotational excitation: collect more poses with multi-axis rotations"
        )
    va, vb = np.asarray(va), np.asarray(vb)
    singular_a = np.linalg.svd(va, compute_uv=False)
    singular_b = np.linalg.svd(vb, compute_uv=False)
    if min(singular_a[1] / singular_a[0], singular_b[1] / singular_b[0]) < 0.1:
        raise ValueError(
            "Degenerate rotation axes: include rotations about at least two nonparallel axes"
        )
    if min(
        np.linalg.norm(va, axis=1).max(), np.linalg.norm(vb, axis=1).max()
    ) < np.deg2rad(15):
        raise ValueError(
            "Rotation range is too small: include at least 15 degrees of relative rotation"
        )
    RY, _ = Rotation.align_vectors(va, vb)
    RY = RY.as_matrix()
    lhs = np.concatenate([A[:3, :3] - np.eye(3) for A in As], axis=0)
    rhs = np.concatenate([RY @ B[:3, 3] - A[:3, 3] for A, B in zip(As, Bs)])
    tY, _, rank, singular_t = np.linalg.lstsq(lhs, rhs, rcond=None)
    condition = (
        float(singular_t[0] / singular_t[-1]) if singular_t[-1] > 0 else float("inf")
    )
    if rank < 3 or condition > 100:
        raise ValueError(
            "Ill-conditioned hand-eye translation: collect more diverse multi-axis poses"
        )
    Y = _transform(RY, tY)
    Xs = np.asarray([g @ Y @ np.linalg.inv(v) for g, v in zip(G, V)])
    X = _transform(
        Rotation.from_matrix(Xs[:, :3, :3]).mean().as_matrix(),
        Xs[:, :3, 3].mean(axis=0),
    )
    diagnostics = {
        "relative_motion_count": len(As),
        "robot_rotation_singular_values": singular_a.tolist(),
        "camera_rotation_singular_values": singular_b.tolist(),
        "translation_condition_number": condition,
    }
    return X, Y, diagnostics


def _pack(T: np.ndarray) -> np.ndarray:
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def _unpack(parameters: np.ndarray) -> np.ndarray:
    return _transform(Rotation.from_rotvec(parameters[:3]).as_matrix(), parameters[3:6])


def _statistics(values) -> dict | None:
    if not len(values):
        return None
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _evaluate(samples: list[dict], X: np.ndarray, Y: np.ndarray, board, K, D) -> dict:
    rows = []
    objects = np.asarray(board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
    for sample in samples:
        G, V = sample["G"], sample["V"]
        observed_Y = np.linalg.inv(G) @ X @ V
        row = {
            "sample_id": sample["id"],
            "translation_error_mm": float(
                np.linalg.norm(observed_Y[:3, 3] - Y[:3, 3]) * 1000
            ),
            "rotation_error_deg": float(
                np.rad2deg(
                    Rotation.from_matrix(Y[:3, :3].T @ observed_Y[:3, :3]).magnitude()
                )
            ),
            "reprojection_rms_px": None,
        }
        detection = sample["data"]["detection"]
        ids = np.asarray(detection.get("corner_ids", []), dtype=int).reshape(-1)
        points = np.asarray(detection.get("corners_px", []), dtype=float).reshape(-1, 2)
        if (
            ids.size
            and len(ids) == len(points)
            and np.isfinite(points).all()
            and np.all((ids >= 0) & (ids < len(objects)))
        ):
            predicted_V = np.linalg.inv(X) @ G @ Y
            if (
                np.min(
                    (objects[ids] @ predicted_V[:3, :3].T + predicted_V[:3, 3])[:, 2]
                )
                > 0
            ):
                predicted = cv2.projectPoints(
                    objects[ids],
                    cv2.Rodrigues(predicted_V[:3, :3])[0],
                    predicted_V[:3, 3],
                    K,
                    D,
                )[0].reshape(-1, 2)
                row["reprojection_rms_px"] = float(
                    np.sqrt(np.mean(np.sum((predicted - points) ** 2, axis=1)))
                )
        rows.append(row)
    return {
        "count": len(samples),
        "translation_error_mm": _statistics(
            [row["translation_error_mm"] for row in rows]
        ),
        "rotation_error_deg": _statistics([row["rotation_error_deg"] for row in rows]),
        "reprojection_rms_px": _statistics(
            [
                row["reprojection_rms_px"]
                for row in rows
                if row["reprojection_rms_px"] is not None
            ]
        ),
        "samples": rows,
    }


def solve_session(
    session_dir,
    output_path=None,
    method: str = "scipy",
    *,
    visualization_dir=None,
    axis_length_m: float = 0.04,
) -> dict:
    """Fit camera-to-base and end-to-board using only ``split=train`` samples.

    Eight fitting poses is a hard minimum, not a recommendation; collect roughly
    30--40 diverse fitting poses plus 8--10 independently reserved validation
    poses.  Validation never participates in initialization or optimization.
    After solving, export SDK_END axes and board-corner checks on each paired
    raw image. Missing images are reported separately from the numeric fit.
    """
    if not np.isfinite(axis_length_m) or axis_length_m <= 0:
        raise ValueError("axis_length_m must be finite and positive")
    if visualization_dir is not None and Path(visualization_dir).exists():
        raise FileExistsError(f"验收图片目录已存在，请使用新目录：{visualization_dir}")
    if method not in ("scipy", "scipy_joint"):
        raise ValueError(
            "Supported method: scipy (joint SE(3) fit with relative-motion initialization)"
        )
    session_dir = Path(session_dir)
    session = json.loads((session_dir / "session.json").read_text())
    if session.get("arm") not in ("left", "right"):
        raise ValueError("Session arm must be left or right")
    K, D, image_size = _camera_parameters(session["camera"])
    board = _make_board(session["board"])
    object_points = np.asarray(board.getChessboardCorners()).reshape(-1, 3)
    samples, excluded = [], []
    for path in sorted((session_dir / "samples").glob("*/sample.json")):
        sample_id = path.parent.name
        # Path.glob includes dot directories. Only the collector's numeric
        # directory rename commits a sample; never consume a pending write.
        if not sample_id.isdecimal():
            continue
        data = json.loads(path.read_text())
        if (
            data.get("invalidated", False)
            or (path.parent / "invalidated.json").exists()
            or (path.parent / "invalidated").exists()
        ):
            excluded.append({"sample_id": sample_id, "reason": "invalidated"})
            continue
        if not data.get("detection", {}).get("accepted", False):
            excluded.append(
                {"sample_id": sample_id, "reason": "detection_not_accepted"}
            )
            continue
        split = data.get("split", "train")
        if split not in ("train", "validation"):
            raise ValueError(f"{sample_id}: split must be train or validation")
        if data.get("arm", session.get("arm")) != session.get("arm"):
            raise ValueError(f"{sample_id}: arm differs from session arm")
        G = _check_transform(data["T_base_end"], f"{sample_id}.T_base_end")
        V = _check_transform(
            data["detection"]["T_camera_board"], f"{sample_id}.T_camera_board"
        )
        if np.min((object_points @ V[:3, :3].T + V[:3, 3])[:, 2]) <= 0:
            raise ValueError(
                f"{sample_id}: observed board must have positive camera depth"
            )
        ids = np.asarray(data["detection"].get("corner_ids", []))
        corners = np.asarray(data["detection"].get("corners_px", []))
        if ids.size or corners.size:
            if (
                ids.ndim != 1
                or corners.shape != (len(ids), 2)
                or not np.isfinite(ids).all()
                or not np.isfinite(corners).all()
                or np.any(ids != ids.astype(int))
                or np.any((ids < 0) | (ids >= len(object_points)))
                or len(set(ids.tolist())) != len(ids)
            ):
                raise ValueError(f"{sample_id}: invalid detection corner IDs or pixels")
        samples.append({"id": sample_id, "G": G, "V": V, "split": split, "data": data})
    train = [sample for sample in samples if sample["split"] == "train"]
    validation = [sample for sample in samples if sample["split"] == "validation"]
    if len(train) < 8:
        raise ValueError(f"Need at least 8 accepted training poses; found {len(train)}")
    G, V = (
        np.asarray([sample["G"] for sample in train]),
        np.asarray([sample["V"] for sample in train]),
    )
    X0, Y0, excitation = _initial_handeye(G, V)

    def residual(parameters):
        X, Y = _unpack(parameters[:6]), _unpack(parameters[6:])
        left, right = X @ V, G @ Y
        xyz = (left[:, :3, 3] - right[:, :3, 3]) / 0.002
        rot = (
            Rotation.from_matrix(
                np.swapaxes(right[:, :3, :3], 1, 2) @ left[:, :3, :3]
            ).as_rotvec()
            / 0.005
        )
        return np.concatenate([xyz, rot], axis=1).reshape(-1)

    fit = least_squares(
        residual,
        np.r_[_pack(X0), _pack(Y0)],
        loss="soft_l1",
        max_nfev=1000,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Hand-eye optimization did not converge: {fit.message}")
    X, Y = _unpack(fit.x[:6]), _unpack(fit.x[6:])
    training_metrics = _evaluate(train, X, Y, board, K, D)
    validation_metrics = _evaluate(validation, X, Y, board, K, D)
    warnings = []
    if len(train) < 25:
        warnings.append(
            "Fewer than 25 training poses; 30-40 varied poses are recommended."
        )
    if len(validation) < 3:
        warnings.append(
            "Insufficient held-out validation: at least 3 poses required to report validation available; 8-10 recommended."
        )
    warnings.append(
        "Pixel residual and board/end consistency do not independently establish robot absolute positioning accuracy; validate physically before deployment."
    )
    result = {
        "schema_version": 1,
        "arm": session.get("arm"),
        "camera_type": session["camera"].get("type", "pico"),
        "camera_name": session["camera"].get(
            "name", session["camera"].get("camera_name", "pico_left")
        ),
        "solve_status": "solved",
        "extrinsics_convention": "T_base_camera",
        "method": "scipy_joint_se3",
        "convention": "T_A_B maps coordinates in B to A; X=T_base_camera, Y=T_end_board; X T_camera_board_i = T_base_end_i Y",
        "translation_unit": "metre",
        "T_base_camera": X.tolist(),
        "T_camera_base": np.linalg.inv(X).tolist(),
        "T_end_board": Y.tolist(),
        "camera": session["camera"],
        "board": session["board"],
        "training_sample_ids": [sample["id"] for sample in train],
        "validation_sample_ids": [sample["id"] for sample in validation],
        "excluded_samples": excluded,
        "excitation": excitation,
        "optimization": {
            "success": True,
            "cost": float(fit.cost),
            "evaluations": int(fit.nfev),
            "translation_scale_m": 0.002,
            "rotation_scale_rad": 0.005,
            "loss": "soft_l1",
        },
        "training": training_metrics,
        "validation": validation_metrics,
        "validation_available": len(validation) >= 3,
        "production_approved": False,
        "warnings": warnings,
    }
    if result["camera_type"] == "pico":
        result["camera_eye"] = session["camera"].get("eye", "left")
    if "camera_stream" in session:
        result["camera_stream"] = session["camera_stream"]
    destination = (
        Path(output_path)
        if output_path is not None
        else session_dir / "calibration_result.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    try:
        from egomimic.robot.piper_pico.calibration_report import (
            export_verification_images,
        )

        result["visualization"] = export_verification_images(
            session_dir,
            result,
            samples,
            object_points,
            K,
            D,
            image_size,
            output_dir=visualization_dir,
            axis_length_m=axis_length_m,
        )
    except (OSError, ValueError, cv2.error) as exc:
        # Numeric calibration is already on disk. A picture-writing failure
        # must never hide that result or falsely report that all images exist.
        result["visualization"] = {
            "status": "failed",
            "directory": str(visualization_dir) if visualization_dir else None,
            "sample_count": 0,
            "image_count": 0,
            "error": str(exc),
        }
    if result["visualization"]["status"] != "complete":
        result["warnings"].append(
            "Verification images are incomplete; inspect visualization and its summary.json for omitted samples."
        )
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    return result
