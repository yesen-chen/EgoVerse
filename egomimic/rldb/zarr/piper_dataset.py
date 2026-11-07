"""Strict per-episode calibration for Piper; invalid episodes never disappear."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import zarr

from egomimic.rldb.zarr.action_chunk_transforms import Transform
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
from egomimic.robot.piper_pico.image_geometry import validate_dataset_image_geometry
from egomimic.utils.pose_utils import _matrix_to_xyzwxyz

SIDES = ("left", "right")
CAMERAS = ("front_1", "left_wrist", "right_wrist")
NUMERIC_SHAPES = {
    f"{side}.{field}": (7,) if field.endswith("ee_pose") else (1,)
    for side in SIDES
    for field in ("obs_ee_pose", "cmd_ee_pose", "obs_gripper", "cmd_gripper")
}


def _rigid(value, label):
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected a finite rigid SE(3) matrix") from exc
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
        or not np.allclose(
            matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5, rtol=0
        )
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError(f"{label}: expected a finite rigid SE(3) matrix")
    return matrix.copy()


def _intrinsics(value, label):
    if value is None:
        return
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label}: expected camera-keyed pinhole matrices")
    if "front_1" not in value:
        raise ValueError(
            f"{label}: front_1 intrinsics are required when intrinsics are supplied"
        )
    for name, matrix in value.items():
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape == (3, 3):
            matrix = np.column_stack([matrix, np.zeros(3)])
        if (
            matrix.shape != (3, 4)
            or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0
            or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1, 0], atol=1e-7, rtol=0)
            or not np.allclose(matrix[:, 3], 0, atol=1e-7, rtol=0)
        ):
            raise ValueError(f"{label}.{name}: invalid pinhole matrix")


def observation_to_send(metadata: Mapping, *, episode_path="<episode>") -> float:
    """Recorded median delay from a row's observation to sending its command."""
    alignment = metadata.get("time_alignment")
    value = (
        alignment.get("observation_to_send_s")
        if isinstance(alignment, Mapping)
        else None
    )
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f"{episode_path}: time_alignment.observation_to_send_s must be finite and nonnegative"
        )
    return float(value)


def validate_piper_metadata(
    metadata: Mapping,
    *,
    episode_path: str | Path = "<episode>",
    allow_synthetic_calibration: bool = False,
) -> dict[str, np.ndarray]:
    """Return validated T_base_camera copies or raise an episode-specific error.

    The converter stores original calibration provenance in calibration.data,
    plus scaled intrinsics and redundant extrinsics at the episode root. Both
    copies of the extrinsics must agree. Intrinsics may differ by image resize.
    """
    label = str(episode_path)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{label}: metadata must be a mapping")
    expected = {
        "embodiment": "piper_bimanual",
        "schema": "piper_base_v3",
        "pose_representation": "xyz_m+quat_wxyz",
        "reference_camera": "front_1",
        "extrinsics_convention": "T_base_camera",
        "pose_coordinates": {side: f"piper_{side}_base" for side in SIDES},
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{label}: {key} must be {value!r}")
    encoding = metadata.get("gripper_encoding", {})
    if not isinstance(encoding, Mapping) or encoding.get("unit") != "normalized_width":
        raise ValueError(f"{label}: gripper_encoding.unit must be normalized_width")
    observation_to_send(metadata, episode_path=label)
    calibration = metadata.get("calibration")
    data = calibration.get("data") if isinstance(calibration, Mapping) else None
    status = metadata.get("calibration_status")
    if not isinstance(data, Mapping) or status not in ("provided", "synthetic"):
        raise ValueError(f"{label}: missing or unusable calibration")
    synthetic = status == "synthetic" or data.get("synthetic", False) is True
    if synthetic and not allow_synthetic_calibration:
        raise ValueError(
            f"{label}: synthetic calibration requires explicit allow_synthetic_calibration=True"
        )
    if (
        not isinstance(data.get("calibration_id"), str)
        or not data["calibration_id"].strip()
        or data.get("reference_camera") != "front_1"
        or data.get("extrinsics_convention") != "T_base_camera"
    ):
        raise ValueError(
            f"{label}: invalid calibration_id/reference_camera/extrinsics_convention"
        )
    for key in ("extrinsics",):
        if not isinstance(data.get(key), Mapping) or set(data[key]) != set(SIDES):
            raise ValueError(
                f"{label}: calibration.{key} must contain exactly left and right"
            )
        if not isinstance(metadata.get(key), Mapping) or set(metadata[key]) != set(
            SIDES
        ):
            raise ValueError(f"{label}: root {key} must contain exactly left and right")
    matrices = {}
    for side in SIDES:
        matrices[side] = _rigid(
            data["extrinsics"][side], f"{label}: calibration.extrinsics.{side}"
        )
        root = _rigid(metadata["extrinsics"][side], f"{label}: extrinsics.{side}")
        if not np.allclose(matrices[side], root, atol=1e-8, rtol=0):
            raise ValueError(
                f"{label}: extrinsics.{side} disagrees with calibration provenance"
            )
    _intrinsics(metadata.get("intrinsics"), f"{label}: intrinsics")
    _intrinsics(data.get("intrinsics"), f"{label}: calibration.intrinsics")
    validate_dataset_image_geometry(metadata)
    return matrices


def _validate_store(store, metadata, label):
    count, fps = metadata.get("total_frames"), metadata.get("fps")
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise ValueError(f"{label}: total_frames must be a positive integer")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, Real)
        or not np.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError(f"{label}: fps must be finite and positive")
    features = metadata.get("features")
    if not isinstance(features, Mapping):
        raise ValueError(f"{label}: missing features mapping")
    for key, shape in {**NUMERIC_SHAPES, "timestamp": ()}.items():
        if key not in store or key not in features:
            raise ValueError(f"{label}: missing required array {key}")
        arr = store[key]
        if (
            arr.shape[0] < count
            or arr.shape[1:] != shape
            or tuple(features[key].get("shape", [])) != shape
        ):
            raise ValueError(
                f"{label}: {key} must have at least {count} rows and per-row shape {shape}"
            )
        values = arr[:count]
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError(f"{label}: {key} contains non-finite or nonnumeric data")
        if key.endswith("ee_pose") and not np.allclose(
            np.linalg.norm(values[:, 3:], axis=1), 1, atol=1e-3, rtol=0
        ):
            raise ValueError(f"{label}: {key} contains non-unit quaternions")
        # Rows are control ticks: nominally 1/fps apart with jitter and
        # occasional recorded gaps, so only strict ordering is required here.
        if key == "timestamp" and count > 1 and np.any(np.diff(values) <= 0):
            raise ValueError(f"{label}: timestamps must be strictly increasing")
    for camera in CAMERAS:
        key = f"images.{camera}"
        info = features.get(key, {})
        if key not in store or info.get("dtype") != "jpeg":
            raise ValueError(f"{label}: missing JPEG camera {key}")
        shape = info.get("shape", [])
        if (
            len(shape) != 3
            or shape[-1] != 3
            or any(not isinstance(x, Integral) or x <= 0 for x in shape)
        ):
            raise ValueError(f"{label}: invalid image shape for {key}")
        if len(store[key].shape) != 1 or store[key].shape[0] < count:
            raise ValueError(f"{label}: {key} has insufficient image rows")


def preflight_piper_episode(
    path: str | Path, *, allow_synthetic_calibration=False
) -> dict:
    """Validate metadata and numeric arrays before constructing a training set."""
    store = zarr.open_group(str(path), mode="r")
    metadata = dict(store.attrs)
    validate_piper_metadata(
        metadata,
        episode_path=path,
        allow_synthetic_calibration=allow_synthetic_calibration,
    )
    _validate_store(store, metadata, str(path))
    return metadata


class _EpisodeCalibration(Transform):
    def __init__(self, matrices):
        self.poses = {
            f"piper.{side}.T_base_camera_pose": _matrix_to_xyzwxyz(matrix[None])[0]
            for side, matrix in matrices.items()
        }

    def transform(self, batch):
        # Each sample owns its arrays even when another transform changes them.
        batch.update({key: pose.copy() for key, pose in self.poses.items()})
        return batch


class PiperDataset(ZarrDataset):
    strict_read = True

    def __init__(
        self,
        Episode_path,
        key_map,
        transform_list=None,
        *,
        allow_synthetic_calibration=False,
    ):
        self.allow_synthetic_calibration = allow_synthetic_calibration
        super().__init__(Path(Episode_path), key_map, transform_list)
        matrices = validate_piper_metadata(
            self.metadata,
            episode_path=Episode_path,
            allow_synthetic_calibration=allow_synthetic_calibration,
        )
        _validate_store(self.episode_reader._store, self.metadata, str(Episode_path))
        self.calibration_matrices = {
            key: matrix.copy() for key, matrix in matrices.items()
        }
        self.transform = [
            _EpisodeCalibration(matrices),
            *copy.deepcopy(transform_list or []),
        ]

    def __getitem__(self, idx, **kwargs):
        if not isinstance(idx, Integral) or idx < 0 or idx >= len(self):
            raise IndexError(
                f"{self.episode_path}: frame index {idx!r} outside episode"
            )
        return super().__getitem__(idx, **kwargs)
