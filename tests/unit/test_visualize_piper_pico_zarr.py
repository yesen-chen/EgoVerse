"""Helpers and logging for the Piper Zarr Rerun viewer."""

import json

import numpy as np
import pytest
import rerun as rr
import simplejpeg
import zarr
from zarr.core.dtype import VariableLengthBytes

from egomimic.scripts.data_visualization import visualize_piper_pico_zarr as viz


def test_find_episodes_accepts_store_or_batch(tmp_path):
    episode = tmp_path / "episode_000000.zarr"
    episode.mkdir()
    (episode / "zarr.json").write_text(json.dumps({"zarr_format": 3}))
    assert viz.find_episodes(episode) == [episode]
    assert viz.find_episodes(tmp_path) == [episode]
    with pytest.raises(FileNotFoundError):
        viz.find_episodes(tmp_path / "missing")


def test_quaternion_and_jpeg_helpers():
    xyzw = viz.wxyz_to_xyzw(np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(xyzw, [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    image = np.full((10, 12, 3), 90, dtype=np.uint8)
    jpeg = simplejpeg.encode_jpeg(image, colorspace="RGB")
    decoded = viz.decode_jpeg(jpeg)
    assert decoded.shape == (10, 12, 3)
    stored = np.empty(2, dtype=object)
    stored[:] = [jpeg, jpeg]
    assert viz.decode_jpeg(viz.encoded_image(stored, 1)).shape == (10, 12, 3)
    assert viz.gripper_width_to_color(0.0)[0] == 255
    assert viz.gripper_width_to_color(1.0)[1] == 255


def test_encoded_image_reads_zarr_variable_length_bytes(tmp_path):
    image = np.full((8, 10, 3), 70, dtype=np.uint8)
    jpeg = simplejpeg.encode_jpeg(image, colorspace="RGB")
    store = zarr.open_group(str(tmp_path / "images.zarr"), mode="w")
    store.create_array("front_1", shape=(2,), chunks=(1,), dtype=VariableLengthBytes())
    store["front_1"][:] = np.array([jpeg, jpeg], dtype=object)
    decoded = viz.decode_jpeg(viz.encoded_image(store["front_1"], 0))
    assert decoded.shape == (8, 10, 3)


def _episode(tmp_path, frames=3):
    jpeg = simplejpeg.encode_jpeg(
        np.full((8, 8, 3), 40, dtype=np.uint8), colorspace="RGB"
    )
    images = np.array([jpeg] * frames, dtype=object)
    pose = np.zeros((frames, 7))
    pose[:, 3] = 1.0
    pose[:, 0] = np.linspace(0.0, 0.1, frames)
    episode = {
        "path": tmp_path / "episode_000000.zarr",
        "attrs": {
            "schema": "piper_base_v3",
            "embodiment": "piper_bimanual",
            "calibration_status": "missing",
            "pose_coordinates": {side: f"piper_{side}_base" for side in viz.SIDES},
        },
        "count": frames,
        "timestamp": 10.0 + np.arange(frames) / 30.0,
    }
    for side in viz.SIDES:
        episode[f"{side}.obs_ee_pose"] = pose + (0.0 if side == "left" else 0.2)
        episode[f"{side}.cmd_ee_pose"] = pose + (0.05 if side == "left" else 0.25)
        episode[f"{side}.obs_gripper"] = np.linspace(0.8, 0.2, frames)
        episode[f"{side}.cmd_gripper"] = np.linspace(0.7, 0.1, frames)
    for camera in viz.CAMERAS:
        episode[f"images.{camera}"] = images
        episode[f"source.{camera}.age"] = np.full(frames, 0.08)
    return episode


def test_log_episode_writes_rerun_without_viewer(tmp_path):
    rr.init("piper_pico_zarr_test", spawn=False)
    logged = viz.log_episode(
        _episode(tmp_path), stride=1, marker_stride=2, send_layout=False
    )
    assert logged == 3
    logged = viz.log_episode(
        _episode(tmp_path), stride=2, marker_stride=2, send_layout=False
    )
    assert logged == 2
