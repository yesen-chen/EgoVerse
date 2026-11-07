"""Offline tests for world-to-camera projection and converted-episode preview."""

import json

import av
import numpy as np
import pytest
import simplejpeg
from scipy.spatial.transform import Rotation

from egomimic.scripts.data_visualization import visualize_pico_ego_zarr as viz


def test_projection_uses_inverse_camera_pose_and_wxyz():
    rotation = Rotation.from_euler("z", 90, degrees=True)
    camera_position = np.array([1.0, -2.0, 3.0])
    camera = np.r_[camera_position, rotation.as_quat()[[3, 0, 1, 2]]][None]
    points_camera = np.array([[0.2, 0.4, 2.0], [0.0, 0.0, -1.0]])
    points_world = rotation.apply(points_camera) + camera_position
    k = np.array([[100.0, 0, 50], [0, 200, 60], [0, 0, 1]])
    uv, depth = viz.project_world_points(
        points_world[None], np.linalg.inv(viz.pose_matrices(camera)), k
    )
    np.testing.assert_allclose(uv[0, 0], [60, 100])
    np.testing.assert_allclose(depth[0], [2, -1])
    assert np.isnan(uv[0, 1]).all()


def test_preview_reads_saved_images_and_emits_chain_report(tmp_path, monkeypatch):
    n = 3
    identity_pose = np.tile([0, 0, 0, 1, 0, 0, 0], (n, 1)).astype(float)
    point = np.tile([0.0, 0.0, 1.0], (n, 26, 1))
    ee = identity_pose.copy()
    ee[:, 2] = 1.0
    stamp = np.array([0, 33_333_333, 66_666_667], dtype=np.int64)
    image = np.full((24, 32, 3), 100, dtype=np.uint8)
    jpeg = simplejpeg.encode_jpeg(image, colorspace="RGB")

    class Group(dict):
        attrs = {
            "total_frames": n,
            "fps": 30,
            "intrinsics": {"front_1": [[20, 0, 16], [0, 20, 12], [0, 0, 1]]},
            "pico_ego": {"camera_from_head": np.eye(4).tolist()},
        }

    group = Group(
        {
            "obs_head_pose": identity_pose,
            "source.head_pose": identity_pose,
            "timestamp_ns": stamp,
            "obs_rgb_timestamps_ns": np.array(
                [0, 40_000_000, 40_000_000], dtype=np.int64
            ),
            "source.grid_timestamp_ns": stamp,
            "source.video_frame_index": np.array([0, 1, 1]),
            "images.front_1": np.array([jpeg] * n, dtype=object),
        }
    )
    for side in viz.SIDES:
        group.update(
            {
                f"{side}.obs_xr_keypoints": point.reshape(n, 78),
                f"{side}.obs_ee_pose": ee,
                f"{side}.hand_filled": np.array([False, True, False]),
                f"{side}.hand_valid": np.array([True, False, True]),
                f"{side}.obs_gripper": np.full((n, 1), 0.5),
                f"{side}.source_age_s": np.array([0.0, 0.033, 0.0]),
            }
        )
    monkeypatch.setattr(viz.zarr, "open_group", lambda *a, **kw: group)
    episode = tmp_path / "converted" / "episode_0"
    episode.mkdir(parents=True)
    output = tmp_path / "preview"
    result = viz.render_episode(episode, output)
    summary = json.loads((result / "summary.json").read_text())
    assert summary["total_frames"] == n
    assert summary["camera_matrix_chain_max_abs_error"] == 0
    assert summary["hands"]["left"]["filled_frames"] == 1
    assert summary["hands"]["right"]["chain_projection_difference_px"]["max"] == 0
    assert summary["repeated_source_images"] == 1
    assert summary["grid_to_image_abs_dt_ms"]["max"] == pytest.approx(26.666667)
    assert (result / "contact_sheet.jpg").is_file()
    with np.load(result / "projections.npz") as projections:
        assert projections["left_uv"].shape == (n, 26, 2)
    with av.open(str(result / "overlay.mp4")) as video:
        frames = list(video.decode(video=0))
        assert len(frames) == n
        assert (frames[0].width, frames[0].height) == (32, 24)
        assert float(frames[-1].time) == pytest.approx(2 / 30, abs=1e-5)
    with pytest.raises(FileExistsError):
        viz.render_episode(episode, output)


def test_preview_rejects_overwriting_input(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir()
    with pytest.raises(ValueError, match="outside"):
        viz.render_episode(episode, episode / "viz")
