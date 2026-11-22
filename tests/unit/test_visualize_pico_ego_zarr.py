"""Offline tests for world-to-camera projection and converted-episode preview."""

import json
from copy import deepcopy

import av
import cv2
import numpy as np
import pytest
import simplejpeg
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_pico.image_geometry import make_image_geometry
from egomimic.scripts.data_visualization import visualize_pico_ego_zarr as viz


def external_attrs():
    raw_k = [[20.0, 0, 16], [0, 20, 12], [0, 0, 1]]
    raw_d = [-0.15, 0.025, 0.001, -0.002, 0.0]
    geometry = make_image_geometry([32, 24], [32, 24], K=raw_k, D=raw_d)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler(
        "xyz", [4, -7, 12], degrees=True
    ).as_matrix()
    transform[:3, 3] = [0.02, -0.03, 0.04]
    return {
        "image_space": "processed",
        "image_geometry": {"front_1": geometry},
        "intrinsics": {"front_1": geometry["K_output"]},
        "pico_ego": {
            "source_format": "uvc_joint_v1",
            "camera_from_head": transform.tolist(),
            "fit": {
                "data": {
                    "schema": "seeded_external_uvc_spacetime_v1",
                    "parameters": {
                        "R_CH": transform[:3, :3].tolist(),
                        "t_CH_m": transform[:3, 3].tolist(),
                    },
                    "intrinsics": {
                        "raw_K": raw_k,
                        "raw_D": raw_d,
                        "image_size": [32, 24],
                        "mirror": False,
                    },
                }
            },
        },
    }


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


@pytest.mark.parametrize("external_uvc", [False, True])
def test_preview_reads_saved_images_and_emits_chain_report(
    tmp_path, monkeypatch, external_uvc
):
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
    if external_uvc:
        group.attrs = {**group.attrs, **external_attrs()}
        head = identity_pose.copy()
        head[:, 0] = [0, 0.01, -0.02]
        head[:, 3:] = Rotation.from_euler(
            "z", [[0], [10], [20]], degrees=True
        ).as_quat()[:, [3, 0, 1, 2]]
        camera_from_head = np.asarray(group.attrs["pico_ego"]["camera_from_head"])
        camera_world = viz.pose_matrices(head) @ np.linalg.inv(camera_from_head)
        group["source.head_pose"] = head
        group["obs_head_pose"] = np.column_stack(
            [
                camera_world[:, :3, 3],
                Rotation.from_matrix(camera_world[:, :3, :3]).as_quat()[
                    :, [3, 0, 1, 2]
                ],
            ]
        )
    monkeypatch.setattr(viz.zarr, "open_group", lambda *a, **kw: group)
    episode = tmp_path / "converted" / "episode_0"
    episode.mkdir(parents=True)
    output = tmp_path / "preview"
    result = viz.render_episode(episode, output)
    summary = json.loads((result / "summary.json").read_text())
    assert summary["total_frames"] == n
    assert summary["image_selection_time_basis"] == (
        "frame receive UTC" if external_uvc else "video PTS"
    )
    assert summary["image_selection_time_basis"] in summary["video_time_basis"]
    assert summary["camera_matrix_chain_max_abs_error"] < 1e-12
    assert summary["hands"]["left"]["filled_frames"] == 1
    assert summary["hands"]["right"]["chain_projection_difference_px"]["max"] < 1e-10
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
    if external_uvc:
        assert summary["camera_reference_basis"] == (
            "frozen external fit parameters R_CH and t_CH_m"
        )
        assert summary["image_geometry"]["D_output"] == [0.0] * 5
        # A self-consistent saved derived transform cannot hide a wrong extrinsic.
        group.attrs["pico_ego"]["camera_from_head"][0][3] += 0.1
        with pytest.raises(ValueError, match="differs from the frozen fit"):
            viz.render_episode(episode, tmp_path / "wrong_extrinsic")
        group.attrs["pico_ego"]["camera_from_head"][0][3] -= 0.1
        group["obs_head_pose"][1, 0] += 0.1
        with pytest.raises(ValueError, match="camera poses disagree"):
            viz.render_episode(episode, tmp_path / "wrong_camera_pose")


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda a: a["intrinsics"]["front_1"][0].__setitem__(0, 999), "K_output"),
        (
            lambda a: a["pico_ego"]["fit"]["data"]["parameters"]["R_CH"][0].__setitem__(
                0, 2
            ),
            "proper rigid",
        ),
        (
            lambda a: a["pico_ego"]["fit"]["data"]["intrinsics"]["raw_D"].__setitem__(
                0, -0.3
            ),
            "raw_D differs",
        ),
        (lambda a: a.__setitem__("image_space", "raw"), "rectified processed"),
        (
            lambda a: a["pico_ego"]["fit"]["data"]["intrinsics"].__setitem__(
                "mirror", True
            ),
            "mirror=false",
        ),
        (
            lambda a: a.__setitem__(
                "distortion_coefficients", {"front_1": [0.1, 0, 0, 0, 0]}
            ),
            "zero output distortion",
        ),
    ],
)
def test_external_geometry_rejects_contradictory_metadata(mutation, match):
    attrs = deepcopy(external_attrs())
    # The helper returns aliased output K values; saved JSON has no aliasing.
    attrs = json.loads(json.dumps(attrs))
    mutation(attrs)
    with pytest.raises(ValueError, match=match):
        viz.external_uvc_geometry(attrs, [32, 24])


def test_external_geometry_rejects_wrong_jpeg_size():
    with pytest.raises(ValueError, match="JPEG size"):
        viz.external_uvc_geometry(external_attrs(), [64, 48])


def test_rectified_projection_matches_raw_projection_then_undistort():
    """Changed focal lengths and nonzero D must agree in processed pixel space."""
    attrs = external_attrs()
    raw = attrs["pico_ego"]["fit"]["data"]["intrinsics"]
    geometry = make_image_geometry(
        [32, 24], [64, 48], K=raw["raw_K"], D=raw["raw_D"], alpha=0.3
    )
    attrs["image_geometry"]["front_1"] = geometry
    attrs["intrinsics"]["front_1"] = geometry["K_output"]
    camera_from_world, _ = viz.external_uvc_geometry(attrs, [64, 48])
    camera_points = np.array([[-0.2, -0.1, 0.8], [0.15, 0.2, 1.0], [0, 0, 1.2]])
    world_points = (camera_points - camera_from_world[:3, 3]) @ camera_from_world[
        :3, :3
    ]
    processed_uv, _ = viz.project_world_points(
        world_points[None], camera_from_world[None], np.asarray(geometry["K_output"])
    )
    raw_uv, _ = cv2.projectPoints(
        camera_points,
        np.zeros(3),
        np.zeros(3),
        np.asarray(raw["raw_K"]),
        np.asarray(raw["raw_D"]),
    )
    expected = cv2.undistortPoints(
        raw_uv,
        np.asarray(raw["raw_K"]),
        np.asarray(raw["raw_D"]),
        P=np.asarray(geometry["K_output"]),
    ).reshape(-1, 2)
    np.testing.assert_allclose(processed_uv[0], expected, atol=1e-7, rtol=0)


def test_preview_rejects_overwriting_input(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir()
    with pytest.raises(ValueError, match="outside"):
        viz.render_episode(episode, episode / "viz")
