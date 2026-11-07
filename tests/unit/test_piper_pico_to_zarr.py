"""Offline converter contracts: provenance, tick rows, frames and calibration."""

import json
import shutil
from dataclasses import replace

import av
import numpy as np
import pytest
import simplejpeg
import zarr
from scipy.spatial.transform import Rotation

from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id
from egomimic.scripts.piper_pico_to_zarr import (
    Options,
    convert_dataset,
    load_calibration,
    matching_frame_indices,
    prepare_episode,
)


def test_piper_identity_is_distinct_and_preserves_original_ids():
    original_ids = {
        "human_right_arm": 1,
        "human_left_arm": 2,
        "human_bimanual": 3,
        "eva_right_arm": 4,
        "eva_left_arm": 5,
        "eva_bimanual": 6,
    }
    for name, value in original_ids.items():
        assert get_embodiment_id(name) == value
        assert get_embodiment(value) == name.upper()
    assert get_embodiment_id("piper_bimanual") == 7
    assert get_embodiment(7) == "PIPER_BIMANUAL"


def write_video(path, frames=8):
    with av.open(str(path), mode="w") as out:
        stream = out.add_stream("mpeg4", rate=40)
        stream.width, stream.height, stream.pix_fmt = 16, 12, "yuv420p"
        for i in range(frames):
            rgb = np.full((12, 16, 3), 25 * i + 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


VIDEO_STAMPS = 999.99 + np.arange(8) * 0.02
TICKS = 1000 + np.arange(6) / 30
# Latest frame stamped at or before each tick's observation time.
SELECTED = [0, 2, 3, 5, 7, 7]


def write_ticks(data, tau):
    """Rows as PiperPicoEnv.record_tick stores them."""
    n = len(tau)
    data.create_array("timestamp", data=tau)
    data.create_array("decision_timestamp", data=tau + 0.002)
    data.create_array("deadline_timestamp", data=tau + 0.002 + 1 / 30)
    data.create_array("command_id", data=np.arange(n, dtype=np.int64))
    action = np.zeros((n, 14))
    for i in range(2):
        action[:, i * 7] = 10 + i + np.arange(n)
        action[:, i * 7 + 5] = np.arange(n) * 0.3
        action[:, i * 7 + 6] = np.linspace(0.08, 0, n)
        data.create_array(
            f"robot{i}_eef_pos", data=np.column_stack([tau - 1000 + i, tau * 0, tau * 0])
        )
        data.create_array(
            f"robot{i}_eef_rot_axis_angle",
            data=np.column_stack([tau * 0, tau * 0, (tau - 1000) * 10]),
        )
        data.create_array(
            f"robot{i}_gripper_width", data=np.linspace(0.08, 0.02, n)[:, None]
        )
        data.create_array(f"robot{i}_joint_pos", data=np.zeros((n, 6)) + i)
        data.create_array(f"robot{i}_source_timestamp", data=tau - 0.004)
        send = tau + 0.005
        if i == 1:
            send[2] = np.nan
        data.create_array(f"robot{i}_send_timestamp", data=send)
    data.create_array("action", data=action)
    for i in range(3):
        data.create_array(f"camera{i}_timestamp", data=VIDEO_STAMPS[SELECTED][:n])


@pytest.fixture
def capture(tmp_path):
    source = tmp_path / "capture"
    video_dir = source / "videos/0"
    video_dir.mkdir(parents=True)
    names = ["front_1", "left_wrist", "right_wrist"]
    (source / "videos/cameras.json").write_text(json.dumps(names))
    replay = zarr.open_group(str(source / "replay_buffer.zarr"), mode="w")
    data = replay.create_group("data")
    replay.create_group("meta").create_array("episode_ends", data=np.array([6]))
    write_ticks(data, TICKS)
    for i in range(3):
        np.save(video_dir / f"{i}.npy", VIDEO_STAMPS)
        np.save(video_dir / f"{i}.receive_timestamps.npy", VIDEO_STAMPS + 0.01)
        write_video(video_dir / f"{i}.mp4")
    return source


def calibration_file(path, x=5):
    left = np.eye(4)
    left[:3, :3] = Rotation.from_euler("z", 0.4).as_matrix()
    left[0, 3] = x
    right = np.eye(4)
    right[1, 3] = -2
    value = {
        "calibration_id": f"bench-{x}",
        "reference_camera": "front_1",
        "extrinsics_convention": "T_base_camera",
        "extrinsics": {"left": left.tolist(), "right": right.tolist()},
        "intrinsics": {"front_1": [[10, 0, 8], [0, 10, 6], [0, 0, 1]]},
        "image_sizes": {"front_1": [16, 12]},
    }
    path.write_text(json.dumps(value))
    return path, value


def test_calibrated_export_preserves_base_pose_and_command_semantics(capture, tmp_path):
    path, calibration = calibration_file(tmp_path / "cal.json")
    output = convert_dataset(
        capture,
        tmp_path / "export",
        calibration_path=path,
        options=Options(width=8, height=6),
    )[0]
    group = zarr.open_group(str(output), mode="r")
    assert group.attrs["embodiment"] == "piper_bimanual"
    assert group.attrs["schema"] == "piper_base_v3"
    report = json.loads((output / "conversion_report.json").read_text())
    assert report["embodiment"] == "piper_bimanual"
    assert report["schema"] == "piper_base_v3"
    assert report["missing_required_cameras"] == []
    assert report == group.attrs["conversion"]
    n = group.attrs["total_frames"]
    assert n == 6
    t = group["timestamp"][:n]
    # Rows are the recorded ticks themselves, not a resampled grid.
    np.testing.assert_array_equal(t, TICKS)
    # No calibration translation, rotation, or Eva orientation remap is applied.
    np.testing.assert_allclose(group["left.obs_ee_pose"][:n, 0], t - 1000)
    np.testing.assert_allclose(group["right.obs_ee_pose"][:n, 0], t - 1000 + 1)
    actual_rotation = Rotation.from_quat(group["left.obs_ee_pose"][:n, [4, 5, 6, 3]])
    np.testing.assert_allclose(
        actual_rotation.as_rotvec()[:, 2], (t - 1000) * 10, atol=1e-8
    )
    # Each row's command is the one decided after that row's observation.
    np.testing.assert_allclose(group["left.cmd_ee_pose"][:n, 0], 10 + np.arange(6))
    np.testing.assert_allclose(group["left.cmd_gripper"][:n, 0], np.linspace(1, 0, 6))
    np.testing.assert_array_equal(group["source.action_index"][:n], np.arange(6))
    np.testing.assert_array_equal(group["source.command_id"][:n], np.arange(6))
    np.testing.assert_allclose(group["left.obs_joints"][:n], 0)
    np.testing.assert_allclose(group["right.obs_joints"][:n], 1)
    np.testing.assert_allclose(group["source.left.robot_timestamp"][:n], t - 0.004)
    assert np.isnan(group["source.right.send_timestamp"][2])
    assert report["arms"]["right"]["unsent_commands"] == 1
    alignment = group.attrs["time_alignment"]
    assert alignment["schema"] == "control_tick_v1"
    assert alignment["observation_to_send_s"] == pytest.approx(0.005)
    assert alignment["observation_to_send_source"] == "arm_send_timestamp"
    assert report["tick_gaps"]["count"] == 0
    assert group.attrs["extrinsics"] == calibration["extrinsics"]
    np.testing.assert_allclose(
        group.attrs["intrinsics"]["front_1"], [[5, 0, 4, 0], [0, 5, 3, 0], [0, 0, 1, 0]]
    )
    assert group.attrs["pose_coordinates"]["right"] == "piper_right_base"
    assert group.attrs["calibration"]["data"]["calibration_id"] == "bench-5"
    indices = group["source.front_1.frame_index"][:n]
    np.testing.assert_array_equal(indices, SELECTED)
    assert np.all(group["source.front_1.age"][:n] >= 0)
    for row, index in enumerate(indices):
        image = simplejpeg.decode_jpeg(group["images.front_1"][row : row + 1][0])
        assert image.shape == (6, 8, 3)
        assert abs(float(image.mean()) - (25 * index + 20)) < 7
    np.testing.assert_allclose(
        group["source.front_1.receive_timestamp"][:n]
        - group["source.front_1.timestamp"][:n],
        0.01,
    )
    # A different batch calibration cannot be replaced with a global Eva constant.
    other, value = calibration_file(tmp_path / "other.json", x=9)
    second = convert_dataset(
        capture,
        tmp_path / "second",
        calibration_path=other,
        options=Options(width=8, height=6),
    )[0]
    assert (
        zarr.open_group(str(second), mode="r").attrs["extrinsics"]
        == value["extrinsics"]
    )
    with pytest.raises(FileExistsError):
        convert_dataset(capture, tmp_path / "export", calibration_path=path)


def test_uncalibrated_requires_opt_in_and_never_invents_matrices(capture, tmp_path):
    with pytest.raises(ValueError, match="Calibration is required"):
        convert_dataset(capture, tmp_path / "export")
    options = Options(allow_uncalibrated=True, width=16, height=12)
    output = convert_dataset(capture, tmp_path / "export", options=options)[0]
    attrs = dict(zarr.open_group(str(output), mode="r").attrs)
    assert attrs["calibration_status"] == "missing"
    assert attrs["embodiment"] == "piper_bimanual"
    assert "extrinsics" not in attrs and "intrinsics" not in attrs
    assert "hardcoded" in attrs["requires_training_transform"]


def test_invalid_calibration_direction_and_rotation(tmp_path):
    path, data = calibration_file(tmp_path / "cal.json")
    data["extrinsics_convention"] = "T_camera_base"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="T_base_camera"):
        load_calibration(path, Options())
    data["extrinsics_convention"] = "T_base_camera"
    data["extrinsics"]["left"][0][0] = 4
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="rigid"):
        load_calibration(path, Options())


def _args(capture):
    data = zarr.open_group(str(capture / "replay_buffer.zarr"), mode="r")["data"]
    return (data, 0, 6, capture / "videos/0", ["front_1", "left_wrist", "right_wrist"])


def _set(capture, key, value):
    data = zarr.open_group(str(capture / "replay_buffer.zarr"), mode="a")["data"]
    data[key][:] = value


def test_frame_matching_is_exact_and_marks_missing():
    stamps = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        matching_frame_indices(stamps, np.array([1.0, 2.5, 3.0, 4.0])), [0, -1, 2, -1]
    )


def test_frames_before_video_start_trim_only_leading_ticks(capture):
    selected = VIDEO_STAMPS[SELECTED].copy()
    selected[:2] = [999.95, 999.97]
    _set(capture, "camera1_timestamp", selected)
    numeric, _, chosen, report = prepare_episode(*_args(capture), Options())
    assert report["trimmed_leading_rows"] == 2
    np.testing.assert_array_equal(numeric["timestamp"], TICKS[2:])
    np.testing.assert_array_equal(numeric["source.action_index"], np.arange(2, 6))
    np.testing.assert_array_equal(chosen["front_1"], SELECTED[2:])
    selected = VIDEO_STAMPS[SELECTED].copy()
    selected[3] = 1000.555
    _set(capture, "camera1_timestamp", selected)
    with pytest.raises(ValueError, match="row 3 is not in the recorded video"):
        prepare_episode(*_args(capture), Options())


def test_stale_sources_and_acausal_rows_are_rejected(capture):
    with pytest.raises(ValueError, match="max-frame-age"):
        prepare_episode(*_args(capture), Options(max_frame_age=0.005))
    with pytest.raises(ValueError, match="max-robot-age"):
        prepare_episode(*_args(capture), Options(max_robot_age=0.001))
    _set(capture, "decision_timestamp", TICKS - 0.001)
    with pytest.raises(ValueError, match="decided before its observation"):
        prepare_episode(*_args(capture), Options())
    np.save(capture / "videos/0/0.npy", [1000.0, 1000.01, 1000.01])
    with pytest.raises(ValueError, match="strictly increasing"):
        prepare_episode(*_args(capture), Options())


def test_tick_gaps_are_reported_not_resampled(capture, tmp_path):
    tau = TICKS.copy()
    tau[4:] += 0.1
    root = zarr.open_group(str(capture / "replay_buffer.zarr"), mode="w")
    root.create_group("meta").create_array("episode_ends", data=np.array([6]))
    write_ticks(root.create_group("data"), tau)
    stamps = np.sort(np.r_[VIDEO_STAMPS, tau[4:] - 0.001])
    selected = VIDEO_STAMPS[SELECTED].copy()
    selected[4:] = tau[4:] - 0.001
    for i in range(3):
        root["data"][f"camera{i}_timestamp"][:] = selected
        np.save(capture / f"videos/0/{i}.npy", stamps)
        np.save(capture / f"videos/0/{i}.receive_timestamps.npy", stamps + 0.01)
        write_video(capture / f"videos/0/{i}.mp4", frames=len(stamps))
    out = convert_dataset(
        capture, tmp_path / "export", options=Options(allow_uncalibrated=True)
    )[0]
    group = zarr.open_group(str(out), mode="r")
    np.testing.assert_array_equal(group["timestamp"][:6], tau)
    gaps = group.attrs["conversion"]["tick_gaps"]
    assert gaps["count"] == 1
    assert gaps["first"][0]["after_row"] == 3
    assert gaps["first"][0]["interval_s"] == pytest.approx(0.1 + 1 / 30)


def test_collector_tick_rows_convert_end_to_end(tmp_path):
    from egomimic.robot.piper_pico.env import PiperPicoEnv
    from egomimic.robot.piper_pico.record import (
        ReplayBuffer,
        TickAccumulator,
        episode_from_ticks,
    )

    env = PiperPicoEnv.__new__(PiperPicoEnv)
    env.arms, env.cameras = [None, None], [None, None, None]
    env.tick_accumulator = TickAccumulator()
    for k, tau in enumerate(TICKS):
        obs = {"timestamp": np.array([tau - 1 / 30, tau])}
        for i in range(3):
            obs[f"camera{i}_timestamp"] = np.array([0.0, VIDEO_STAMPS[SELECTED[k]]])
        for i in range(2):
            obs[f"robot{i}_eef_pos"] = np.array([[0, 0, 0], [tau - 1000, i, 0]])
            obs[f"robot{i}_eef_rot_axis_angle"] = np.zeros((2, 3))
            obs[f"robot{i}_gripper_width"] = np.full((2, 1), 0.04)
            obs[f"robot{i}_joint_pos"] = np.zeros((2, 6))
            obs[f"robot{i}_source_timestamp"] = np.float64(tau - 0.003)
        action = np.zeros(14)
        action[0], action[6], action[13] = k, 0.08, 0.02
        env.record_tick(obs, action, tau + 0.002, tau + 0.035, k)
    stream = {
        "last_command_id": np.arange(6),
        "last_command_send_timestamp": TICKS + 0.006,
    }
    source = tmp_path / "capture"
    (source / "videos/0").mkdir(parents=True)
    names = ["front_1", "left_wrist", "right_wrist"]
    (source / "videos/cameras.json").write_text(json.dumps(names))
    ReplayBuffer(source / "replay_buffer.zarr").add_episode(
        episode_from_ticks(env.tick_accumulator, [stream, stream])
    )
    for i in range(3):
        np.save(source / f"videos/0/{i}.npy", VIDEO_STAMPS)
        write_video(source / f"videos/0/{i}.mp4")
    out = convert_dataset(
        source, tmp_path / "export", options=Options(allow_uncalibrated=True)
    )[0]
    group = zarr.open_group(str(out), mode="r")
    np.testing.assert_array_equal(group["timestamp"][:6], TICKS)
    np.testing.assert_allclose(group["left.cmd_ee_pose"][:6, 0], np.arange(6))
    np.testing.assert_allclose(group["right.obs_ee_pose"][:6, 1], 1)
    np.testing.assert_array_equal(group["source.front_1.frame_index"][:6], SELECTED)
    assert group.attrs["time_alignment"]["observation_to_send_s"] == pytest.approx(0.006)


def test_recording_without_tick_fields_is_refused(capture):
    data = zarr.open_group(str(capture / "replay_buffer.zarr"), mode="a")["data"]
    del data["camera0_timestamp"]
    with pytest.raises(ValueError, match="tick collector"):
        prepare_episode(*_args(capture), Options())


def test_short_video_and_bad_intrinsics_never_commit_partial_episode(capture, tmp_path):
    path, data = calibration_file(tmp_path / "cal.json")
    data["image_sizes"]["front_1"] = [32, 24]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="calibration image size"):
        convert_dataset(capture, tmp_path / "export", calibration_path=path)
    write_video(capture / "videos/0/0.mp4", frames=7)
    with pytest.raises(ValueError, match="decoded 7 frames"):
        convert_dataset(
            capture, tmp_path / "export", options=Options(allow_uncalibrated=True)
        )
    assert not list((tmp_path / "export").glob("**/*.zarr"))


def test_missing_camera_is_explicit_not_fabricated(capture, tmp_path):
    (capture / "videos/cameras.json").write_text('["front_1", "left_wrist"]')
    options = Options(allow_uncalibrated=True, width=16, height=12)
    with pytest.raises(ValueError, match="Missing required Piper cameras"):
        convert_dataset(capture, tmp_path / "export", options=options)
    out = convert_dataset(
        capture,
        tmp_path / "export",
        options=replace(options, allow_missing_cameras=True),
    )[0]
    group = zarr.open_group(str(out), mode="r")
    assert "images.right_wrist" not in group
    assert group.attrs["conversion"]["missing_required_cameras"] == ["right_wrist"]


def test_episode_selection_keeps_boundaries_and_global_source_indices(
    capture, tmp_path
):
    root = zarr.open_group(str(capture / "replay_buffer.zarr"), mode="a")
    for key, value in root["data"].arrays():
        first = value[:]
        second = first.copy()
        if key == "action":
            second[:, 0] += 100
        elif key.endswith("timestamp"):
            second += 10
        value.resize((12,) + value.shape[1:])
        value[:] = np.concatenate([first, second])
    root["meta/episode_ends"].resize((2,))
    root["meta/episode_ends"][:] = [6, 12]
    shutil.copytree(capture / "videos/0", capture / "videos/1")
    for path in (capture / "videos/1").glob("*.npy"):
        np.save(path, np.load(path) + 10)
    out = convert_dataset(
        capture,
        tmp_path / "export",
        episodes=[1],
        options=Options(allow_uncalibrated=True, width=16, height=12),
    )
    assert len(out) == 1 and out[0].name == "episode_000001.zarr"
    group = zarr.open_group(str(out[0]), mode="r")
    np.testing.assert_array_equal(group["source.action_index"][:6], np.arange(6, 12))
    np.testing.assert_allclose(group["left.cmd_ee_pose"][:6, 0], 110 + np.arange(6))
    assert group.attrs["conversion"]["source_row_range"] == [6, 12]
    assert not (out[0].parent / "episode_000000.zarr").exists()


def test_reader_consumes_exported_images_and_pose(capture, tmp_path):
    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset, ZarrEpisode

    out = convert_dataset(
        capture,
        tmp_path / "export",
        options=Options(allow_uncalibrated=True, width=16, height=12),
    )[0]
    episode = ZarrEpisode(out)
    assert episode.metadata["embodiment"] == "piper_bimanual"
    assert len(episode) == 6
    sample = episode.read({"images.front_1": (1, None), "left.cmd_ee_pose": (0, 4)})
    assert simplejpeg.decode_jpeg(sample["images.front_1"]).shape == (12, 16, 3)
    assert sample["left.cmd_ee_pose"].shape == (4, 7)
    dataset = ZarrDataset(
        out,
        key_map={
            "left.obs_ee_pose": {
                "zarr_key": "left.obs_ee_pose",
                "key_type": "state_keys",
            }
        },
    )
    assert dataset[0]["embodiment"] == 7


@pytest.mark.parametrize("target", ["parent", "self", "child"])
def test_output_cannot_write_inside_raw_dataset(capture, target):
    destinations = {
        "parent": capture.parent,
        "self": capture,
        "child": capture / "export",
    }
    with pytest.raises(ValueError, match="separate from the raw"):
        convert_dataset(
            capture, destinations[target], options=Options(allow_uncalibrated=True)
        )
