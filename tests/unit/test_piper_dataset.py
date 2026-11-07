"""Independent SE(3) oracles and strict schema checks for per-episode Piper data."""

import copy
from dataclasses import asdict

import numpy as np
import pytest
import torch
import zarr
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data._utils.collate import default_collate

from egomimic.rldb.action_timing import ActionTiming, validate_dataset_timing
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.piper import Piper
from egomimic.rldb.zarr.piper_dataset import (
    PiperDataset,
    preflight_piper_episode,
    validate_piper_metadata,
)
from egomimic.rldb.zarr.zarr_writer import ZarrWriter


def se3(position, euler):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("ZYX", euler).as_matrix()
    result[:3, 3] = position
    return result


def pose_matrices(poses):
    result = np.repeat(np.eye(4)[None], len(poses), axis=0)
    result[:, :3, :3] = Rotation.from_quat(poses[:, [4, 5, 6, 3]]).as_matrix()
    result[:, :3, 3] = poses[:, :3]
    return result


def ypr_matrices(poses):
    result = np.repeat(np.eye(4)[None], len(poses), axis=0)
    result[:, :3, :3] = Rotation.from_euler("ZYX", poses[:, 3:6]).as_matrix()
    result[:, :3, 3] = poses[:, :3]
    return result


def metadata(extrinsics, synthetic=False):
    data = {
        "calibration_id": "synthetic-fixture" if synthetic else "fixture-measured",
        "reference_camera": "front_1",
        "extrinsics_convention": "T_base_camera",
        "extrinsics": {key: matrix.tolist() for key, matrix in extrinsics.items()},
        "intrinsics": {"front_1": [[100, 0, 8], [0, 120, 6], [0, 0, 1]]},
        "synthetic": synthetic,
    }
    return {
        "schema": "piper_base_v3",
        "time_alignment": {"observation_to_send_s": 0.004},
        "embodiment": "piper_bimanual",
        "pose_representation": "xyz_m+quat_wxyz",
        "pose_coordinates": {side: f"piper_{side}_base" for side in ("left", "right")},
        "reference_camera": "front_1",
        "extrinsics_convention": "T_base_camera",
        "gripper_encoding": {"unit": "normalized_width", "open_m": 0.08},
        "calibration_status": "synthetic" if synthetic else "provided",
        "calibration": {"data": data, "path": "fixture-only", "sha256": "fixture-only"},
        "extrinsics": copy.deepcopy(data["extrinsics"]),
        "intrinsics": copy.deepcopy(data["intrinsics"]),
    }


@pytest.fixture
def extrinsics():
    return {
        "left": se3([0.2, -0.1, 0.7], [0.4, -0.3, 0.15]),
        "right": se3([-0.6, 0.2, 0.5], [-0.5, 0.2, -0.25]),
    }


def write_episode(path, transforms, fps=30, synthetic=False):
    frames = 8
    data = {"timestamp": 1234 + np.arange(frames) / fps}
    for side_index, side in enumerate(("left", "right")):
        for command in (False, True):
            prefix = "cmd" if command else "obs"
            x = np.arange(frames) / 10
            xyz = np.column_stack(
                [x + 0.2 * side_index + 0.3 * command, 0.2 + x / 2, 1.5 - x / 3]
            )
            # Cross pi with alternating quaternion signs; output rotations must
            # remain continuous and must not acquire Eva's constant transform.
            rot = Rotation.from_euler("ZYX", np.column_stack([2.9 + x, x / 8, -x / 5]))
            quat = rot.as_quat()[:, [3, 0, 1, 2]]
            quat[::2] *= -1
            data[f"{side}.{prefix}_ee_pose"] = np.column_stack([xyz, quat])
            data[f"{side}.{prefix}_gripper"] = (
                x / 2 + side_index * 0.1 + command * 0.05
            )[:, None]
    images = {
        f"images.{name}": np.full((frames, 12, 16, 3), 80 + 20 * i, dtype=np.uint8)
        for i, name in enumerate(("front_1", "left_wrist", "right_wrist"))
    }
    meta = metadata(transforms, synthetic)
    ZarrWriter(
        path,
        embodiment="piper_bimanual",
        fps=fps,
        intrinsics=meta["intrinsics"],
        extrinsics=meta["extrinsics"],
    ).write(numeric_data=data, image_data=images, metadata_override=meta)
    return data


def make_dataset(path, timing, **kwargs):
    return PiperDataset(
        path,
        Piper.get_keymap(timing=timing),
        Piper.get_transform_list(timing=timing),
        **kwargs,
    )


def oracle(data, transforms, spec, index):
    state, action = [], []
    indices = np.minimum(
        index + np.asarray(spec.support_indices), len(data["timestamp"]) - 1
    )
    query = np.linspace(0, len(indices) - 1, spec.output_chunk_length)
    for side in ("left", "right"):
        camera_base = np.linalg.inv(transforms[side])
        obs = (
            camera_base
            @ pose_matrices(data[f"{side}.obs_ee_pose"][index : index + 1])[0]
        )
        cmd = camera_base @ pose_matrices(data[f"{side}.cmd_ee_pose"][indices])
        pos = np.column_stack(
            [
                np.interp(query, np.arange(len(indices)), cmd[:, axis, 3])
                for axis in range(3)
            ]
        )
        rot = Slerp(np.arange(len(indices)), Rotation.from_matrix(cmd[:, :3, :3]))(
            query
        )
        gripper = np.interp(
            query, np.arange(len(indices)), data[f"{side}.cmd_gripper"][indices, 0]
        )
        state.extend(
            [
                *obs[:3, 3],
                *Rotation.from_matrix(obs[:3, :3]).as_euler("ZYX"),
                data[f"{side}.obs_gripper"][index, 0],
            ]
        )
        action.append(np.column_stack([pos, rot.as_euler("ZYX"), gripper]))
    return np.asarray(state), np.concatenate(action, axis=1)


@pytest.mark.parametrize("index", [0, 2, 6, 7])
@pytest.mark.parametrize(
    "timing", [ActionTiming(30, 5, 2, 9), ActionTiming(30, 4, 1, 7)]
)
def test_numeric_se3_oracle_current_frame_stride_padding_and_inverse(
    tmp_path, extrinsics, index, timing
):
    path = tmp_path / "episode.zarr"
    raw = write_episode(path, extrinsics)
    dataset = make_dataset(path, timing)
    sample = dataset[index]
    state, action = oracle(raw, extrinsics, timing, index)
    assert sample["embodiment"] == 7
    assert sample["observations.state.ee_pose"].shape == (14,)
    assert sample["actions_cartesian"].shape == (timing.output_chunk_length, 14)
    np.testing.assert_allclose(sample["observations.state.ee_pose"], state, atol=3e-6)
    np.testing.assert_allclose(sample["actions_cartesian"], action, atol=3e-6)
    for side, offset in (("left", 0), ("right", 7)):
        obs_camera = ypr_matrices(
            sample["observations.state.ee_pose"].numpy()[None, offset : offset + 6]
        )[0]
        cmd_camera = ypr_matrices(
            sample["actions_cartesian"].numpy()[:1, offset : offset + 6]
        )[0]
        np.testing.assert_allclose(
            extrinsics[side] @ obs_camera,
            pose_matrices(raw[f"{side}.obs_ee_pose"])[index],
            atol=3e-6,
        )
        np.testing.assert_allclose(
            extrinsics[side] @ cmd_camera,
            pose_matrices(raw[f"{side}.cmd_ee_pose"])[index],
            atol=3e-6,
        )
        # Width is already normalized by the converter; it is neither divided
        # by 0.08 again nor normalized using this episode's extrema.
        assert sample["actions_cartesian"][0, offset + 6].item() == pytest.approx(
            raw[f"{side}.cmd_gripper"][index, 0]
        )
    for camera in ("front_img_1", "left_wrist_img", "right_wrist_img"):
        assert sample[f"observations.images.{camera}"].shape == (3, 12, 16)
    assert not any(key.startswith("piper.") for key in sample)
    validate_dataset_timing({"piper_bimanual": timing}, {"piper_bimanual": dataset}, {})


def test_alternating_episodes_and_collated_batch_never_share_calibration(
    tmp_path, extrinsics
):
    other = {
        "left": se3([-1, 0.5, 2], [-0.4, 0.2, 0.8]),
        "right": se3([0.7, -0.5, 1], [0.7, -0.2, 0.3]),
    }
    spec = ActionTiming(30, 5, 1, 9)
    shared_pipeline = Piper.get_transform_list(timing=spec)
    original_eva = copy.deepcopy(Eva.EXTRINSICS)
    episodes, raw = [], []
    for i, transforms in enumerate((extrinsics, other)):
        path = tmp_path / f"batch{i}" / "episode_000000.zarr"
        raw.append(write_episode(path, transforms))
        episodes.append(
            PiperDataset(path, Piper.get_keymap(timing=spec), shared_pipeline)
        )
    snapshots = []
    for i in (0, 1, 0, 1):
        sample = episodes[i][2]
        state, actions = oracle(raw[i], (extrinsics, other)[i], spec, 2)
        np.testing.assert_allclose(
            sample["observations.state.ee_pose"], state, atol=3e-6
        )
        np.testing.assert_allclose(sample["actions_cartesian"], actions, atol=3e-6)
        snapshots.append(sample)
    batch = default_collate(snapshots)
    assert batch["actions_cartesian"].shape == (4, 9, 14)
    torch.testing.assert_close(
        batch["actions_cartesian"][0], batch["actions_cartesian"][2]
    )
    assert not torch.allclose(
        batch["actions_cartesian"][0], batch["actions_cartesian"][1]
    )
    assert len(shared_pipeline) == len(episodes[0].transform) - 1
    for side in ("left", "right"):
        np.testing.assert_array_equal(Eva.EXTRINSICS[side], original_eva[side])


def test_mapping_timing_and_norm_keymap_are_explicit(extrinsics):
    timing = ActionTiming(30, 6, 2, 13)
    keymap = Piper.get_keymap(timing=asdict(timing), norm_mode=True)
    assert len(keymap) == 8
    assert all(
        value.get("horizon") == 6
        for value in keymap.values()
        if value["key_type"] == "action_keys"
    )
    assert len(Piper.get_transform_list(timing=asdict(timing))) > 0
    with pytest.raises(TypeError):
        Piper.get_keymap()
    with pytest.raises(ValueError, match="cartesian"):
        Piper.get_keymap("other", timing=timing)


@pytest.mark.parametrize(
    "field,value",
    [
        ("embodiment", "eva_bimanual"),
        ("schema", "piper_eva_base_v1"),
        ("pose_representation", "xyz+ypr"),
        ("reference_camera", "right"),
        ("extrinsics_convention", "T_camera_base"),
        ("pose_coordinates", {"left": "world", "right": "world"}),
        ("gripper_encoding", {"unit": "meters"}),
        ("calibration", None),
        ("calibration_status", "missing"),
        ("time_alignment", {}),
        ("time_alignment", {"observation_to_send_s": -0.01}),
        ("extrinsics", {"left": np.eye(4).tolist()}),
        ("intrinsics", {"front_1": [[0, 0, 1], [0, 2, 1], [0, 0, 1]]}),
        ("intrinsics", {"right_wrist": np.eye(3).tolist()}),
    ],
)
def test_bad_metadata_names_episode(extrinsics, field, value):
    meta = metadata(extrinsics)
    meta[field] = value
    with pytest.raises(ValueError, match="episode_broken"):
        validate_piper_metadata(meta, episode_path="episode_broken")


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, 4)),
        np.eye(3),
        np.diag([-1, 1, 1, 1]),
        np.diag([2, 1, 1, 1]),
        np.full((4, 4), np.nan),
    ],
)
def test_invalid_extrinsics_are_rejected(extrinsics, bad):
    meta = metadata(extrinsics)
    meta["calibration"]["data"]["extrinsics"]["left"] = bad.tolist()
    with pytest.raises(ValueError, match="rigid"):
        validate_piper_metadata(meta)


def test_redundant_extrinsics_must_agree_and_do_not_alias(extrinsics):
    meta = metadata(extrinsics)
    valid = validate_piper_metadata(meta)
    valid["left"][0, 3] += 4
    assert meta["extrinsics"]["left"][0][3] == extrinsics["left"][0, 3]
    meta["extrinsics"]["left"][0][3] += 0.1
    with pytest.raises(ValueError, match="disagrees"):
        validate_piper_metadata(meta)


def test_synthetic_calibration_requires_explicit_opt_in(tmp_path, extrinsics):
    path = tmp_path / "synthetic.zarr"
    write_episode(path, extrinsics, synthetic=True)
    spec = ActionTiming(30, 5, 1, 7)
    with pytest.raises(ValueError, match="synthetic calibration"):
        make_dataset(path, spec)
    meta = preflight_piper_episode(path, allow_synthetic_calibration=True)
    assert meta["calibration_status"] == "synthetic"
    assert (
        make_dataset(path, spec, allow_synthetic_calibration=True)[0]["embodiment"] == 7
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_camera",
        "missing_pose",
        "nan_pose",
        "bad_quaternion",
        "nonincreasing_time",
        "short_array",
        "missing_width",
    ],
)
def test_preflight_rejects_bad_array_data_before_sampling(
    tmp_path, extrinsics, corruption
):
    path = tmp_path / "broken.zarr"
    write_episode(path, extrinsics)
    group = zarr.open_group(str(path), mode="a")
    if corruption == "missing_camera":
        del group["images.left_wrist"]
    elif corruption == "missing_pose":
        del group["right.cmd_ee_pose"]
    elif corruption == "missing_width":
        del group["left.obs_gripper"]
    elif corruption == "nan_pose":
        group["left.obs_ee_pose"][3, 1] = np.nan
    elif corruption == "bad_quaternion":
        group["right.cmd_ee_pose"][2, 3:] = 0
    elif corruption == "nonincreasing_time":
        group["timestamp"][2] = group["timestamp"][1]
    elif corruption == "short_array":
        del group["left.cmd_gripper"]
        group.create_array("left.cmd_gripper", data=np.zeros((2, 1)))
    with pytest.raises(ValueError, match="broken.zarr"):
        preflight_piper_episode(path)


def test_tick_jitter_and_recorded_gaps_are_accepted(tmp_path, extrinsics):
    path = tmp_path / "ticks.zarr"
    write_episode(path, extrinsics)
    group = zarr.open_group(str(path), mode="a")
    stamps = group["timestamp"][:]
    stamps[3:] += 0.1
    stamps[1] += 0.002
    group["timestamp"][:] = stamps
    assert preflight_piper_episode(path)["schema"] == "piper_base_v3"


def test_invalid_indices_do_not_silently_switch_to_another_frame(tmp_path, extrinsics):
    path = tmp_path / "episode.zarr"
    write_episode(path, extrinsics)
    dataset = make_dataset(path, ActionTiming(30, 4, 1, 7))
    for index in (-1, 8, 0.5):
        with pytest.raises(IndexError):
            dataset[index]


def test_corrupt_jpeg_does_not_relabel_a_random_frame(tmp_path, extrinsics):
    from egomimic.rldb.zarr.local_recordings import (
        RecordingEpisode,
        RecordingMultiDataset,
    )

    path = tmp_path / "episode.zarr"
    write_episode(path, extrinsics)
    group = zarr.open_group(str(path), mode="a")
    group.attrs["conversion"] = {"source_dataset": "synthetic_bad_jpeg_recording"}
    group["images.front_1"][0:1] = np.array([b"broken JPEG"], dtype=object)
    dataset = make_dataset(path, ActionTiming(30, 4, 1, 7))
    wrapped = RecordingEpisode(dataset, "batch/episode")
    with pytest.raises(ValueError, match="does not substitute"):
        wrapped[0]
    assert wrapped[1]["sample_index"] == 1
    combined = RecordingMultiDataset(datasets={"batch/episode": wrapped}, mode="total")
    with pytest.raises(RuntimeError, match="Strict recording read failed"):
        combined[0]
