"""Moving-camera timing/geometry and gripper contracts for PICO Ego loading."""

import copy
from dataclasses import asdict

import numpy as np
import pytest
import torch
import zarr
from scipy.spatial.transform import Rotation, Slerp

from egomimic.rldb.action_timing import ActionTiming, validate_dataset_timing
from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.embodiment.pico_ego import (
    PicoEgo,
    PicoEgoDataset,
    preflight_pico_ego_episode,
    validate_pico_ego_metadata,
)
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
from egomimic.rldb.zarr.zarr_writer import ZarrWriter


def poses(position, ypr):
    quat = Rotation.from_euler("ZYX", ypr).as_quat()[:, [3, 0, 1, 2]]
    quat[::2] *= -1
    return np.column_stack([position, quat])


def matrices(pose):
    result = np.repeat(np.eye(4)[None], len(pose), axis=0)
    result[:, :3, 3] = pose[:, :3]
    result[:, :3, :3] = Rotation.from_quat(pose[:, [4, 5, 6, 3]]).as_matrix()
    return result


def metadata():
    return {
        "schema": "pico_ego_world_v1",
        "embodiment": "human_bimanual",
        "pose_representation": "xyz_m+quat_wxyz",
        "reference_camera": "front_1",
        "pose_coordinates": {
            "left": "tracking_world",
            "right": "tracking_world",
            "obs_head_pose": "T_world_left_optical_camera",
        },
        "extrinsics": None,
        "pico_ego": {
            "tool_frame": {"axis_convention": "piper_x"},
            "source_tracking": {"sha256": "synthetic-source", "path": "fixture-only"},
            "gripper_encoding": {
                "open_m": 0.08,
                "normalized": "clip(distance,0,open_m)/open_m; fixed across episodes",
            },
            # Deliberately substantial extrinsics: they were already applied by
            # conversion and must never be applied by the training loader.
            "camera_from_head": [
                [1, 0, 0, 1.0],
                [0, 1, 0, -2.0],
                [0, 0, 1, 3.0],
                [0, 0, 0, 1],
            ],
            "options": {"max_hold_s": 4 / 30, "pose_sampling": "nearest"},
        },
        "intrinsics": {"front_1": [[120, 0, 8, 0], [0, 125, 6, 0], [0, 0, 1, 0]]},
    }


def write_episode(path, *, moving_hands=True, held=False, fps=30):
    frames = 12
    x = np.arange(frames) / 10
    timestamp_ns = np.int64(1700000000000000000) + np.rint(
        np.arange(frames) * (1e9 / fps)
    ).astype(np.int64)
    data = {
        "timestamp_ns": timestamp_ns,
        "timestamp": timestamp_ns.astype(np.float64) / 1e9,
        "obs_head_pose": poses(
            np.column_stack([x / 3, -x / 4, x / 5]),
            np.column_stack([x / 5, -x / 10, x / 8]),
        ),
    }
    for i, side in enumerate(("left", "right")):
        motion = x if moving_hands else x * 0
        data[f"{side}.obs_ee_pose"] = poses(
            np.column_stack([motion / 2 + i * 0.3, 0.1 - motion / 4, 1.8 + motion / 5]),
            np.column_stack([2.9 + motion, -0.2 + motion / 8, motion / 10]),
        )
        data[f"{side}.obs_gripper"] = (0.1 + x / 3 + i * 0.1)[:, None]
        valid = np.ones(frames, dtype=bool)
        filled = np.zeros(frames, dtype=bool)
        if held:
            valid[3:7], filled[3:7] = False, True
            data[f"{side}.obs_ee_pose"][3:7] = data[f"{side}.obs_ee_pose"][2]
            data[f"{side}.obs_gripper"][3:7] = data[f"{side}.obs_gripper"][2]
        data[f"{side}.obs_gripper_width_m"] = data[f"{side}.obs_gripper"] * 0.08
        data[f"{side}.hand_valid"], data[f"{side}.hand_filled"] = valid, filled
    meta = metadata()
    images = {"images.front_1": np.full((frames, 12, 16, 3), 80, dtype=np.uint8)}
    ZarrWriter(
        path, embodiment="human_bimanual", fps=fps, intrinsics=meta["intrinsics"]
    ).write(numeric_data=data, image_data=images, metadata_override=meta)
    return data


def dataset(path, spec, mode="zero"):
    return PicoEgoDataset(
        path,
        PicoEgo.get_keymap(timing=spec, gripper_mode=mode),
        PicoEgo.get_transform_list(timing=spec, gripper_mode=mode),
        gripper_mode=mode,
    )


def oracle(raw, spec, index, mode):
    current_inverse = np.linalg.inv(matrices(raw["obs_head_pose"])[index])
    support = np.minimum(
        index + np.asarray(spec.support_indices), len(raw["timestamp"]) - 1
    )
    query = np.linspace(0, len(support) - 1, spec.output_chunk_length)
    state, action = [], []
    for side in ("left", "right"):
        world = matrices(raw[f"{side}.obs_ee_pose"])
        current, future = (
            current_inverse @ world[index],
            current_inverse @ world[support],
        )
        state_part = [
            *current[:3, 3],
            *Rotation.from_matrix(current[:3, :3]).as_euler("ZYX"),
        ]
        positions = np.column_stack(
            [
                np.interp(query, np.arange(len(support)), future[:, axis, 3])
                for axis in range(3)
            ]
        )
        rotations = Slerp(
            np.arange(len(support)), Rotation.from_matrix(future[:, :3, :3])
        )(query).as_euler("ZYX")
        action_part = np.column_stack([positions, rotations])
        if mode == "width":
            state_part.append(raw[f"{side}.obs_gripper"][index, 0])
            grip = np.interp(
                query, np.arange(len(support)), raw[f"{side}.obs_gripper"][support, 0]
            )
            action_part = np.column_stack([action_part, grip])
        state.extend(state_part)
        action.append(action_part)
    return np.asarray(state), np.concatenate(action, axis=1)


@pytest.mark.parametrize("mode", ["zero", "width"])
@pytest.mark.parametrize("index", [0, 3, 10, 11])
def test_moving_camera_and_hand_independent_oracle_tail_and_current(
    tmp_path, mode, index
):
    path = tmp_path / "episode.zarr"
    raw = write_episode(path)
    spec = ActionTiming(30, 6, 2, 9)
    loaded = dataset(path, spec, mode)
    sample = loaded[index]
    expected_state, expected_action = oracle(raw, spec, index, mode)
    dimension = 12 if mode == "zero" else 14
    assert sample["embodiment"] == 3
    assert sample["observations.state.ee_pose"].shape == (dimension,)
    assert sample["actions_cartesian"].shape == (9, dimension)
    np.testing.assert_allclose(
        sample["observations.state.ee_pose"], expected_state, atol=3e-6
    )
    np.testing.assert_allclose(sample["actions_cartesian"], expected_action, atol=3e-6)
    torch.testing.assert_close(
        sample["actions_cartesian"][0], sample["observations.state.ee_pose"]
    )
    assert sample["observations.images.front_img_1"].shape == (3, 12, 16)
    assert not any("wrist_img" in key for key in sample)
    assert "obs_head_pose" not in sample
    validate_dataset_timing({"human_bimanual": spec}, {"human_bimanual": loaded}, {})


@pytest.mark.parametrize("mode", ["zero", "width"])
def test_static_world_hand_has_constant_future_pose_in_current_camera(tmp_path, mode):
    path = tmp_path / "episode.zarr"
    raw = write_episode(path, moving_hands=False)
    spec = ActionTiming(30, 6, 1, 11)
    sample = dataset(path, spec, mode)[2]
    actions = sample["actions_cartesian"].numpy()
    stride = 6 if mode == "zero" else 7
    for i, side in enumerate(("left", "right")):
        actual = actions[:, i * stride : i * stride + 6]
        np.testing.assert_allclose(
            actual, np.repeat(actual[:1], len(actual), axis=0), atol=1e-6
        )
        # An erroneous per-future-frame camera transform creates a moving target
        # even though this hand is stationary in world coordinates.
        wrong = (
            np.linalg.inv(matrices(raw["obs_head_pose"])[7])
            @ matrices(raw[f"{side}.obs_ee_pose"])[7]
        )
        assert np.linalg.norm(wrong[:3, 3] - actual[-1, :3]) > 0.05
    # The next observation uses the next current camera, so state does move.
    following = dataset(path, spec, mode)[3]["observations.state.ee_pose"]
    assert not torch.allclose(sample["observations.state.ee_pose"][:3], following[:3])


def test_zero_pose_matches_width_pose_and_original_human_behavior(tmp_path):
    path = tmp_path / "episode.zarr"
    write_episode(path)
    original_horizon = Human.ACTION_HORIZON
    original_intrinsics = Human.INTRINSICS.copy()
    spec = ActionTiming(30, 30, 3, 100)
    zero = dataset(path, spec, "zero")[2]
    width = dataset(path, spec, "width")[2]
    pose_indices = [*range(6), *range(7, 13)]
    torch.testing.assert_close(
        zero["actions_cartesian"], width["actions_cartesian"][:, pose_indices]
    )
    torch.testing.assert_close(
        zero["observations.state.ee_pose"],
        width["observations.state.ee_pose"][pose_indices],
    )
    old = ZarrDataset(
        path, Human.get_keymap("cartesian"), Human.get_transform_list("cartesian")
    )[2]
    torch.testing.assert_close(
        zero["actions_cartesian"],
        old["actions_cartesian"],
        check_dtype=False,
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        zero["observations.state.ee_pose"],
        old["observations.state.ee_pose"],
        check_dtype=False,
        atol=1e-6,
        rtol=1e-6,
    )
    assert Human.ACTION_HORIZON == original_horizon
    np.testing.assert_array_equal(Human.INTRINSICS, original_intrinsics)


def test_four_world_held_rows_are_preserved_without_holding_camera_frame(tmp_path):
    path = tmp_path / "held.zarr"
    raw = write_episode(path, held=True)
    spec = ActionTiming(30, 5, 1, 5)
    loaded = dataset(path, spec, "width")
    sample = loaded[2]
    for offset in (0, 7):
        chunk = sample["actions_cartesian"][:, offset : offset + 7].numpy()
        np.testing.assert_allclose(chunk, np.repeat(chunk[:1], 5, axis=0), atol=1e-6)
    assert np.count_nonzero(raw["left.hand_filled"]) == 4
    # Held world hand reprojects differently as current camera moves; storing
    # a fixed camera-space hand during loss would be the wrong convention.
    assert not torch.allclose(
        loaded[3]["observations.state.ee_pose"][:3],
        loaded[6]["observations.state.ee_pose"][:3],
    )
    group = zarr.open_group(str(path), mode="r")
    np.testing.assert_array_equal(
        group["left.obs_ee_pose"][:12], raw["left.obs_ee_pose"]
    )


def test_width_uses_fixed_saved_scale_not_episode_minmax_or_second_division(tmp_path):
    path = tmp_path / "width.zarr"
    raw = write_episode(path)
    spec = ActionTiming(30, 4, 1, 4)
    sample = dataset(path, spec, "width")[0]
    np.testing.assert_allclose(
        sample["actions_cartesian"][:, 6], raw["left.obs_gripper"][:4, 0]
    )
    np.testing.assert_allclose(
        sample["actions_cartesian"][:, 13], raw["right.obs_gripper"][:4, 0]
    )
    assert sample["actions_cartesian"][0, 6].item() == pytest.approx(0.1)
    assert sample["actions_cartesian"][-1, 6].item() < 1


def test_explicit_timing_and_modes_validate_interface(tmp_path):
    path = tmp_path / "episode.zarr"
    write_episode(path)
    spec = ActionTiming(30, 5, 2, 9)
    norm = PicoEgo.get_keymap(timing=asdict(spec), norm_mode=True, gripper_mode="width")
    assert not any(item["key_type"] == "camera_keys" for item in norm.values())
    assert norm["left.action_ee_pose"]["horizon"] == 5
    for mode in ("none", "zeroes", None):
        with pytest.raises(ValueError, match="gripper_mode"):
            PicoEgo.get_transform_list(timing=spec, gripper_mode=mode)
    with pytest.raises(TypeError):
        PicoEgo.get_keymap()
    with pytest.raises(ValueError, match="keymap disagrees"):
        PicoEgoDataset(
            path,
            PicoEgo.get_keymap(timing=spec),
            PicoEgo.get_transform_list(timing=spec),
            gripper_mode="width",
        )
    loaded = PicoEgoDataset(
        path,
        PicoEgo.get_keymap(timing=spec, gripper_mode="width"),
        PicoEgo.get_transform_list(timing=spec),
        gripper_mode="width",
    )
    with pytest.raises(ValueError, match="transforms disagree"):
        loaded[0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("embodiment", "piper_bimanual"),
        ("schema", "human"),
        ("reference_camera", "right"),
        ("pose_representation", "xyz+ypr"),
        ("pose_coordinates", {"left": "camera", "right": "camera"}),
        ("pico_ego", None),
        ("intrinsics", None),
        ("extrinsics", {"left": np.eye(4).tolist()}),
    ],
)
def test_bad_metadata_is_reported_with_episode_path(field, value):
    meta = metadata()
    meta[field] = value
    with pytest.raises(ValueError, match="specific_episode"):
        validate_pico_ego_metadata(meta, episode_path="specific_episode")


@pytest.mark.parametrize("bad", [0, -0.08, float("nan"), True, "0.08"])
def test_invalid_gripper_scale_rejected(bad):
    meta = metadata()
    meta["pico_ego"]["gripper_encoding"]["open_m"] = bad
    with pytest.raises(ValueError, match="open_m"):
        validate_pico_ego_metadata(meta)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_head",
        "bad_quat",
        "nan_pose",
        "width_mismatch",
        "width_outside_range",
        "both_invalid",
        "both_true",
        "time_gap",
        "time_mismatch",
        "missing_image",
        "timestamp_float_ns",
    ],
)
def test_invalid_data_rejected_before_iteration(tmp_path, corruption):
    path = tmp_path / "bad_segment.zarr"
    write_episode(path)
    group = zarr.open_group(str(path), mode="a")
    if corruption == "missing_head":
        del group["obs_head_pose"]
    elif corruption == "bad_quat":
        group["obs_head_pose"][1, 3:] = 0
    elif corruption == "nan_pose":
        group["right.obs_ee_pose"][1, 2] = np.nan
    elif corruption == "width_mismatch":
        group["left.obs_gripper"][2] = 0.5
    elif corruption == "width_outside_range":
        group["left.obs_gripper"][2] = 1.1
    elif corruption == "both_invalid":
        group["left.hand_valid"][2] = False
    elif corruption == "both_true":
        group["right.hand_filled"][2] = True
    elif corruption == "time_gap":
        group["timestamp_ns"][2] += 100_000_000
        group["timestamp"][2] = group["timestamp_ns"][2] / 1e9
    elif corruption == "time_mismatch":
        group["timestamp"][2] += 0.1
    elif corruption == "missing_image":
        del group["images.front_1"]
    elif corruption == "timestamp_float_ns":
        values = group["timestamp_ns"][:].astype(float)
        del group["timestamp_ns"]
        group.create_array("timestamp_ns", data=values)
    with pytest.raises(ValueError, match="bad_segment.zarr"):
        preflight_pico_ego_episode(path)


def test_does_not_read_next_episode_when_window_reaches_end(tmp_path):
    path = tmp_path / "first.zarr"
    raw = write_episode(path)
    other = tmp_path / "second.zarr"
    write_episode(other)
    second = zarr.open_group(str(other), mode="a")
    second["left.obs_ee_pose"][:, 0] = 100
    spec = ActionTiming(30, 20, 2, 17)
    actual = dataset(path, spec)[11]["actions_cartesian"]
    expected = oracle(raw, spec, 11, "zero")[1]
    np.testing.assert_allclose(actual, expected, atol=3e-6)
    np.testing.assert_allclose(
        actual, np.repeat(actual[:1], len(actual), axis=0), atol=3e-6
    )


def test_corrupt_jpeg_raises_instead_of_changing_frame(tmp_path):
    path = tmp_path / "episode.zarr"
    write_episode(path)
    group = zarr.open_group(str(path), mode="a")
    group["images.front_1"][2:3] = np.asarray([b"broken-jpeg"], dtype=object)
    loaded = dataset(path, ActionTiming(30, 5, 1, 9))
    assert loaded.strict_read is True
    with pytest.raises(Exception, match="JPEG|decode|strict"):
        loaded[2]


def test_metadata_validation_does_not_mutate_input():
    original = metadata()
    snapshot = copy.deepcopy(original)
    assert validate_pico_ego_metadata(original) == 0.08
    assert original == snapshot


@pytest.mark.parametrize("actual", [None, "piper", "unknown"])
def test_axes_must_match_explicit_robot_convention(actual):
    meta = metadata()
    meta["pico_ego"]["tool_frame"]["axis_convention"] = actual
    with pytest.raises(ValueError, match="expected_axis_convention"):
        validate_pico_ego_metadata(meta)


def test_explicit_non_x_axes_supported_without_any_hidden_rotation(tmp_path):
    path = tmp_path / "non_x.zarr"
    write_episode(path)
    store = zarr.open_group(str(path), mode="a")
    pico = copy.deepcopy(store.attrs["pico_ego"])
    pico["tool_frame"]["axis_convention"] = "piper"
    store.attrs["pico_ego"] = pico
    spec = ActionTiming(30, 5, 1, 9)
    with pytest.raises(ValueError, match="expected_axis_convention"):
        dataset(path, spec)
    loaded = PicoEgoDataset(
        path,
        PicoEgo.get_keymap(timing=spec),
        PicoEgo.get_transform_list(timing=spec),
        expected_axis_convention="piper",
    )
    assert loaded[0]["actions_cartesian"].shape == (9, 12)
    preflight_pico_ego_episode(path, expected_axis_convention="piper")
    with pytest.raises(ValueError, match="must be piper_x or piper"):
        preflight_pico_ego_episode(path, expected_axis_convention="unknown")
