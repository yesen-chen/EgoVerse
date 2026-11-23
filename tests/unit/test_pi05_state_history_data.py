"""CPU regression tests for timestamp-based Pi0.5 robot/Ego history inputs."""

import copy
import json
import shutil

import numpy as np
import pytest
import zarr
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_pico.pi05_data import (
    Pi05Dataset,
    export_manifest,
    history_to_camera,
    load_manifest,
)
from egomimic.robot.piper_pico.pi05_ego_data import (
    EgoPi05Dataset,
    export_ego_manifest,
    load_ego_manifest,
)
from egomimic.robot.piper_pico.pi05_training_data import (
    TrainingDataset,
    export_mixed_manifest,
    load_training_manifest,
)
from egomimic.robot.piper_pico.state_history import write_feedback_sidecar
from tests.unit.test_pi05_ego_data import write_ego
from tests.unit.test_piper_dataset import se3, write_episode


def piper_episode(root, *, feedback=True, first_offset=-0.2, future_shift=0):
    path = root / "episode_000000.zarr"
    transforms = {
        "left": se3([0.1, -0.2, 0.3], [0.2, 0.1, -0.3]),
        "right": se3([-0.1, 0.3, 0.2], [-0.2, 0.2, 0.1]),
    }
    write_episode(path, transforms)
    group = zarr.open_group(str(path), mode="a")
    group.attrs["task_description"] = "Pick the object."
    anchors = group["timestamp"][:8]
    for side in ("left", "right"):
        group.create_array(f"source.{side}.robot_timestamp", data=anchors - 0.011)
    if feedback:
        raw = root / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        offsets = np.arange(first_offset, 0.31, 0.01)
        for arm in range(2):
            poses = np.column_stack(
                [
                    offsets + arm,
                    offsets * 2,
                    offsets * -0.5,
                    np.zeros_like(offsets),
                    np.zeros_like(offsets),
                    offsets * 0.4,
                ]
            )
            poses[offsets > 0.06, 0] += future_shift
            np.savez(
                raw / f"robot{arm}_state.npz",
                eef_pose=poses,
                eef_pose__timestamp=1234 + offsets,
                gripper_width=(0.03 + offsets * 0.02)[:, None],
                gripper_width__timestamp=1234 + offsets,
            )
        write_feedback_sidecar(raw, path, allow_missing=False)
    return path, transforms


def export_piper(root, path, offsets=()):
    return export_manifest(
        [path],
        root / "manifest.json",
        validation_ratio=0,
        state_history_offsets_s=offsets,
    )


@pytest.mark.parametrize(
    "offsets", [(-0.025,), (-0.05, -0.025), (-0.09, -0.047, -0.021)]
)
def test_robot_arbitrary_seconds_history_gripper_and_camera_transform(
    tmp_path, offsets
):
    path, transforms = piper_episode(tmp_path / "data")
    manifest = export_piper(tmp_path, path, offsets)
    dataset = Pi05Dataset(manifest, include_images=False)
    sample = dataset[2]
    times = 2 / 30 + np.array(offsets)
    actual = sample["observation.state_history"]
    assert actual.shape == (len(offsets), 14)
    for arm, side in enumerate(("left", "right")):
        world_positions = np.column_stack([times + arm, times * 2, times * -0.5])
        camera_base = np.linalg.inv(transforms[side])
        expected_positions = (
            world_positions @ camera_base[:3, :3].T + camera_base[:3, 3]
        )
        np.testing.assert_allclose(
            actual[:, arm * 7 : arm * 7 + 3], expected_positions, atol=1e-6
        )
        expected_rot = Rotation.from_matrix(camera_base[:3, :3]) * Rotation.from_euler(
            "Z", times[:, None] * 0.4
        )
        got_rot = Rotation.from_euler("ZYX", actual[:, arm * 7 + 3 : arm * 7 + 6])
        np.testing.assert_allclose(
            (expected_rot.inv() * got_rot).magnitude(), 0, atol=1e-6
        )
        np.testing.assert_allclose(
            actual[:, arm * 7 + 6], (0.03 + times * 0.02) / 0.08, atol=1e-6
        )
    old = Pi05Dataset(export_piper(tmp_path / "old", path), include_images=False)[2]
    np.testing.assert_array_equal(sample["observation.state"], old["observation.state"])
    np.testing.assert_array_equal(sample["actions"], old["actions"])


def test_default_contract_and_samples_stay_identical_with_sidecars(tmp_path):
    path, _ = piper_episode(tmp_path / "data")
    first = export_piper(tmp_path / "first", path)
    second = export_piper(tmp_path / "second", path, [])
    a, b = load_manifest(first), load_manifest(second)
    assert a == b and "state_history" not in a
    assert "eligible_indices" not in a["episodes"][0]
    assert set(Pi05Dataset(first, include_images=False)[0]) == {
        "observation.state",
        "actions",
        "prompt",
    }


def test_future_feedback_cannot_change_history(tmp_path):
    a, _ = piper_episode(tmp_path / "a")
    b, _ = piper_episode(tmp_path / "b", future_shift=1000)
    da = Pi05Dataset(
        export_piper(tmp_path / "ma", a, (-0.05, -0.025)), include_images=False
    )
    db = Pi05Dataset(
        export_piper(tmp_path / "mb", b, (-0.05, -0.025)), include_images=False
    )
    np.testing.assert_array_equal(
        da[2]["observation.state_history"], db[2]["observation.state_history"]
    )


def test_availability_cutoff_is_bound_and_short_history_is_not_clipped(tmp_path):
    path, _ = piper_episode(tmp_path / "data", first_offset=0)
    manifest = export_piper(tmp_path, path, (-0.05, -0.025))
    record = load_manifest(manifest)["episodes"][0]
    assert record["eligible_indices"] == [2, 3, 4, 5, 6, 7]
    ds = Pi05Dataset(manifest, include_images=False)
    assert len(ds) == 6
    assert ds.state_history_offsets_s == (-0.05, -0.025)
    assert ds.source_location(0) == (0, 2)
    assert ds.index_for_source(0, 2) == 0
    assert ds.index_for_source(0, 7) == 5
    with pytest.raises(IndexError, match="no eligible"):
        ds.index_for_source(0, 1)
    legacy = Pi05Dataset(export_piper(tmp_path / "old", path), include_images=False)
    np.testing.assert_array_equal(
        ds[0]["observation.state"], legacy[2]["observation.state"]
    )
    # A target inside the observation/query gap has no causal right endpoint.
    with pytest.raises(ValueError, match="no sample"):
        export_piper(tmp_path / "too_recent", path, (-0.001,))
    group = zarr.open_group(str(path), mode="a")
    group["source.left.robot_timestamp"][2] -= 0.001
    with pytest.raises(ValueError, match="source/coverage changed"):
        Pi05Dataset(manifest, include_images=False)


@pytest.mark.parametrize("offsets", [(), (-0.05, -0.025)])
def test_source_index_roundtrip_across_episode_boundary(tmp_path, offsets):
    paths = [piper_episode(tmp_path / str(i), first_offset=0)[0] for i in range(2)]
    manifest = export_manifest(
        paths,
        tmp_path / "manifest.json",
        validation_ratio=0,
        state_history_offsets_s=offsets,
    )
    ds = Pi05Dataset(manifest, include_images=False)
    expected_rows = list(range(2 if offsets else 0, 8))
    expected = [(episode, row) for episode in range(2) for row in expected_rows]
    assert [ds.source_location(i) for i in range(len(ds))] == expected
    assert [ds.index_for_source(*location) for location in expected] == list(
        range(len(ds))
    )
    if offsets:
        # Each episode loses its own uncovered leading rows; the preceding
        # episode must never supply history for row zero of the next one.
        for episode in range(2):
            with pytest.raises(IndexError, match="no eligible"):
                ds.index_for_source(episode, 0)
    for index in (-1, len(ds), 0.5):
        with pytest.raises(IndexError):
            ds.source_location(index)
    for episode, row in [(-1, 0), (2, 0), (0, -1), (0, 8), (0, 1.5)]:
        with pytest.raises(IndexError):
            ds.index_for_source(episode, row)


@pytest.mark.parametrize("case", ["no_sidecar", "no_cutoff", "corrupt_sidecar"])
def test_required_history_sources_are_not_silently_replaced(tmp_path, case):
    path, _ = piper_episode(tmp_path / "data", feedback=case != "no_sidecar")
    if case == "no_cutoff":
        del zarr.open_group(str(path), mode="a")["source.left.robot_timestamp"]
    if case == "corrupt_sidecar":
        sidecar = path / "feedback_history" / "left.npz"
        sidecar.write_bytes(sidecar.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="history|checksum"):
        export_piper(tmp_path, path, (-0.05,))


def test_sidecar_bytes_change_manifest_identity_and_portable_move_works(tmp_path):
    a, _ = piper_episode(tmp_path / "a")
    b, _ = piper_episode(tmp_path / "b", future_shift=1)
    ma = export_piper(tmp_path / "ma", a, (-0.05,))
    mb = export_piper(tmp_path / "mb", b, (-0.05,))
    assert (
        load_manifest(ma)["data_identity_sha256"]
        != load_manifest(mb)["data_identity_sha256"]
    )
    expected = Pi05Dataset(ma, include_images=False)[0]
    moved = tmp_path / "relocated"
    moved.mkdir()
    shutil.move(str(tmp_path / "a"), moved / "a")
    shutil.move(str(tmp_path / "ma"), moved / "ma")
    restored = Pi05Dataset(moved / "ma" / "manifest.json", include_images=False)[0]
    np.testing.assert_array_equal(
        expected["observation.state_history"], restored["observation.state_history"]
    )


@pytest.mark.parametrize("mode", ["width", "zero"])
def test_ego_history_uses_seconds_world_motion_and_current_camera(tmp_path, mode):
    path = write_ego(tmp_path / "data", frames=36)
    group = zarr.open_group(str(path), mode="a")
    times = group["timestamp"][:36]
    t = times - times[0]
    head = group["obs_head_pose"][:36]
    head[:, :3] = np.column_stack([t, t * -2, t * 0])
    head[:, 3:] = Rotation.from_euler("Z", t[:, None]).as_quat()[:, [3, 0, 1, 2]]
    group["obs_head_pose"][:36] = head
    maximum = group.attrs["pico_ego"]["gripper_encoding"]["open_m"]
    for side in ("left", "right"):
        poses = group[f"{side}.obs_ee_pose"][:36]
        poses[:, :3], poses[:, 3:] = [0.3, 0.2, 1], [1, 0, 0, 0]
        group[f"{side}.obs_ee_pose"][:36] = poses
        group[f"{side}.obs_gripper"][:36] = (0.2 + t * 0.1)[:, None]
        group[f"{side}.obs_gripper_width_m"][:36] = ((0.2 + t * 0.1) * maximum)[:, None]
    manifest = export_ego_manifest(
        [path],
        tmp_path / "ego.json",
        validation_ratio=0,
        gripper_mode=mode,
        state_history_offsets_s=(-0.05, -0.025),
    )
    ds = EgoPi05Dataset(manifest, include_images=False)
    assert ds.episodes[0]["eligible_indices"] == [2, 3, 4, 5, 6]
    sample = ds[2]  # Original row 4, not historical camera row 2 or 3.
    for arm in range(2):
        history = sample["observation.state_history"][:, arm * 7 : (arm + 1) * 7]
        np.testing.assert_allclose(
            history[:, :6],
            np.repeat(
                sample["observation.state"][None, arm * 7 : arm * 7 + 6], 2, axis=0
            ),
            atol=1e-6,
        )
        expected = (
            0.2 + (t[4] + np.array([-0.05, -0.025])) * 0.1
            if mode == "width"
            else [0, 0]
        )
        np.testing.assert_allclose(history[:, 6], expected, atol=1e-6)
    legacy = export_ego_manifest(
        [path], tmp_path / "old.json", validation_ratio=0, gripper_mode=mode
    )
    old = EgoPi05Dataset(legacy, include_images=False)[4]
    np.testing.assert_array_equal(sample["observation.state"], old["observation.state"])
    np.testing.assert_array_equal(sample["actions"], old["actions"])


def test_ego_history_never_borrows_previous_episode(tmp_path):
    paths = [
        write_ego(tmp_path / f"data{i}", name=f"episode{i}", reason=None)
        for i in range(2)
    ]
    manifest = export_ego_manifest(
        paths,
        tmp_path / "ego.json",
        validation_ratio=0,
        state_history_offsets_s=(-0.05,),
    )
    value = load_ego_manifest(manifest)
    assert all(r["eligible_indices"][0] == 2 for r in value["episodes"])
    with pytest.raises(ValueError, match="no sample"):
        export_ego_manifest(
            paths,
            tmp_path / "bad.json",
            validation_ratio=0,
            state_history_offsets_s=(-100,),
        )


@pytest.mark.parametrize("ego_offsets", [(-0.05,), (), (-0.08,)])
def test_mixed_history_contract_must_match(tmp_path, ego_offsets):
    robot, _ = piper_episode(tmp_path / "robot")
    p = export_piper(tmp_path / "pm", robot, (-0.05,))
    human = write_ego(tmp_path / "human")
    e = export_ego_manifest(
        [human],
        tmp_path / "ego.json",
        validation_ratio=0,
        state_history_offsets_s=ego_offsets,
    )
    if ego_offsets != (-0.05,):
        with pytest.raises(ValueError, match="identical state_history"):
            export_mixed_manifest(p, e, tmp_path / "mixed.json")
        return
    mixed = export_mixed_manifest(p, e, tmp_path / "mixed.json")
    manifest = load_training_manifest(mixed)
    assert manifest["state_history"] == load_manifest(p)["state_history"]
    dataset = TrainingDataset(mixed, include_images=False)
    assert all(
        dataset[i]["observation.state_history"].shape == (1, 14)
        for i in range(len(dataset))
    )
    changed = copy.deepcopy(manifest)
    changed["state_history"]["offsets_s"] = [-0.1]
    # Contract is tied to the whole manifest, not a CLI override at train time.
    (tmp_path / "tampered.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="checksum"):
        load_training_manifest(tmp_path / "tampered.json")


def test_feedback_gripper_overshoot_is_preserved_like_current_state():
    history = np.zeros((2, 14))
    history[:, [6, 13]] = [[-0.001, 0.081], [0.09, -0.002]]
    result = history_to_camera(history, {"left": np.eye(4), "right": np.eye(4)}, 0.08)
    np.testing.assert_allclose(result[:, [6, 13]], history[:, [6, 13]] / 0.08)
