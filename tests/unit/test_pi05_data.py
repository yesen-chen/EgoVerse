"""CPU-only parity and corruption checks for the JAX-independent Pi0.5 data boundary."""

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import simplejpeg
import zarr
from scipy.spatial.transform import Rotation

from egomimic.rldb.action_timing import ActionTiming
from egomimic.rldb.embodiment.piper import Piper
from egomimic.rldb.zarr.piper_dataset import PiperDataset
from egomimic.robot.piper_pico.pi05_data import (
    CAMERAS,
    Pi05Dataset,
    canonical_hash,
    compute_norm_stats,
    export_manifest,
    load_manifest,
)
from tests.unit.test_piper_dataset import se3, write_episode


@pytest.fixture
def episodes(tmp_path):
    transforms = {
        "left": se3([0.2, -0.1, 0.7], [0.4, -0.3, 0.15]),
        "right": se3([-0.6, 0.2, 0.5], [-0.5, 0.2, -0.25]),
    }
    paths = []
    for index in range(3):
        path = tmp_path / "converted" / f"episode_{index:06d}.zarr"
        write_episode(path, transforms)
        group = zarr.open_group(str(path), mode="a")
        group.attrs["task_description"] = f"Move object {index} into the container."
        # Deliberately different numeric distributions for split-leak checks.
        for side in ("left", "right"):
            for field in ("obs_ee_pose", "cmd_ee_pose"):
                values = group[f"{side}.{field}"][:8]
                values[:, 0] += index * 10
                group[f"{side}.{field}"][:8] = values
        paths.append(path)
    return paths


def manifest(tmp_path, episodes, **kwargs):
    return export_manifest(episodes, tmp_path / "pi05" / "manifest.json", **kwargs)


def resign(value):
    value["manifest_sha256"] = canonical_hash(
        {k: v for k, v in value.items() if k != "manifest_sha256"}
    )


@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("index", [0, 5, 7])
def test_parity_existing_training_pipeline_and_100_point_span(
    tmp_path, episodes, stride, index
):
    path = manifest(
        tmp_path, episodes, source_horizon=7, action_stride=stride, validation_ratio=0
    )
    dataset = Pi05Dataset(path)
    timing = ActionTiming(30, 7, stride, 100)
    original = PiperDataset(
        episodes[0],
        Piper.get_keymap(timing=timing),
        Piper.get_transform_list(timing=timing),
    )
    sample, expected = dataset[index], original[index]
    assert sample["actions"].shape == (100, 14)
    assert sample["observation.state"].shape == (14,)
    assert sample["actions"].dtype == np.float32
    np.testing.assert_allclose(
        sample["actions"], expected["actions_cartesian"].numpy(), atol=2e-6
    )
    np.testing.assert_allclose(
        sample["observation.state"],
        expected["observations.state.ee_pose"].numpy(),
        atol=2e-6,
    )
    assert dataset.manifest["action_timing"] == timing.to_dict()
    assert sample["prompt"] == "Move object 0 into the container."
    # Pixel processing happened in the original converter, not this adapter.
    group = zarr.open_group(str(episodes[0]), mode="r")
    for camera in CAMERAS:
        value = group[f"images.{camera}"][index : index + 1][0]
        if isinstance(value, np.ndarray):
            value = value.item()
        pixels = simplejpeg.decode_jpeg(bytes(value), colorspace="RGB")
        np.testing.assert_array_equal(sample[f"observation.images.{camera}"], pixels)


def test_terminal_repeat_never_reads_next_episode(tmp_path, episodes):
    path = manifest(tmp_path, episodes, validation_ratio=0)
    ds = Pi05Dataset(path, include_images=False)
    final_first, start_second = ds[7], ds[8]
    np.testing.assert_allclose(
        final_first["actions"],
        np.repeat(final_first["actions"][:1], 100, axis=0),
        atol=1e-6,
    )
    assert not np.allclose(final_first["actions"][0], start_second["actions"][0])
    assert set(final_first) == {"actions", "observation.state", "prompt"}
    assert len(ds) == 24
    for bad in (-1, len(ds), 0.5):
        with pytest.raises(IndexError):
            ds[bad]


def test_so3_short_arc_at_plus_minus_179_degrees(tmp_path, episodes):
    source = episodes[0]
    group = zarr.open_group(str(source), mode="a")
    for side in ("left", "right"):
        camera = np.asarray(group.attrs["extrinsics"][side])
        pose = group[f"{side}.cmd_ee_pose"][:8]
        for idx, yaw in enumerate([179, -179]):
            R = Rotation.from_matrix(camera[:3, :3]) * Rotation.from_euler(
                "Z", yaw, degrees=True
            )
            pose[idx, 3:] = R.as_quat()[[3, 0, 1, 2]]
        group[f"{side}.cmd_ee_pose"][:8] = pose
    path = manifest(tmp_path, [source], source_horizon=2, validation_ratio=0)
    action = Pi05Dataset(path, include_images=False)[0]["actions"]
    for column in (3, 10):
        rotations = Rotation.from_euler("ZYX", action[:, column : column + 3])
        steps = (rotations[:-1].inv() * rotations[1:]).magnitude()
        assert np.max(steps) < np.deg2rad(0.021)
        assert np.all(np.abs(action[:, column]) > np.deg2rad(178.9))


def test_split_is_deterministic_disjoint_and_norm_stats_train_only(tmp_path, episodes):
    path = manifest(tmp_path, episodes, validation_ratio=0.34, seed=7)
    other = export_manifest(
        list(reversed(episodes)), tmp_path / "other.json", validation_ratio=0.34, seed=7
    )
    a, b = load_manifest(path), load_manifest(other)
    split_a = {Path(r["path"]).name: r["split"] for r in a["episodes"]}
    split_b = {Path(r["path"]).name: r["split"] for r in b["episodes"]}
    assert split_a == split_b
    train = Pi05Dataset(path, split="train", include_images=False)
    validation = Pi05Dataset(path, split="validation", include_images=False)
    assert len(train) == 16 and len(validation) == 8
    assert {r["path"] for r in train.episodes}.isdisjoint(
        r["path"] for r in validation.episodes
    )
    stats = compute_norm_stats(path)
    for name, key in (("state", "observation.state"), ("actions", "actions")):
        values = np.concatenate(
            [train[i][key].reshape(-1, 14) for i in range(len(train))]
        ).astype(float)
        assert np.asarray(stats[name]["mean"]).shape == (14,)
        np.testing.assert_allclose(stats[name]["mean"], values.mean(axis=0))
        np.testing.assert_allclose(stats[name]["std"], values.std(axis=0))
        np.testing.assert_allclose(
            stats[name]["q01"], np.quantile(values, 0.01, axis=0)
        )
        np.testing.assert_allclose(
            stats[name]["q99"], np.quantile(values, 0.99, axis=0)
        )
    stats_small = compute_norm_stats(path, max_windows=3)
    assert len(stats_small["actions"]["mean"]) == 14


def test_manifest_is_portable_and_does_not_copy_images(tmp_path, episodes):
    path = manifest(tmp_path, episodes, validation_ratio=0)
    original = Pi05Dataset(path)[0]
    assert list(path.parent.iterdir()) == [path]
    assert all(
        not Path(r["path"]).is_absolute() for r in load_manifest(path)["episodes"]
    )
    moved = tmp_path / "relocated"
    moved.mkdir()
    shutil.move(str(tmp_path / "converted"), moved)
    shutil.move(str(path.parent), moved)
    restored = Pi05Dataset(moved / "pi05" / "manifest.json")[0]
    for key in original.keys() - {"prompt"}:
        np.testing.assert_array_equal(original[key], restored[key])


@pytest.mark.parametrize(
    "mutation", ["manifest", "timing", "dimension", "numeric", "metadata"]
)
def test_corruption_is_rejected(tmp_path, episodes, mutation):
    path = manifest(tmp_path, episodes, validation_ratio=0)
    value = load_manifest(path)
    if mutation in ("manifest", "timing", "dimension"):
        if mutation == "manifest":
            value["episodes"][0]["prompt"] = "different task"
        elif mutation == "timing":
            value["action_timing"]["effective_span_s"] = 100 / 30
            resign(value)
        else:
            value["action_contract"]["dimension"] = 7
            resign(value)
        path.write_text(json.dumps(value))
    else:
        group = zarr.open_group(str(episodes[0]), mode="a")
        if mutation == "numeric":
            group["left.obs_gripper"][0] = [0.25]
        else:
            group.attrs["task_description"] = "changed after export"
    with pytest.raises(ValueError, match="checksum|span|contract|changed"):
        Pi05Dataset(path)


def test_prompt_and_geometry_must_be_explicit(tmp_path, episodes):
    source = episodes[0]
    group = zarr.open_group(str(source), mode="a")
    group.attrs["task_description"] = ""
    with pytest.raises(ValueError, match="prompt required"):
        manifest(tmp_path, [source], validation_ratio=0)
    path = manifest(tmp_path, [source], validation_ratio=0, prompt="Pick the object.")
    assert Pi05Dataset(path)[0]["prompt"] == "Pick the object."
    with pytest.raises(FileExistsError):
        export_manifest([source], path, validation_ratio=0, prompt="x")
    changed = copy.deepcopy(group.attrs["calibration"])
    changed["data"]["extrinsics"]["left"][0][3] += 0.5
    group.attrs["calibration"] = changed
    with pytest.raises(ValueError, match="extrinsics disagree"):
        export_manifest([source], tmp_path / "bad.json", validation_ratio=0, prompt="x")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"validation_ratio": 1}, "validation_ratio"),
        ({"action_stride": 30}, "support"),
        ({"source_fps": 25}, "fps disagrees"),
        ({"validation_ratio": 0.1}, "at least two episodes"),
    ],
)
def test_invalid_contract_options(tmp_path, episodes, kwargs, message):
    with pytest.raises(ValueError, match=message):
        manifest(tmp_path, episodes[:1], **kwargs)


def test_incompatible_gripper_scales_are_not_silently_pooled(tmp_path, episodes):
    group = zarr.open_group(str(episodes[1]), mode="a")
    grip = dict(group.attrs["gripper_encoding"])
    grip["open_m"] = 0.07
    group.attrs["gripper_encoding"] = grip
    with pytest.raises(ValueError, match="contract differs"):
        manifest(tmp_path, episodes)


def test_multiple_calibrations_preserve_per_episode_transform(tmp_path, episodes):
    group = zarr.open_group(str(episodes[1]), mode="a")
    calibration = copy.deepcopy(group.attrs["calibration"])
    for side in ("left", "right"):
        calibration["data"]["extrinsics"][side] = np.eye(4).tolist()
    group.attrs["calibration"] = calibration
    group.attrs["extrinsics"] = calibration["data"]["extrinsics"]
    path = manifest(tmp_path, episodes, validation_ratio=0)
    value = load_manifest(path)
    assert len(value["training_contracts"]) == 2
    sample = Pi05Dataset(path, include_images=False)[8]
    np.testing.assert_allclose(
        sample["observation.state"][:3], group["left.obs_ee_pose"][0][:3]
    )
    first = Pi05Dataset(path, include_images=False)[0]
    source = zarr.open_group(str(episodes[0]), mode="r")
    assert not np.allclose(
        first["observation.state"][:3], source["left.obs_ee_pose"][0][:3]
    )


def test_export_cli_and_converter_opt_in(tmp_path, episodes, monkeypatch):
    from egomimic.scripts import export_piper_pi05, piper_pico_to_zarr

    export_piper_pi05.main(
        [
            "--input",
            str(episodes[0].parent),
            "--output",
            str(tmp_path / "cli.json"),
            "--source-horizon",
            "6",
            "--action-stride",
            "2",
            "--validation-ratio",
            "0",
        ]
    )
    assert (
        load_manifest(tmp_path / "cli.json")["action_timing"]["output_chunk_length"]
        == 100
    )
    monkeypatch.setattr(
        piper_pico_to_zarr, "convert_dataset", lambda *a, **kw: episodes
    )
    output = tmp_path / "converted_manifest.json"
    piper_pico_to_zarr.main(
        [
            "--input",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "unused"),
            "--pi05-manifest",
            str(output),
            "--pi05-source-horizon",
            "7",
            "--pi05-action-stride",
            "3",
            "--pi05-validation-ratio",
            "0",
            "--pi05-prompt",
            "A converted task",
        ]
    )
    result = load_manifest(output)
    assert result["action_timing"]["support_indices"] == [0, 3, 6]
    assert result["episodes"][0]["prompt"] == "A converted task"


def test_import_has_no_training_framework_or_hardware_dependency():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import egomimic.robot.piper_pico.pi05_data; assert not ({'torch', 'jax', 'piper_sdk'} & sys.modules.keys())",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
