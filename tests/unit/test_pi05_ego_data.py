"""Offline Ego Pi0.5 geometry, clock, boundary, provenance and split acceptance."""

import copy
import hashlib
import shutil
import subprocess
import sys

import numpy as np
import pytest
import zarr
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_pico.image_geometry import make_image_geometry
from egomimic.robot.piper_pico.pi05_data import canonical_hash
from egomimic.robot.piper_pico.pi05_ego_data import (
    EgoPi05Dataset,
    export_ego_manifest,
    load_ego_manifest,
    validate_ego_manifest,
)
from tests.fixtures.piper_pico_training import write_recording


def write_ego(root, name="session", *, frames=36, reason="ui_trim", source=None):
    path = write_recording(root, "pico_ego", name, frames=frames)
    store = zarr.open_group(str(path), mode="a")
    K = [[25.0, 0, 16], [0, 25.0, 16], [0, 0, 1]]
    geometry = make_image_geometry((32, 32), (32, 32), K=K, D=[0.0] * 5)
    store.attrs.update(
        {
            "task_description": "Pick up the object and place it in the container.",
            "image_space": "processed",
            "image_geometry": {"front_1": geometry},
            "intrinsics": {"front_1": geometry["K_output"]},
            "distortion_coefficients": {"front_1": [0.0] * 5},
        }
    )
    if reason is not None:
        store.attrs["preprocessing"] = {
            "schema": "ego_preprocessing_v1",
            "end_reason": reason,
            "start_reason": "ui_trim",
        }
    if source is not None:
        pico = copy.deepcopy(store.attrs["pico_ego"])
        pico["source_tracking"]["sha256"] = hashlib.sha256(source.encode()).hexdigest()
        store.attrs["pico_ego"] = pico
    return path


def export(tmp_path, paths, **kwargs):
    return export_ego_manifest(
        paths, tmp_path / "manifest.json", validation_ratio=0, **kwargs
    )


def resign(value):
    value["manifest_sha256"] = canonical_hash(
        {k: v for k, v in value.items() if k != "manifest_sha256"}
    )


def test_default_30_to_100_preserves_grid_and_current_camera_frame(tmp_path):
    path = write_ego(tmp_path)
    store = zarr.open_group(str(path), mode="a")
    camera = store["obs_head_pose"][:36]
    camera[:, :3] = np.column_stack([np.arange(36) / 30, np.zeros((36, 2))])
    camera[:, 3:] = Rotation.from_euler("Z", np.arange(36)[:, None] * 0.01).as_quat()[
        :, [3, 0, 1, 2]
    ]
    store["obs_head_pose"][:36] = camera
    for side in ("left", "right"):
        pose = store[f"{side}.obs_ee_pose"][:36]
        pose[:, :3] = [0.3, 0.1, 1.0]
        pose[:, 3:] = [1, 0, 0, 0]
        store[f"{side}.obs_ee_pose"][:36] = pose
    original_ns = store["timestamp_ns"][:36].copy()
    manifest = export(tmp_path, [path])
    dataset = EgoPi05Dataset(manifest)
    assert len(dataset) == 36 - 30 + 1
    sample = dataset[5]
    assert sample["actions"].shape == (100, 14)
    assert sample["observation.state"].shape == (14,)
    assert sample["actions"].dtype == np.float32
    assert sample["observation.images.front_1"].shape == (32, 32, 3)
    assert "observation.images.left_wrist" not in sample
    assert sample["domain"] == "ego" and sample["gripper_mode"] == "zero"
    # A stationary world hand must remain stationary throughout this chunk,
    # despite the camera moving at every future source row.
    np.testing.assert_allclose(
        sample["actions"],
        np.repeat(sample["observation.state"][None], 100, axis=0),
        atol=1e-6,
    )
    expected_xyz = Rotation.from_euler("Z", -0.05).apply(
        np.array([0.3, 0.1, 1.0]) - camera[5, :3]
    )
    np.testing.assert_allclose(sample["observation.state"][:3], expected_xyz, atol=1e-6)
    assert not np.allclose(dataset[0]["actions"], sample["actions"])
    timing = load_ego_manifest(manifest)["action_timing"]
    assert timing["source_span_s"] == pytest.approx(29 / 30)
    assert timing["effective_span_s"] == pytest.approx(29 / 30)
    assert timing["output_offsets_s"][-1] == pytest.approx(29 / 30)
    np.testing.assert_array_equal(store["timestamp_ns"][:36], original_ns)


@pytest.mark.parametrize("mode", ["zero", "width"])
def test_linear_motion_and_width_interpolation(tmp_path, mode):
    path = write_ego(tmp_path)
    store = zarr.open_group(str(path), mode="a")
    camera = store["obs_head_pose"][:36]
    camera[:] = [0, 0, 0, 1, 0, 0, 0]
    store["obs_head_pose"][:36] = camera
    for side in ("left", "right"):
        pose = store[f"{side}.obs_ee_pose"][:36]
        pose[:, :3] = np.column_stack([np.arange(36) / 30, np.zeros(36), np.ones(36)])
        pose[:, 3:] = Rotation.from_euler("Z", np.arange(36)[:, None] * 0.02).as_quat()[
            :, [3, 0, 1, 2]
        ]
        store[f"{side}.obs_ee_pose"][:36] = pose
        width = np.arange(36)[:, None] / 36
        store[f"{side}.obs_gripper"][:36] = width
        store[f"{side}.obs_gripper_width_m"][:36] = width * 0.08
    dataset = EgoPi05Dataset(
        export(tmp_path, [path], gripper_mode=mode), include_images=False
    )
    sample = dataset[3]
    np.testing.assert_allclose(
        sample["actions"][:, 0], np.linspace(3, 32, 100) / 30, atol=1e-6
    )
    np.testing.assert_allclose(
        sample["actions"][:, 3], np.linspace(3, 32, 100) * 0.02, atol=1e-6
    )
    for column in (6, 13):
        np.testing.assert_allclose(
            sample["actions"][:, column],
            np.linspace(3, 32, 100) / 36 if mode == "width" else 0,
            atol=1e-6,
        )
        assert sample["observation.state"][column] == pytest.approx(
            3 / 36 if mode == "width" else 0
        )


@pytest.mark.parametrize(
    "trim,inactivity", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize(
    "reason", ["ui_trim", "inactivity", "tracking_gap", "video_gap", "source_end", None]
)
def test_boundary_switches_use_full_source_horizon_before_stride(
    tmp_path, trim, inactivity, reason
):
    path = write_ego(tmp_path, frames=12, reason=reason)
    strict = (
        trim
        if reason == "ui_trim"
        else inactivity
        if reason in ("inactivity", "tracking_gap", "video_gap")
        else False
    )
    manifest = export(
        tmp_path,
        [path],
        source_horizon=6,
        action_stride=2,
        require_full_window_after_trim=trim,
        require_full_window_after_inactivity=inactivity,
    )
    dataset = EgoPi05Dataset(manifest, include_images=False)
    assert len(dataset) == (7 if strict else 12)
    assert dataset.episodes[0]["total_frames"] == 12
    assert dataset.manifest["action_timing"]["effective_span_s"] == pytest.approx(
        4 / 30
    )
    if not strict:
        action = dataset[11]["actions"]
        np.testing.assert_allclose(
            action, np.repeat(action[:1], 100, axis=0), atol=1e-6
        )
    for bad in (-1, len(dataset), True, 0.5):
        with pytest.raises(IndexError):
            dataset[bad]
    assert zarr.open_group(str(path), mode="r")["timestamp_ns"].shape[0] >= 12


def test_short_strict_fragment_rejected_without_deleting_source(tmp_path):
    path = write_ego(tmp_path, frames=29)
    with pytest.raises(ValueError, match="29 rows.*30"):
        export(tmp_path, [path])
    assert not (tmp_path / "manifest.json").exists()
    assert zarr.open_group(str(path), mode="r").attrs["total_frames"] == 29


def test_source_recording_fragments_split_together_and_deterministically(tmp_path):
    paths = [
        write_ego(tmp_path, f"{source}/segment{segment}", source=source)
        for source in ("a", "b", "c")
        for segment in range(2)
    ]
    a = export_ego_manifest(paths, tmp_path / "a.json", validation_ratio=0.34, seed=17)
    b = export_ego_manifest(
        list(reversed(paths)), tmp_path / "b.json", validation_ratio=0.34, seed=17
    )
    va, vb = load_ego_manifest(a), load_ego_manifest(b)
    assert va["episodes"] == vb["episodes"]
    groups = {}
    for record in va["episodes"]:
        groups.setdefault(record["source_recording_sha256"], set()).add(record["split"])
    assert len(groups) == 3 and all(len(splits) == 1 for splits in groups.values())
    assert len(EgoPi05Dataset(a, "train", False)) == 4 * 7
    assert len(EgoPi05Dataset(a, "validation", False)) == 2 * 7
    one_source = paths[:2]
    with pytest.raises(ValueError, match="two source recordings"):
        export_ego_manifest(one_source, tmp_path / "bad.json", validation_ratio=0.1)


@pytest.mark.parametrize(
    "mutation", ["checksum", "timing", "eligible", "split", "numeric", "metadata"]
)
def test_corruption_and_split_leakage_rejected(tmp_path, mutation):
    paths = [write_ego(tmp_path, f"segment{i}", source="same") for i in range(2)]
    manifest = export(tmp_path, paths)
    if mutation in ("numeric", "metadata"):
        store = zarr.open_group(str(paths[0]), mode="a")
        if mutation == "numeric":
            values = store["left.obs_ee_pose"][:36]
            values[2, 0] += 0.01
            store["left.obs_ee_pose"][:36] = values
        else:
            store.attrs["task_description"] = "changed prompt"
        with pytest.raises(ValueError, match="source changed"):
            EgoPi05Dataset(manifest, include_images=False)
        return
    value = load_ego_manifest(manifest)
    if mutation == "checksum":
        value["episodes"][0]["prompt"] = "changed prompt"
    elif mutation == "timing":
        value["action_timing"]["effective_span_s"] = 1.6333
    elif mutation == "eligible":
        value["episodes"][0]["eligible_starts"] += 1
    else:
        value["episodes"][0]["split"] = "validation"
    if mutation != "checksum":
        resign(value)
    with pytest.raises(ValueError):
        validate_ego_manifest(value)


@pytest.mark.parametrize(
    "mutation", ["timestamp", "quaternion", "width", "validity", "geometry", "axes"]
)
def test_numerical_and_geometry_invariants(tmp_path, mutation):
    path = write_ego(tmp_path)
    store = zarr.open_group(str(path), mode="a")
    if mutation == "timestamp":
        values = store["timestamp_ns"][:36]
        values[3] += 1_000_000
        store["timestamp_ns"][:36] = values
    elif mutation == "quaternion":
        values = store["obs_head_pose"][:36]
        values[2, 3:] = 0
        store["obs_head_pose"][:36] = values
    elif mutation == "width":
        values = store["left.obs_gripper"][:36]
        values[2] = 1.5
        store["left.obs_gripper"][:36] = values
    elif mutation == "validity":
        values = store["left.hand_valid"][:36]
        values[2] = False
        store["left.hand_valid"][:36] = values
    elif mutation == "geometry":
        store.attrs["intrinsics"] = {"front_1": np.eye(3).tolist()}
    else:
        pico = copy.deepcopy(store.attrs["pico_ego"])
        pico["tool_frame"]["axis_convention"] = "piper"
        store.attrs["pico_ego"] = pico
    with pytest.raises(ValueError):
        export(tmp_path, [path])


def test_rotation_interpolation_uses_so3_short_arc(tmp_path):
    path = write_ego(tmp_path)
    store = zarr.open_group(str(path), mode="a")
    camera = store["obs_head_pose"][:36]
    camera[:] = [0, 0, 0, 1, 0, 0, 0]
    store["obs_head_pose"][:36] = camera
    for side in ("left", "right"):
        pose = store[f"{side}.obs_ee_pose"][:36]
        pose[:2, 3:] = Rotation.from_euler(
            "Z", [[179], [-179]], degrees=True
        ).as_quat()[:, [3, 0, 1, 2]]
        store[f"{side}.obs_ee_pose"][:36] = pose
    action = EgoPi05Dataset(
        export(tmp_path, [path], source_horizon=2), include_images=False
    )[0]["actions"]
    for start in (3, 10):
        rotations = Rotation.from_euler("ZYX", action[:, start : start + 3])
        assert np.max((rotations[:-1].inv() * rotations[1:]).magnitude()) < np.deg2rad(
            0.021
        )


def test_portable_manifest_and_export_cli(tmp_path):
    path = write_ego(tmp_path / "source")
    manifest = tmp_path / "export" / "manifest.json"
    from egomimic.scripts.export_pico_ego_pi05 import main

    main(
        [
            "--input",
            str(path),
            "--output",
            str(manifest),
            "--validation-ratio",
            "0",
            "--gripper-mode",
            "width",
        ]
    )
    original = EgoPi05Dataset(manifest)[0]
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    shutil.move(str(tmp_path / "source"), relocated)
    shutil.move(str(tmp_path / "export"), relocated)
    restored = EgoPi05Dataset(relocated / "export" / "manifest.json")[0]
    for key in original:
        if isinstance(original[key], np.ndarray):
            np.testing.assert_array_equal(original[key], restored[key])
        else:
            assert original[key] == restored[key]


def test_adapter_import_does_not_import_torch_or_jax():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import egomimic.robot.piper_pico.pi05_ego_data; assert 'torch' not in sys.modules; assert 'jax' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
