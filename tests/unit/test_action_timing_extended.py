"""Time contracts against real on-disk reads, including padded episode tails."""

from __future__ import annotations

import numpy as np
import pytest

from egomimic.rldb.action_timing import ActionTiming, validate_dataset_timing
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.zarr.action_chunk_transforms import (
    InterpolateLinear,
    InterpolatePose,
)
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
from egomimic.rldb.zarr.zarr_writer import ZarrWriter


@pytest.mark.parametrize("embodiment,horizon,stride", [(Human, 30, 3), (Eva, 45, 1)])
def test_contract_matches_original_factories_without_replacing_their_defaults(
    embodiment, horizon, stride
):
    # Inspect the untouched factories themselves, rather than only instantiating
    # independent interpolation objects with the values we expect them to use.
    keymap = embodiment._get_keymap("cartesian")
    action_windows = [
        entry["horizon"]
        for entry in keymap.values()
        if entry.get("key_type") == "action_keys" and "horizon" in entry
    ]
    assert action_windows and set(action_windows) == {horizon}
    interpolators = [
        transform
        for transform in embodiment.get_transform_list("cartesian")
        if isinstance(transform, (InterpolateLinear, InterpolatePose))
    ]
    assert interpolators
    for transform in interpolators:
        assert transform.stride == stride
        assert transform.new_chunk_length == 100
        timing = ActionTiming(30, horizon, transform.stride, transform.new_chunk_length)
        source_seconds = np.arange(horizon, dtype=np.float64) / 30
        if isinstance(transform, InterpolatePose):
            payload = np.zeros((horizon, 7))
            payload[:, 0] = source_seconds
            payload[:, 3] = 1
        else:
            payload = source_seconds[:, None]
        actual = transform.transform({transform.action_key: payload})[
            transform.output_action_key
        ]
        np.testing.assert_allclose(actual[:, 0], timing.output_offsets_s, atol=1e-12)


@pytest.fixture
def recorded_episode(tmp_path):
    """127 real rows, padded by the writer to 128; row IDs remain distinguishable."""
    n = 127
    row = np.arange(n, dtype=np.float64)
    pose = np.zeros((n, 7), dtype=np.float64)
    pose[:, 0] = row / 30
    pose[:, 1] = 2 * row / 30
    pose[:, 3] = 1
    command = pose.copy()
    command[:, :3] += [10, 20, 30]
    values = {
        "left.obs_ee_pose": pose,
        "right.obs_ee_pose": pose + [0, 0, 1, 0, 0, 0, 0],
        "left.cmd_ee_pose": command,
        "right.cmd_ee_pose": command + [0, 0, 1, 0, 0, 0, 0],
        "left.obs_gripper": (row / n)[:, None],
        "right.obs_gripper": (row / n)[:, None],
        "left.cmd_gripper": (0.1 + row / (2 * n))[:, None],
        "right.cmd_gripper": (0.1 + row / (2 * n))[:, None],
        "row_id": row[:, None],
    }
    path = tmp_path / "recording" / "episode_000000.zarr"
    ZarrWriter(path, embodiment="piper_bimanual", fps=30, chunk_timesteps=16).write(
        numeric_data=values
    )
    return path, values


@pytest.mark.parametrize("embodiment", [Human, Eva])
@pytest.mark.parametrize("row", [0, 51, 125, 126])
def test_original_keymaps_start_at_current_row_and_never_read_storage_padding(
    recorded_episode, embodiment, row
):
    path, values = recorded_episode
    keymap = {
        key: spec
        for key, spec in embodiment._get_keymap("cartesian").items()
        if spec["zarr_key"] in values
    }
    reader = ZarrDataset(path, keymap)
    assert len(reader) == 127
    sample = reader[row]
    for key, spec in keymap.items():
        source = values[spec["zarr_key"]]
        horizon = spec.get("horizon")
        if horizon is None:
            expected = source[row]
        else:
            assert horizon == (30 if embodiment is Human else 45)
            expected = source[np.minimum(row + np.arange(horizon), len(source) - 1)]
            np.testing.assert_allclose(sample[key][0], source[row], atol=2e-6)
        np.testing.assert_allclose(sample[key], expected, atol=2e-6)
    # Robot command at t is not replaced with measured state at t, even though
    # both chunks use the same current-index anchor. Human does use future obs.
    action_key = "left.action_ee_pose" if embodiment is Human else "left.cmd_ee_pose"
    delta = sample[action_key][0, :3] - sample["left.obs_ee_pose"][:3]
    np.testing.assert_allclose(
        delta, [0, 0, 0] if embodiment is Human else [10, 20, 30]
    )


@pytest.mark.parametrize(
    "horizon,stride,points",
    [(2, 1, 2), (11, 4, 17), (30, 3, 100), (45, 1, 100), (45, 4, 2), (90, 7, 257)],
)
@pytest.mark.parametrize("row", [0, 51, 125, 126])
def test_disk_read_stride_and_resampling_keep_physical_time_at_head_and_tail(
    recorded_episode, horizon, stride, points, row
):
    path, values = recorded_episode
    spec = ActionTiming(30, horizon, stride, points)
    keymap = {
        "action_pose": {
            "zarr_key": "left.cmd_ee_pose",
            "key_type": "action_keys",
            "horizon": horizon,
        },
        "action_index": {
            "zarr_key": "row_id",
            "key_type": "action_keys",
            "horizon": horizon,
        },
    }
    reader = ZarrDataset(
        path,
        keymap,
        [
            InterpolatePose(points, "action_pose", "action_pose", stride),
            InterpolateLinear(points, "action_index", "action_index", stride),
        ],
    )
    validate_dataset_timing({"piper_bimanual": spec}, {"piper_bimanual": reader}, {})
    sample = reader[row]
    # Build an independent physical-time interpolation oracle after explicit
    # repeat-last source lookup. Do not assume clamping occurs after interpolation:
    # a stride can bridge the last actual row and the first padded support.
    support_times = np.arange(0, horizon, stride) / 30
    actual_indices = np.minimum(row + np.arange(0, horizon, stride), 126)
    expected_indices = np.interp(spec.output_offsets_s, support_times, actual_indices)
    np.testing.assert_allclose(
        sample["action_index"][:, 0], expected_indices, atol=1e-5
    )
    np.testing.assert_allclose(
        sample["action_pose"][:, 0], 10 + expected_indices / 30, atol=1e-6
    )
    np.testing.assert_allclose(
        sample["action_pose"][0], values["left.cmd_ee_pose"][row], atol=2e-6
    )
    np.testing.assert_allclose(
        sample["action_pose"][-1],
        values["left.cmd_ee_pose"][actual_indices[-1]],
        atol=2e-6,
    )
    assert spec.output_offsets_s[0] == 0
    assert spec.output_offsets_s[-1] == pytest.approx(support_times[-1])
    assert np.all(np.isfinite(sample["action_pose"].numpy()))
    # Nominal future offsets remain positive at the tail; repeated poses do not
    # manufacture future measured observations or shrink the contract's duration.
    if row == 126:
        assert spec.output_offsets_s[-1] > 0
        np.testing.assert_array_equal(
            sample["action_index"][:, 0], np.full(points, 126)
        )


@pytest.mark.parametrize("fps", [15, 29.97, 30, 45, 60, 90])
@pytest.mark.parametrize("points", [2, 17, 100, 257])
def test_physical_duration_scales_with_source_fps_and_not_model_density(fps, points):
    timing = ActionTiming(fps, 46, 4, points)
    # Samples 0..45 are available but the last retained sample is number 44.
    assert timing.source_span_s == pytest.approx(45 / fps)
    assert timing.effective_span_s == pytest.approx(44 / fps)
    assert len(timing.output_offsets_s) == points
    np.testing.assert_allclose(
        np.diff(timing.output_offsets_s), 44 / fps / (points - 1)
    )
    assert timing.to_dict()["source_sampling"] == "uniform"
