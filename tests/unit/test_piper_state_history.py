"""History interpolation/availability contracts without hardware or ML runtimes."""

import copy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_pico.state_history import (
    HistoryCoverageError,
    load_feedback_streams,
    make_history_contract,
    portable_feedback_metadata,
    prepare_feedback_streams,
    sample_state_history,
    validate_history_contract,
    validate_history_offsets,
    write_feedback_sidecar,
)


def streams(times=(0, 0.01, 0.02, 0.03, 0.04)):
    times = np.asarray(times, dtype=float)
    result = []
    for index in range(2):
        pose = np.zeros((len(times), 6))
        pose[:, 0] = times * (index + 1)
        pose[:, 5] = times * (index + 2)
        result.append(
            {
                "eef_pose": pose,
                "eef_pose__timestamp": times,
                "gripper_width": (times * (index + 0.5))[:, None],
                "gripper_width__timestamp": times.copy(),
            }
        )
    return result


def test_arbitrary_offsets_interpolate_each_arm_and_gripper_without_current_state():
    result = sample_state_history(streams(), 0.04, [-0.025, -0.009])
    assert result.shape == (2, 14)
    targets = np.array([0.015, 0.031])
    for index in range(2):
        np.testing.assert_allclose(result[:, index * 7], targets * (index + 1))
        np.testing.assert_allclose(result[:, index * 7 + 5], targets * (index + 2))
        np.testing.assert_allclose(result[:, index * 7 + 6], targets * (index + 0.5))
    np.testing.assert_array_equal(sample_state_history([], 0.04, []), np.empty((0, 14)))


def test_rotation_wrap_uses_short_so3_path():
    data = streams([0, 1])
    for arm in data:
        arm["eef_pose"][:, 3:] = Rotation.from_euler(
            "z", [[179], [-179]], degrees=True
        ).as_rotvec()
    result = sample_state_history(data, 1, [-0.5])
    for index in range(2):
        rotation = Rotation.from_rotvec(result[0, index * 7 + 3 : index * 7 + 6])
        assert (
            rotation.inv() * Rotation.from_euler("z", 180, degrees=True)
        ).magnitude() < 1e-9


def test_future_samples_cannot_supply_interpolation_endpoint():
    data = streams([0, 0.02, 0.04, 0.06])
    # Although an offline recording has t=.06, it was not available at .05.
    with pytest.raises(HistoryCoverageError):
        sample_state_history(data, 0.05, [-0.005])
    first = sample_state_history(data, 0.05, [-0.02])
    data[0]["eef_pose"][-1] = 100
    data[1]["gripper_width"][-1] = 500
    np.testing.assert_array_equal(first, sample_state_history(data, 0.05, [-0.02]))


def test_per_arm_cutoff_excludes_samples_received_after_buffer_read():
    data = streams([0, 0.02, 0.04, 0.06])
    # query=.06, left buffer ended .04 and right buffer ended .02.
    with pytest.raises(HistoryCoverageError, match="right"):
        sample_state_history(data, 0.06, [-0.03], available_until=[0.04, 0.02])
    result = sample_state_history(
        data, 0.06, [-0.05], available_until={"left": 0.04, "right": 0.02}
    )
    np.testing.assert_allclose(result[0, [0, 7]], [0.01, 0.02])
    with pytest.raises(ValueError, match="available_until"):
        sample_state_history(data, 0.06, [-0.03], available_until=[0.07, 0.02])


def test_each_sensor_stream_needs_its_own_coverage():
    data = streams()
    data[1]["gripper_width__timestamp"] = np.array([0.03, 0.04])
    data[1]["gripper_width"] = np.array([[0.01], [0.02]])
    with pytest.raises(HistoryCoverageError, match="right.gripper_width"):
        sample_state_history(data, 0.04, [-0.025])
    with pytest.raises(HistoryCoverageError):
        sample_state_history(streams(), 0.04, [-0.05])


def test_single_exact_available_sample_is_valid_but_not_extrapolated():
    data = streams([0.02])
    result = sample_state_history(data, 0.04, [-0.02])
    assert result[0, 0] == pytest.approx(0.02)
    with pytest.raises(HistoryCoverageError):
        sample_state_history(data, 0.04, [-0.01])


@pytest.mark.parametrize(
    "offsets",
    [
        [0],
        [0.1],
        [-0.01, -0.02],
        [-0.02, -0.02],
        [float("nan")],
        [float("inf")],
        [True],
        "-0.1",
        0.1,
    ],
)
def test_rejects_invalid_offsets(offsets):
    with pytest.raises(ValueError):
        validate_history_offsets(offsets)


def test_contract_is_explicit_and_round_trips():
    assert make_history_contract([]) is None
    assert validate_history_contract(None) == ()
    assert validate_history_offsets([-0.1, -0.025]) == (-0.1, -0.025)
    contract = make_history_contract([-0.1, -0.025])
    assert validate_history_contract(json.loads(json.dumps(contract))) == (-0.1, -0.025)
    assert contract["coordinates"] == "current_observation_camera"
    for key, value in (
        ("causal", False),
        ("gripper_unit", "m"),
        ("offsets_s", []),
        ("extra", "unknown"),
    ):
        with pytest.raises(ValueError):
            validate_history_contract({**contract, key: value})


def test_bad_data_raises_value_error_not_coverage_error():
    for case in ("duplicate", "nan", "shape", "missing"):
        data = copy.deepcopy(streams())
        if case == "duplicate":
            data[0]["eef_pose__timestamp"][1] = 0
        elif case == "nan":
            data[0]["eef_pose"][0, 0] = np.nan
        elif case == "shape":
            data[0]["eef_pose"] = data[0]["eef_pose"][:, :5]
        else:
            del data[0]["eef_pose"]
        with pytest.raises(ValueError) as error:
            sample_state_history(data, 0.04, [-0.025])
        assert not isinstance(error.value, HistoryCoverageError)


def test_prepared_streams_are_owned_immutable_arrays():
    data = streams()
    prepared = prepare_feedback_streams(data)
    expected = sample_state_history(prepared, 0.04, [-0.02])
    data[0]["eef_pose"][2] = 999
    np.testing.assert_array_equal(
        expected, sample_state_history(prepared, 0.04, [-0.02])
    )
    assert prepare_feedback_streams(prepared) is prepared
    with pytest.raises(ValueError, match="read-only"):
        prepared[0]["eef_pose"][0] = 999


def test_sidecars_are_portable_checksummed_and_idempotent(tmp_path):
    raw, converted = tmp_path / "raw", tmp_path / "converted"
    raw.mkdir()
    for index, data in enumerate(streams()):
        np.savez(raw / f"robot{index}_state.npz", **data)
    metadata = write_feedback_sidecar(raw, converted)
    assert portable_feedback_metadata(converted) == metadata
    assert write_feedback_sidecar(raw, converted) == metadata
    expected = sample_state_history(streams(), 0.04, [-0.025])
    moved = tmp_path / "moved"
    converted.rename(moved)
    np.testing.assert_allclose(
        sample_state_history(load_feedback_streams(moved), 0.04, [-0.025]), expected
    )
    (moved / "feedback_history/left.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_feedback_streams(moved)


def test_missing_history_marker_can_be_upgraded_but_existing_data_not_replaced(
    tmp_path,
):
    raw, converted = tmp_path / "raw", tmp_path / "converted"
    raw.mkdir()
    descriptor = write_feedback_sidecar(raw, converted)
    assert descriptor["status"] == "unavailable"
    with pytest.raises(HistoryCoverageError):
        load_feedback_streams(converted)
    with pytest.raises(FileNotFoundError):
        write_feedback_sidecar(raw, converted, allow_missing=False)
    for index, data in enumerate(streams()):
        np.savez(raw / f"robot{index}_state.npz", **data)
    assert write_feedback_sidecar(raw, converted)["status"] == "available"
    data = streams()[0]
    data["eef_pose"][0, 0] = 10
    np.savez(raw / "robot0_state.npz", **data)
    with pytest.raises(FileExistsError):
        write_feedback_sidecar(raw, converted)


def test_metadata_paths_and_ranges_cannot_lie(tmp_path):
    raw, converted = tmp_path / "raw", tmp_path / "converted"
    raw.mkdir()
    for index, data in enumerate(streams()):
        np.savez(raw / f"robot{index}_state.npz", **data)
    metadata = write_feedback_sidecar(raw, converted)
    invalid = copy.deepcopy(metadata)
    invalid["arms"]["left"]["path"] = "../raw/robot0_state.npz"
    with pytest.raises(ValueError, match="within"):
        load_feedback_streams(converted, invalid)
    invalid = copy.deepcopy(metadata)
    invalid["arms"]["left"]["pose_samples"] += 1
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_feedback_streams(converted, invalid)
