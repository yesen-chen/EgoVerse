"""Handoff comparisons must separate plan disagreement, motion and smoothness."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pi05/src"))
from egoverse_pi05.rtc_evaluation import (  # noqa: E402
    BaseCameraAdapter,
    evaluate_handoff,
)

from egomimic.rldb.action_timing import ActionTiming
from egomimic.robot.piper_pico.policy_continuity import ContinuousPlan, sample_plan
from egomimic.robot.piper_pico.policy_streaming import ChunkPlan

TIMING = ActionTiming(30, 50, 1, 100)
OFFSETS = np.asarray(TIMING.output_offsets_s)


def actions(anchor=10, velocity=0.03):
    values = np.zeros((100, 14), dtype=float)
    values[:, 0] = velocity * (anchor + OFFSETS - 10)
    values[:, 7] = -2 * values[:, 0]
    return values


def evaluate(new, *, old=None, **kwargs):
    if old is None:
        old = ChunkPlan(actions(10), TIMING, 10)
    return evaluate_handoff(old, new, TIMING, 10.2, 10.45, **kwargs)


def test_constant_velocity_aligned_by_wall_time_is_not_a_switch():
    result = evaluate(actions(10.2))
    for side in ("left", "right"):
        m = result["metrics"][side]
        assert m["raw_switch_mm"] < 1e-10
        assert m["raw_prefix_max_mm"] < 1e-10
        assert m["smoothed_switch_mm"] < 1e-10
        assert m["transition_d2_p95_mm"] < 1e-10
        assert m["raw_internal_d2_p95_mm"] < 1e-10
    assert result["metrics"]["left"]["transition_max_step_mm"] == pytest.approx(1)
    assert result["metrics"]["right"]["transition_max_step_mm"] == pytest.approx(2)
    json.dumps(result["metrics"], allow_nan=False)
    json.dumps(result["sampling"], allow_nan=False)


def test_known_translation_reports_plan_gap_not_one_control_step():
    shifted = actions(10.2)
    shifted[:, :3] += [0.003, 0.004, 0]
    shifted[:, 7] += 0.020
    result = evaluate(shifted)
    left, right = (result["metrics"][side] for side in ("left", "right"))
    assert left["raw_switch_mm"] == pytest.approx(5)
    assert left["raw_prefix_max_mm"] == pytest.approx(5)
    assert right["raw_switch_mm"] == pytest.approx(20)
    assert right["raw_prefix_max_mm"] == pytest.approx(20)
    # Before the handoff the right arm moves -2 mm/control tick, so the actual
    # switch step is 20 - 2 = 18 mm, not the same-time plan gap of 20 mm.
    assert right["raw_handoff_max_step_mm"] == pytest.approx(18)
    assert right["transition_max_step_mm"] < right["raw_handoff_max_step_mm"]


def test_transition_preserves_first_pose_but_does_not_filter_gripper():
    shifted = actions(10.2)
    shifted[:, 0] += 0.025
    shifted[:, 6] = 0.040
    result = evaluate(shifted)
    m = result["metrics"]["left"]
    assert m["executed_switch_mm"] == pytest.approx(0, abs=1e-12)
    trace = result["traces"]
    at = np.flatnonzero(np.isclose(trace["times_s"], 10.45))[0]
    np.testing.assert_allclose(
        trace["executed_base_actions"][at, :6],
        trace["old_base_actions"][at, :6],
    )
    assert trace["executed_base_actions"][at, 6] == 0.040
    assert m["transition_max_gripper_step_mm"] == pytest.approx(40)


def test_rotation_wrap_is_physical_two_degrees_not_358():
    old_actions = actions(10)
    old_actions[:, 5] = np.deg2rad(179)
    new = actions(10.2)
    new[:, 5] = np.deg2rad(-179)
    result = evaluate(new, old=ChunkPlan(old_actions, TIMING, 10))
    m = result["metrics"]["left"]
    assert m["raw_switch_deg"] == pytest.approx(2)
    assert m["raw_prefix_max_deg"] == pytest.approx(2)
    assert m["transition_max_step_deg"] < 1
    assert m["raw_internal_d2_p95_deg"] < 1e-10


def test_smooth_but_incorrect_constant_prediction_has_target_motion_error():
    reference = actions(10.2, velocity=0.1)
    still = np.zeros_like(reference)
    old = ChunkPlan(still, TIMING, 10)
    stationary = evaluate(still, old=old, reference_base_actions=reference)
    correct = evaluate(reference, old=old, reference_base_actions=reference)
    m = stationary["metrics"]["left"]
    assert m["raw_internal_d2_p95_mm"] == 0
    assert m["raw_switch_mm"] == 0
    assert m["raw_target_position_rmse_mm"] > 40
    assert correct["metrics"]["left"]["raw_target_position_rmse_mm"] < 1e-10
    # Smoothness alone cannot select the static candidate as the better policy.
    assert (
        m["executed_target_position_rmse_mm"]
        > correct["metrics"]["left"]["executed_target_position_rmse_mm"]
    )


def test_zero_transition_exposes_unblended_discontinuity():
    shifted = actions(10.2)
    shifted[:, 0] += 0.025
    result = evaluate(shifted, smoothing_s=0, transition_s=0)
    m = result["metrics"]["left"]
    assert m["executed_switch_mm"] == pytest.approx(25)
    assert m["transition_max_step_mm"] == pytest.approx(26)
    assert m["raw_switch_mm"] == pytest.approx(25)


def test_nonintegral_transition_has_two_complete_post_transition_intervals():
    result = evaluate(actions(10.2), transition_s=0.11)
    times = result["traces"]["times_s"]
    assert times[0] == pytest.approx(10.45 - 2 / 30)
    assert times[-1] >= 10.45 + 0.11 + 2 / 30
    np.testing.assert_allclose(np.diff(times), 1 / 30)


@pytest.mark.parametrize("activation", [10.21, 11.5])
def test_insufficient_overlap_is_not_silently_endpoint_padded(activation):
    with pytest.raises(ValueError, match="Insufficient valid"):
        evaluate_handoff(
            ChunkPlan(actions(10), TIMING, 10),
            actions(10.2),
            TIMING,
            10.2,
            activation,
        )


@pytest.mark.parametrize("frequency", [0, -30, np.nan, np.inf])
def test_invalid_frequency_rejected(frequency):
    with pytest.raises(ValueError, match="finite"):
        evaluate(actions(10.2), control_frequency=frequency)


def test_neither_previous_plan_nor_candidates_are_mutated():
    old = ChunkPlan(actions(10), TIMING, 10)
    original_old = old.actions.copy()
    new = actions(10.2)
    original_new = new.copy()
    evaluate(new, old=old)
    np.testing.assert_array_equal(old.actions, original_old)
    np.testing.assert_array_equal(new, original_new)


def test_current_state_only_affects_explicit_drift_metrics():
    state = np.zeros(14)
    result = evaluate(actions(10.2), current_base_state=state)
    assert result["metrics"]["left"]["raw_max_from_current_mm"] == pytest.approx(55)
    assert result["metrics"]["left"]["raw_switch_mm"] < 1e-10
    with pytest.raises(ValueError, match="Current measured"):
        evaluate(actions(10.2), current_base_state=np.zeros(13))


def test_old_prefix_includes_the_already_committed_transition():
    held = np.zeros((100, 14))
    shifted = held.copy()
    shifted[:, 0] = 0.020
    old = ContinuousPlan(
        ChunkPlan(shifted, TIMING, 10),
        ChunkPlan(held, TIMING, 10),
        10.3,
        0.6,
    )
    new = held.copy()
    new[:, 0] = 0.010
    result = evaluate(new, old=old)
    expected_switch = abs(0.010 - old.sample(10.45)[0]) * 1000
    assert result["metrics"]["left"]["raw_switch_mm"] == pytest.approx(expected_switch)
    assert expected_switch != pytest.approx(10)
    trace = result["traces"]
    np.testing.assert_allclose(
        trace["old_prefix_base_actions"], sample_plan(old, trace["prefix_times_s"])
    )


@pytest.mark.parametrize("nonidentity", [False, True])
def test_geometry_adapter_matches_production_in_both_directions(nonidentity):
    # Only this parity test imports the heavier production image adapter; the
    # offline adapter itself needs neither projectaria_tools nor torch.
    from egomimic.robot.piper_pico.policy_inference import PiperObservationAdapter

    transforms = {side: np.eye(4) for side in ("left", "right")}
    if nonidentity:
        for index, side in enumerate(("left", "right")):
            transforms[side][:3, :3] = Rotation.from_euler(
                "ZYX", [0.3 + index * 0.2, -0.4, 0.1 - index * 0.3]
            ).as_matrix()
            transforms[side][:3, 3] = [0.4 - index * 0.3, -0.13, 0.2 + index * 0.1]
    production = PiperObservationAdapter.__new__(PiperObservationAdapter)
    production.extrinsics = transforms
    production.gripper_open_m = 0.08
    lightweight = BaseCameraAdapter(transforms, 0.08)
    rng = np.random.default_rng(42)
    states = rng.normal(size=(9, 14)) * 0.2
    states[:, 6] = 0.081  # Overshoot must survive normalization and round trip.
    states[:, 13] = -0.001
    camera = np.array([lightweight.base_to_camera_state(row) for row in states])
    expected = np.array([production.base_to_camera_state(row) for row in states])
    assert camera.dtype == np.float64
    np.testing.assert_allclose(camera, expected, atol=1e-14, rtol=0)
    assert camera[0, 6] > 1
    assert camera[0, 13] < 0
    np.testing.assert_allclose(
        lightweight.camera_to_base_actions(camera),
        production.camera_to_base_actions(camera),
        atol=1e-14,
        rtol=0,
    )
    np.testing.assert_allclose(
        lightweight.camera_to_base_actions(camera), states, atol=1e-14
    )


def test_geometry_adapter_rejects_nonrigid_or_invalid_units():
    transforms = {side: np.eye(4) for side in ("left", "right")}
    with pytest.raises(ValueError, match="positive"):
        BaseCameraAdapter(transforms, 0)
    transforms["left"][0, 0] = -1
    with pytest.raises(ValueError, match="rigid"):
        BaseCameraAdapter(transforms, 0.08)


def test_geometry_adapter_copies_transforms_and_validates_inputs():
    transforms = {side: np.eye(4) for side in ("left", "right")}
    adapter = BaseCameraAdapter(transforms, 0.08)
    transforms["left"][0, 3] = 100
    assert adapter.base_to_camera_state(np.zeros(14))[0] == 0
    with pytest.raises(ValueError, match="base_state"):
        adapter.base_to_camera_state(np.ones(13))
    with pytest.raises(ValueError, match="base_state"):
        adapter.base_to_camera_state(np.full(14, np.nan))
    with pytest.raises(ValueError, match="camera actions"):
        adapter.camera_to_base_actions(np.zeros((0, 14)))
