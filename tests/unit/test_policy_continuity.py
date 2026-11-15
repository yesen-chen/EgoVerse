"""Execution continuity invariants without hardware, RPC, or model weights."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egomimic.rldb.action_timing import ActionTiming
from egomimic.robot.piper_pico.policy_continuity import (
    ConstantPosePlan,
    ContinuousPlan,
    conditioned_plan,
    sample_plan,
    smooth_pose_actions,
    transition_disagreement,
)
from egomimic.robot.piper_pico.policy_streaming import ChunkPlan

TIMING = ActionTiming(30, 30, 1, 100)
OFFSETS = np.asarray(TIMING.output_offsets_s)


def moving_actions(*, variant=0):
    """Linear xyz and nonidentity rotations with constant spatial velocity."""
    result = np.zeros((100, 14), np.float64)
    for arm, k in enumerate((0, 7)):
        origin = np.array([0.2, -0.03, 0.3]) + arm * 0.03 + variant * 0.003
        velocity = np.array([0.03, -0.02, 0.01]) * (1 + arm + variant)
        result[:, k : k + 3] = origin + OFFSETS[:, None] * velocity
        initial = Rotation.from_rotvec([0.2 + variant * 0.1, -0.3, 0.4])
        omega = np.array([0.1, -0.07, 0.04]) * (1 + arm + variant)
        result[:, k + 3 : k + 6] = (
            Rotation.from_rotvec(OFFSETS[:, None] * omega) * initial
        ).as_rotvec()
        # Discontinuous widths deliberately test that grippers are not filtered.
        result[:, k + 6] = (np.arange(100) % (3 + variant)) * 0.01 + arm * 0.005
    return result


def plan(*, anchor=10.0, variant=0):
    return ChunkPlan(moving_actions(variant=variant), TIMING, anchor)


def assert_pose_equal(actual, expected, *, atol=1e-10):
    for k in (0, 7):
        np.testing.assert_allclose(actual[k : k + 3], expected[k : k + 3], atol=atol)
        angular_error = (
            Rotation.from_rotvec(actual[k + 3 : k + 6])
            * Rotation.from_rotvec(expected[k + 3 : k + 6]).inv()
        ).magnitude()
        assert angular_error < atol


def velocity_between(first, second, dt, k):
    linear = (second[k : k + 3] - first[k : k + 3]) / dt
    angular = (
        Rotation.from_rotvec(second[k + 3 : k + 6])
        * Rotation.from_rotvec(first[k + 3 : k + 6]).inv()
    ).as_rotvec() / dt
    return np.concatenate((linear, angular))


def test_transition_preserves_entry_exit_pose_and_velocity():
    old, new = plan(), plan(variant=1)
    start, duration = 10.2137, 0.16
    end = start + duration
    joined = ContinuousPlan(new, old, start, duration)
    assert_pose_equal(joined.sample(start), old.sample(start))
    assert_pose_equal(joined.sample(end), new.sample(end))
    eps = 1e-6
    for k in (0, 7):
        before_entry = velocity_between(
            old.sample(start - eps), old.sample(start), eps, k
        )
        after_entry = velocity_between(
            joined.sample(start), joined.sample(start + eps), eps, k
        )
        before_exit = velocity_between(
            joined.sample(end - eps), joined.sample(end), eps, k
        )
        after_exit = velocity_between(new.sample(end), new.sample(end + eps), eps, k)
        np.testing.assert_allclose(after_entry, before_entry, atol=2e-6, rtol=0)
        np.testing.assert_allclose(before_exit, after_exit, atol=2e-6, rtol=0)
    for timestamp in np.linspace(start, new.end_walltime, 20):
        assert np.isfinite(joined.sample(timestamp)).all()
    for timestamp in (end, end + 0.1, new.end_walltime):
        np.testing.assert_array_equal(joined.sample(timestamp), new.sample(timestamp))


def test_transition_uses_short_rotation_arc_across_plus_minus_pi():
    old_actions, new_actions = moving_actions(), moving_actions(variant=1)
    for k in (0, 7):
        old_actions[:, k + 3 : k + 6] = [0, 0, np.deg2rad(179)]
        new_actions[:, k + 3 : k + 6] = [0, 0, np.deg2rad(-179)]
    old = ChunkPlan(old_actions, TIMING, 10)
    new = ChunkPlan(new_actions, TIMING, 10)
    joined = ContinuousPlan(new, old, 10.2, 0.2)
    middle = joined.sample(10.3)
    expected = Rotation.from_rotvec([0, 0, np.pi])
    for k in (0, 7):
        actual = Rotation.from_rotvec(middle[k + 3 : k + 6])
        assert (actual * expected.inv()).magnitude() < 1e-10
        assert (
            actual * Rotation.from_rotvec([0, 0, np.deg2rad(179)]).inv()
        ).magnitude() == pytest.approx(np.deg2rad(1))


def test_gripper_is_never_smoothed_or_blended_including_entry():
    raw = moving_actions(variant=1)
    smoothed = smooth_pose_actions(raw, TIMING, 0.1)
    np.testing.assert_array_equal(smoothed[:, [6, 13]], raw[:, [6, 13]])
    old_actions = moving_actions()
    old_actions[:, [6, 13]] = 0.079
    old = ChunkPlan(old_actions, TIMING, 10)
    new = ChunkPlan(smoothed, TIMING, 10)
    joined = ContinuousPlan(new, old, 10.2, 0.2)
    for timestamp in np.linspace(10.19, 10.5, 23):
        np.testing.assert_array_equal(
            joined.sample(timestamp)[[6, 13]], new.sample(timestamp)[[6, 13]]
        )


def test_smoothing_off_returns_exact_independent_copy():
    raw = moving_actions()
    original = raw.copy()
    output = smooth_pose_actions(raw, TIMING, 0)
    np.testing.assert_array_equal(output, raw)
    assert not np.shares_memory(raw, output)
    output[0, 0] = -100
    np.testing.assert_array_equal(raw, original)


def test_smoothing_preserves_constant_pose_and_linear_xyz():
    row = moving_actions()[0]
    constant = np.repeat(row[None], 100, axis=0)
    smoothed_constant = smooth_pose_actions(constant, TIMING, 0.1)
    for actual in smoothed_constant:
        assert_pose_equal(actual, row)
    linear = moving_actions()
    for k in (0, 7):
        linear[:, k + 3 : k + 6] = linear[0, k + 3 : k + 6]
    output = smooth_pose_actions(linear, TIMING, 0.1)
    np.testing.assert_allclose(
        output[:, [0, 1, 2, 7, 8, 9]], linear[:, [0, 1, 2, 7, 8, 9]], atol=1e-12, rtol=0
    )


def test_smoothing_reduces_high_frequency_position_and_rotation_noise():
    clean = np.repeat(moving_actions()[0:1], 100, axis=0)
    noisy = clean.copy()
    alternating = (-1.0) ** np.arange(100)
    for k in (0, 7):
        noisy[:, k] += 0.003 * alternating
        base = Rotation.from_rotvec(clean[0, k + 3 : k + 6])
        noisy[:, k + 3 : k + 6] = (
            Rotation.from_rotvec(
                np.column_stack((np.zeros(100), 0.02 * alternating, np.zeros(100)))
            )
            * base
        ).as_rotvec()
    result = smooth_pose_actions(noisy, TIMING, 0.1)
    interior = slice(15, -15)
    for k in (0, 7):
        assert np.std(result[interior, k] - clean[interior, k]) < 0.3 * np.std(
            noisy[interior, k] - clean[interior, k]
        )
        reference = Rotation.from_rotvec(clean[interior, k + 3 : k + 6])
        before = (
            Rotation.from_rotvec(noisy[interior, k + 3 : k + 6]) * reference.inv()
        ).magnitude()
        after = (
            Rotation.from_rotvec(result[interior, k + 3 : k + 6]) * reference.inv()
        ).magnitude()
        assert after.mean() < 0.3 * before.mean()


def test_smoothing_equivalent_rotation_branches_are_consistent():
    raw = moving_actions()
    equivalent = raw.copy()
    angles = np.deg2rad(np.linspace(178, 182, 100))
    for k in (0, 7):
        vectors = np.column_stack((np.zeros(100), np.zeros(100), angles))
        raw[:, k + 3 : k + 6] = vectors
        equivalent[:, k + 3 : k + 6] = Rotation.from_rotvec(vectors).as_rotvec()
    first = smooth_pose_actions(raw, TIMING, 0.1)
    second = smooth_pose_actions(equivalent, TIMING, 0.1)
    for k in (0, 7):
        error = (
            Rotation.from_rotvec(first[:, k + 3 : k + 6])
            * Rotation.from_rotvec(second[:, k + 3 : k + 6]).inv()
        ).magnitude()
        assert error.max() < 1e-10
        assert Rotation.from_rotvec(
            first[:, k + 3 : k + 6]
        ).magnitude().min() > np.deg2rad(177)


def test_conditioned_plan_preserves_timing_and_input_arrays():
    raw = moving_actions()
    original = raw.copy()
    current = conditioned_plan(raw, TIMING, 10.0, 0.1)
    previous = plan(variant=1)
    old_original = previous.actions.copy()
    new_original = current.actions.copy()
    joined = ContinuousPlan(current, previous, 10.2, 0.15)
    for timestamp in np.linspace(10.2, 10.7, 20):
        output = joined.sample(timestamp)
        output[:] = -1  # Samples must not alias either stored plan.
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(previous.actions, old_original)
    np.testing.assert_array_equal(current.actions, new_original)
    np.testing.assert_array_equal(current.offsets, OFFSETS)
    assert current.anchor == 10
    assert current.span == TIMING.effective_span_s
    assert joined.anchor == current.anchor
    assert joined.end_walltime == current.end_walltime


def test_completed_previous_transitions_are_flattened_without_mutation():
    first, second, third = plan(), plan(variant=1), plan(variant=2)
    old_join = ContinuousPlan(second, first, 10.1, 0.1)
    new_join = ContinuousPlan(third, old_join, 10.4, 0.15)
    assert new_join.previous is second
    assert old_join.previous is first
    assert_pose_equal(new_join.sample(10.4), old_join.sample(10.4))


def test_active_previous_transition_is_preserved():
    first, second, third = plan(), plan(variant=1), plan(variant=2)
    old_join = ContinuousPlan(second, first, 10.1, 0.4)
    new_join = ContinuousPlan(third, old_join, 10.3, 0.15)
    assert new_join.previous is old_join
    assert_pose_equal(new_join.sample(10.3), old_join.sample(10.3))


@pytest.mark.parametrize("short_side", ["old", "new"])
def test_transition_rejects_insufficient_old_or_new_end_support(short_side):
    old = plan(anchor=9.7 if short_side == "old" else 10)
    new = plan(anchor=9.7 if short_side == "new" else 10)
    with pytest.raises(ValueError):
        ContinuousPlan(new, old, 10.5, 0.2)


@pytest.mark.parametrize("late_side", ["old", "new"])
def test_transition_rejects_start_before_old_or_new_anchor(late_side):
    old = plan(anchor=10.4 if late_side == "old" else 10)
    new = plan(anchor=10.4 if late_side == "new" else 10)
    with pytest.raises(ValueError):
        ContinuousPlan(new, old, 10.3, 0.2)


@pytest.mark.parametrize("duration", [0, -0.1, float("nan"), float("inf")])
def test_transition_rejects_invalid_duration(duration):
    with pytest.raises(ValueError):
        ContinuousPlan(plan(), plan(), 10.2, duration)


@pytest.mark.parametrize("start", [float("nan"), float("inf"), -float("inf")])
def test_transition_rejects_nonfinite_start(start):
    with pytest.raises(ValueError):
        ContinuousPlan(plan(), plan(), start, 0.1)


@pytest.mark.parametrize("window", [-0.1, float("nan"), float("inf")])
def test_smoothing_rejects_invalid_window(window):
    with pytest.raises(ValueError):
        smooth_pose_actions(moving_actions(), TIMING, window)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((99, 14)),
        np.zeros((100, 13)),
        np.full((100, 14), np.nan),
        np.full((100, 14), np.inf),
    ],
)
def test_smoothing_rejects_wrong_shape_or_nonfinite_actions(bad):
    with pytest.raises(ValueError):
        smooth_pose_actions(bad, TIMING, 0.1)


def test_smoothing_rejects_nonuniform_or_too_short_grid():
    uneven = OFFSETS.copy()
    uneven[10] += 0.001
    with pytest.raises(ValueError, match="uniform"):
        smooth_pose_actions(
            moving_actions(),
            SimpleNamespace(
                output_chunk_length=100,
                output_offsets_s=uneven,
                effective_span_s=TIMING.effective_span_s,
            ),
            0.1,
        )
    with pytest.raises(ValueError, match="five"):
        smooth_pose_actions(
            np.zeros((4, 14)),
            SimpleNamespace(
                output_chunk_length=4,
                output_offsets_s=np.linspace(0, 1, 4),
                effective_span_s=1.0,
            ),
            0.1,
        )


@pytest.mark.parametrize(
    "window", [TIMING.effective_span_s, 2 * TIMING.effective_span_s]
)
def test_smoothing_rejects_window_as_long_as_horizon(window):
    with pytest.raises(ValueError, match="horizon"):
        smooth_pose_actions(moving_actions(), TIMING, window)


def test_transition_disagreement_detects_interior_spike_when_entry_matches():
    old_actions = np.zeros((100, 14))
    new_actions = old_actions.copy()
    # This model-grid peak does not coincide with the acceptance helper's
    # 100 Hz checks, so inspecting both model grids matters.
    new_actions[30, 0] = 0.03
    new_actions[30, 10] = np.deg2rad(8)
    old = ChunkPlan(old_actions, TIMING, 10)
    new = ChunkPlan(new_actions, TIMING, 10)
    assert_pose_equal(new.sample(10.2), old.sample(10.2))
    report = transition_disagreement(new, old, 10.2, 0.2)
    assert report["left"]["max_position_m"] == pytest.approx(0.03)
    assert report["right"]["max_rotation_rad"] == pytest.approx(np.deg2rad(8))
    assert report["left"]["max_rotation_rad"] == pytest.approx(0)
    assert report["right"]["max_position_m"] == pytest.approx(0)


def test_transition_disagreement_measures_physical_rotation_across_pi():
    old_actions = np.zeros((100, 14))
    new_actions = old_actions.copy()
    old_actions[:, 5] = np.deg2rad(179)
    new_actions[:, 5] = np.deg2rad(-179)
    old = ChunkPlan(old_actions, TIMING, 10)
    new = ChunkPlan(new_actions, TIMING, 10)
    report = transition_disagreement(new, old, 10.2, 0.2)
    assert report["left"]["max_rotation_rad"] == pytest.approx(np.deg2rad(2))


def test_constant_pose_plan_copies_inputs_and_samples():
    pose = moving_actions()[0].copy()
    expected = pose.copy()
    held = ConstantPosePlan(pose, 11)
    pose[:] = -100
    output = held.sample(10)
    np.testing.assert_array_equal(output, expected)
    output[:] = 100
    np.testing.assert_array_equal(held.sample(10.5), expected)
    joined = ContinuousPlan(plan(variant=1), held, 10.2, 0.15)
    assert_pose_equal(joined.sample(10.2), expected)


@pytest.mark.parametrize(
    "pose", [np.zeros(13), np.zeros((1, 14)), np.full(14, np.nan), np.full(14, np.inf)]
)
def test_constant_pose_plan_rejects_invalid_pose(pose):
    with pytest.raises(ValueError):
        ConstantPosePlan(pose, 11)


@pytest.mark.parametrize("end", [float("nan"), float("inf"), -float("inf")])
def test_constant_pose_plan_rejects_nonfinite_end(end):
    with pytest.raises(ValueError):
        ConstantPosePlan(np.zeros(14), end)


@pytest.mark.parametrize(
    "kind", ["chunk", "constant", "transition", "nested_transition"]
)
def test_vectorized_plan_samples_match_scalar_samples_and_do_not_alias(kind):
    first, second, third = plan(), plan(variant=1), plan(variant=2)
    if kind == "chunk":
        selected = first
    elif kind == "constant":
        selected = ConstantPosePlan(first.sample(10.2), 11)
    else:
        selected = ContinuousPlan(second, first, 10.2, 0.3)
        if kind == "nested_transition":
            selected = ContinuousPlan(third, selected, 10.3, 0.15)
    # Mixed order, duplicates, exact boundaries, and clamped endpoint samples.
    times = np.array([10.5, 9.9, 10.2, 10.3, 10.31, 10.45, 10.5, 10.8, 11.1])
    original_times = times.copy()
    expected = np.array([selected.sample(t) for t in times])
    actual = sample_plan(selected, times)
    assert actual.shape == (len(times), 14)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(times, original_times)
    actual[:] = -100
    np.testing.assert_allclose(
        sample_plan(selected, times.tolist()), expected, atol=1e-12, rtol=0
    )


def test_vectorized_short_arc_across_pi_matches_scalar_physical_rotations():
    old_actions = moving_actions()
    new_actions = moving_actions(variant=1)
    for k, sign in ((0, 1), (7, -1)):
        old_actions[:, k + 3 : k + 6] = [0, 0, sign * np.deg2rad(179)]
        new_actions[:, k + 3 : k + 6] = [0, 0, -sign * np.deg2rad(179)]
    old = ChunkPlan(old_actions, TIMING, 10)
    new = ChunkPlan(new_actions, TIMING, 10)
    joined = ContinuousPlan(new, old, 10.2, 0.2)
    times = [10.19, 10.2, 10.21, 10.3, 10.39, 10.4, 10.5]
    batch = sample_plan(joined, times)
    scalar = np.array([joined.sample(t) for t in times])
    for actual, expected in zip(batch, scalar):
        assert_pose_equal(actual, expected)
        np.testing.assert_array_equal(actual[[6, 13]], expected[[6, 13]])
    for k in (0, 7):
        midpoint = Rotation.from_rotvec(batch[3, k + 3 : k + 6])
        assert midpoint.magnitude() == pytest.approx(np.pi)


@pytest.mark.parametrize("kind", ["chunk", "constant", "transition"])
def test_vectorized_plan_sampling_accepts_empty_times(kind):
    first = plan()
    selected = {
        "chunk": first,
        "constant": ConstantPosePlan(first.sample(10.2), 11),
        "transition": ContinuousPlan(plan(variant=1), first, 10.2, 0.2),
    }[kind]
    result = sample_plan(selected, [])
    assert result.shape == (0, 14)
    assert result.dtype.kind == "f"
