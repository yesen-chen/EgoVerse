"""Smooth individual plans can require a fast correction when blended."""

import numpy as np
import pytest

from egomimic.rldb.action_timing import ActionTiming
from egomimic.robot.piper_pico.policy_continuity import (
    ContinuousPlan,
    transition_disagreement,
)
from egomimic.robot.piper_pico.policy_streaming import ChunkPlan


def constant_velocity_plan(speed_m_s, offset_m=0):
    timing = ActionTiming(
        source_fps=30,
        source_horizon=50,
        action_stride=1,
        output_chunk_length=100,
        includes_current=True,
    )
    actions = np.zeros((100, 14))
    actions[:, 0] = speed_m_s * np.asarray(timing.output_offsets_s) + offset_m
    actions[:, 6] = actions[:, 13] = 0.04
    return ChunkPlan(actions, timing, 0)


def velocity_x(plan, time, step=1e-5):
    return (plan.sample(time + step)[0] - plan.sample(time - step)[0]) / (2 * step)


def test_zero_executed_entry_gap_does_not_imply_low_transition_speed():
    old = constant_velocity_plan(0.1)
    new = constant_velocity_plan(0.1, 0.08)
    start, duration = 0.5, 0.3
    blended = ContinuousPlan(new, old, start, duration)

    # The executed first target exactly follows the old plan. Both plans have
    # constant 100 mm/s speed and zero acceleration when considered alone.
    np.testing.assert_allclose(blended.sample(start), old.sample(start), atol=1e-12)
    for time in (start - 0.05, start + duration / 2, start + duration + 0.05):
        assert velocity_x(old, time) == pytest.approx(0.1, abs=1e-9)
        assert velocity_x(new, time) == pytest.approx(0.1, abs=1e-9)
    assert velocity_x(blended, start - 0.05) == pytest.approx(0.1, abs=1e-9)
    assert velocity_x(blended, start + duration + 0.05) == pytest.approx(0.1, abs=1e-9)
    # Joining trajectories 80 mm apart in 300 ms adds 500 mm/s at mid-transition.
    peak = velocity_x(blended, start + duration / 2)
    assert peak == pytest.approx(0.6, abs=1e-8)
    assert peak > 0.5


@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_quintic_gap_derivative_matches_production_finite_difference(fraction):
    old = constant_velocity_plan(0.1)
    new = constant_velocity_plan(0.2, 0.04)
    start, duration = 0.5, 0.3
    blended = ContinuousPlan(new, old, start, duration)
    time = start + fraction * duration
    weight = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
    weight_derivative = 30 * fraction**2 * (1 - fraction) ** 2 / duration
    gap = new.sample(time)[0] - old.sample(time)[0]
    naive_velocity = (1 - weight) * 0.1 + weight * 0.2
    analytic_velocity = naive_velocity + weight_derivative * gap
    actual_velocity = velocity_x(blended, time)
    assert actual_velocity == pytest.approx(analytic_velocity, abs=2e-8)
    # Blending endpoint velocities alone would omit the correction term.
    assert actual_velocity - naive_velocity == pytest.approx(
        weight_derivative * gap, abs=2e-8
    )


def test_guard_checks_growing_disagreement_after_small_entry_gap():
    start, duration = 0.5, 0.3
    old = constant_velocity_plan(0.1)
    # At activation the plans are only 1 mm apart, then diverge at 300 mm/s.
    new = constant_velocity_plan(0.4, 0.001 - 0.3 * start)
    entry_gap = abs(new.sample(start)[0] - old.sample(start)[0])
    assert entry_gap == pytest.approx(0.001)
    measured = transition_disagreement(new, old, start, duration)
    assert measured["left"]["max_position_m"] == pytest.approx(0.091)
    assert measured["left"]["max_position_m"] > 0.060 > entry_gap
    assert measured["right"]["max_position_m"] == pytest.approx(0)
    assert measured["left"]["max_rotation_rad"] == pytest.approx(0)
