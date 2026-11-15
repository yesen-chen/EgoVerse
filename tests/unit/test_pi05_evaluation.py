import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pi05/src"))
from egoverse_pi05.evaluation import (  # noqa: E402
    aggregate_metrics,
    repeat_metrics,
    select_static_repeats,
    static_window,
    trajectory_metrics,
)


def test_constant_bias_is_drift_not_jitter():
    target = np.zeros((30, 14))
    predicted = target.copy()
    predicted[:, 0] = 0.01
    m = trajectory_metrics(predicted, target, target[0], 0)
    assert np.mean(m["position_error_mm"]) == pytest.approx(10)
    assert np.max(m["second_difference_mm"]) == 0
    assert m["max_from_current_mm"] == pytest.approx([10])
    assert m["max_from_first_mm"] == [0]


def test_zigzag_and_linear_motion_are_distinguished():
    target = np.zeros((30, 14))
    p = target.copy()
    p[:, 0] = np.arange(30) * 0.001
    assert (
        np.max(trajectory_metrics(p, target, target[0], 0)["second_difference_mm"])
        < 1e-10
    )
    p[:, 0] = np.arange(30) % 2 * 0.001
    m = trajectory_metrics(p, target, target[0], 0)
    assert m["reversals"] == m["eligible"] == 28
    assert np.mean(m["second_difference_mm"]) == pytest.approx(2)
    assert aggregate_metrics([m, m])["reversals"] == 56


def test_rotation_wrap_uses_geodesic_distance():
    p = np.zeros((30, 14))
    t = p.copy()
    p[:, 5] = np.deg2rad(179)
    t[:, 5] = np.deg2rad(-179)
    assert np.mean(
        trajectory_metrics(p, t, t[0], 0)["rotation_error_deg"]
    ) == pytest.approx(2)


def test_static_selection_rejects_padding_motion_and_tracking_lag():
    a = {}
    for src in ("cmd", "obs"):
        a[f"left.{src}_ee_pose"] = np.tile([0.0, 0, 0, 1, 0, 0, 0], (40, 1))
        a[f"left.{src}_gripper"] = np.zeros((40, 1))
    assert static_window(a, "left", 0, 30, 0.08)["static"]
    assert not static_window(a, "left", 20, 30, 0.08)["static"]
    a["left.obs_ee_pose"][:, 0] = 0.006
    assert not static_window(a, "left", 0, 30, 0.08)["static"]
    a["left.obs_ee_pose"][:, 0] = 0
    a["left.cmd_gripper"][10, 0] = 0.1
    assert not static_window(a, "left", 0, 30, 0.08)["static"]


def test_repeat_variability_is_not_temporal_jitter():
    p = np.zeros((2, 30, 14))
    p[0, :, 0] = -0.002
    p[1, :, 0] = 0.002
    assert repeat_metrics(p, 0)["position_std_rms_mm"] == pytest.approx(2)
    assert repeat_metrics(p, 0)["successive_prediction_difference_mm"][
        "mean"
    ] == pytest.approx(4)


def test_empty_and_nonfinite():
    assert aggregate_metrics([]) == {"windows": 0}
    p = np.full((30, 14), np.nan)
    with pytest.raises(ValueError):
        trajectory_metrics(p, p, np.zeros(14), 0)


def test_repeat_selection_covers_both_arms_and_distinct_episodes():
    windows = [
        dict(
            episode=episode,
            static={"left": {"static": arm == 0}, "right": {"static": arm == 1}},
        )
        for episode in range(4)
        for arm in range(2)
    ]
    selected = select_static_repeats(windows, 2)
    assert selected == [0, 1, 6, 7]
    assert select_static_repeats(windows, 0) == []
