"""Drawing may be throttled; per-control-tick command jump checks may not."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from egomimic.robot.piper_pico import sim_view


@pytest.fixture
def view(monkeypatch):
    # Prevent figure creation entirely, then attach a tiny fake canvas. No
    # backend event loop, display server, or robot is needed by these tests.
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "figure", Mock(side_effect=RuntimeError("test has no display")))
    instance = sim_view.SimBimanualView(refresh_hz=30)
    instance._ok = True
    instance._plt = SimpleNamespace(fignum_exists=Mock(return_value=True))
    instance._fig = SimpleNamespace(number=123, canvas=SimpleNamespace(draw_idle=Mock(), flush_events=Mock()))
    instance._draw_arm = Mock()
    instance._draw_jumps_and_hud = Mock()
    return instance


def _environment():
    poses = [np.zeros(6), np.zeros(6)]
    arms = [SimpleNamespace(get_state=lambda i=i: {"ee_pose": poses[i]}) for i in range(2)]
    return SimpleNamespace(arms=arms), poses


def test_ninety_hz_sampling_draws_only_thirty_hz(view, monkeypatch):
    env, poses = _environment()
    now = [0.0]
    monkeypatch.setattr(sim_view.time, "monotonic", lambda: now[0])
    for index in range(90):
        now[0] = index / 90
        poses[0][0] = index * 0.001
        assert view.update(env)
    assert view._fig.canvas.draw_idle.call_count == 30
    assert view._fig.canvas.flush_events.call_count == 30
    assert view._draw_arm.call_count == 60
    assert len(view._trails["left"]) == 90
    assert view._prev["left"][0] == pytest.approx(0.089)
    assert view.jump_count == 0


def test_jump_and_return_between_draws_are_both_detected(view, monkeypatch):
    env, poses = _environment()
    now = [0.0]
    monkeypatch.setattr(sim_view.time, "monotonic", lambda: now[0])
    for index, x in enumerate((0.0, 0.2, 0.0, 0.0)):
        now[0] = index / 90
        poses[0][0] = x
        assert view.update(env)
    assert view._fig.canvas.draw_idle.call_count == 2
    assert view.jump_count == 2
    assert len(view._jump_points) == 2
    assert len(view._trails["left"]) == 4
    np.testing.assert_allclose(view._jump_points[0], sim_view.LEFT_BASE + [0.2, 0.0, 0.0])
    np.testing.assert_allclose(view._jump_points[1], sim_view.LEFT_BASE)


def test_closed_figure_is_detected_between_draws(view, monkeypatch):
    env, _ = _environment()
    monkeypatch.setattr(sim_view.time, "monotonic", lambda: 0.0)
    assert view.update(env)
    view._plt.fignum_exists.return_value = False
    assert view.update(env) is False
    assert view._fig.canvas.draw_idle.call_count == 1


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_invalid_refresh_rate_fails_before_initializing_a_gui(rate):
    with pytest.raises(ValueError, match="positive and finite"):
        sim_view.SimBimanualView(refresh_hz=rate)
