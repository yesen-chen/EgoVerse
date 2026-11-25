import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def runner(monkeypatch):
    folder = Path(__file__).resolve().parents[2] / "pi05/scripts"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "sensitivity_test", folder / "debug_pi05_rtc_sensitivity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_is_one_moving_pair_per_episode_without_metric_ranking(runner):
    rows = [
        dict(episode=e, current_row=r, static={"left": r == 0})
        for e in range(3)
        for r in (0, 27, 54, 81)
    ]
    assert [
        (r["episode"], r["current_row"]) for r in runner.select_moving_pairs(rows)
    ] == [(0, 54), (1, 54), (2, 54)]


@pytest.mark.parametrize(
    "kind,state_changed,image_changed",
    [
        ("unchanged", False, False),
        ("state_100ms_older", True, False),
        ("images_100ms_older", False, True),
        ("inputs_100ms_older", True, True),
    ],
)
def test_only_selected_observation_parts_change(
    runner, kind, state_changed, image_changed
):
    current = {
        "observation.state": np.zeros(14),
        "observation.images.front_1": np.zeros((2, 2, 3)),
        "prompt": "task",
    }
    older = {
        "observation.state": np.ones(14),
        "observation.images.front_1": np.ones((2, 2, 3)),
        "prompt": "other",
    }
    result = runner.observation_variant(current, older, kind)
    assert np.all(result["observation.state"] == state_changed)
    assert np.all(result["observation.images.front_1"] == image_changed)
    assert result["prompt"] == "task"
    result["observation.state"][:] = 99
    assert np.all(current["observation.state"] == 0)
    assert np.all(older["observation.state"] == 1)


def test_prefix_shift_changes_only_left_x_and_does_not_mutate(runner):
    original = np.arange(1400).reshape(100, 14) / 1000
    result = runner.translated_actions(original, 0.030)
    np.testing.assert_allclose(result[:, 0], original[:, 0] + 0.030)
    np.testing.assert_array_equal(result[:, 1:], original[:, 1:])
    assert result is not original


@pytest.mark.parametrize(
    "actions,amount",
    [
        (np.zeros((10, 7)), 0.03),
        (np.full((10, 14), np.nan), 0.03),
        (np.zeros((10, 14)), np.inf),
    ],
)
def test_invalid_perturbation_is_rejected(runner, actions, amount):
    with pytest.raises(ValueError):
        runner.translated_actions(actions, amount)
