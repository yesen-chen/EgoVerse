"""Teacher-forced RTC chains: recursively carry plans, never model-generated state."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "pi05/scripts"), str(ROOT / "pi05/src")]
SPEC = importlib.util.spec_from_file_location(
    "rtc_chain_debug", ROOT / "pi05/scripts/debug_pi05_rtc_chain.py"
)
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)

from egoverse_pi05.rtc_evaluation import BaseCameraAdapter  # noqa: E402

from egomimic.rldb.action_timing import ActionTiming  # noqa: E402

TIMING = ActionTiming(30, 50, 1, 100)


def dataset(*, n=220):
    records = []
    numeric = []
    for episode in range(3):
        poses = np.zeros((n, 7))
        poses[:, 0] = np.arange(n) / 30 * (episode + 1)
        poses[:, 3] = 1
        records.append(
            dict(
                path=f"episode_{episode}",
                total_frames=n,
                contract={"time_alignment": {"observation_to_send_s": 0.007}},
            )
        )
        numeric.append(
            {
                "timestamp": 100 + np.arange(n) / 30,
                "left.cmd_ee_pose": poses,
                "right.cmd_ee_pose": poses,
            }
        )
    return SimpleNamespace(timing=TIMING, episodes=records, _numeric=numeric)


def observations(count):
    return [
        dict(
            _test_tick=i,
            actions=np.zeros((100, 14)),
            **{
                "observation.state": np.zeros(14),
                "observation.images.front": np.full((3, 2, 2), i, dtype=np.uint8),
            },
        )
        for i in range(count)
    ]


def adapter():
    return BaseCameraAdapter({s: np.eye(4) for s in ("left", "right")}, 0.08)


def test_activation_uses_initial_send_phase_and_rounds_deadline_up():
    assert SCRIPT.activation_time(100.9, 100.007, 30, 0.17) == pytest.approx(
        101.07366666666667
    )
    t = SCRIPT.activation_time(100.9, 100.087, 30, 0.17)
    assert t >= 101.07
    assert (t - 100.087) * 30 == pytest.approx(round((t - 100.087) * 30))


def test_selection_is_bounded_prioritizes_motion_and_uses_full_horizons():
    source = dataset()
    chains = SCRIPT.select_chains(source, max_episodes=2, max_handoffs=3)
    assert [x["episode"] for x in chains] == [2, 1]
    assert all(x["rows"] == [0, 27, 54, 81] for x in chains)
    assert all(
        x["rows"][-1] + 50 <= source.episodes[x["episode"]]["total_frames"]
        for x in chains
    )


def test_selection_stops_at_missing_history_or_timestamp_stall():
    source = dataset()
    source.episodes[2]["eligible_indices"] = [i for i in range(220) if i != 54]
    source._numeric[1]["timestamp"][54:] += 0.1
    chains = SCRIPT.select_chains(source, max_episodes=3)
    byid = {x["episode"]: x for x in chains}
    assert byid[2]["rows"] == [0, 27]
    assert byid[1]["rows"] == [0, 27]
    assert len(byid[0]["rows"]) > 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_episodes": 0},
        {"max_handoffs": False},
        {"frequency": 25},
        {"start_delay_s": -1},
    ],
)
def test_selection_rejects_invalid_protocol(kwargs):
    with pytest.raises(ValueError):
        SCRIPT.select_chains(dataset(), **kwargs)


def test_chain_propagates_guided_plan_but_keeps_recorded_observations_and_paired_noise():
    calls = []

    def predict(sample, noise):
        calls.append(
            (
                sample["_test_tick"],
                sample["observation.state"].copy(),
                noise.copy(),
                sample.get("rtc"),
            )
        )
        out = np.zeros((100, 14))
        if "rtc" in sample:
            p = sample["rtc"]
            out[:, 0] = p["previous_actions"][p["weights"] == 1, 0].mean() + 0.001
        return out

    samples = observations(4)
    results = list(
        SCRIPT.replay_chain(
            samples,
            [100, 100.9, 101.8, 102.7],
            predict=predict,
            adapter=adapter(),
            timing=TIMING,
            observation_to_send_s=0.007,
        )
    )
    assert len(results) == 3
    assert (
        len(calls) == 10
    )  # One first prediction, then ordinary + two RTC per observation.
    for step, (detail, arrays) in enumerate(results, 1):
        np.testing.assert_allclose(
            arrays["independent_pair_rtc_actions"][:, 0], 0.001, atol=1e-9
        )
        np.testing.assert_allclose(
            arrays["recursive_rtc_actions"][:, 0], 0.001 * step, atol=1e-9
        )
        np.testing.assert_allclose(arrays["recorded_target"], 0)
        assert detail["metrics"]["recursive_rtc"]["left"][
            "raw_target_position_rmse_mm"
        ] == pytest.approx(step, abs=1e-6)
    for tick in range(1, 4):
        these = [c for c in calls if c[0] == tick]
        assert len(these) == 3
        np.testing.assert_array_equal(these[0][2], these[1][2])
        np.testing.assert_array_equal(these[1][2], these[2][2])
    for _, state, _, _ in calls:
        np.testing.assert_array_equal(state, 0)
    assert all("actions" in s for s in samples)  # Input dictionaries were not consumed.
    np.testing.assert_array_equal(
        results[0][1]["recursive_rtc_prefix"],
        results[0][1]["independent_pair_rtc_prefix"],
    )
    assert not np.array_equal(
        results[1][1]["recursive_rtc_prefix"],
        results[1][1]["independent_pair_rtc_prefix"],
    )


def test_chain_retains_held_initial_transition_for_an_early_next_request():
    # Nondefault short request spacing exercises the recursive composed plan,
    # not just its underlying raw actions. No hardware scheduling is implied.
    def predict(sample, noise):
        out = np.zeros((100, 14))
        out[:, 0] = 0.020
        return out

    result = list(
        SCRIPT.replay_chain(
            observations(2),
            [100, 100.10],
            predict=predict,
            adapter=adapter(),
            timing=TIMING,
            observation_to_send_s=0,
            delay_s=0.10,
            transition_s=0.30,
        )
    )
    arrays = result[0][1]
    prefix = arrays["recursive_rtc_prefix"][:, 0]
    assert 0 < prefix[0] < 0.020
    assert np.max(prefix) == pytest.approx(0.020)


@pytest.mark.parametrize(
    "count,times", [(1, [100, 100.9]), (3, [100, 100.9]), (2, [100, 99])]
)
def test_chain_refuses_missing_extra_or_nonmonotonic_input(count, times):
    with pytest.raises(ValueError):
        list(
            SCRIPT.replay_chain(
                observations(count),
                times,
                predict=lambda sample, noise: np.zeros((100, 14)),
                adapter=adapter(),
                timing=TIMING,
                observation_to_send_s=0,
            )
        )
