"""Offline RTC comparison uses paired noise and real, unpadded source windows."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.rldb.action_timing import ActionTiming


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "pi05" / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "rtc_history_comparison_tested", scripts / "evaluate_pi05_rtc_history.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_dataset(lengths=(130,), *, eligible=None, send=0.006, jitter=False):
    records, numeric = [], []
    for ep, length in enumerate(lengths):
        stamps = 1000.0 * (ep + 1) + np.arange(length) / 30
        if jitter:
            stamps += 0.002 * np.sin(np.arange(length))
        record = {
            "path": f"../../recordings/episode_{ep:06d}.zarr",
            "total_frames": length,
            "contract": {"time_alignment": {"observation_to_send_s": send}},
        }
        if eligible is not None:
            record["eligible_indices"] = eligible[ep]
        records.append(record)
        numeric.append({"timestamp": stamps})
    return SimpleNamespace(
        timing=ActionTiming(30, 50, 1, 100),
        episodes=records,
        _numeric=numeric,
    )


def select(runner, dataset, count=6, **overrides):
    options = dict(steps=27, delay=0.170, transition=0.300, frequency=30)
    options.update(overrides)
    return runner.select_pairs(dataset, count=count, **options)


def test_noise_is_paired_and_independent_of_global_rng(runner):
    first = runner.paired_noise(42, 7, 1)
    assert first.shape == (100, 32)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.random.seed(314)
    expected_next = np.random.random(4)
    np.random.seed(314)
    again = runner.paired_noise(42, 7, 1)
    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(np.random.random(4), expected_next)
    for seed, pair, role in ((43, 7, 1), (42, 8, 1), (42, 7, 0)):
        assert not np.array_equal(first, runner.paired_noise(seed, pair, role))


def test_noise_custom_shape_and_role_repeat(runner):
    first = runner.paired_noise(2, 3, 0, horizon=17, dimension=14)
    assert first.shape == (17, 14)
    np.testing.assert_array_equal(
        first, runner.paired_noise(2, 3, 0, horizon=17, dimension=14)
    )


def test_selection_preserves_source_rows_actual_times_and_tick_guard(runner):
    dataset = fake_dataset(jitter=True)
    pairs = select(runner, dataset)
    assert len(pairs) == 6
    assert [p["previous_row"] for p in pairs] == [0, 10, 21, 31, 42, 53]
    stamps = dataset._numeric[0]["timestamp"]
    interval = 1 / 30
    for pair in pairs:
        before, after = pair["previous_row"], pair["current_row"]
        assert after == before + 27
        assert pair["previous_time"] == stamps[before]
        assert pair["current_time"] == stamps[after]
        assert pair["previous_anchor"] == stamps[before] + 0.006
        assert pair["new_anchor"] == stamps[after] + 0.006
        assert pair["episode_name"] == "episode_000000.zarr"
        deadline = stamps[after] + 0.170
        assert deadline - 1e-10 <= pair["activation"] < deadline + interval
        ticks = (pair["activation"] - pair["previous_anchor"]) / interval
        assert ticks == pytest.approx(round(ticks), abs=1e-9)
        assert pair["hard_until"] == pytest.approx(pair["activation"] + interval)
        assert (
            pair["hard_until"] + dataset.timing.output_offsets_s[1] + 0.300
            < pair["previous_anchor"] + dataset.timing.effective_span_s
        )


def test_exact_tick_deadline_does_not_round_up_a_whole_extra_tick(runner):
    dataset = fake_dataset(send=0)
    # Both the previous/current observation and this 200 ms deadline are on
    # the control grid. Floating-point division must not add another tick.
    for pair in select(runner, dataset, delay=0.200):
        assert pair["activation"] == pytest.approx(
            pair["current_time"] + 0.200, abs=1e-10
        )


def test_no_episode_end_padding_or_cross_episode_pairing(runner):
    dataset = fake_dataset((130, 77, 76))
    pairs = select(runner, dataset, count=1000)
    assert len(pairs) == 54 + 1
    assert {p["episode"] for p in pairs} == {0, 1}
    for pair in pairs:
        record = dataset.episodes[pair["episode"]]
        assert (
            pair["current_row"] + dataset.timing.source_horizon
            <= record["total_frames"]
        )
        assert pair["previous_time"] >= 1000 * (pair["episode"] + 1)
        assert pair["current_time"] < 1000 * (pair["episode"] + 2)
    assert pairs[-1]["previous_row"] == 0
    assert pairs[-1]["current_row"] == 27


def test_history_eligibility_is_applied_to_both_original_source_rows(runner):
    eligible = [[2, 4, 6, 7, 29, 31, 33]]
    pairs = select(runner, fake_dataset(eligible=eligible), count=100)
    assert [(p["previous_row"], p["current_row"]) for p in pairs] == [
        (2, 29),
        (4, 31),
        (6, 33),
    ]


def test_recording_stalls_are_excluded_without_faking_uniform_times(runner):
    dataset = fake_dataset()
    dataset._numeric[0]["timestamp"][20:] += 0.200
    pairs = select(runner, dataset, count=100)
    assert pairs
    assert all(p["previous_row"] >= 20 for p in pairs)
    for pair in pairs:
        assert (
            pair["previous_time"]
            == dataset._numeric[0]["timestamp"][pair["previous_row"]]
        )
        assert pair["current_time"] - pair["previous_time"] == pytest.approx(0.9)


def test_empty_or_ineligible_recordings_raise(runner):
    with pytest.raises(ValueError, match="No complete, timing-valid pairs"):
        select(runner, fake_dataset((76,)))
    with pytest.raises(ValueError, match="No complete, timing-valid pairs"):
        select(runner, fake_dataset(eligible=[[0, 1, 2]]))


@pytest.mark.parametrize("count", [0, -1, True, False, 1.5, float("nan"), "6"])
def test_invalid_count_rejected(runner, count):
    with pytest.raises(ValueError, match="pairs-per-episode"):
        select(runner, fake_dataset(), count=count)


def test_positive_numpy_integer_count_is_allowed(runner):
    assert len(select(runner, fake_dataset(), count=np.int64(3))) == 3


def test_mismatched_source_and_control_frequency_is_explicitly_rejected(runner):
    with pytest.raises(ValueError, match="control frequency.*source_fps"):
        select(runner, fake_dataset(), frequency=35)


@pytest.mark.parametrize(
    "override",
    [
        {"steps": 0},
        {"steps": True},
        {"steps": 2.5},
        {"delay": 0},
        {"delay": float("nan")},
        {"transition": -0.1},
        {"frequency": 0},
        {"frequency": float("inf")},
        {"steps": 40},  # Insufficient old-plan overlap at takeover.
        {"steps": 6},  # A new request before the preceding one can take over.
    ],
)
def test_invalid_schedule_rejected_before_selection(runner, override):
    with pytest.raises(ValueError, match="RTC"):
        select(runner, fake_dataset(), **override)
