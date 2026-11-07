"""Deterministic offline QR timing analysis; no display or camera is opened."""

import json

import pytest

from egomimic.robot.piper_pico.latency_analysis import analyze_decoded


@pytest.fixture
def variable_software_costs():
    # All emitted codes, including one never seen by the camera, have different
    # preparation costs. Subtracting their global mean corrupts these samples.
    return {
        "decoded": [(10.12, "10.0"), (20.19, "20.0")],
        "submissions": {
            "10.0": {"generated_unix": 10.0, "submitted_unix": 10.02},
            "20.0": {"generated_unix": 20.0, "submitted_unix": 20.09},
            "30.0": {"generated_unix": 30.0, "submitted_unix": 30.80},
        },
    }


def test_each_qr_uses_its_own_submission(variable_software_costs):
    report = analyze_decoded(**variable_software_costs, min_samples=2)
    assert [sample["raw_sec"] for sample in report["samples"]] == pytest.approx(
        [0.12, 0.19]
    )
    assert [sample["software_sec"] for sample in report["samples"]] == pytest.approx(
        [0.02, 0.09]
    )
    assert [sample["corrected_sec"] for sample in report["samples"]] == pytest.approx(
        [0.10, 0.10]
    )
    assert report["latency"]["mean_sec"] == pytest.approx(0.10)
    assert report["latency"]["std_sec"] == pytest.approx(0, abs=1e-12)
    assert report["valid_sample_count"] == 2
    assert report["submitted_qrs"] == 3
    assert report["unique_decoded_qrs"] == 2
    assert report["recommendation_allowed"] is True


def test_explicit_display_compensation_applies_exactly_once(variable_software_costs):
    baseline = analyze_decoded(**variable_software_costs, min_samples=2)
    report = analyze_decoded(
        **variable_software_costs, display_latency_sec=0.025, min_samples=2
    )
    assert report["raw_latency"] == baseline["raw_latency"]
    assert (
        report["latency_before_display_correction"]
        == baseline["latency_before_display_correction"]
    )
    assert report["latency"]["mean_sec"] == pytest.approx(0.075)
    assert [sample["corrected_sec"] for sample in report["samples"]] == pytest.approx(
        [0.075, 0.075]
    )


def test_earliest_finite_receive_is_used_even_when_unsorted():
    report = analyze_decoded(
        [(5.9, "5"), (float("nan"), "5"), (5.3, "5"), (5.5, "5")],
        {"5": {"generated_unix": 5, "submitted_unix": 5.1}},
        min_samples=1,
    )
    assert report["samples"][0]["receive_unix"] == 5.3
    assert report["latency"]["mean_sec"] == pytest.approx(0.2)
    assert report["rejected_counts"]["nonfinite_receive"] == 1
    assert report["paired_qrs"] == 1


def test_buffer_start_excludes_equal_and_older_submissions():
    report = analyze_decoded(
        [(5.2, "4"), (5.3, "4.9"), (5.4, "5.1"), (6, "5.8")],
        {
            "4": {"generated_unix": 4, "submitted_unix": 4.1},
            "4.9": {"generated_unix": 4.9, "submitted_unix": 5},
            "5.1": {"generated_unix": 5.1, "submitted_unix": 5.2},
            "5.8": {"generated_unix": 5.8, "submitted_unix": 5.9},
        },
        min_samples=2,
        buffer_start=5,
        buffer_end=6,
    )
    assert [sample["qr_text"] for sample in report["samples"]] == ["5.1", "5.8"]
    assert report["rejected_counts"]["left_censored_qrs"] == 2
    assert report["recommendation_allowed"] is True


def test_failures_unknown_and_left_censored_are_separate():
    report = analyze_decoded(
        [(5.1, ""), (5.2, "999"), (5.3, "4"), (5.4, "999"), (5.5, "junk")],
        {"4": {"generated_unix": 4, "submitted_unix": 4.5}},
        buffer_start=5,
    )
    assert report["rejected_counts"]["decode_failures"] == 1
    assert report["rejected_counts"]["unknown_codes"] == 3
    assert report["rejected_counts"]["malformed_payloads"] == 1
    assert report["rejected_counts"]["left_censored_qrs"] == 1
    assert report["unique_decoded_qrs"] == 3
    assert report["samples"] == []
    assert report["raw_latency"] is None
    assert report["latency_before_display_correction"] is None
    assert report["latency"] is None
    assert report["recommendation_allowed"] is False


def test_impossible_first_receive_is_rejected_not_replaced_by_later_receive():
    report = analyze_decoded(
        [(3.8, "4"), (4.5, "4"), (5.05, "5"), (5.3, "5")],
        {
            "4": {"generated_unix": 4, "submitted_unix": 4.1},
            "5": {"generated_unix": 5, "submitted_unix": 5.1},
        },
        min_samples=1,
    )
    assert report["rejected_counts"]["pre_submit_receive_qrs"] == 2
    assert report["rejected_counts"]["negative_raw_qrs"] == 1
    assert report["samples"] == []
    assert report["recommendation_allowed"] is False


def test_negative_corrected_samples_are_retained_and_block_recommendation():
    report = analyze_decoded(
        [(1.12, "1"), (2.3, "2")],
        {
            "1": {"generated_unix": 1, "submitted_unix": 1.1},
            "2": {"generated_unix": 2, "submitted_unix": 2.1},
        },
        display_latency_sec=0.05,
        min_samples=1,
    )
    assert [sample["corrected_sec"] for sample in report["samples"]] == pytest.approx(
        [-0.03, 0.15]
    )
    assert [sample["valid"] for sample in report["samples"]] == [False, True]
    assert report["latency"]["count"] == 2
    assert report["latency"]["min_sec"] == pytest.approx(-0.03)
    assert report["latency_before_display_correction"]["min_sec"] == pytest.approx(0.02)
    assert report["valid_sample_count"] == 1
    assert report["invalid_sample_count"] == 1
    assert report["recommendation_allowed"] is False
    assert any(
        "negative_corrected_samples" in warning
        for warning in report["quality_warnings"]
    )


def test_too_few_samples_preserves_statistics_but_blocks_recommendation(
    variable_software_costs,
):
    report = analyze_decoded(**variable_software_costs)
    assert report["latency"]["mean_sec"] == pytest.approx(0.1)
    assert report["min_samples"] == 30
    assert report["recommendation_allowed"] is False
    assert any(
        "insufficient_valid_samples: 2 < 30" in warning
        for warning in report["quality_warnings"]
    )


def test_stale_and_nonfinite_values_do_not_leak_into_json():
    report = analyze_decoded(
        [
            (float("nan"), "1"),
            (float("inf"), "1"),
            (5, "nan"),
            (5, "2"),
            (9, "8"),
            (7, "3"),
        ],
        {
            "1": {"generated_unix": 1, "submitted_unix": 1.1},
            "nan": {"generated_unix": 2, "submitted_unix": 2.1},
            "2": {"generated_unix": 2, "submitted_unix": float("inf")},
            "8": {"generated_unix": 8, "submitted_unix": 8.1},
            "3": {"generated_unix": 3, "submitted_unix": 3.1},
        },
        buffer_start=2,
        buffer_end=6,
    )
    assert report["rejected_counts"]["nonfinite_receive"] == 2
    assert report["rejected_counts"]["invalid_submission_qrs"] == 2
    assert report["rejected_counts"]["submission_after_buffer_qrs"] == 1
    assert report["rejected_counts"]["receive_outside_buffer_qrs"] == 2
    json.dumps(report, allow_nan=False)


def test_empty_input_is_a_useful_json_report():
    report = analyze_decoded([], {})
    assert report["decoded_frames"] == report["valid_sample_count"] == 0
    assert report["latency"] is None
    assert report["quality_warnings"]
    assert report["recommendation_allowed"] is False
    json.dumps(report, allow_nan=False)


def test_malformed_records_and_negative_software_are_rejected():
    report = analyze_decoded(
        [None, (1,), (2, {}), (3, None), (4, "3"), (6, "5")],
        {
            "3": {"generated_unix": 3, "submitted_unix": 2.9},
            "5": {"generated_unix": 5, "submitted_unix": 5.1},
        },
        min_samples=1,
    )
    assert report["rejected_counts"]["malformed_frames"] == 2
    assert report["rejected_counts"]["malformed_payloads"] == 1
    assert report["rejected_counts"]["decode_failures"] == 1
    assert report["rejected_counts"]["negative_software_qrs"] == 1
    assert report["valid_sample_count"] == 1
    assert report["recommendation_allowed"] is False
    json.dumps(report, allow_nan=False)


def test_summary_has_population_std_and_linearly_interpolated_p95():
    report = analyze_decoded(
        [(2, "1"), (4, "2"), (6, "3"), (8, "4"), (15, "5")],
        {
            str(value): {"generated_unix": value, "submitted_unix": value}
            for value in range(1, 6)
        },
        min_samples=5,
    )
    assert report["latency"] == pytest.approx(
        {
            "count": 5,
            "mean_sec": 4,
            "std_sec": 10**0.5,
            "median_sec": 3,
            "p95_sec": 8.8,
            "min_sec": 1,
            "max_sec": 10,
        }
    )


@pytest.mark.parametrize(
    "value", [True, False, 0, -1, 1.5, float("nan"), float("inf"), "3"]
)
def test_invalid_min_samples(value):
    with pytest.raises(ValueError, match="min_samples"):
        analyze_decoded([], {}, min_samples=value)


@pytest.mark.parametrize("value", [True, False, -0.1, float("nan"), float("inf"), None])
def test_invalid_display_compensation(value):
    with pytest.raises(ValueError, match="display_latency_sec"):
        analyze_decoded([], {}, display_latency_sec=value)


@pytest.mark.parametrize(
    "bounds",
    [
        {"buffer_start": float("nan")},
        {"buffer_end": float("inf")},
        {"buffer_start": 2, "buffer_end": 1},
    ],
)
def test_invalid_buffer_bounds(bounds):
    with pytest.raises(ValueError, match="buffer_"):
        analyze_decoded([], {}, **bounds)


def test_finite_extreme_arithmetic_cannot_emit_nonfinite_summary():
    report = analyze_decoded(
        [(1e308, "0"), (1e308, "1"), (1e308, "-1e308")],
        {
            "0": {"generated_unix": 0, "submitted_unix": 0},
            "1": {"generated_unix": 1, "submitted_unix": 1},
            "-1e308": {"generated_unix": -1e308, "submitted_unix": -1e308},
        },
        min_samples=2,
    )
    assert report["rejected_counts"]["nonfinite_offsets_qrs"] == 1
    assert report["latency"]["mean_sec"] == 1e308
    assert report["latency"]["median_sec"] == 1e308
    json.dumps(report, allow_nan=False)
