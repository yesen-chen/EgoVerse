"""Raw-frequency boundaries must survive output resampling without tail deletion."""

import numpy as np
import pytest

from egomimic.rldb.ego_boundary_policy import EgoBoundaryPolicy
from egomimic.robot.piper_pico.ego_preprocessing import PreprocessOptions, split_rows

EPOCH = 1_794_729_600_123_456_789


def streams(duration=6):
    t = EPOCH + np.arange(duration * 100 + 1, dtype=np.int64) * 10_000_000
    grid = EPOCH + np.rint(np.arange(duration * 30 + 1) / 30 * 1e9).astype(np.int64)
    raw = {
        "timestamp_ns": t,
        "head_status": np.full(len(t), 3),
        "head_pose": np.tile([0.0, 0, 0, 0, 0, 0, 1], (len(t), 1)),
    }
    lo = np.searchsorted(t, grid, side="right") - 1
    hi = np.searchsorted(t, grid)
    support = np.c_[t[lo], t[hi]]
    numeric = {
        "timestamp_ns": grid,
        "obs_rgb_timestamps_ns": grid.copy(),
        "source.head_support_timestamp_ns": support.copy(),
    }
    for side in ("left", "right"):
        raw.update(
            {
                f"{side}_active": np.ones(len(t), bool),
                f"{side}_status": np.full((len(t), 26), 15),
                f"{side}_pose": np.zeros((len(t), 26, 7)),
            }
        )
        numeric.update(
            {
                f"{side}.hand_valid": np.ones(len(grid), bool),
                f"{side}.hand_filled": np.zeros(len(grid), bool),
                f"{side}.source_support_timestamp_ns": support.copy(),
            }
        )
    return numeric, raw, grid


def split(numeric, raw, video, **opts):
    return split_rows(
        numeric,
        raw,
        video,
        head_offset_ns=0,
        hand_offset_ns=0,
        options=PreprocessOptions(**opts),
    )


def test_one_second_each_end_and_two_second_minimum():
    numeric, raw, video = streams()
    segments, boundaries, report = split(numeric, raw, video)
    assert segments == [(30, 151)]
    assert boundaries[0]["start_reason"] == boundaries[0]["end_reason"] == "ui_trim"
    assert boundaries[0]["span_s"] == 4
    assert report["output_rows"] == 121
    # Selection has not physically removed any source/support row.
    assert len(numeric["timestamp_ns"]) == 181


def test_sub_tick_invalid_sample_splits_even_when_all_grid_rows_claim_valid():
    numeric, raw, video = streams(8)
    raw["left_active"][401] = False  # 4.01s is never a 30Hz output tick.
    segments, boundaries, _ = split(numeric, raw, video)
    assert len(segments) == 2
    assert boundaries[0]["end_reason"] == "inactivity"
    assert boundaries[1]["end_reason"] == "ui_trim"
    assert boundaries[1]["start_reason"] == "inactivity"
    assert numeric["timestamp_ns"][segments[0][1] - 1] <= EPOCH + 4_000_000_000
    assert numeric["timestamp_ns"][segments[1][0]] >= EPOCH + 4_020_000_000


def test_support_may_not_cross_gap_even_with_nearest_or_interpolation():
    numeric, raw, video = streams(8)
    raw["right_status"][401] = 0
    # Deliberately bad interpolator output at a later otherwise valid tick.
    numeric["right.source_support_timestamp_ns"][121] = [
        raw["timestamp_ns"][400],
        raw["timestamp_ns"][404],
    ]
    segments, _, _ = split(numeric, raw, video)
    assert segments[1][0] == 122


def test_short_fragments_drop_and_never_stitch():
    numeric, raw, video = streams(10)
    raw["left_active"][400:411] = False
    raw["right_active"][450:461] = False
    segments, boundaries, report = split(numeric, raw, video)
    assert len(segments) == 2
    assert report["discarded_short_segments"] == 1
    assert all(x["span_s"] >= 2 for x in boundaries)
    assert segments[0][1] < segments[1][0]


def test_offsets_move_inactivity_boundaries_into_image_clock_once():
    numeric, raw, video = streams(8)
    raw["left_active"][401] = False
    shift = 100_000_000
    for side in ("left", "right"):
        numeric[f"{side}.source_support_timestamp_ns"] += shift
    segments, boundaries, _ = split_rows(
        numeric,
        raw,
        video,
        head_offset_ns=0,
        hand_offset_ns=shift,
        options=PreprocessOptions(),
    )
    assert boundaries[0]["supported_end_ns"] == EPOCH + 3_900_000_000
    assert segments[0][1] == 118


def test_selected_image_must_also_be_inside_trim():
    numeric, raw, video = streams()
    numeric["obs_rgb_timestamps_ns"][30] -= 1
    numeric["obs_rgb_timestamps_ns"][150] += 1
    assert split(numeric, raw, video)[0] == [(31, 150)]


def test_selected_image_cannot_land_inside_inactive_interval():
    numeric, raw, video = streams(8)
    raw["left_active"][401] = False
    numeric["obs_rgb_timestamps_ns"][119] = EPOCH + 4_010_000_000
    segments, boundaries, report = split(numeric, raw, video)
    assert segments[0][1] == 119
    assert boundaries[0]["end_reason"] == "inactivity"
    assert report["rows_image_crosses_source_boundary"] == 1


def test_video_or_tracking_timestamp_holes_split():
    numeric, raw, video = streams(8)
    video = np.delete(video, np.arange(115, 125))
    segments, boundaries, _ = split(numeric, raw, video)
    assert len(segments) == 2
    assert boundaries[0]["end_reason"] == "video_gap"
    numeric, raw, video = streams(8)
    for key in list(raw):
        raw[key] = np.delete(raw[key], np.arange(395, 405), axis=0)
    segments, boundaries, _ = split(numeric, raw, video)
    assert len(segments) == 2
    assert boundaries[0]["end_reason"] == "tracking_gap"


@pytest.mark.parametrize("trim_end", [0, 1])
def test_tracking_ending_before_video_is_artificial_cut(trim_end):
    numeric, raw, video = streams(8)
    raw = {key: value[:501] for key, value in raw.items()}
    segments, boundaries, _ = split(
        numeric, raw, video, trim_start_s=0, trim_end_s=trim_end
    )
    assert segments == [(0, 151)]
    assert boundaries[0]["end_reason"] == "tracking_gap"
    assert EgoBoundaryPolicy().resolve({"preprocessing": boundaries[0]})[1]


def test_nearest_image_crossing_terminal_tracking_is_artificial_cut():
    numeric, raw, video = streams(8)
    raw = {key: value[:498] for key, value in raw.items()}  # through 4.97s
    numeric["obs_rgb_timestamps_ns"][149] = EPOCH + 4_980_000_000
    video = video.copy()
    video[149] = EPOCH + 4_980_000_000
    segments, boundaries, _ = split(numeric, raw, video, trim_start_s=0, trim_end_s=1)
    assert segments == [(0, 149)]
    assert boundaries[0]["end_reason"] == "tracking_gap"
    assert EgoBoundaryPolicy().resolve({"preprocessing": boundaries[0]})[1]


def test_untrimmed_full_video_with_tracking_coverage_is_source_end():
    numeric, raw, video = streams(8)
    _, boundaries, _ = split(numeric, raw, video, trim_start_s=0, trim_end_s=0)
    assert boundaries[0]["end_reason"] == "source_end"
    assert not EgoBoundaryPolicy().resolve({"preprocessing": boundaries[0]})[1]


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_options_fail(value):
    with pytest.raises(ValueError):
        PreprocessOptions(trim_end_s=value).validate()


def test_too_short_recording_is_explicit_empty_report():
    numeric, raw, video = streams(2)
    segments, _, report = split(numeric, raw, video)
    assert not segments
    assert report["discarded_short_segments"] == 1
