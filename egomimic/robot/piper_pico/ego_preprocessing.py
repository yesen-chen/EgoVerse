"""Non-destructive trim/split decisions for high-rate joint Ego recordings.

Intervals are computed before downsampling. Even one invalid tracking sample
between two output ticks separates episodes; surviving rows are never joined.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PreprocessOptions:
    trim_start_s: float = 1.0
    trim_end_s: float = 1.0
    min_segment_s: float = 2.0
    max_tracking_gap_s: float = 0.05
    max_video_gap_s: float = 0.05

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.max_tracking_gap_s == 0 or self.max_video_gap_s == 0:
            raise ValueError("Gap thresholds must be positive")


def valid_runs(times, valid, *, gap_ns, invalid_reason):
    """Closed supported intervals with provenance for both boundaries."""
    times, valid = np.asarray(times, np.int64), np.asarray(valid, bool)
    if times.ndim != 1 or valid.shape != times.shape or len(times) < 2:
        raise ValueError("Expected matching timestamp/validity arrays")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Source timestamps must increase strictly")
    runs, start, start_reason = [], None, "source_start"
    for i in range(len(times) + 1):
        gap = i > 0 and i < len(times) and times[i] - times[i - 1] > gap_ns
        if start is not None and (i == len(times) or not valid[i] or gap):
            runs.append(
                {
                    "first": start,
                    "last": i - 1,
                    "start_ns": int(times[start]),
                    "end_ns": int(times[i - 1]),
                    "start_reason": start_reason,
                    "end_reason": (
                        "source_end"
                        if i == len(times)
                        else "tracking_gap"
                        if gap
                        else invalid_reason
                    ),
                }
            )
            start = None
        if i < len(times) and valid[i] and start is None:
            start = i
            start_reason = (
                "source_start" if i == 0 else "tracking_gap" if gap else invalid_reason
            )
    return runs


def _membership(runs, queries, shift_ns=0):
    ids = np.full(len(queries), -1, dtype=np.int64)
    for index, interval in enumerate(runs):
        lo, hi = interval["start_ns"] - shift_ns, interval["end_ns"] - shift_ns
        ids[(queries >= lo) & (queries <= hi)] = index
    return ids


def split_rows(
    numeric,
    raw,
    video_utc_ns,
    *,
    head_offset_ns,
    hand_offset_ns,
    options=None,
    max_image_dt_ns=None,
):
    """Return kept [start,end) grid ranges, boundary metadata, and audit report.

    A minimum duration refers to the first-to-last retained row span, not N/fps.
    Trimming applies to both the sample time and its selected image timestamp.
    The high-rate support samples must belong to the same uninterrupted run.
    """
    options = options or PreprocessOptions()
    options.validate()
    t = np.asarray(raw["timestamp_ns"], np.int64)
    grid = np.asarray(numeric["timestamp_ns"], np.int64)
    images = np.asarray(numeric["obs_rgb_timestamps_ns"], np.int64)
    video = np.asarray(video_utc_ns, np.int64)
    hand_valid = np.ones(len(t), bool)
    for side in ("left", "right"):
        hand_valid &= (
            raw[f"{side}_active"]
            & ((raw[f"{side}_status"] & 2) != 0).all(axis=1)
            & np.isfinite(raw[f"{side}_pose"][..., :3]).all(axis=(1, 2))
        )
        if f"{side}_geometry_valid" in raw:
            hand_valid &= raw[f"{side}_geometry_valid"]
    head = raw["head_pose"]
    head_valid = (
        ((raw["head_status"] & 3) == 3)
        & np.isfinite(head).all(axis=1)
        & (np.linalg.norm(head[:, 3:], axis=1) > 1e-8)
    )
    gap = round(options.max_tracking_gap_s * 1e9)
    hand_runs = valid_runs(t, hand_valid, gap_ns=gap, invalid_reason="inactivity")
    head_runs = valid_runs(t, head_valid, gap_ns=gap, invalid_reason="tracking_gap")
    video_runs = valid_runs(
        video,
        np.ones(len(video), bool),
        gap_ns=round(options.max_video_gap_s * 1e9),
        invalid_reason="video_gap",
    )
    for run in video_runs:
        for field in ("start_reason", "end_reason"):
            if run[field] == "tracking_gap":
                run[field] = "video_gap"
    streams = [
        (hand_runs, hand_offset_ns),
        (head_runs, head_offset_ns),
        (video_runs, 0),
    ]
    memberships = np.array(
        [_membership(runs, grid, shift) for runs, shift in streams]
    ).T
    image_memberships = np.array(
        [_membership(runs, images, shift) for runs, shift in streams]
    ).T
    image_interval_ok = (memberships == image_memberships).all(axis=1)
    lower = int(video[0]) + round(options.trim_start_s * 1e9)
    upper = int(video[-1]) - round(options.trim_end_s * 1e9)
    crop_ok = (grid >= lower) & (grid <= upper) & (images >= lower) & (images <= upper)
    image_ok = (
        np.ones(len(grid), bool)
        if max_image_dt_ns is None
        else (np.abs(images - grid) <= max_image_dt_ns)
    )
    good = crop_ok & image_ok & image_interval_ok & (memberships >= 0).all(axis=1)
    for side in ("left", "right"):
        good &= numeric[f"{side}.hand_valid"] & ~numeric[f"{side}.hand_filled"]
    # Finite output is necessary but cannot on its own detect sub-tick gaps.
    for value in numeric.values():
        if value.dtype.kind == "f":
            good &= np.isfinite(value).reshape(len(grid), -1).all(axis=1)
    for column, keys in (
        (0, ("left.source_support_timestamp_ns", "right.source_support_timestamp_ns")),
        (1, ("source.head_support_timestamp_ns",)),
    ):
        runs, _ = streams[column]
        for index, run in enumerate(runs):
            rows = memberships[:, column] == index
            for key in keys:
                support = numeric[key]
                good[rows] &= (
                    (support[rows] >= run["start_ns"])
                    & (support[rows] <= run["end_ns"])
                ).all(axis=1)
    segments, decisions, start = [], [], None
    for i in range(len(grid) + 1):
        changed = (
            i > 0 and i < len(grid) and np.any(memberships[i] != memberships[i - 1])
        )
        if start is not None and (i == len(grid) or not good[i] or changed):
            span = int(grid[i - 1] - grid[start])
            starts = [
                (lower, "ui_trim" if options.trim_start_s > 0 else "source_start")
            ]
            ends = [(upper, "ui_trim" if options.trim_end_s > 0 else "source_end")]
            for column, (runs, shift) in enumerate(streams):
                run = runs[int(memberships[start, column])]
                starts.append((run["start_ns"] - shift, run["start_reason"]))
                end_ns, end_reason = run["end_ns"] - shift, run["end_reason"]
                # A tracking file ending before the retained video ends is
                # missing future coverage, not a completed demonstration.
                # This also covers a nearest image crossing that terminal
                # tracking boundary before the uniform grid crosses it.
                if column < 2 and end_reason == "source_end" and end_ns < upper:
                    end_reason = "tracking_gap"
                ends.append((end_ns, end_reason))
            begin = max(starts, key=lambda item: item[0])
            end = min(ends, key=lambda item: item[0])
            if (
                start > 0
                and crop_ok[start - 1]
                and not good[start - 1]
                and image_interval_ok[start - 1]
                and np.array_equal(memberships[start - 1], memberships[start])
            ):
                begin = (
                    int(grid[start]),
                    "tracking_gap" if image_ok[start - 1] else "video_gap",
                )
            # Numeric/support rejection within a raw valid interval is itself
            # an artificial cut, never a certified natural task ending.
            if (
                i < len(grid)
                and not changed
                and not good[i]
                and crop_ok[i]
                and image_interval_ok[i]
            ):
                end = (int(grid[i]), "tracking_gap" if image_ok[i] else "video_gap")
            keep = i - start >= 2 and span >= round(options.min_segment_s * 1e9)
            decision = {
                "schema": "ego_preprocessing_v1",
                "candidate_range": [start, i],
                "start_reason": begin[1],
                "end_reason": end[1],
                "supported_start_ns": int(begin[0]),
                "supported_end_ns": int(end[0]),
                "first_row_timestamp_ns": int(grid[start]),
                "last_row_timestamp_ns": int(grid[i - 1]),
                "frames": i - start,
                "span_s": span / 1e9,
                "kept": keep,
                "decision": "keep" if keep else "below_min_segment_duration",
                "options": asdict(options),
            }
            decisions.append(decision)
            if keep:
                segments.append((start, i))
            start = None
        if i < len(grid) and good[i] and start is None:
            start = i
    report = {
        "schema": "ego_preprocessing_report_v1",
        "options": asdict(options),
        "trim_reference": "first/last saved frame receive UTC; both grid and selected image must be inside",
        "source_start_ns": int(video[0]),
        "source_end_ns": int(video[-1]),
        "trimmed_start_ns": lower,
        "trimmed_end_ns": upper,
        "source_hand_invalid_rows": int((~hand_valid).sum()),
        "raw_hand_runs": hand_runs,
        "raw_head_runs": head_runs,
        "raw_video_runs": video_runs,
        "candidate_rows": len(grid),
        "rows_outside_ui_trim": int((~crop_ok).sum()),
        "rows_outside_image_time_bound": int((~image_ok).sum()),
        "rows_image_crosses_source_boundary": int((~image_interval_ok).sum()),
        "candidate_segments": len(decisions),
        "kept_segments": len(segments),
        "output_rows": sum(b - a for a, b in segments),
        "kept_span_s": sum(d["span_s"] for d in decisions if d["kept"]),
        "discarded_short_segments": sum(not d["kept"] for d in decisions),
        "segments": decisions,
    }
    return segments, [d for d in decisions if d["kept"]], report
