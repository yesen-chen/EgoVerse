"""Offline analysis of emitted QR codes and camera receive timestamps.

The measured interval starts at the host's display submission, not at camera
exposure. Display scanout, camera frame phase, and image decoding can all remain
in this interval. Only an explicitly supplied display delay is subtracted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral
from statistics import fmean


def _finite_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _summary(values: list[float]) -> dict | None:
    """Return population statistics using bounded, finite intermediate values."""
    if not values:
        return None
    ordered = sorted(values)
    # Scale before summing/squaring so even finite extreme inputs cannot produce
    # an infinity/NaN in the JSON report.
    scale = max(abs(value) for value in values)
    scaled = [value / scale for value in values] if scale else values
    mean_scaled = fmean(scaled)
    std_scaled = math.sqrt(fmean((value - mean_scaled) ** 2 for value in scaled))
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    weight = position - lower
    # Weighted sums avoid overflowing the difference between extreme endpoints.
    p95 = ordered[lower] * (1 - weight) + ordered[math.ceil(position)] * weight
    midpoint = len(ordered) // 2
    center = (
        ordered[midpoint]
        if len(ordered) % 2
        else ordered[midpoint - 1] / 2 + ordered[midpoint] / 2
    )
    return {
        "count": len(values),
        "mean_sec": mean_scaled * scale,
        "std_sec": min(std_scaled, 1.0) * scale,
        "median_sec": center,
        "p95_sec": p95,
        "min_sec": ordered[0],
        "max_sec": ordered[-1],
    }


def analyze_decoded(
    decoded: list[tuple[float, str]],
    submissions: Mapping[str, Mapping[str, float]],
    *,
    display_latency_sec: float = 0.0,
    min_samples: int = 30,
    buffer_start: float | None = None,
    buffer_end: float | None = None,
) -> dict:
    """Pair each emitted QR with its earliest finite receive time.

    ``submissions`` records the generation and display-submission host Unix time
    for each exact payload. A payload submitted at or before ``buffer_start`` is
    left-censored: its actual first received frame may have left the buffer.
    Such payloads are excluded even if they decode in the retained frames.

    Rejection counts describe frames unless their name ends in ``_qrs``. Some
    diagnostics overlap: a nonempty unknown payload with a nonfinite receive
    time increments both counters. Corrected negative measurements are retained
    in ``samples`` and summaries, flagged invalid, and prevent a recommendation.
    Summary ``latency`` is the corrected interval for report compatibility.
    """
    compensation = _finite_number(display_latency_sec)
    if compensation is None or compensation < 0:
        raise ValueError("display_latency_sec must be finite and nonnegative")
    if (
        isinstance(min_samples, bool)
        or not isinstance(min_samples, Integral)
        or min_samples <= 0
    ):
        raise ValueError("min_samples must be a positive integer")
    min_samples = int(min_samples)
    start = None if buffer_start is None else _finite_number(buffer_start)
    end = None if buffer_end is None else _finite_number(buffer_end)
    if buffer_start is not None and start is None:
        raise ValueError("buffer_start must be finite")
    if buffer_end is not None and end is None:
        raise ValueError("buffer_end must be finite")
    if start is not None and end is not None and start > end:
        raise ValueError("buffer_start must not exceed buffer_end")
    if not isinstance(submissions, Mapping):
        raise ValueError("submissions must be a mapping of QR payloads to timestamps")

    rejected = dict.fromkeys(
        (
            "decode_failures",
            "malformed_frames",
            "malformed_payloads",
            "nonfinite_receive",
            "unknown_codes",
            "invalid_submission_qrs",
            "negative_software_qrs",
            "left_censored_qrs",
            "submission_after_buffer_qrs",
            "receive_outside_buffer_qrs",
            "pre_submit_receive_qrs",
            "negative_raw_qrs",
            "nonfinite_offsets_qrs",
        ),
        0,
    )
    emitted = {}
    for code, record in submissions.items():
        qr_time = _finite_number(code) if isinstance(code, str) and code else None
        generated = (
            _finite_number(record.get("generated_unix"))
            if isinstance(record, Mapping)
            else None
        )
        submitted = (
            _finite_number(record.get("submitted_unix"))
            if isinstance(record, Mapping)
            else None
        )
        if qr_time is None or generated is None or submitted is None:
            rejected["invalid_submission_qrs"] += 1
            continue
        if submitted < generated:
            rejected["negative_software_qrs"] += 1
            continue
        software = submitted - generated
        if not math.isfinite(software):
            rejected["invalid_submission_qrs"] += 1
            continue
        emitted[code] = (qr_time, generated, submitted, software)

    first_receive: dict[str, float] = {}
    unique_decoded = set()
    decoded_frames_with_code = 0
    for frame in decoded:
        if not isinstance(frame, (tuple, list)) or len(frame) != 2:
            rejected["malformed_frames"] += 1
            continue
        received, code = frame
        received = _finite_number(received)
        if received is None:
            rejected["nonfinite_receive"] += 1
        if code is None:
            rejected["decode_failures"] += 1
            continue
        if not isinstance(code, str):
            rejected["malformed_payloads"] += 1
            continue
        if code == "":
            rejected["decode_failures"] += 1
            continue
        decoded_frames_with_code += 1
        unique_decoded.add(code)
        if code not in submissions:
            rejected["unknown_codes"] += 1
        if _finite_number(code) is None:
            rejected["malformed_payloads"] += 1
            continue
        if received is None or code not in emitted:
            continue
        if code not in first_receive or received < first_receive[code]:
            first_receive[code] = received

    samples = []
    for code, received in sorted(
        first_receive.items(), key=lambda item: (item[1], item[0])
    ):
        qr_time, generated, submitted, software = emitted[code]
        raw = received - qr_time
        before_correction = received - submitted
        corrected = before_correction - compensation
        invalid = False
        # Count all applicable causes rather than allowing a left-censoring
        # check to hide an independently impossible negative measurement.
        conditions = {
            "left_censored_qrs": start is not None and submitted <= start,
            "submission_after_buffer_qrs": end is not None and submitted > end,
            "receive_outside_buffer_qrs": (
                (start is not None and received < start)
                or (end is not None and received > end)
            ),
            "pre_submit_receive_qrs": received < submitted,
            "negative_raw_qrs": received < qr_time,
            "nonfinite_offsets_qrs": not all(
                math.isfinite(value) for value in (raw, before_correction, corrected)
            ),
        }
        for reason, applies in conditions.items():
            if applies:
                rejected[reason] += 1
                invalid = True
        if invalid:
            continue
        samples.append(
            {
                "qr_text": code,
                "qr_timestamp_unix": qr_time,
                "generated_unix": generated,
                "submitted_unix": submitted,
                "receive_unix": received,
                "software_sec": software,
                "raw_sec": raw,
                "submitted_to_receive_sec": before_correction,
                "corrected_sec": corrected,
                "valid": corrected >= 0,
            }
        )

    negative_corrected = sum(not sample["valid"] for sample in samples)
    valid_count = len(samples) - negative_corrected
    warnings = [f"{reason}: {count}" for reason, count in rejected.items() if count]
    if negative_corrected:
        warnings.append(
            f"negative_corrected_samples: {negative_corrected}; negative values are retained; "
            "check the display compensation and timestamp ordering"
        )
    if valid_count < min_samples:
        warnings.append(f"insufficient_valid_samples: {valid_count} < {min_samples}")
    if not samples:
        warnings.append(
            "no_usable_samples: no emitted QR has a usable first receive time"
        )
    negative_timing = negative_corrected or any(
        rejected[reason]
        for reason in (
            "negative_software_qrs",
            "pre_submit_receive_qrs",
            "negative_raw_qrs",
        )
    )
    return {
        "definition": (
            "earliest received frame containing an emitted QR minus that QR's display-submission "
            "host timestamp minus explicit display_latency_sec; not absolute camera exposure latency"
        ),
        "display_latency_sec": compensation,
        "min_samples": min_samples,
        "buffer_start_unix": start,
        "buffer_end_unix": end,
        "decoded_frames": len(decoded),
        "decoded_frames_with_code": decoded_frames_with_code,
        "unique_decoded_qrs": len(unique_decoded),
        "submitted_qrs": len(submissions),
        "valid_submitted_qrs": len(emitted),
        "paired_qrs": len(samples),
        "valid_sample_count": valid_count,
        "invalid_sample_count": negative_corrected,
        "negative_corrected_samples": negative_corrected,
        "rejected_counts": rejected,
        "quality_warnings": warnings,
        "recommendation_allowed": valid_count >= min_samples and not negative_timing,
        "raw_latency": _summary([sample["raw_sec"] for sample in samples]),
        "latency_before_display_correction": _summary(
            [sample["submitted_to_receive_sec"] for sample in samples]
        ),
        "latency": _summary([sample["corrected_sec"] for sample in samples]),
        "samples": samples,
    }
