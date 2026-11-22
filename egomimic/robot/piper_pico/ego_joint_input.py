"""Strict input contract for external-UVC / PICO joint recordings.

This module only reads files. It neither rectifies images nor applies time
compensation: callers query tracking UTC at image-receive UTC + fitted offset.
The fit's embedded intrinsics are frozen input, not a path to reload silently.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np

FIT_SCHEMA = "seeded_external_uvc_spacetime_v1"
PTS_TOLERANCE_NS = 250_000
MAX_QUERY_DURATION_NS = 20_000_000


@dataclass(frozen=True)
class JointInput:
    source_dir: Path
    video_path: Path
    tracking_path: Path
    fit: dict[str, Any]
    metadata: dict[str, Any]
    raw: dict[str, np.ndarray]
    video_pts_ns: np.ndarray
    video_pts_absolute_ns: np.ndarray
    video_utc_ns: np.ndarray
    video_frame_ids: np.ndarray
    source_size: tuple[int, int]
    T_C_H: np.ndarray
    raw_K: np.ndarray
    raw_D: np.ndarray

    @property
    def head_offset_ns(self) -> int:
        return round(self.fit["data"]["parameters"]["delta_head_ms"] * 1_000_000)

    @property
    def hand_offset_ns(self) -> int:
        return round(self.fit["data"]["parameters"]["delta_hand_ms"] * 1_000_000)


def _provenance(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def _json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return data


def _integer(value: Any, label: str) -> int:
    # Reject float conversion: UTC nanoseconds cannot survive a float round trip.
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{label}: expected an exact integer")
    try:
        result = int(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{label}: expected an exact integer") from error
    if not -(1 << 63) <= result < (1 << 63):
        raise ValueError(f"{label}: outside int64")
    return result


def _flag(value: Any, label: str) -> bool:
    if value in (False, 0, "0", "false"):
        return False
    if value in (True, 1, "1", "true"):
        return True
    raise ValueError(f"{label}: expected a boolean flag")


def _increasing(values: np.ndarray, label: str) -> None:
    if len(values) < 2 or any(int(b) <= int(a) for a, b in zip(values, values[1:])):
        raise ValueError(f"{label}: needs at least two strictly increasing values")


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label}: expected finite array of shape {shape}")
    return result


def load_joint_fit(path: str | Path) -> dict[str, Any]:
    """Validate an immutable fitted head-to-camera transform and time offsets."""
    path = Path(path)
    data = _json(path)
    if data.get("schema") != FIT_SCHEMA:
        raise ValueError(f"Unsupported joint fit schema: {data.get('schema')!r}")
    try:
        parameters, intrinsics = data["parameters"], data["intrinsics"]
        rotation = _array(parameters["R_CH"], (3, 3), "R_CH")
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-6
        ) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
            raise ValueError("R_CH: expected a proper rotation")
        _array(parameters["t_CH_m"], (3,), "t_CH_m")
        for name in ("delta_head_ms", "delta_hand_ms"):
            value = parameters[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name}: expected a finite millisecond offset")
        intrinsic = _array(intrinsics["raw_K"], (3, 3), "raw_K")
        if (
            not np.allclose(intrinsic[2], [0, 0, 1])
            or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0
        ):
            raise ValueError("raw_K: invalid pinhole intrinsics")
        if not np.isclose(intrinsic[0, 1], 0) or not np.isclose(intrinsic[1, 0], 0):
            raise ValueError("raw_K: skewed intrinsics are unsupported")
        _array(intrinsics["raw_D"], (5,), "raw_D (OpenCV Brown5)")
        size = intrinsics["image_size"]
        if len(size) != 2 or any(
            isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in size
        ):
            raise ValueError("image_size: expected positive integer width and height")
        if intrinsics.get("mirror") is not False:
            raise ValueError("Only explicitly unmirrored raw UVC images are supported")
    except KeyError as error:
        raise ValueError(f"Joint fit missing {error.args[0]}") from error
    return {**_provenance(path), "data": data}


def _check_metadata(session: dict, joint: dict, summary: dict) -> tuple[str, int]:
    if (
        session.get("schema") != "pico_uvc_probe_v2"
        or joint.get("schema") != "pico_uvc_joint_v1"
    ):
        raise ValueError("Unsupported joint recording metadata schema")
    for name, record in (("session", session), ("joint_session", joint)):
        if record.get("complete") is not True or record.get("status") != "complete":
            raise ValueError(f"{name}: recording is not complete")
        if record.get("synthetic") is not False:
            raise ValueError(f"{name}: synthetic or unspecified camera recording")
        if record.get("alignment_clock") != "unix_utc":
            raise ValueError(f"{name}: expected unix_utc alignment clock")
        for key in ("clock_discontinuities", "clock_outside_anchor_samples"):
            if _integer(record.get(key, 0), f"{name}.{key}") != 0:
                raise ValueError(f"{name}: {key} requires clock review")
    if session.get("utc_alignment_requires_review", False):
        raise ValueError("session: UTC alignment requires review")
    if (
        session.get("mp4_finalized") is not True
        or joint.get("video_complete") is not True
        or joint.get("tracking_complete") is not True
    ):
        raise ValueError("Joint video/tracking files were not finalized")
    if session.get("session_id") != joint.get("session_id"):
        raise ValueError("session_id mismatch")
    clock_id = session.get("clock_id")
    if (
        not isinstance(clock_id, str)
        or not clock_id
        or joint.get("clock_id") != clock_id
    ):
        raise ValueError("session/joint clock_id mismatch")
    if summary.get("clock_id", clock_id) != clock_id:
        raise ValueError("tracking summary clock_id mismatch")
    if summary.get("session_id", session["session_id"]) != session["session_id"]:
        raise ValueError("tracking summary session_id mismatch")
    for key in (
        "clock_discontinuities",
        "clock_outside_anchor_samples",
        "unaccounted_samples",
    ):
        if _integer(summary.get(key, 0), f"tracking_summary.{key}") != 0:
            raise ValueError(f"tracking_summary: {key} requires review")
    if (
        summary.get("complete", True) is not True
        or summary.get("finalized", True) is not True
    ):
        raise ValueError("tracking_summary: recording is not complete/finalized")
    if (
        joint.get("tracking_config", {}).get("source_sensor_latency_applied")
        is not False
    ):
        raise ValueError("Source sensor compensation is unknown or already applied")
    threshold = _integer(
        session.get("clock_jump_threshold_ns", 10_000_000), "clock_jump_threshold_ns"
    )
    if threshold <= 0:
        raise ValueError("clock_jump_threshold_ns must be positive")
    return clock_id, threshold


def _clock_rows(
    rows: list[dict], clock_id: str, threshold: int, tracking: bool
) -> tuple[np.ndarray, np.ndarray]:
    label = "tracking" if tracking else "frames"
    utc, monotonic = [], []
    for i, row in enumerate(rows):
        prefix = f"{label} row {i}"
        if tracking and row.get("clock_id") != clock_id:
            raise ValueError(f"{prefix}: clock_id mismatch")
        if row.get("utc_source") != "direct_wall_clock":
            raise ValueError(f"{prefix}: UTC must be directly recorded, not inferred")
        for flag in ("utc_discontinuity", "utc_outside_anchor_tolerance"):
            if _flag(row.get(flag), f"{prefix}.{flag}"):
                raise ValueError(f"{prefix}: {flag}")
        utc.append(_integer(row["receive_utc_ns"], f"{prefix}.receive_utc_ns"))
        monotonic.append(_integer(row["receive_time_ns"], f"{prefix}.receive_time_ns"))
        if (
            tracking
            and _integer(row["timeStampNs"], f"{prefix}.timeStampNs") != utc[-1]
        ):
            raise ValueError(f"{prefix}: timeStampNs disagrees with receive_utc_ns")
        for name in ("utc_step_ns", "utc_offset_from_anchor_ns"):
            if (
                name in row
                and abs(_integer(row[name], f"{prefix}.{name}")) >= threshold
            ):
                raise ValueError(f"{prefix}: {name} exceeds clock threshold")
    utc = np.asarray(utc, np.int64)
    monotonic = np.asarray(monotonic, np.int64)
    _increasing(utc, f"{label} UTC")
    _increasing(monotonic, f"{label} monotonic")
    # Test the timeline independently of the recorded discontinuity flags.
    residual = np.array(
        [
            (int(u) - int(utc[0])) - (int(m) - int(monotonic[0]))
            for u, m in zip(utc, monotonic)
        ],
        np.int64,
    )
    if np.max(np.abs(residual)) >= threshold:
        raise ValueError(f"{label}: UTC drift/step disagrees with monotonic timeline")
    return utc, monotonic


def _pose(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, str):
        raise ValueError(f"{label}: expected comma-separated xyz+xyzw")
    try:
        return _array([float(part) for part in value.split(",")], (7,), label)
    except ValueError as error:
        raise ValueError(f"{label}: malformed xyz+xyzw") from error


def _tracking(path: Path, clock_id: str, threshold: int) -> tuple[dict, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    utc, monotonic = _clock_rows(rows, clock_id, threshold, tracking=True)
    n = len(rows)
    raw = {
        "timestamp_ns": utc,
        "head_pose": np.empty((n, 7)),
        "head_status": np.zeros(n, np.uint32),
    }
    for side in ("left", "right"):
        raw[side + "_pose"] = np.empty((n, 26, 7))
        raw[side + "_status"] = np.zeros((n, 26), np.uint32)
        raw[side + "_active"] = np.zeros(n, bool)
    raw["head_source_status"] = np.zeros(n, np.uint32)
    raw["query_duration_ns"] = np.empty(n, np.int64)
    sample_ids = []
    for i, row in enumerate(rows):
        end = _integer(row["sample_end_time_ns"], "sample_end_time_ns")
        duration = end - int(monotonic[i])
        if duration < 0:
            raise ValueError(f"tracking row {i}: negative query duration")
        raw["query_duration_ns"][i] = duration
        sample_ids.append(_integer(row["sample_id"], "sample_id"))
        sample = row["tracking"]
        good = row.get("status") == "saved" and duration <= MAX_QUERY_DURATION_NS
        good &= sample.get("appState", {}).get("focus", True) is True
        head = sample["Head"]
        raw["head_pose"][i] = _pose(head["pose"], f"tracking row {i} Head")
        status = _integer(head.get("status", 0), "head.status")
        if not 0 <= status <= np.iinfo(np.uint32).max:
            raise ValueError("head.status outside uint32")
        raw["head_source_status"][i] = status
        good_head = good and head.get("sdk_result", 0) == 0
        good_head &= abs(np.linalg.norm(raw["head_pose"][i, 3:]) - 1) < 0.05
        raw["head_status"][i] = status if good_head else 0
        for side in ("left", "right"):
            hand = sample["Hand"][side + "Hand"]
            joints = hand["HandJointLocations"]
            if hand["count"] != 26 or len(joints) != 26:
                raise ValueError(f"tracking row {i} {side}: expected 26 XR joints")
            raw[side + "_pose"][i] = [
                _pose(j["p"], f"{side} joint {k}") for k, j in enumerate(joints)
            ]
            flags = [_integer(j["s"], f"{side}.joint.status") for j in joints]
            if any(not 0 <= flag <= np.iinfo(np.uint32).max for flag in flags):
                raise ValueError("joint status outside uint32")
            valid = good and hand.get("sdk_success", False) is True
            raw[side + "_status"][i] = flags if valid else 0
            raw[side + "_active"][i] = valid and _flag(
                hand["isActive"], f"{side}.isActive"
            )
    _increasing(np.asarray(sample_ids, np.int64), "tracking sample_id")
    raw["sample_id"] = np.asarray(sample_ids, np.int64)
    raw["receive_monotonic_ns"] = monotonic
    return raw, {
        "samples": n,
        "query_duration_over_20ms": int(
            (raw["query_duration_ns"] > MAX_QUERY_DURATION_NS).sum()
        ),
        "first_utc_ns": int(utc[0]),
        "last_utc_ns": int(utc[-1]),
    }


def _video(path: Path) -> tuple[np.ndarray, tuple[int, int], list[int], list[str]]:
    absolute, ticks, bases = [], [], []
    size = None
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError("Expected exactly one video stream")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Video frame has no PTS/time base")
            current_size = (frame.width, frame.height)
            if size is not None and size != current_size:
                raise ValueError("Video resolution changed")
            size = current_size
            ticks.append(int(frame.pts))
            bases.append(str(frame.time_base))
            absolute.append(round(frame.pts * frame.time_base * 1_000_000_000))
    absolute = np.asarray(absolute, np.int64)
    _increasing(absolute, "decoded video PTS")
    return absolute, size, ticks, bases


def load_joint_input(source: str | Path, fit_path: str | Path) -> JointInput:
    """Read one complete recording, fail closed on ambiguous frame/clock mapping.

    Video PTS is matched to the saved CSV mux PTS, never inferred from a frame
    index or requested FPS. A one-to-one match within 250 us permits ordinary
    MP4 timescale quantization observed in the recorder; it does not allow
    missing/duplicate timestamps or a different frame count.
    """
    source = Path(source).resolve()
    session_path, joint_path, summary_path = (
        source / name
        for name in ("session.json", "joint_session.json", "tracking_summary.json")
    )
    session, joint, summary = (
        _json(path) for path in (session_path, joint_path, summary_path)
    )
    clock_id, threshold = _check_metadata(session, joint, summary)
    fit = load_joint_fit(fit_path)
    video_path, csv_path, tracking_path = (
        source / name for name in ("ego.mp4", "frames.csv", "tracking.jsonl")
    )
    with csv_path.open(newline="") as stream:
        terminal = list(csv.DictReader(stream))
    ids = [_integer(row["frame_id"], "frame_id") for row in terminal]
    if len(ids) != len(set(ids)) or any(i < 0 for i in ids):
        raise ValueError("frames.csv: duplicate or negative frame_id")
    if any(row["status"] not in ("saved", "dropped", "error") for row in terminal):
        raise ValueError("frames.csv: unknown terminal status")
    saved = sorted(
        (row for row in terminal if row["status"] == "saved"),
        key=lambda r: _integer(r["mux_pts_us"], "mux_pts_us"),
    )
    mux = np.asarray(
        [_integer(r["mux_pts_us"], "mux_pts_us") * 1000 for r in saved], np.int64
    )
    _increasing(mux, "CSV mux PTS")
    saved_ids = np.asarray(
        [_integer(row["frame_id"], "frame_id") for row in saved], np.int64
    )
    _increasing(saved_ids, "saved frame_id")
    utc, monotonic = _clock_rows(saved, clock_id, threshold, tracking=False)
    for key, expected in (
        ("saved_frames", len(saved)),
        ("terminal_csv_rows", len(terminal)),
        ("received_frames", len(terminal)),
    ):
        if key in session and _integer(session[key], key) != expected:
            raise ValueError(f"session {key} disagrees with frames.csv")
    pts, size, ticks, bases = _video(video_path)
    if len(pts) != len(saved):
        raise ValueError(
            f"MP4/CSV saved frame count mismatch: {len(pts)} != {len(saved)}"
        )
    match = []
    for value in pts:
        insertion = int(np.searchsorted(mux, value))
        choices = [j for j in (insertion - 1, insertion) if 0 <= j < len(mux)]
        distances = [abs(int(mux[j]) - int(value)) for j in choices]
        if (
            not distances
            or min(distances) > PTS_TOLERANCE_NS
            or distances.count(min(distances)) != 1
        ):
            raise ValueError("MP4 PTS has no unique CSV mux PTS match within 250 us")
        match.append(choices[distances.index(min(distances))])
    if len(set(match)) != len(saved) or match != list(range(len(saved))):
        raise ValueError("MP4/CSV PTS mapping is not one-to-one and ordered")
    intrinsics = fit["data"]["intrinsics"]
    if size != tuple(intrinsics["image_size"]):
        raise ValueError("Video resolution disagrees with frozen fit intrinsics")
    for record in (session, joint):
        if (record.get("width"), record.get("height")) != size:
            raise ValueError("Video resolution disagrees with recording metadata")
    raw, tracking_stats = _tracking(tracking_path, clock_id, threshold)
    if "saved_samples" in summary and _integer(
        summary["saved_samples"], "saved_samples"
    ) != len(raw["timestamp_ns"]):
        raise ValueError("tracking summary saved_samples disagrees with tracking.jsonl")
    if "clock_anchor_utc_ns" in session and "clock_anchor_monotonic_ns" in session:
        anchor_offset = _integer(
            session["clock_anchor_utc_ns"], "clock_anchor_utc_ns"
        ) - _integer(session["clock_anchor_monotonic_ns"], "clock_anchor_monotonic_ns")
        first_offset = int(utc[0]) - int(monotonic[0])
        if abs(first_offset - anchor_offset) >= threshold:
            raise ValueError("Video UTC/monotonic origin disagrees with session anchor")
    # Same clock_id is necessary but not sufficient: independently check that
    # both streams' UTC/monotonic clock offsets agree at the session scale.
    frame_offset = int(utc[0]) - int(monotonic[0])
    tracking_offset = int(raw["timestamp_ns"][0]) - int(raw["receive_monotonic_ns"][0])
    if abs(frame_offset - tracking_offset) >= threshold:
        raise ValueError("Video/tracking UTC and monotonic origins disagree")
    transform = np.eye(4)
    transform[:3, :3] = fit["data"]["parameters"]["R_CH"]
    transform[:3, 3] = fit["data"]["parameters"]["t_CH_m"]
    pts_delta = pts - mux
    metadata = {
        "schema": "external_uvc_joint_input_v1",
        "session": session,
        "joint_session": joint,
        "tracking_summary": summary,
        "source_files": {
            path.name: _provenance(path)
            for path in (
                video_path,
                csv_path,
                tracking_path,
                session_path,
                joint_path,
                summary_path,
            )
        },
        "tracking_validation": tracking_stats,
        "frame_mapping": {
            "method": "actual_decoded_pts_to_unique_saved_csv_mux_pts",
            "tolerance_ns": PTS_TOLERANCE_NS,
            "max_abs_pts_error_ns": int(np.max(np.abs(pts_delta))),
            "decoded_frames": len(pts),
            "saved_csv_rows": len(saved),
            "dropped_csv_rows": sum(row["status"] == "dropped" for row in terminal),
            "error_csv_rows": sum(row["status"] == "error" for row in terminal),
            "original_pts_ticks": ticks,
            "original_time_bases": bases,
            "original_pts_ns": pts.tolist(),
            "csv_mux_pts_ns": mux.tolist(),
            "csv_frame_ids": saved_ids.tolist(),
            "csv_receive_utc_ns": utc.tolist(),
            "actual_minus_csv_pts_ns": pts_delta.tolist(),
        },
        "coordinate_convention": "T_C_W = T_C_H @ inverse(T_W_H); raw XR world poses, no reflection, no eye swap",
        "time_convention": "query_tracking_utc_ns = image_receive_utc_ns + fitted_offset_ns; compensation not yet applied",
    }
    return JointInput(
        source,
        video_path,
        tracking_path,
        fit,
        metadata,
        raw,
        pts - pts[0],
        pts,
        utc,
        saved_ids,
        size,
        transform,
        np.asarray(intrinsics["raw_K"], float),
        np.asarray(intrinsics["raw_D"], float),
    )
