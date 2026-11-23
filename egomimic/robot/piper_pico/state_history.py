"""Causal feedback history shared by offline conversion and live inference.

This module never replaces the recorded current state. Raw streams use each
arm's base frame, metres and rotation vectors; callers apply the same camera
transform and gripper normalization as for their current observation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

SIDES = ("left", "right")
FEEDBACK_METADATA = "feedback_history.json"
_STREAM_SCHEMA = "piper_feedback_streams_v1"
_FIELDS = ("eef_pose", "gripper_width")


class HistoryCoverageError(ValueError):
    """The requested past time is not covered by causally available feedback."""


def validate_history_offsets(offsets=None) -> tuple[float, ...]:
    """Strictly increasing negative seconds, oldest first; no fixed time grid."""
    if offsets is None:
        return ()
    if isinstance(offsets, (str, bytes)):
        raise ValueError("history offsets must be a sequence of negative seconds")
    try:
        values = tuple(offsets)
    except TypeError as exc:
        raise ValueError("history offsets must be a sequence") from exc
    if any(isinstance(v, (bool, np.bool_)) for v in values):
        raise ValueError("history offsets must be numbers, not booleans")
    try:
        result = tuple(float(v) for v in values)
    except (ValueError, TypeError) as exc:
        raise ValueError("history offsets must be finite negative numbers") from exc
    if any(not np.isfinite(v) or v >= 0 for v in result):
        raise ValueError("history offsets must be finite and strictly negative")
    if any(b <= a for a, b in zip(result, result[1:])):
        raise ValueError("history offsets must be unique and increasing, oldest first")
    return result


def make_history_contract(offsets=None) -> dict | None:
    offsets = validate_history_offsets(offsets)
    if not offsets:
        return None
    return {
        "schema": "egoverse_state_history_v1",
        "offsets_s": list(offsets),
        "dimension": 14,
        "arm_order": list(SIDES),
        "representation": "absolute_xyz_ypr_gripper",
        "rotation_convention": "intrinsic_ZYX",
        "gripper_unit": "normalized_width",
        "coordinates": "current_observation_camera",
        "anchor": "observation_timestamp",
        "interpolation": "linear_xyz_gripper+slerp_rotation",
        "causal": True,
        "missing": "error",
        "source": "measured_pose_and_gripper",
    }


def validate_history_contract(value) -> tuple[float, ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError("state history contract must be an object or null")
    offsets = validate_history_offsets(value.get("offsets_s"))
    if not offsets or value != make_history_contract(offsets):
        raise ValueError("unsupported or inconsistent state history contract")
    return offsets


class FeedbackStreams(tuple):
    """Validated, immutable left/right arrays, suitable for reuse by a loader."""


def prepare_feedback_streams(streams) -> FeedbackStreams:
    if isinstance(streams, FeedbackStreams):
        return streams
    if isinstance(streams, Mapping):
        if set(streams) != set(SIDES):
            raise ValueError("feedback streams need exactly left and right arms")
        streams = [streams[side] for side in SIDES]
    if len(streams) != 2:
        raise ValueError("feedback streams need exactly two arms")
    result = []
    for side, stream in zip(SIDES, streams, strict=True):
        data = {}
        for field in _FIELDS:
            stamp_key = field + "__timestamp"
            if field not in stream or stamp_key not in stream:
                raise ValueError(f"{side} feedback lacks {field} or its timestamps")
            times = np.array(stream[stamp_key], dtype=np.float64, copy=True)
            values = np.array(stream[field], dtype=np.float64, copy=True)
            if field == "gripper_width" and values.ndim == 1:
                values = values[:, None]
            width = 6 if field == "eef_pose" else 1
            if (
                times.ndim != 1
                or not len(times)
                or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)
                or values.shape != (len(times), width)
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    f"{side}.{field}: invalid finite values/strict timestamps"
                )
            times.flags.writeable = values.flags.writeable = False
            data[stamp_key], data[field] = times, values
        result.append(data)
    return FeedbackStreams(result)


def sample_state_history(
    streams, observation_timestamp, offsets_s, *, available_until=None
) -> np.ndarray:
    """Return (history,14) base-frame feedback, never clamping or extrapolating.

    ``available_until`` is each arm's latest timestamp from the actual buffer
    read used for this observation. Offline callers must use recorded per-arm
    source timestamps, rather than the later observation query time. Future
    samples cannot supply the right interpolation endpoint.
    """
    offsets = validate_history_offsets(offsets_s)
    anchor = float(observation_timestamp)
    if not np.isfinite(anchor):
        raise ValueError("observation timestamp must be finite")
    if not offsets:
        return np.empty((0, 14), dtype=np.float64)
    streams = prepare_feedback_streams(streams)
    if available_until is None:
        cutoffs = np.full(2, anchor)
    elif isinstance(available_until, Mapping):
        if set(available_until) != set(SIDES):
            raise ValueError("available_until needs left and right")
        cutoffs = np.asarray([available_until[s] for s in SIDES], dtype=float)
    else:
        cutoffs = np.asarray(available_until, dtype=float)
        if cutoffs.ndim == 0:
            cutoffs = np.repeat(cutoffs, 2)
    if (
        cutoffs.shape != (2,)
        or not np.isfinite(cutoffs).all()
        or np.any(cutoffs > anchor)
    ):
        raise ValueError(
            "available_until must be finite per-arm timestamps <= observation"
        )
    targets = anchor + np.asarray(offsets)
    output = np.empty((len(offsets), 14), dtype=np.float64)
    for index, (side, stream, cutoff) in enumerate(
        zip(SIDES, streams, cutoffs, strict=True)
    ):
        for field in _FIELDS:
            all_times = stream[field + "__timestamp"]
            stop = int(np.searchsorted(all_times, cutoff, side="right"))
            if (
                not stop
                or targets[0] < all_times[0]
                or targets[-1] > all_times[stop - 1]
            ):
                raise HistoryCoverageError(
                    f"{side}.{field}: history is not covered before available cutoff"
                )
            # Only construct SO(3) interpolators over the requested short window,
            # not an entire episode for every minibatch item.
            first = max(
                0, int(np.searchsorted(all_times, targets[0], side="right")) - 1
            )
            last = min(
                stop, int(np.searchsorted(all_times, targets[-1], side="left")) + 1
            )
            times, values = all_times[first:last], stream[field][first:last]
            if field == "gripper_width":
                output[:, index * 7 + 6] = np.interp(targets, times, values[:, 0])
            else:
                for axis in range(3):
                    output[:, index * 7 + axis] = np.interp(
                        targets, times, values[:, axis]
                    )
                if len(times) == 1:
                    rotations = np.repeat(values[:, 3:], len(targets), axis=0)
                else:
                    rotations = Slerp(
                        times - times[0], Rotation.from_rotvec(values[:, 3:].copy())
                    )(targets - times[0]).as_rotvec()
                output[:, index * 7 + 3 : index * 7 + 6] = rotations
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_feedback_metadata(episode_path) -> dict | None:
    """Read a sidecar descriptor; absent means an older export without history."""
    path = Path(episode_path) / FEEDBACK_METADATA
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != _STREAM_SCHEMA:
        raise ValueError(f"{path}: unsupported feedback descriptor")
    if value.get("status") not in ("available", "unavailable"):
        raise ValueError(f"{path}: invalid feedback status")
    return value


def load_feedback_streams(episode_path, metadata=None) -> FeedbackStreams:
    """Load portable sidecars and verify both file hashes before interpolation."""
    episode_path = Path(episode_path).resolve()
    metadata = (
        portable_feedback_metadata(episode_path) if metadata is None else metadata
    )
    if not isinstance(metadata, dict) or metadata.get("status") != "available":
        raise HistoryCoverageError(
            f"{episode_path}: no raw feedback history; backfill it first"
        )
    expected = _stream_header()
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("feedback metadata representation/timestamp contract mismatch")
    if set(metadata.get("arms", {})) != set(SIDES):
        raise ValueError("feedback metadata requires both arms")
    result = []
    for side in SIDES:
        item = metadata["arms"][side]
        relative = Path(item.get("path", ""))
        path = (episode_path / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or episode_path not in path.parents
        ):
            raise ValueError("feedback file path must stay within its episode")
        if not path.is_file() or _sha256(path) != item.get("sha256"):
            raise ValueError(f"{side} feedback checksum mismatch or file missing")
        with np.load(path, allow_pickle=False) as archive:
            stream = {
                key: archive[key] for f in _FIELDS for key in (f, f + "__timestamp")
            }
        result.append(stream)
    streams = prepare_feedback_streams(result)
    for side, stream in zip(SIDES, streams, strict=True):
        item = metadata["arms"][side]
        for field, label in (("eef_pose", "pose"), ("gripper_width", "gripper")):
            times = stream[field + "__timestamp"]
            if item.get(label + "_samples") != len(times) or item.get(
                label + "_range_s"
            ) != [float(times[0]), float(times[-1])]:
                raise ValueError(f"{side}: feedback count/time range metadata mismatch")
    return streams


def _stream_header():
    return {
        "schema": _STREAM_SCHEMA,
        "status": "available",
        "representation": "xyz_m+rotvec_rad+gripper_m",
        "coordinates": {side: f"piper_{side}_base" for side in SIDES},
        "timestamp_unit": "s",
        "timestamp_clock": "recorded_robot_timestamp_host_wall_clock",
        "source": "actual_robot_feedback",
        "arm_order": list(SIDES),
    }


def write_feedback_sidecar(video_dir, episode_path, *, allow_missing=True) -> dict:
    """Copy raw feedback without modifying Zarr metadata, arrays or image bytes.

    Already populated sidecars are immutable and idempotent for the same raw
    bytes. Missing-history markers may later be upgraded by explicit backfill.
    """
    video_dir, episode_path = Path(video_dir), Path(episode_path)
    sources = [video_dir / f"robot{i}_state.npz" for i in range(2)]
    missing = [
        side for side, path in zip(SIDES, sources, strict=True) if not path.is_file()
    ]
    existing = portable_feedback_metadata(episode_path)
    if missing:
        if not allow_missing:
            raise FileNotFoundError(
                f"missing raw feedback for {missing} in {video_dir}"
            )
        proposed = {
            "schema": _STREAM_SCHEMA,
            "status": "unavailable",
            "reason": "missing_raw_feedback",
            "missing_arms": missing,
        }
    else:
        raw = []
        for path in sources:
            with np.load(path, allow_pickle=False) as archive:
                raw.append(
                    {
                        key: archive[key]
                        for f in _FIELDS
                        for key in (f, f + "__timestamp")
                    }
                )
        streams = prepare_feedback_streams(raw)
        proposed = {**_stream_header(), "arms": {}}
        for side, path, stream in zip(SIDES, sources, streams, strict=True):
            item = {"path": f"feedback_history/{side}.npz", "sha256": _sha256(path)}
            for field, label in (("eef_pose", "pose"), ("gripper_width", "gripper")):
                times = stream[field + "__timestamp"]
                item[label + "_samples"] = len(times)
                item[label + "_range_s"] = [float(times[0]), float(times[-1])]
            proposed["arms"][side] = item
    if existing == proposed:
        if proposed["status"] == "available":
            load_feedback_streams(episode_path, proposed)
        return proposed
    if existing is not None and existing.get("status") == "available":
        raise FileExistsError(
            "refusing to replace existing feedback with different raw data"
        )
    episode_path.mkdir(parents=True, exist_ok=True)
    if proposed["status"] == "available":
        directory = episode_path / "feedback_history"
        if directory.exists():
            raise FileExistsError(f"refusing to overwrite {directory}")
        temporary = Path(tempfile.mkdtemp(prefix=".feedback_", dir=episode_path))
        try:
            for side, path in zip(SIDES, sources, strict=True):
                shutil.copyfile(path, temporary / f"{side}.npz")
                if (
                    _sha256(temporary / f"{side}.npz")
                    != proposed["arms"][side]["sha256"]
                ):
                    raise ValueError("raw feedback changed during copying")
            temporary.rename(directory)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    descriptor = episode_path / FEEDBACK_METADATA
    fd, name = tempfile.mkstemp(prefix=".feedback_metadata_", dir=episode_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(proposed, file, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(name, descriptor)
    finally:
        if Path(name).exists():
            Path(name).unlink()
    return proposed
