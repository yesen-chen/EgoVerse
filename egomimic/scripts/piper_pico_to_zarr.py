"""Convert recorded Piper/PICO robot episodes to Piper Zarr datasets.

One output row is one control tick of the collector: the observation that
``env.get_obs`` returned at that tick and the command decided after it. Rows
are not resampled. Images are the exact frames ``get_obs`` selected, found in
the recorded video by their timestamp.

Poses remain in each Piper's base frame; extrinsics are never applied here.
Measured lens distortion is corrected on images using a saved image contract.
Low-dimensional/image field names follow the original Eva Zarr layout, but the
embodiment is piper_bimanual and requires a Piper-specific training loader.
No hardware, PICO SDK, CUDA, or Eva-specific rotation constants are imported.
Run with: python -m egomimic.scripts.piper_pico_to_zarr --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np
import simplejpeg
import zarr
from scipy.spatial.transform import Rotation

from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.robot.piper_pico.image_geometry import (
    RgbImageProcessor,
    camera_image_geometry,
)
from egomimic.robot.piper_pico.state_history import (
    validate_history_offsets,
    write_feedback_sidecar,
)

EXPECTED_CAMERAS = {"front_1", "left_wrist", "right_wrist"}
SIDES = ("left", "right")
OUTPUT_EMBODIMENT = "piper_bimanual"
OUTPUT_SCHEMA = "piper_base_v3"
FRAME_MATCH_TOLERANCE_S = 1e-6
MAX_REPORTED_GAPS = 20


@dataclass(frozen=True)
class Options:
    fps: float = 30.0
    width: int = 640
    height: int = 480
    max_frame_age: float = 0.1
    max_robot_age: float = 0.05
    max_tick_gap: float = 0.05
    gripper_max_m: float = 0.08
    allow_uncalibrated: bool = False
    allow_missing_cameras: bool = False
    task_name: str = ""
    task_description: str = ""
    undistort_alpha: float = 0.0

    def validate(self):
        for key in (
            "fps",
            "max_frame_age",
            "max_robot_age",
            "max_tick_gap",
            "gripper_max_m",
        ):
            value = getattr(self, key)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if not np.isfinite(self.undistort_alpha) or not 0 <= self.undistort_alpha <= 1:
            raise ValueError("undistort_alpha must be finite in [0,1]")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def timestamps(value, label):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or len(value) < 2 or not np.isfinite(value).all():
        raise ValueError(f"{label}: expected at least two finite timestamps")
    if np.any(np.diff(value) <= 0):
        raise ValueError(f"{label}: timestamps must be strictly increasing")
    return value


def array(value, shape, label):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{label}: expected finite array {shape}, got {value.shape}")
    return value


def matching_frame_indices(video_stamps, selected_stamps):
    """Video frame index of each selected stamp; -1 where the video lacks it."""
    right = np.searchsorted(video_stamps, selected_stamps, side="left")
    left = (right - 1).clip(0)
    right = right.clip(0, len(video_stamps) - 1)
    nearest = np.where(
        np.abs(video_stamps[left] - selected_stamps)
        <= np.abs(video_stamps[right] - selected_stamps),
        left,
        right,
    )
    found = np.abs(video_stamps[nearest] - selected_stamps) <= FRAME_MATCH_TOLERANCE_S
    return np.where(found, nearest, -1)


def pose_to_quaternion(position, rotvec):
    quat = Rotation.from_rotvec(rotvec).as_quat()[:, [3, 0, 1, 2]]
    # A continuous sign branch avoids q/-q discontinuities in stored trajectories.
    for i in range(1, len(quat)):
        if np.dot(quat[i - 1], quat[i]) < 0:
            quat[i] *= -1
    return np.concatenate([position, quat], axis=1)


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {
        "median_ms": float(1000 * np.median(values)),
        "p95_ms": float(1000 * np.percentile(values, 95)),
        "max_ms": float(1000 * values.max()),
        "min_ms": float(1000 * values.min()),
    }


def load_calibration(path: Path | None, options: Options):
    if path is None:
        if not options.allow_uncalibrated:
            raise ValueError(
                "Calibration is required: use --calibration FILE (or input/calibration.json). "
                "For diagnostic conversion only, use --allow-uncalibrated."
            )
        return None
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("calibration must be a JSON object")
    if not data.get("calibration_id") or data.get("reference_camera") != "front_1":
        raise ValueError(
            "calibration needs calibration_id and reference_camera='front_1'"
        )
    if data.get("extrinsics_convention") != "T_base_camera":
        raise ValueError(
            "extrinsics_convention must be T_base_camera (camera -> arm base)"
        )
    extrinsics = data.get("extrinsics", {})
    if not isinstance(extrinsics, dict) or set(extrinsics) != set(SIDES):
        raise ValueError("extrinsics must contain exactly left and right")
    for side, value in extrinsics.items():
        transform = array(value, (4, 4), f"extrinsics.{side}")
        rotation = transform[:3, :3]
        if not (
            np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7)
            and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            and np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
        ):
            raise ValueError(f"extrinsics.{side}: expected a rigid SE(3) transform")
    intrinsics = data.get("intrinsics", {})
    sizes = data.get("image_sizes", {})
    if not isinstance(intrinsics, dict) or not isinstance(sizes, dict):
        raise ValueError(
            "intrinsics and image_sizes must be JSON objects keyed by camera name"
        )
    if "front_1" not in intrinsics:
        raise ValueError(
            "calibration needs front_1 intrinsics and its image_sizes [width,height]"
        )
    for name, value in intrinsics.items():
        k = np.asarray(value, dtype=float)
        if k.shape == (3, 3):
            k = np.column_stack([k, np.zeros(3)])
        k = array(k, (3, 4), f"intrinsics.{name}")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1, 0]):
            raise ValueError(f"intrinsics.{name}: invalid pinhole matrix")
        if not np.allclose(k[:, 3], 0):
            raise ValueError(f"intrinsics.{name}: expected zero last column")
        size = array(sizes.get(name), (2,), f"image_sizes.{name}")
        if np.any(size <= 0) or np.any(size != size.astype(int)):
            raise ValueError(
                f"image_sizes.{name}: expected positive integer width,height"
            )
        intrinsics[name] = k.tolist()
    return {
        "data": data,
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def camera_names(source: Path, options: Options):
    names = read_json(source / "videos/cameras.json")
    if (
        not isinstance(names, list)
        or not names
        or any(
            not isinstance(n, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", n)
            for n in names
        )
        or len(set(names)) != len(names)
    ):
        raise ValueError("cameras.json must contain unique camera names")
    missing = sorted(EXPECTED_CAMERAS - set(names))
    if missing and not options.allow_missing_cameras:
        raise ValueError(
            f"Missing required Piper cameras {missing}; diagnostic use: --allow-missing-cameras"
        )
    return names, missing


def prepare_episode(data, start, end, video_dir, names, options):
    """One row per recorded control tick; nothing is resampled or interpolated."""
    tau_all = timestamps(data["timestamp"][start:end], "observation")
    n_input = len(tau_all)
    camera_times = {
        name: timestamps(np.load(video_dir / f"{i}.npy", allow_pickle=False), name)
        for i, name in enumerate(names)
    }
    # get_obs may pick a frame received just before the video writer started.
    # Such ticks can only lead the episode; a missing frame later is corruption.
    frame_indices = {}
    first = 0
    for i, name in enumerate(names):
        key = f"camera{i}_timestamp"
        if key not in data:
            raise ValueError(
                f"{name}: recording lacks {key}; re-record with the tick collector"
            )
        chosen = array(data[key][start:end], (n_input,), key)
        indices = matching_frame_indices(camera_times[name], chosen)
        missing = indices < 0
        leading = int(np.argmax(~missing)) if np.any(~missing) else n_input
        if np.any(missing[leading:]):
            row = leading + int(np.argmax(missing[leading:]))
            raise ValueError(
                f"{name}: selected frame at row {row} is not in the recorded video"
            )
        first = max(first, leading)
        frame_indices[name] = indices
    keep = slice(first, n_input)
    tau = tau_all[keep]
    n = len(tau)
    if n < 2:
        raise ValueError("Less than two ticks whose frames are in every recorded video")
    rows = np.arange(start, end)[keep]

    def column(key, shape, label=None):
        return array(data[key][start:end], (n_input, *shape), label or key)[keep]

    decision = column("decision_timestamp", ())
    if np.any(decision < tau):
        raise ValueError("A command was decided before its observation")
    action = column("action", (14,))
    numeric = {
        "timestamp": tau,
        "source.action_index": rows.astype(np.int64),
        "source.decision_timestamp": decision,
        "source.deadline_timestamp": column("deadline_timestamp", ()),
        "source.command_id": np.asarray(data["command_id"][start:end], dtype=np.int64)[
            keep
        ],
    }
    grip_stats, send_times, arm_stats = {}, [], {}
    for i, side in enumerate(SIDES):
        prefix = f"robot{i}"
        robot_time = column(f"{prefix}_source_timestamp", ())
        robot_age = tau - robot_time
        if np.any(robot_age < 0) or np.max(robot_age) > options.max_robot_age:
            raise ValueError(
                f"{side} arm state age {np.max(robot_age):.6f}s exceeds --max-robot-age "
                "or is in the future"
            )
        numeric[f"source.{side}.robot_timestamp"] = robot_time
        send = np.asarray(
            data[f"{prefix}_send_timestamp"][start:end], dtype=np.float64
        )[keep]
        numeric[f"source.{side}.send_timestamp"] = send
        send_times.append(send)
        arm_stats[side] = {
            "state_age": _summary(robot_age),
            "observation_to_send": _summary(send - tau),
            "unsent_commands": int(np.count_nonzero(~np.isfinite(send))),
        }
        numeric[f"{side}.obs_ee_pose"] = pose_to_quaternion(
            column(f"{prefix}_eef_pos", (3,), prefix + " position"),
            column(f"{prefix}_eef_rot_axis_angle", (3,), prefix + " rotation"),
        )
        offset = 7 * i
        cmd = action[:, offset : offset + 6]
        numeric[f"{side}.cmd_ee_pose"] = pose_to_quaternion(cmd[:, :3], cmd[:, 3:])
        for kind, width in (
            ("obs", column(f"{prefix}_gripper_width", (1,), prefix + " gripper")),
            ("cmd", action[:, offset + 6 : offset + 7]),
        ):
            # Keep physical values as diagnostics; never silently clamp feedback.
            numeric[f"{side}.{kind}_gripper_width_m"] = width
            numeric[f"{side}.{kind}_gripper"] = width / options.gripper_max_m
            grip_stats[f"{side}.{kind}"] = {
                "min_m": float(width.min()),
                "max_m": float(width.max()),
                "out_of_range_samples": int(
                    np.count_nonzero((width < 0) | (width > options.gripper_max_m))
                ),
            }
        numeric[f"{side}.obs_joints"] = column(f"{prefix}_joint_pos", (6,))
    selected, camera_stats = {}, {}
    for i, (name, stamps) in enumerate(camera_times.items()):
        indices = frame_indices[name][keep]
        age = tau - stamps[indices]
        if np.any(age < 0):
            raise ValueError(f"{name}: a selected frame is later than its observation")
        if np.max(age) > options.max_frame_age:
            raise ValueError(
                f"{name}: selected frame age {np.max(age):.6f}s exceeds --max-frame-age"
            )
        selected[name] = indices
        numeric[f"source.{name}.frame_index"] = indices.astype(np.int64)
        numeric[f"source.{name}.timestamp"] = stamps[indices]
        numeric[f"source.{name}.age"] = age
        receive_path = video_dir / f"{i}.receive_timestamps.npy"
        if receive_path.exists():
            receive = timestamps(
                np.load(receive_path, allow_pickle=False), name + " receive"
            )
            if receive.shape != stamps.shape:
                raise ValueError(
                    f"{name}: receive and corrected timestamp counts differ"
                )
            numeric[f"source.{name}.receive_timestamp"] = receive[indices]
        camera_stats[name] = {
            "source_frames": len(stamps),
            "source_fps": float((len(stamps) - 1) / (stamps[-1] - stamps[0])),
            "selected_unique_frames": len(np.unique(indices)),
            "repeated_rows": len(indices) - len(np.unique(indices)),
            "frame_age": _summary(age),
        }
    intervals = np.diff(tau)
    gaps = np.flatnonzero(intervals > options.max_tick_gap)
    sent = np.concatenate(send_times)
    sent_offsets = sent - np.tile(tau, len(send_times))
    sent_offsets = sent_offsets[np.isfinite(sent_offsets)]
    if sent_offsets.size:
        observation_to_send, send_source = (
            float(np.median(sent_offsets)),
            "arm_send_timestamp",
        )
    else:
        observation_to_send, send_source = (
            float(np.median(decision - tau)),
            "decision_timestamp",
        )
    report = {
        "input_rows": n_input,
        "output_rows": n,
        "trimmed_leading_rows": first,
        "output_start_time": float(tau[0]),
        "output_end_time": float(tau[-1]),
        "tick_interval": _summary(intervals),
        "tick_gaps": {
            "threshold_s": options.max_tick_gap,
            "count": int(gaps.size),
            "first": [
                {"after_row": int(row), "interval_s": float(intervals[row])}
                for row in gaps[:MAX_REPORTED_GAPS]
            ],
        },
        "observation_to_decision": _summary(decision - tau),
        "observation_to_send_s": observation_to_send,
        "observation_to_send_source": send_source,
        "arms": arm_stats,
        "cameras": camera_stats,
        "grippers": grip_stats,
    }
    return numeric, camera_times, selected, report


def encode_selected_video(
    path, indices, expected_frames, options, *, image_processor=None
):
    """Decode sequentially on CPU, retaining only selected compressed frames."""
    wanted = set(indices.tolist())
    encoded = {}
    source_size = None
    count = 0
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"{path}: expected exactly one video stream")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for count, frame in enumerate(container.decode(stream), start=1):
            size = (frame.width, frame.height)
            if source_size is None:
                source_size = size
            if source_size != size:
                raise ValueError(f"{path}: resolution changed within the episode")
            if count > expected_frames:
                raise ValueError(
                    f"{path}: more decoded frames than timestamps ({expected_frames})"
                )
            index = count - 1
            if index in wanted:
                rgb = frame.to_ndarray(format="rgb24")
                if image_processor is not None:
                    rgb = image_processor.apply(rgb)
                elif size != (options.width, options.height):
                    rgb = cv2.resize(
                        rgb,
                        (options.width, options.height),
                        interpolation=cv2.INTER_AREA,
                    )
                encoded[index] = simplejpeg.encode_jpeg(
                    np.ascontiguousarray(rgb),
                    quality=ZarrWriter.JPEG_QUALITY,
                    colorspace="RGB",
                )
    if count != expected_frames:
        raise ValueError(
            f"{path}: decoded {count} frames but has {expected_frames} timestamps"
        )
    frames = np.empty(len(indices), dtype=object)
    frames[:] = [encoded[int(index)] for index in indices]
    return frames, source_size


def scaled_intrinsics(calibration, source_sizes, options):
    if calibration is None:
        return None
    data = calibration["data"]
    result = {}
    for name, value in data["intrinsics"].items():
        if name not in source_sizes:
            continue
        width, height = data["image_sizes"][name]
        if (width, height) != source_sizes[name]:
            raise ValueError(
                f"{name}: calibration image size differs from source video"
            )
        k = np.asarray(value).copy()
        k[0] *= options.width / width
        k[1] *= options.height / height
        result[name] = k.tolist()
    return result or None


def convert_dataset(
    source: Path,
    output_dir: Path,
    *,
    options=None,
    calibration_path=None,
    episodes=None,
):
    options = options or Options()
    options.validate()
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    destination = (output_dir / source.name).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Output must be separate from the raw input dataset")
    if calibration_path is None and (source / "calibration.json").exists():
        calibration_path = source / "calibration.json"
    calibration = load_calibration(
        Path(calibration_path) if calibration_path else None, options
    )
    names, missing = camera_names(source, options)
    root = zarr.open_group(str(source / "replay_buffer.zarr"), mode="r")
    data = root["data"]
    ends = np.asarray(root["meta/episode_ends"][:])
    if (
        ends.ndim != 1
        or not len(ends)
        or ends.dtype.kind not in "iu"
        or np.any(np.diff(np.r_[0, ends]) <= 0)
    ):
        raise ValueError("Invalid or empty meta/episode_ends")
    if any(v.shape[0] != int(ends[-1]) for _, v in data.arrays()):
        raise ValueError("Replay arrays do not match committed episode_ends")
    ids = list(range(len(ends))) if episodes is None else list(episodes)
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(i < 0 or i >= len(ends) for i in ids)
    ):
        raise ValueError("episodes must select unique existing episode indices")
    config_path = source / "videos/camera_config.json"
    camera_config = read_json(config_path) if config_path.exists() else None
    for episode in ids:
        path = destination / f"episode_{episode:06d}.zarr"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    results = []
    for episode in ids:
        start, end = (0 if episode == 0 else int(ends[episode - 1])), int(ends[episode])
        video_dir = source / "videos" / str(episode)
        numeric, camera_times, selected, report = prepare_episode(
            data, start, end, video_dir, names, options
        )
        images, sizes, processing = {}, {}, {}
        specs = {spec["name"]: spec for spec in (camera_config or [])}
        for i, name in enumerate(names):
            video_path = video_dir / f"{i}.mp4"
            with av.open(str(video_path)) as container:
                if len(container.streams.video) != 1:
                    raise ValueError(f"{video_path}: expected one video stream")
                stream = container.streams.video[0]
                source_size = (stream.width, stream.height)
            contract = camera_image_geometry(
                name,
                specs.get(name),
                calibration["data"] if calibration else None,
                source_size,
                (options.width, options.height),
                alpha=options.undistort_alpha,
            )
            processing[name] = contract
            frames, size = encode_selected_video(
                video_path,
                selected[name],
                len(camera_times[name]),
                options,
                image_processor=RgbImageProcessor(contract),
            )
            images[f"images.{name}"] = (frames, [options.height, options.width, 3])
            sizes[name] = size
        intrinsics = {
            name: np.column_stack(
                [np.asarray(contract["K_output"]), np.zeros(3)]
            ).tolist()
            for name, contract in processing.items()
            if contract["K_output"] is not None
        } or None
        report.update(
            {
                "source_dataset": str(source),
                "source_episode": episode,
                "source_row_range": [start, end],
                "embodiment": OUTPUT_EMBODIMENT,
                "schema": OUTPUT_SCHEMA,
                "missing_required_cameras": missing,
                "calibration_status": "provided" if calibration else "missing",
                "image_processing": processing,
            }
        )
        metadata = {
            "schema": OUTPUT_SCHEMA,
            "pose_coordinates": {side: f"piper_{side}_base" for side in SIDES},
            "pose_representation": "xyz_m+quat_wxyz",
            "tool_frame": "piper_sdk_end_pose; no additional TCP offset applied",
            "extrinsics_convention": "T_base_camera",
            "reference_camera": "front_1",
            "gripper_encoding": {
                "unit": "normalized_width",
                "closed_m": 0.0,
                "open_m": options.gripper_max_m,
                "clipped": False,
            },
            "calibration_status": report["calibration_status"],
            "calibration": calibration,
            "image_processing": processing,
            "image_space": "processed",
            "distortion_coefficients": {
                name: contract["D_output"]
                for name, contract in processing.items()
                if contract["D_output"] is not None
            },
            "requires_training_transform": "piper_bimanual per_episode_base_to_camera; do not use hardcoded robot extrinsics",
            "conversion": report,
            "source_capture_config": {
                "replay_attributes": dict(root.attrs),
                "cameras": camera_config,
            },
            "time_alignment": {
                "schema": "control_tick_v1",
                "row": "one collector control tick; nothing resampled",
                "timestamp": "observation time taken by env.get_obs (wall clock after reading device buffers)",
                "observation": "arm state taken by env.get_obs at the observation time",
                "image": "latest frame stamped at or before the observation time, as selected by env.get_obs",
                "command": "the command decided after this row's observation and sent at once",
                "action_chunk": "following rows' commands by index; rows are nominally 1/fps apart, see conversion.tick_interval/tick_gaps",
                "observation_to_send_s": report["observation_to_send_s"],
                "observation_to_send_source": report["observation_to_send_source"],
                "frame_max_age_s": options.max_frame_age,
                "robot_max_age_s": options.max_robot_age,
                "camera_latency": "use stored .npy as-is; no second latency subtraction; deployment must use the same per-camera latency",
            },
        }
        destination.mkdir(parents=True, exist_ok=True)
        final = destination / f"episode_{episode:06d}.zarr"
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".episode_{episode:06d}_", suffix=".partial", dir=destination
            )
        )
        try:
            # Low-level writer permits explicitly uncalibrated diagnostic episodes.
            # Normal exports validate calibration above; never invent intrinsic matrices.
            writer = ZarrWriter(
                temporary,
                embodiment=OUTPUT_EMBODIMENT,
                fps=options.fps,
                task_name=options.task_name,
                task_description=options.task_description,
                intrinsics=intrinsics,
                extrinsics=calibration["data"]["extrinsics"] if calibration else None,
                annotations=(
                    [(options.task_description, 0, len(numeric["timestamp"]))]
                    if options.task_description
                    else []
                ),
            )
            writer.write(
                numeric_data=numeric,
                pre_encoded_image_data=images,
                metadata_override=metadata,
            )
            # Portable high-rate feedback is independent of the row/image schema.
            # Select history offsets at training time, without re-encoding images.
            write_feedback_sidecar(video_dir, temporary)
            (temporary / "conversion_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            if final.exists():
                raise FileExistsError(f"Refusing to overwrite {final}")
            temporary.rename(final)
        except BaseException:
            shutil.rmtree(temporary)
            raise
        results.append(final)
        print(
            f"{final}: {report['output_rows']} rows, {len(names)} cameras, calibration={report['calibration_status']}",
            flush=True,
        )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="One capture batch containing replay_buffer.zarr and videos/",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        help="This batch's calibration JSON; default input/calibration.json",
    )
    parser.add_argument(
        "--episodes", type=int, nargs="+", help="Default: all committed episodes"
    )
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--undistort-alpha",
        type=float,
        default=0.0,
        help="OpenCV rectification alpha [0,1]; keeps full output canvas without ROI crop",
    )
    parser.add_argument(
        "--max-frame-age",
        type=float,
        default=0.1,
        help="Reject a selected frame older than this at its observation, seconds",
    )
    parser.add_argument(
        "--max-robot-age",
        type=float,
        default=0.05,
        help="Reject arm state older than this at its observation, seconds",
    )
    parser.add_argument(
        "--max-tick-gap",
        type=float,
        default=0.05,
        help="Report tick intervals above this (not rejected), seconds",
    )
    parser.add_argument("--gripper-max-m", type=float, default=0.08)
    parser.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="Diagnostic output with missing calibration explicitly marked",
    )
    parser.add_argument(
        "--allow-missing-cameras",
        action="store_true",
        help="Diagnostic conversion of older one/two-camera recordings",
    )
    parser.add_argument("--task-name", default="")
    parser.add_argument("--task-description", default="")
    parser.add_argument(
        "--pi05-manifest",
        type=Path,
        help="Also export a Pi0.5 reference manifest over these converted episodes; JPEGs are not duplicated",
    )
    parser.add_argument("--pi05-source-horizon", type=int, default=30)
    parser.add_argument("--pi05-action-stride", type=int, default=1)
    parser.add_argument("--pi05-validation-ratio", type=float, default=0.1)
    parser.add_argument("--pi05-split-seed", type=int, default=42)
    parser.add_argument(
        "--pi05-prompt", help="Override recorded task description for Pi0.5"
    )
    parser.add_argument(
        "--pi05-state-history-offsets-ms",
        type=float,
        nargs="*",
        default=[],
        help="Optional negative feedback history offsets in milliseconds, oldest first; requires --pi05-manifest",
    )
    args = parser.parse_args(argv)
    try:
        history_offsets = validate_history_offsets(
            [value / 1000 for value in args.pi05_state_history_offsets_ms]
        )
    except ValueError as exc:
        parser.error(str(exc))
    if history_offsets and args.pi05_manifest is None:
        parser.error("--pi05-state-history-offsets-ms requires --pi05-manifest")
    options = Options(
        **{name: getattr(args, name) for name in Options.__dataclass_fields__}
    )
    cv2.setNumThreads(1)
    try:
        converted = convert_dataset(
            args.input,
            args.output_dir,
            options=options,
            calibration_path=args.calibration,
            episodes=args.episodes,
        )
        if args.pi05_manifest is not None:
            from egomimic.robot.piper_pico.pi05_data import export_manifest

            manifest = export_manifest(
                converted,
                args.pi05_manifest,
                source_fps=args.fps,
                source_horizon=args.pi05_source_horizon,
                action_stride=args.pi05_action_stride,
                validation_ratio=args.pi05_validation_ratio,
                seed=args.pi05_split_seed,
                prompt=args.pi05_prompt,
                state_history_offsets_s=history_offsets,
            )
            print(
                f"Pi0.5 manifest: {manifest}; 100-point windows are interpolated at training read time"
            )
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as exc:
        parser.exit(1, f"Conversion failed: {exc}\n")


if __name__ == "__main__":
    main()
