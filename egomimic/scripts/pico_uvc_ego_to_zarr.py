"""Preprocess external-UVC joint recordings and export rectified world-pose Zarr.

Raw files are never changed. Uses a supplied frozen head/camera/time fit, not
the old PICO stereo fit. --preprocess-only writes decisions without images.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import av
import cv2
import numpy as np
import simplejpeg

from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.robot.piper_pico.ego_joint_input import load_joint_input
from egomimic.robot.piper_pico.ego_preprocessing import PreprocessOptions, split_rows
from egomimic.robot.piper_pico.image_geometry import (
    RgbImageProcessor,
    make_image_geometry,
)
from egomimic.scripts.pico_ego_to_zarr import (
    Options,
    hand_geometry,
    prepare_recording,
    provenance,
)


def prepare_joint(recording, *, options=None, preprocessing=None):
    """Reuse legacy pose mathematics, with explicit frame-UTC and fit adaptation."""
    options = options or Options(video_layout="left", max_hold_s=0)
    if options.video_layout != "left" or options.max_hold_s != 0:
        raise ValueError("Joint UVC conversion requires whole images and no hand holds")
    preprocessing = preprocessing or PreprocessOptions()
    options.validate()
    params = recording.fit["data"]["parameters"]
    K = recording.raw_K
    # This internal adapter supplies the same T_CH to the shared geometry code.
    # Only the original external fit is persisted, never this stereo-shaped shim.
    internal_fit = {
        "data": {
            "head_offset_ms": params["delta_head_ms"],
            "hand_offset_ms": params["delta_hand_ms"],
            "post_axis": np.eye(3).tolist(),
            "rig_rotation": recording.T_C_H[:3, :3].tolist(),
            "rig_translation_m": recording.T_C_H[:3, 3].tolist(),
            "extrinsics": "direct",
            "eye_swap": False,
            "K_fx_fy_cx_cy": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
        }
    }
    meta = {
        "timeStampNs": int(recording.video_utc_ns[0]),
        "cameraExtrinsics": "|".join([json.dumps(np.eye(4).ravel().tolist())] * 2),
    }
    numeric, _, sampling_report, T_C_H, _ = prepare_recording(
        meta,
        recording.raw,
        recording.video_utc_ns - recording.video_utc_ns[0],
        internal_fit,
        options,
        recording.source_size,
    )
    chosen = numeric["source.video_frame_index"]
    # Image matching above uses measured receive UTC. Keep true MP4 PTS as a
    # separate field; receive UTC and PTS are NOT interchangeable timestamps.
    numeric["source.video_pts_ns"] = recording.video_pts_ns[chosen]
    numeric["source.video_pts_absolute_ns"] = recording.video_pts_absolute_ns[chosen]
    numeric["source.video_frame_id"] = recording.video_frame_ids[chosen]
    numeric["source.image_grid_delta_ns"] = (
        numeric["obs_rgb_timestamps_ns"] - numeric["timestamp_ns"]
    )
    raw_validity = dict(recording.raw)
    for side in ("left", "right"):
        _, raw_validity[f"{side}_geometry_valid"] = hand_geometry(
            recording.raw[f"{side}_pose"][..., :3], side, options.axis_convention
        )
    segments, boundaries, report = split_rows(
        numeric,
        raw_validity,
        recording.video_utc_ns,
        head_offset_ns=round(params["delta_head_ms"] * 1e6),
        hand_offset_ns=round(params["delta_hand_ms"] * 1e6),
        options=preprocessing,
        max_image_dt_ns=round(options.max_image_dt_s * 1e9),
    )
    report["resampling"] = sampling_report
    report["resampling"]["time_basis"] = "frame receive UTC, not MP4 PTS clock"
    report["resampling"]["source_receive_fps"] = sampling_report.pop("source_video_fps")
    report["source_video_fps"] = (
        (len(recording.video_pts_ns) - 1) * 1e9 / int(recording.video_pts_ns[-1])
    )
    report["source_dataset"] = str(recording.source_dir)
    report["fit"] = {k: v for k, v in recording.fit.items() if k != "data"}
    report["input_validation"] = recording.metadata
    report["pose_options"] = asdict(options)
    report["status"] = "ready" if segments else "no_retained_segments"
    return numeric, segments, boundaries, report, T_C_H


def _encode(recording, wanted, geometry):
    processor = RgbImageProcessor(geometry)
    cv2.setNumThreads(1)
    encoded, count = {}, 0
    with av.open(str(recording.video_path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for count, frame in enumerate(container.decode(stream), 1):
            if (frame.width, frame.height) != recording.source_size:
                raise ValueError("Video size changed between passes")
            if count - 1 in wanted:
                rgb = processor.apply(frame.to_ndarray(format="rgb24"))
                encoded[count - 1] = simplejpeg.encode_jpeg(
                    rgb, quality=85, colorspace="RGB"
                )
    if count != len(recording.video_pts_ns) or set(encoded) != wanted:
        raise ValueError("Video frame selection changed between passes")
    return encoded


def convert_dataset(
    source,
    output_dir,
    *,
    fit_path,
    options=None,
    preprocessing=None,
    preprocess_only=False,
    alpha=0.0,
):
    options = options or Options(video_layout="left", max_hold_s=0)
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    destination = output_dir / source.name
    if destination == source or source in destination.parents:
        raise ValueError("Output must be separate from raw input")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    recording = load_joint_input(source, fit_path)
    numeric, segments, boundaries, report, T_C_H = prepare_joint(
        recording,
        options=options,
        preprocessing=preprocessing,
    )
    geometry = make_image_geometry(
        recording.source_size,
        (options.width, options.height),
        K=recording.raw_K,
        D=recording.raw_D,
        alpha=alpha,
    )
    report["image_geometry"] = geometry
    report["preprocess_only"] = bool(preprocess_only)
    report["episodes_written"] = 0 if preprocess_only else len(segments)
    wanted = {
        int(i) for a, b in segments for i in numeric["source.video_frame_index"][a:b]
    }
    encoded = (
        {} if preprocess_only or not wanted else _encode(recording, wanted, geometry)
    )
    metadata = {
        "schema": "pico_ego_world_v1",
        "pose_representation": "xyz_m+quat_wxyz",
        "pose_coordinates": {
            "left": "tracking_world",
            "right": "tracking_world",
            "obs_head_pose": "T_world_left_optical_camera",
        },
        "reference_camera": "front_1",
        "reference_camera_description": "head-mounted external UVC optical frame; legacy pose key name retained",
        "image_space": "processed",
        "image_geometry": {"front_1": geometry},
        "distortion_coefficients": {"front_1": geometry["D_output"]},
        "pico_ego": {
            "source_format": "uvc_joint_v1",
            "fit": recording.fit,
            "camera_from_head": T_C_H.tolist(),
            "source_tracking": provenance(recording.tracking_path),
            "source_video": str(recording.video_path),
            "joint_input": recording.metadata,
            "options": asdict(options),
            "tool_frame": {
                "origin": "mean XR wrist/index/middle/ring MCP [1,7,12,17]; no fingertip translation",
                "orientation": "bone-derived proper rotation",
                "axis_convention": options.axis_convention,
                "piper_x_axes": "X lateral/opening; Y signed palm normal; Z wrist toward palm",
            },
            "gripper_encoding": {
                "raw": "thumb tip 5 to index tip 10 distance in metres",
                "open_m": options.gripper_max_m,
                "normalized": "clip(distance,0,open_m)/open_m; fixed across episodes",
            },
            "time_alignment": {
                "selection": "uniform 30Hz/configured grid -> nearest frame receive UTC; ties earlier",
                "nominal": "first saved frame receive UTC + integer uniform grid offsets",
                "pose_query": "grid UTC + frozen delta_head_ms/delta_hand_ms exactly once",
                "video_mapping": "actual MP4 PTS paired one-to-one to saved frames.csv mux_pts_us within tolerance",
                "source_video_pts_ns": "true MP4 PTS relative to first decoded PTS",
                "head_source_timestamp": "application query UTC; SDK zero source timestamp is not used",
                "physical_exposure": "unknown; fit offsets are empirical alignment, not certified exposure latency",
                "pose_sampling": options.pose_sampling,
                "max_hold_s": 0,
                "gap_policy": "strict high-rate invalidity split before output downsampling",
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{source.name}_", suffix=".partial", dir=output_dir)
    )
    paths = []
    try:
        if not preprocess_only:
            for index, ((a, b), boundary) in enumerate(
                zip(segments, boundaries, strict=True)
            ):
                values = {k: v[a:b].copy() for k, v in numeric.items()}
                for key in (
                    "source.head_pose",
                    "obs_head_pose",
                    "left.obs_ee_pose",
                    "right.obs_ee_pose",
                ):
                    for i in range(1, b - a):
                        if values[key][i - 1, 3:] @ values[key][i, 3:] < 0:
                            values[key][i, 3:] *= -1
                if any(
                    v.dtype.kind == "f" and not np.isfinite(v).all()
                    for v in values.values()
                ):
                    raise ValueError("Non-finite accepted pose output")
                images = np.empty(b - a, dtype=object)
                images[:] = [
                    encoded[int(i)] for i in values["source.video_frame_index"]
                ]
                name = f"episode_{index:06d}.zarr"
                writer = ZarrWriter(
                    temporary / name,
                    embodiment="human_bimanual",
                    fps=options.fps,
                    task_name=options.task_name,
                    task_description=options.task_description,
                    intrinsics={"front_1": geometry["K_output"]},
                    extrinsics=None,
                    annotations=[(options.task_description, 0, b - a)]
                    if options.task_description
                    else [],
                )
                writer.write(
                    numeric_data=values,
                    pre_encoded_image_data={
                        "images.front_1": (images, [options.height, options.width, 3])
                    },
                    metadata_override={
                        **metadata,
                        "preprocessing": boundary,
                        "conversion": {
                            "segment_index": index,
                            "segment_candidate_range": [a, b],
                        },
                    },
                )
                paths.append(destination / name)
        (temporary / "preprocessing_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return paths, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="One session or parent containing sessions",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fit", type=Path, required=True, help="Frozen external UVC fit.json"
    )
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--trim-start-s", type=float, default=1.0)
    parser.add_argument("--trim-end-s", type=float, default=1.0)
    parser.add_argument("--min-segment-s", type=float, default=2.0)
    parser.add_argument(
        "--trim-overrides",
        type=Path,
        help="JSON mapping session name to trim_start_s/trim_end_s/min_segment_s",
    )
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--pose-sampling", choices=["interpolate", "nearest"], default="interpolate"
    )
    parser.add_argument("--max-pose-dt-s", type=float, default=0.025)
    parser.add_argument("--max-interp-gap-s", type=float, default=0.05)
    parser.add_argument("--max-image-dt-s", type=float, default=1 / 30)
    parser.add_argument("--max-video-gap-s", type=float, default=0.05)
    parser.add_argument("--gripper-max-m", type=float, default=0.08)
    parser.add_argument(
        "--axis-convention", choices=["piper_x", "piper"], default="piper_x"
    )
    parser.add_argument("--undistort-alpha", type=float, default=0.0)
    parser.add_argument("--task-name", default="")
    parser.add_argument("--task-description", default="")
    args = parser.parse_args()
    sources = (
        [args.input]
        if (args.input / "tracking.jsonl").is_file()
        else sorted(p.parent for p in args.input.glob("*/tracking.jsonl"))
    )
    if not sources:
        parser.error("No joint recording sessions found")
    overrides = (
        json.loads(args.trim_overrides.read_text()) if args.trim_overrides else {}
    )
    if not isinstance(overrides, dict) or set(overrides) - {p.name for p in sources}:
        parser.error("Trim overrides must map selected session names")
    options = Options(
        **{
            k: getattr(args, k)
            for k in Options.__dataclass_fields__
            if hasattr(args, k)
        },
        video_layout="left",
        max_hold_s=0,
    )
    preprocessing = PreprocessOptions(
        args.trim_start_s,
        args.trim_end_s,
        args.min_segment_s,
        args.max_interp_gap_s,
        args.max_video_gap_s,
    )
    reports = []
    for source in sources:
        changes = overrides.get(source.name, {})
        if not isinstance(changes, dict) or set(changes) - {
            "trim_start_s",
            "trim_end_s",
            "min_segment_s",
        }:
            parser.error(f"Invalid trim override for {source.name}")
        paths, report = convert_dataset(
            source,
            args.output_dir,
            fit_path=args.fit,
            options=options,
            preprocessing=replace(preprocessing, **changes),
            preprocess_only=args.preprocess_only,
            alpha=args.undistort_alpha,
        )
        summary = {
            "source": source.name,
            "episodes": len(paths),
            "rows": report["output_rows"],
            "kept_segments": report["kept_segments"],
            "short_segments_removed": report["discarded_short_segments"],
            "kept_span_s": report["kept_span_s"],
            "status": report["status"],
        }
        reports.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    summary_path = args.output_dir / "batch_summary.json"
    if not summary_path.exists():
        summary_path.write_text(
            json.dumps(
                {"sessions": reports, "preprocess_only": args.preprocess_only}, indent=2
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
