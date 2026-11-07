"""Offline native-UVC conversion acceptance using synthetic and optional old captures.

No camera, robot, SDK, or GPU is opened. Synthetic calibration is an explicitly
labelled software fixture; legacy sources use only their own recorded metadata.
Raw files are hashed before and after conversion, never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np
import simplejpeg
import zarr

from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.robot.piper_pico.image_geometry import RgbImageProcessor
from egomimic.scripts.piper_pico_to_zarr import Options, convert_dataset

CAMERAS = ("front_1", "left_wrist", "right_wrist")
NATIVE_SIZES = ((1920, 1080), (640, 480), (640, 480))


def hash_tree(root):
    """Hash all source files, including videos and replay data, in bounded memory."""
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[str(path.relative_to(root))] = digest.hexdigest()
    if not result:
        raise ValueError(f"no source files to verify: {root}")
    return result


def pixel_comparison(expected, actual, *, max_mae):
    """Reject shape/type/nonfinite problems before calculating image error."""
    expected, actual = np.asarray(expected), np.asarray(actual)
    if expected.shape != actual.shape or expected.ndim != 3 or expected.shape[2] != 3:
        raise ValueError("image comparison requires matching HxWx3 RGB arrays")
    if expected.dtype != np.uint8 or actual.dtype != np.uint8:
        raise ValueError("image comparison requires uint8 pixels")
    if not np.isfinite(max_mae) or max_mae < 0:
        raise ValueError("max_mae must be finite and nonnegative")
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    metric = {
        "mean_absolute_error": float(difference.mean()),
        "p99_absolute_error": float(np.percentile(difference, 99)),
        "max_absolute_error": float(difference.max()),
        "max_allowed_mae": max_mae,
    }
    if metric["mean_absolute_error"] > max_mae:
        raise ValueError(
            f"image MAE {metric['mean_absolute_error']:.6f} exceeds {max_mae:g}"
        )
    return metric


def _pattern(width, height, camera, frame):
    y, x = np.mgrid[:height, :width]
    image = np.stack(
        (
            (x * 160 / width + frame * 7 + camera * 20) % 256,
            (y * 180 / height + camera * 25) % 256,
            (x * 60 / width + y * 70 / height + frame * 9) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    for column in np.linspace(0, width - 1, 9).astype(int):
        cv2.line(image, (int(column), 0), (int(column), height - 1), (240, 220, 50), 2)
    for row in np.linspace(0, height - 1, 7).astype(int):
        cv2.line(image, (0, int(row)), (width - 1, int(row)), (30, 210, 240), 2)
    cv2.circle(
        image, (width // 3 + frame, height // 2), max(3, height // 8), (230, 40, 80), -1
    )
    return np.ascontiguousarray(image)


def write_synthetic_capture(root, *, native_sizes=NATIVE_SIZES, frames=8):
    """Write an explicit synthetic raw three-camera capture, not a hardware claim."""
    root = Path(root)
    video_dir = root / "videos/0"
    video_dir.mkdir(parents=True, exist_ok=False)
    ticks = 1000.0 + np.arange(frames) / 30
    selected = np.maximum(np.arange(frames) - 1, 0)
    replay = zarr.open_group(str(root / "replay_buffer.zarr"), mode="w")
    replay.attrs["synthetic"] = True
    data = replay.create_group("data")
    replay.create_group("meta").create_array(
        "episode_ends", data=np.array([frames], dtype=np.int64)
    )
    numeric = {
        "timestamp": ticks,
        "decision_timestamp": ticks + 0.002,
        "deadline_timestamp": ticks + 1 / 30,
        "command_id": np.arange(frames, dtype=np.int64),
    }
    action = np.zeros((frames, 14), dtype=np.float64)
    for side in range(2):
        n = np.arange(frames, dtype=np.float64)
        xyz = np.column_stack(
            [0.2 + n * 0.001, n * 0.002 - side * 0.1, np.full(frames, 0.5)]
        )
        rotvec = np.column_stack([n * 0.01, n * 0.005, n * 0.02])
        width = np.linspace(0.02, 0.06, frames)[:, None]
        numeric.update(
            {
                f"robot{side}_eef_pos": xyz,
                f"robot{side}_eef_rot_axis_angle": rotvec,
                f"robot{side}_gripper_width": width,
                f"robot{side}_joint_pos": np.repeat(n[:, None], 6, axis=1),
                f"robot{side}_source_timestamp": ticks - 0.003,
                f"robot{side}_send_timestamp": ticks + 0.006,
            }
        )
        action[:, side * 7 : side * 7 + 3] = xyz + 0.01
        action[:, side * 7 + 3 : side * 7 + 6] = rotvec + 0.02
        action[:, side * 7 + 6 : side * 7 + 7] = width + 0.005
    numeric["action"] = action
    specs, matrices, distortions, sizes = [], {}, {}, {}
    for index, (name, (width, height)) in enumerate(zip(CAMERAS, native_sizes)):
        stamps = ticks - (0.01 + index * 0.003)
        numeric[f"camera{index}_timestamp"] = stamps[selected]
        np.save(video_dir / f"{index}.npy", stamps)
        np.save(video_dir / f"{index}.receive_timestamps.npy", stamps + 0.02)
        with av.open(str(video_dir / f"{index}.mp4"), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=30)
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            stream.codec_context.thread_count = 1
            for frame in range(frames):
                image = av.VideoFrame.from_ndarray(
                    _pattern(width, height, index, frame), format="rgb24"
                )
                for packet in stream.encode(image):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        K = [
            [width * (0.8 + index * 0.02), 0, width * 0.49],
            [0, height * 1.1, height * 0.51],
            [0, 0, 1],
        ]
        D = [-0.22 - index * 0.02, 0.06, 0.001, -0.002, -0.008]
        calibration = {
            "schema": "uvc_intrinsics_calibration_v1",
            "camera_model": "opencv_pinhole_brown5",
            "image_size": [width, height],
            "K": K,
            "D": D,
            "preprocessing": {"crop": None, "resize": None, "mirror": False},
            "synthetic": True,
            "production_approved": False,
        }
        specs.append(
            {
                "name": name,
                "type": "uvc",
                "device": f"/dev/synthetic-{index}",
                "width": width,
                "height": height,
                "fps": 30,
                "fourcc": "MJPG",
                "latency": 0.02,
                "latency_status": "unmeasured",
                "calibration": calibration,
            }
        )
        matrices[name], distortions[name], sizes[name] = K, D, [width, height]
    for key, value in numeric.items():
        data.create_array(key, data=value)
    (root / "videos/cameras.json").write_text(json.dumps(list(CAMERAS)))
    (root / "videos/camera_config.json").write_text(json.dumps(specs, indent=2))
    left, right = np.eye(4), np.eye(4)
    left[:3, 3], right[:3, 3] = [0.1, -0.2, 0.3], [-0.1, 0.2, 0.3]
    calibration = {
        "calibration_id": "SYNTHETIC_SOFTWARE_ACCEPTANCE_ONLY",
        "synthetic": True,
        "reference_camera": "front_1",
        "extrinsics_convention": "T_base_camera",
        "extrinsics": {"left": left.tolist(), "right": right.tolist()},
        "intrinsics": matrices,
        "distortion_coefficients": distortions,
        "camera_models": {name: "opencv_pinhole_brown5" for name in CAMERAS},
        "image_sizes": sizes,
    }
    (root / "calibration.json").write_text(json.dumps(calibration, indent=2))
    return root


def _pose(xyz, rotvec):
    """Independent rotvec -> scalar-first quaternion formula, without converter helpers."""
    angle = np.linalg.norm(rotvec, axis=1)
    factor = np.full(len(angle), 0.5)
    np.divide(np.sin(angle / 2), angle, out=factor, where=angle != 0)
    return np.column_stack([xyz, np.cos(angle / 2), rotvec * factor[:, None]])


def verify_annotations(values, feature, storage_type, *, frames, task_description=""):
    """Check the writer's byte-encoded annotation contract separately from numbers."""
    values = np.asarray(values)
    if storage_type != "variable_length_bytes" or values.ndim != 1:
        raise ValueError("annotations must use a 1-D variable_length_bytes array")
    if (
        feature.get("dtype") != "json"
        or feature.get("format") != "annotation_v1"
        or feature.get("shape") != [len(values)]
        or feature.get("names") != ["json"]
    ):
        raise ValueError("annotations feature metadata is not annotation_v1 JSON")
    decoded = []
    for value in values:
        if not isinstance(value, bytes):
            raise ValueError("annotations must contain JSON bytes, not numeric values")
        decoded.append(json.loads(value))
    expected = (
        [{"text": task_description, "start_idx": 0, "end_idx": frames}]
        if task_description
        else []
    )
    if decoded != expected:
        raise ValueError(
            "annotations differ from the requested task description and frame range"
        )
    return len(decoded)


def verify_numeric(source, episode, store):
    replay = zarr.open_group(str(source / "replay_buffer.zarr"), mode="r")
    data, ends = replay["data"], replay["meta/episode_ends"][:]
    start, end = (0 if episode == 0 else int(ends[episode - 1])), int(ends[episode])
    names = json.loads((source / "videos/cameras.json").read_text())
    indices, source_stamps, first = {}, {}, 0
    for camera, name in enumerate(names):
        stamps = np.load(source / f"videos/{episode}/{camera}.npy", allow_pickle=False)
        chosen = data[f"camera{camera}_timestamp"][start:end]
        right = np.clip(np.searchsorted(stamps, chosen), 0, len(stamps) - 1)
        left = np.maximum(0, right - 1)
        selected = np.where(
            abs(stamps[left] - chosen) <= abs(stamps[right] - chosen), left, right
        )
        matched = np.abs(stamps[selected] - chosen) <= 1e-6
        if not matched.any():
            raise ValueError(f"{name}: no independently matched frames")
        leading = int(np.flatnonzero(matched)[0])
        if not matched[leading:].all():
            raise ValueError(f"{name}: nonleading source frame mismatch")
        first = max(first, leading)
        indices[name], source_stamps[name] = selected, stamps
    rows = np.arange(start + first, end)
    count = int(store.attrs["total_frames"])
    if count != len(rows):
        raise ValueError(
            "converted row count differs from independently matched source ticks"
        )

    def raw(key):
        return np.asarray(data[key][start:end])[first:]

    expected = {
        "timestamp": raw("timestamp"),
        "source.action_index": rows,
        "source.decision_timestamp": raw("decision_timestamp"),
        "source.deadline_timestamp": raw("deadline_timestamp"),
        "source.command_id": raw("command_id"),
    }
    action = raw("action")
    for index, side in enumerate(("left", "right")):
        expected.update(
            {
                f"{side}.obs_ee_pose": _pose(
                    raw(f"robot{index}_eef_pos"),
                    raw(f"robot{index}_eef_rot_axis_angle"),
                ),
                f"{side}.cmd_ee_pose": _pose(
                    action[:, index * 7 : index * 7 + 3],
                    action[:, index * 7 + 3 : index * 7 + 6],
                ),
                f"{side}.obs_joints": raw(f"robot{index}_joint_pos"),
                f"source.{side}.robot_timestamp": raw(f"robot{index}_source_timestamp"),
                f"source.{side}.send_timestamp": raw(f"robot{index}_send_timestamp"),
            }
        )
        for kind, widths in (
            ("obs", raw(f"robot{index}_gripper_width")),
            ("cmd", action[:, index * 7 + 6 : index * 7 + 7]),
        ):
            expected[f"{side}.{kind}_gripper_width_m"] = widths
            expected[f"{side}.{kind}_gripper"] = widths / 0.08
    for camera, name in enumerate(names):
        selected = indices[name][first:]
        stamps = source_stamps[name][selected]
        expected.update(
            {
                f"source.{name}.frame_index": selected,
                f"source.{name}.timestamp": stamps,
                f"source.{name}.age": raw("timestamp") - stamps,
            }
        )
        receive_path = source / f"videos/{episode}/{camera}.receive_timestamps.npy"
        if receive_path.exists():
            expected[f"source.{name}.receive_timestamp"] = np.load(
                receive_path, allow_pickle=False
            )[selected]
    maximum = 0.0
    for key, value in expected.items():
        actual = np.asarray(store[key][:count])
        if actual.shape != value.shape or not np.allclose(
            actual, value, atol=1e-10, rtol=0, equal_nan=True
        ):
            raise ValueError(f"numeric output changed source semantics: {key}")
        finite = np.isfinite(actual) & np.isfinite(value)
        if finite.any():
            maximum = max(
                maximum, float(np.max(np.abs(actual[finite] - value[finite])))
            )
    annotations = store["annotations"]
    annotations_checked = verify_annotations(
        annotations[:],
        store.attrs.get("features", {}).get("annotations", {}),
        annotations.metadata.to_dict().get("data_type"),
        frames=count,
    )
    # Only the explicitly validated annotation array is exempt. Extra arrays,
    # including extra strings or arrays merely named like annotations, still fail.
    numeric_keys = {
        key
        for key, _ in store.arrays()
        if not key.startswith("images.") and key != "annotations"
    }
    if numeric_keys != set(expected):
        raise ValueError(
            f"unchecked or missing numeric fields: {numeric_keys ^ set(expected)}"
        )
    return {
        "rows": count,
        "fields_checked": len(expected),
        "annotations_checked": annotations_checked,
        "max_absolute_error": maximum,
        "source_row_range": [int(rows[0]), int(rows[-1]) + 1],
        "trimmed_leading_rows": first,
    }


def independent_brown_warp(raw, intrinsic, distortion, rectified_K):
    """Evaluate Brown5 inverse sampling without OpenCV's map-generation APIs.

    Float32 maps match the declared processing contract. cv2.undistort instead
    builds fixed-point maps, whose interpolation-bin rounding differs on edges.
    """
    yy, xx = np.mgrid[: raw.shape[0], : raw.shape[1]]
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    rays = pixels @ np.linalg.inv(rectified_K).T
    x, y = rays[..., 0], rays[..., 1]
    radius2 = x * x + y * y
    k1, k2, p1, p2, k3 = distortion
    radial = 1 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
    distorted = (
        np.stack(
            [
                x * radial + 2 * p1 * x * y + p2 * (radius2 + 2 * x * x),
                y * radial + p1 * (radius2 + 2 * y * y) + 2 * p2 * x * y,
                np.ones_like(x),
            ],
            axis=-1,
        )
        @ intrinsic.T
    )
    return cv2.remap(
        raw,
        distorted[..., 0].astype(np.float32),
        distorted[..., 1].astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def independent_geometry(raw, contract):
    """Compute the oracle from analytic Brown5 sampling and OpenCV interpolation."""
    size = (raw.shape[1], raw.shape[0])
    if contract["source_size"] != list(size):
        raise ValueError("image contract source dimensions do not match decoded video")
    digest = hashlib.sha256(
        json.dumps(
            {k: v for k, v in contract.items() if k != "sha256"},
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    if contract.get("sha256") != digest:
        raise ValueError("image geometry hash mismatch")
    if (
        contract.get("input_space") != "raw"
        or contract.get("output_space") != "processed"
        or contract.get("roi_cropped") is not False
    ):
        raise ValueError(
            "image geometry must preserve explicit raw/processed spaces and uncropped ROI"
        )
    K = (
        None
        if contract["K_raw"] is None
        else np.asarray(contract["K_raw"], dtype=np.float64)
    )
    D = (
        None
        if contract["D_raw"] is None
        else np.asarray(contract["D_raw"], dtype=np.float64)
    )
    rectified_K = K
    result = raw
    if D is not None:
        if (
            contract["operation"] != "undistort_resize"
            or contract["D_output"] != [0.0] * 5
        ):
            raise ValueError("rectification must have measured raw D and zero output D")
        rectified_K, _ = cv2.getOptimalNewCameraMatrix(
            K, D, size, contract["alpha"], size
        )
        result = independent_brown_warp(raw, K, D, rectified_K)
    elif contract["operation"] != "resize" or contract["D_output"] is not None:
        raise ValueError("legacy input cannot claim measured zero distortion")
    if K is not None:
        output_K = rectified_K.copy()
        output_K[0] *= contract["output_size"][0] / size[0]
        output_K[1] *= contract["output_size"][1] / size[1]
        if not np.allclose(
            contract["K_rectified"], rectified_K, atol=1e-9, rtol=0
        ) or not np.allclose(contract["K_output"], output_K, atol=1e-9, rtol=0):
            raise ValueError(
                "stored K does not match independent rectification/resize calculation"
            )
    elif contract["K_rectified"] is not None or contract["K_output"] is not None:
        raise ValueError("legacy image contract invented camera intrinsics")
    if list(size) != contract["output_size"]:
        result = cv2.resize(
            result, tuple(contract["output_size"]), interpolation=cv2.INTER_AREA
        )
    return np.ascontiguousarray(result)


def _comparison_png(path, raw, oracle, stored):
    difference = np.minimum(
        255, np.abs(stored.astype(float) - oracle.astype(float)) * 8
    ).astype(np.uint8)
    panels = []
    for title, image in (
        ("RAW preview (native file retained)", raw),
        ("Analytic Brown5 oracle", oracle),
        ("Stored JPEG", stored),
        ("Absolute difference x8", difference),
    ):
        panel = np.zeros((270, 320, 3), dtype=np.uint8)
        ratio = min(320 / image.shape[1], 240 / image.shape[0])
        preview = cv2.resize(
            image,
            (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
            interpolation=cv2.INTER_AREA,
        )
        panel[30 : 30 + preview.shape[0], : preview.shape[1]] = preview
        cv2.putText(
            panel, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )
        panels.append(panel)
    if not cv2.imwrite(str(path), cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR)):
        raise OSError(f"cannot save comparison image {path}")


def verify_episode(source, converted, comparisons, *, synthetic, max_jpeg_mae=8.0):
    store = zarr.open_group(str(converted), mode="r")
    episode = int(store.attrs["conversion"]["source_episode"])
    numeric = verify_numeric(source, episode, store)
    names = json.loads((source / "videos/cameras.json").read_text())
    specs_path = source / "videos/camera_config.json"
    specs = (
        {x["name"]: x for x in json.loads(specs_path.read_text())}
        if specs_path.exists()
        else {}
    )
    cameras = []
    count = numeric["rows"]
    rows = sorted(set([0, count // 2, count - 1]))
    for camera, name in enumerate(names):
        contract = store.attrs["image_processing"][name]
        if store.attrs.get("image_space") != "processed":
            raise ValueError("converted dataset must declare processed image space")
        expected_intrinsic = specs.get(name, {}).get("calibration")
        if expected_intrinsic is not None:
            if not np.allclose(
                contract["K_raw"], expected_intrinsic["K"], atol=1e-10, rtol=0
            ) or not np.array_equal(contract["D_raw"], expected_intrinsic["D"]):
                raise ValueError(f"{name}: output geometry lost raw K/D provenance")
        if synthetic and (expected_intrinsic is None or not np.any(contract["D_raw"])):
            raise ValueError(
                "synthetic acceptance requires explicit nonzero distortion"
            )
        frame_indices = np.asarray(store[f"source.{name}.frame_index"][:count])
        selected_rows = {}
        for row in rows:
            selected_rows.setdefault(int(frame_indices[row]), []).append(row)
        samples, frame_count = [], 0
        processor = RgbImageProcessor(contract)
        with av.open(str(source / f"videos/{episode}/{camera}.mp4")) as container:
            for index, frame in enumerate(container.decode(video=0)):
                frame_count += 1
                if [frame.width, frame.height] != contract["source_size"]:
                    raise ValueError("raw video frame shape changed")
                if index not in selected_rows:
                    continue
                raw = frame.to_ndarray(format="rgb24")
                oracle = independent_geometry(raw, contract)
                shared = processor.apply(raw, image_space="raw")
                transform_error = pixel_comparison(oracle, shared, max_mae=0.05)
                if not np.array_equal(
                    processor.apply(shared, image_space="processed"), shared
                ):
                    raise ValueError("processed image was transformed twice")
                oracle_jpeg = simplejpeg.decode_jpeg(
                    simplejpeg.encode_jpeg(
                        oracle, quality=ZarrWriter.JPEG_QUALITY, colorspace="RGB"
                    )
                )
                for row in selected_rows[index]:
                    stored = simplejpeg.decode_jpeg(
                        store[f"images.{name}"][row : row + 1][0]
                    )
                    oracle_error = pixel_comparison(oracle_jpeg, stored, max_mae=0.5)
                    jpeg_loss = pixel_comparison(oracle, stored, max_mae=max_jpeg_mae)
                    png = (
                        comparisons / f"episode_{episode:06d}_{name}_row_{row:06d}.png"
                    )
                    _comparison_png(png, raw, oracle, stored)
                    samples.append(
                        {
                            "row": row,
                            "source_frame_index": index,
                            "transform_vs_opencv": transform_error,
                            "stored_vs_independent_jpeg": oracle_error,
                            "jpeg_loss": jpeg_loss,
                            "comparison": str(png),
                        }
                    )
        expected_frames = len(
            np.load(source / f"videos/{episode}/{camera}.npy", allow_pickle=False)
        )
        if frame_count != expected_frames or len(samples) != len(rows):
            raise ValueError(
                f"{name}: decoded source count or checked row count differs"
            )
        cameras.append(
            {
                "name": name,
                "source_frames": frame_count,
                "image_geometry": contract,
                "samples": samples,
            }
        )
    return {
        "episode": episode,
        "converted": str(converted),
        "numeric": numeric,
        "cameras": cameras,
    }


def _verify_case(source, folder, *, synthetic, max_frame_age, output_size):
    folder.mkdir()
    comparisons = folder / "comparisons"
    comparisons.mkdir()
    before = hash_tree(source)
    result = {
        "status": "FAIL",
        "source": str(source),
        "synthetic": synthetic,
        "source_calibration_present": (source / "calibration.json").exists(),
        "borrowed_calibration_applied": False,
        "errors": [],
        "episodes": [],
    }
    try:
        paths = convert_dataset(
            source,
            folder / "converted",
            options=Options(
                width=output_size[0],
                height=output_size[1],
                max_frame_age=max_frame_age,
                undistort_alpha=0.0,
                allow_uncalibrated=not synthetic,
            ),
        )
        for converted in paths:
            result["episodes"].append(
                verify_episode(source, converted, comparisons, synthetic=synthetic)
            )
        result["status"] = "PASS"
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        after = hash_tree(source)
        result["source_files_sha256"] = before
        result["source_files_unchanged"] = before == after
        if before != after:
            result["status"] = "FAIL"
            result["errors"].append("source files changed during acceptance")
    return result


def run_acceptance(
    output,
    *,
    real_inputs=(),
    max_frame_age=0.2,
    native_sizes=NATIVE_SIZES,
    output_size=(640, 480),
    frames=8,
):
    output = Path(output).expanduser().resolve()
    sources = [Path(path).expanduser().resolve() for path in real_inputs]
    if output.exists():
        raise FileExistsError(f"output must be a new directory: {output}")
    if not np.isfinite(max_frame_age) or max_frame_age <= 0:
        raise ValueError("max_frame_age must be finite and positive")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 4:
        raise ValueError("synthetic frames must be an integer >= 4")
    if len(native_sizes) != 3 or any(
        len(size) != 2
        or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 2 or v % 2
            for v in size
        )
        for size in native_sizes
    ):
        raise ValueError(
            "native_sizes must have three positive even [width,height] pairs"
        )
    if len(output_size) != 2 or any(
        isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in output_size
    ):
        raise ValueError("output_size must be positive [width,height]")
    if len(set(sources)) != len(sources):
        raise ValueError("real inputs must be distinct captures")
    for source in sources:
        if (
            not (source / "replay_buffer.zarr").is_dir()
            or not (source / "videos/cameras.json").is_file()
        ):
            raise ValueError(f"real input is not a raw capture: {source}")
        if source == output or source in output.parents:
            raise ValueError("output must be outside every real raw capture")
    output.mkdir(parents=True)
    report = {
        "schema": "uvc_pipeline_acceptance_v1",
        "status": "FAIL",
        "synthetic": None,
        "legacy": [],
        "errors": [],
        "limitation": "Offline software and compatibility acceptance only; synthetic calibration is not physical calibration, and no hardware synchronization or policy performance is established.",
    }
    try:
        synthetic_source = write_synthetic_capture(
            output / "synthetic_raw", native_sizes=native_sizes, frames=frames
        )
        report["synthetic"] = _verify_case(
            synthetic_source,
            output / "synthetic",
            synthetic=True,
            max_frame_age=max_frame_age,
            output_size=output_size,
        )
        for index, source in enumerate(sources):
            report["legacy"].append(
                _verify_case(
                    source,
                    output / f"legacy_{index:03d}",
                    synthetic=False,
                    max_frame_age=max_frame_age,
                    output_size=output_size,
                )
            )
        cases = [report["synthetic"], *report["legacy"]]
        report["status"] = (
            "PASS" if all(case["status"] == "PASS" for case in cases) else "FAIL"
        )
        report["errors"] = [error for case in cases for error in case["errors"]]
    except (Exception, KeyboardInterrupt) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--real-input",
        action="append",
        default=[],
        type=Path,
        help="optional old raw capture; repeat for separate captures",
    )
    parser.add_argument("--max-frame-age", type=float, default=0.2)
    parser.add_argument("--synthetic-frames", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        report = run_acceptance(
            args.output,
            real_inputs=args.real_input,
            max_frame_age=args.max_frame_age,
            frames=args.synthetic_frames,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"{report['status']}: {args.output.resolve() / 'report.json'}")
    print(report["limitation"])
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
