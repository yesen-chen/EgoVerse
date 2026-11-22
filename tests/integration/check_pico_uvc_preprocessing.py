#!/usr/bin/env python3
"""Manual, read-only real-recording acceptance (no hardware, model or network).

Example, from the repository's activated environment:
  python tests/integration/check_pico_uvc_preprocessing.py \
    --raw-root PATH_TO_JOINT_SESSIONS --converted-root PATH_TO_CONVERTED \
    --fit PATH_TO_FROZEN_FIT_JSON --report PATH_TO_ACCEPTANCE_JSON

The report is the only output written. Unlike unit tests this explicitly reads
user-selected recordings; it is intentionally not collected as a pytest test.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import av
import cv2
import numpy as np
import simplejpeg
from scipy.spatial.transform import Rotation, Slerp

# Support direct invocation without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from egomimic.robot.piper_pico.ego_joint_input import load_joint_input

REASONS = {
    "ui_trim",
    "inactivity",
    "tracking_gap",
    "video_gap",
    "source_end",
    "source_start",
}
ARTIFICIAL = {"inactivity", "tracking_gap", "video_gap"}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def equal(actual, expected, message):
    require(np.array_equal(actual, expected), message)


def close(actual, expected, message, atol=1e-7):
    require(np.allclose(actual, expected, rtol=0, atol=atol), message)


def quantiles(values):
    values = np.asarray(values, float)
    return (
        {}
        if not values.size
        else {
            "count": int(values.size),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    )


def matrices(poses):
    values = np.repeat(np.eye(4)[None], len(poses), axis=0)
    values[:, :3, :3] = Rotation.from_quat(poses[:, [4, 5, 6, 3]]).as_matrix()
    values[:, :3, 3] = poses[:, :3]
    return values


def rotation_error(left, right):
    relative = np.einsum("nji,njk->nik", left, right)
    return Rotation.from_matrix(relative).magnitude()


def bone_frames(points, side, convention):
    """Independent geometric reference, using only raw joint positions."""
    origin = np.mean(points[:, [1, 7, 12, 17]], axis=1)
    forward = origin - points[:, 1]
    normal = np.cross(points[:, 7] - points[:, 1], points[:, 22] - points[:, 1])
    if side == "left":
        normal = -normal
    valid = (
        np.isfinite(points).all((1, 2))
        & (np.linalg.norm(forward, axis=1) > 1e-8)
        & (np.linalg.norm(normal, axis=1) > 1e-8)
    )
    z = forward / np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), 1e-12)
    x = np.cross(normal, z)
    valid &= np.linalg.norm(x, axis=1) > 1e-8
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = np.cross(z, x)
    rotation = np.stack((x, y, z), axis=-1)
    if convention == "piper":
        rotation = rotation @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    else:
        require(convention == "piper_x", "unsupported virtual tool axes")
    return origin, rotation, valid


def support_reference(times, queries, mode):
    indices = np.searchsorted(times, queries)
    require(
        np.all((indices >= 0) & (indices < len(times))), "query outside source coverage"
    )
    high = indices
    low = np.where(times[high] == queries, high, high - 1)
    require(np.all(low >= 0), "query before source coverage")
    if mode == "nearest":
        chosen = np.where(queries - times[low] <= times[high] - queries, low, high)
        low, high = chosen, chosen
    else:
        require(mode == "interpolate", "unknown pose sampling mode")
    alpha = np.array(
        [
            0.0
            if a == b
            else (int(q) - int(times[a])) / (int(times[b]) - int(times[a]))
            for q, a, b in zip(queries, low, high)
        ],
        float,
    )
    return low, high, alpha


def interpolate_rotation(rotations, low, high, alpha):
    result = np.empty((len(low), 3, 3))
    for i, (a, b, weight) in enumerate(zip(low, high, alpha)):
        result[i] = (
            rotations[a]
            if a == b
            else Slerp([0.0, 1.0], Rotation.from_matrix(rotations[[a, b]]))(
                [weight]
            ).as_matrix()[0]
        )
    return result


def check_support(times, queries, support, valid, options, mode, label):
    low, high, alpha = support_reference(times, queries, mode)
    equal(
        support,
        np.column_stack((times[low], times[high])),
        f"{label}: saved support differs from independent source lookup",
    )
    require(np.all(valid[low] & valid[high]), f"{label}: invalid support endpoint")
    require(
        np.all(times[high] - times[low] <= round(options["max_interp_gap_s"] * 1e9)),
        f"{label}: interpolation crosses oversized source gap",
    )
    if mode == "nearest":
        require(
            np.max(np.abs(times[low] - queries))
            <= round(options["max_pose_dt_s"] * 1e9),
            f"{label}: nearest pose too far from query",
        )
    # Check every source sample, including ones falling between 30 Hz rows.
    lo = int(np.searchsorted(times, min(int(queries[0]), int(support[0, 0]))))
    hi = int(
        np.searchsorted(times, max(int(queries[-1]), int(support[-1, 1])), side="right")
    )
    require(
        np.all(valid[lo:hi]), f"{label}: source invalidity hidden between output rows"
    )
    return low, high, alpha


def load_arrays(group, n):
    arrays = {}
    features = dict(group.attrs).get("features", {})
    for name, value in group.arrays():
        feature = features.get(name, {})
        if feature.get("format") == "annotation_v1" or name == "annotations":
            for encoded in value[:]:
                annotation = json.loads(bytes(encoded).decode("utf-8"))
                require(
                    0 <= annotation["start_idx"] <= annotation["end_idx"] <= n,
                    f"{name}: annotation interval outside episode",
                )
            continue
        if name.startswith("images."):
            continue
        require(feature.get("dtype") != "json", f"{name}: unknown non-frame JSON array")
        array = np.asarray(value[:n])
        require(len(array) == n, f"{name}: wrong stored length")
        if array.dtype.kind == "f":
            require(np.isfinite(array).all(), f"{name}: nonfinite output")
        arrays[name] = array
    return arrays


def check_episode(path, group, recording):
    metadata = dict(group.attrs)
    n = int(metadata["total_frames"])
    require(n > 1, "episode needs at least two rows")
    require(metadata["schema"] == "pico_ego_world_v1", "wrong output schema")
    require(metadata["embodiment"] == "human_bimanual", "wrong embodiment")
    pico, boundary = metadata["pico_ego"], metadata["preprocessing"]
    require(pico["source_format"] == "uvc_joint_v1", "wrong source format")
    require(pico["fit"]["sha256"] == recording.fit["sha256"], "different frozen fit")
    require(
        pico["source_tracking"]["sha256"]
        == recording.metadata["source_files"]["tracking.jsonl"]["sha256"],
        "raw tracking source hash mismatch",
    )
    require(boundary["schema"] == "ego_preprocessing_v1", "wrong preprocessing schema")
    require(
        boundary["kept"] is True and boundary["frames"] == n,
        "boundary count/decision mismatch",
    )
    require(
        boundary["start_reason"] in REASONS
        and boundary["end_reason"] in REASONS - {"source_start"},
        "unknown boundary reason",
    )
    data = load_arrays(group, n)
    times, raw = recording.raw["timestamp_ns"], recording.raw
    grid = data["timestamp_ns"]
    fps = float(metadata["fps"])
    epoch = int(recording.video_utc_ns[0])
    index = np.rint((grid - np.int64(epoch)).astype(float) * fps / 1e9).astype(np.int64)
    expected_grid = np.int64(epoch) + np.rint(index / fps * 1e9).astype(np.int64)
    equal(grid, expected_grid, "timestamps are not on the exact integer uniform grid")
    require(
        np.all(np.diff(index) == 1), "missing/nonconsecutive grid rows inside episode"
    )
    equal(data["source.grid_timestamp_ns"], grid, "grid provenance changed")
    equal(
        data["source.head_query_timestamp_ns"],
        grid + recording.head_offset_ns,
        "head offset not applied exactly once",
    )
    equal(
        data["source.hand_query_timestamp_ns"],
        grid + recording.hand_offset_ns,
        "hand offset not applied exactly once",
    )
    images = data["obs_rgb_timestamps_ns"]
    frame_index = data["source.video_frame_index"]
    right = np.searchsorted(recording.video_utc_ns, grid).clip(
        0, len(recording.video_utc_ns) - 1
    )
    left = (right - 1).clip(0)
    nearest = np.where(
        grid - recording.video_utc_ns[left] <= recording.video_utc_ns[right] - grid,
        left,
        right,
    )
    equal(frame_index, nearest, "images were not matched by nearest receive UTC")
    equal(
        images,
        recording.video_utc_ns[frame_index],
        "image UTC does not identify source frame",
    )
    equal(
        data["source.video_frame_id"],
        recording.video_frame_ids[frame_index],
        "frame_id mismatch",
    )
    equal(
        data["source.video_pts_ns"],
        recording.video_pts_ns[frame_index],
        "relative PTS mismatch",
    )
    equal(
        data["source.video_pts_absolute_ns"],
        recording.video_pts_absolute_ns[frame_index],
        "original absolute PTS mismatch",
    )
    equal(
        data["source.image_grid_delta_ns"],
        images - grid,
        "image-grid difference provenance mismatch",
    )
    prep = boundary["options"]
    lower = epoch + round(prep["trim_start_s"] * 1e9)
    upper = int(recording.video_utc_ns[-1]) - round(prep["trim_end_s"] * 1e9)
    require(
        np.all(
            (grid >= lower) & (grid <= upper) & (images >= lower) & (images <= upper)
        ),
        "retained timestamp/image inside removed UI interval",
    )
    require(
        int(grid[-1] - grid[0]) >= round(prep["min_segment_s"] * 1e9),
        "retained fragment below minimum duration",
    )
    equal(
        [int(grid[0]), int(grid[-1])],
        [boundary["first_row_timestamp_ns"], boundary["last_row_timestamp_ns"]],
        "boundary timestamp mismatch",
    )
    options = pico["options"]
    require(
        options["max_hold_s"] == 0,
        "joint conversion must not synthesize held hand poses",
    )
    require(
        np.max(np.abs(images - grid)) <= round(options["max_image_dt_s"] * 1e9),
        "selected image exceeds configured time mismatch bound",
    )
    mode = options["pose_sampling"]
    max_gap = round(prep["max_tracking_gap_s"] * 1e9)
    max_rotation_error = 0.0
    head_valid = ((raw["head_status"] & 3) == 3) & np.isfinite(raw["head_pose"]).all(1)
    low, high, alpha = check_support(
        times,
        data["source.head_query_timestamp_ns"],
        data["source.head_support_timestamp_ns"],
        head_valid,
        options,
        mode,
        "head",
    )
    expected_position = (
        raw["head_pose"][low, :3] * (1 - alpha[:, None])
        + raw["head_pose"][high, :3] * alpha[:, None]
    )
    source_rotation = np.repeat(np.eye(3)[None], len(times), axis=0)
    quaternion_good = np.linalg.norm(raw["head_pose"][:, 3:], axis=1) > 1e-8
    source_rotation[quaternion_good] = Rotation.from_quat(
        raw["head_pose"][quaternion_good, 3:]
    ).as_matrix()
    expected_rotation = interpolate_rotation(source_rotation, low, high, alpha)
    expected_head = np.repeat(np.eye(4)[None], n, axis=0)
    expected_head[:, :3, :3], expected_head[:, :3, 3] = (
        expected_rotation,
        expected_position,
    )
    saved_head = matrices(data["source.head_pose"])
    close(
        saved_head[:, :3, 3],
        expected_position,
        "head interpolation translation",
        atol=1e-8,
    )
    max_rotation_error = max(
        max_rotation_error,
        float(np.max(rotation_error(expected_rotation, saved_head[:, :3, :3]))),
    )
    expected_camera = expected_head @ np.linalg.inv(recording.T_C_H)
    saved_camera = matrices(data["obs_head_pose"])
    close(
        saved_camera[:, :3, 3],
        expected_camera[:, :3, 3],
        "camera world translation chain",
        atol=2e-6,
    )
    max_rotation_error = max(
        max_rotation_error,
        float(
            np.max(rotation_error(expected_camera[:, :3, :3], saved_camera[:, :3, :3]))
        ),
    )
    close(
        pico["camera_from_head"],
        recording.T_C_H,
        "persisted T_C_H differs from frozen fit",
    )
    invalid_hand = np.zeros(len(times), bool)
    for side in ("left", "right"):
        equal(
            data[f"{side}.hand_valid"],
            np.ones(n, bool),
            f"{side}: invalid exported hand",
        )
        equal(
            data[f"{side}.hand_filled"],
            np.zeros(n, bool),
            f"{side}: exported held hand",
        )
        points = raw[side + "_pose"][..., :3]
        origin, rotations, geometry_good = bone_frames(
            points, side, options["axis_convention"]
        )
        valid = (
            raw[side + "_active"]
            & ((raw[side + "_status"] & 2) != 0).all(1)
            & geometry_good
        )
        invalid_hand |= ~valid
        low, high, alpha = check_support(
            times,
            data["source.hand_query_timestamp_ns"],
            data[f"{side}.source_support_timestamp_ns"],
            valid,
            options,
            mode,
            side,
        )
        weight = alpha[:, None, None]
        expected_points = points[low] * (1 - weight) + points[high] * weight
        close(
            data[f"{side}.obs_xr_keypoints"].reshape(n, 26, 3),
            expected_points,
            f"{side}: keypoint interpolation",
            atol=1e-8,
        )
        expected_position = (
            origin[low] * (1 - alpha[:, None]) + origin[high] * alpha[:, None]
        )
        expected_rotation = interpolate_rotation(rotations, low, high, alpha)
        actual = matrices(data[f"{side}.obs_ee_pose"])
        close(
            actual[:, :3, 3],
            expected_position,
            f"{side}: virtual TCP origin",
            atol=1e-8,
        )
        max_rotation_error = max(
            max_rotation_error,
            float(np.max(rotation_error(expected_rotation, actual[:, :3, :3]))),
        )
        raw_distance = np.linalg.norm(points[:, 5] - points[:, 10], axis=1)
        interpolated_distance = (
            raw_distance[low] * (1 - alpha) + raw_distance[high] * alpha
        )
        maximum = float(options["gripper_max_m"])
        close(
            data[f"{side}.obs_pinch_distance_m"].ravel(),
            interpolated_distance,
            f"{side}: pinch distance interpolation",
            atol=1e-8,
        )
        close(
            data[f"{side}.obs_gripper"].ravel(),
            np.clip(interpolated_distance, 0, maximum) / maximum,
            f"{side}: fixed width normalization",
            atol=1e-8,
        )
    require(
        max_rotation_error < 3e-5,
        "saved SLERP/camera/tool rotation differs from independent reference",
    )
    # The selected image may lie slightly before/after the nominal grid.
    # Its fitted head/hand queries must remain in the SAME uninterrupted raw
    # valid interval as the whole episode, not just select a nearby video frame.
    for label, offset, valid in (
        ("head", recording.head_offset_ns, head_valid),
        ("hand", recording.hand_offset_ns, ~invalid_hand),
    ):
        combined = np.r_[data[f"source.{label}_query_timestamp_ns"], images + offset]
        first, last = int(combined.min()), int(combined.max())
        require(
            first >= int(times[0]) and last <= int(times[-1]),
            f"{label}: selected image query would extrapolate tracking",
        )
        a = int(np.searchsorted(times, first))
        if times[a] != first:
            a -= 1
        b = int(np.searchsorted(times, last)) + 1
        require(
            np.all(valid[a:b]), f"{label}: selected image crosses raw invalid interval"
        )
        require(
            np.all(np.diff(times[a:b]) <= max_gap),
            f"{label}: selected image crosses raw tracking gap",
        )
    for label, queries in (
        ("head", data["source.head_query_timestamp_ns"]),
        ("hand", data["source.hand_query_timestamp_ns"]),
    ):
        a = int(np.searchsorted(times, queries[0]))
        if a > 0 and (a == len(times) or times[a] != queries[0]):
            a -= 1
        b = min(len(times), int(np.searchsorted(times, queries[-1])) + 1)
        require(
            np.all(np.diff(times[a:b]) <= max_gap),
            f"{label}: segment crosses a raw tracking gap",
        )
    video_first = min(int(grid[0]), int(images.min()))
    video_last = max(int(grid[-1]), int(images.max()))
    a = int(np.searchsorted(recording.video_utc_ns, video_first))
    if a > 0 and (
        a == len(recording.video_utc_ns) or recording.video_utc_ns[a] != video_first
    ):
        a -= 1
    b = min(
        len(recording.video_utc_ns),
        int(np.searchsorted(recording.video_utc_ns, video_last)) + 1,
    )
    require(
        np.all(
            np.diff(recording.video_utc_ns[a:b]) <= round(prep["max_video_gap_s"] * 1e9)
        ),
        "segment crosses a raw video gap",
    )
    geometry = metadata["image_geometry"]["front_1"]
    require(
        metadata["image_space"] == "processed"
        and geometry["operation"] == "undistort_resize",
        "UVC images are not rectified",
    )
    close(geometry["K_raw"], recording.raw_K, "raw K mismatch")
    close(geometry["D_raw"], recording.raw_D, "raw D mismatch")
    close(geometry["D_output"], np.zeros(5), "processed distortion is not zero")
    require(geometry["roi_cropped"] is False, "unexpected ROI crop")
    effective, _ = cv2.getOptimalNewCameraMatrix(
        recording.raw_K,
        recording.raw_D,
        recording.source_size,
        float(geometry["alpha"]),
        recording.source_size,
    )
    size = geometry["output_size"]
    effective[0] *= size[0] / recording.source_size[0]
    effective[1] *= size[1] / recording.source_size[1]
    close(geometry["K_output"], effective, "effective K mismatch")
    intrinsics = metadata["intrinsics"]
    close(intrinsics["front_1"], effective, "training K differs from rectified K")
    return {
        "episode": str(path),
        "source_session": recording.source_dir.name,
        "source_tracking_sha256": pico["source_tracking"]["sha256"],
        "rows": n,
        "span_s": float(grid[-1] - grid[0]) / 1e9,
        "start_reason": boundary["start_reason"],
        "end_reason": boundary["end_reason"],
        "image_grid_abs_delta_ms": quantiles(np.abs(images - grid) / 1e6),
        "max_rotation_error_rad": max_rotation_error,
        "source_hand_invalid_rows": int(invalid_hand.sum()),
    }, data


def check_windows(path, metadata, data):
    from egomimic.rldb.action_timing import ActionTiming
    from egomimic.rldb.embodiment.pico_ego import PicoEgo, PicoEgoDataset
    from egomimic.rldb.zarr.local_recordings import RecordingEpisode, recording_group

    spec = ActionTiming(float(metadata["fps"]), 50, 1, 100)
    reason, n = metadata["preprocessing"]["end_reason"], int(metadata["total_frames"])
    results = []
    for trim, inactivity in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        required = (
            trim
            if reason == "ui_trim"
            else inactivity
            if reason in ARTIFICIAL
            else False
        )
        expected_count = n - 49 if required else n
        kwargs = dict(
            gripper_mode="width",
            expected_axis_convention=metadata["pico_ego"]["options"]["axis_convention"],
            require_full_window_after_trim=trim,
            require_full_window_after_inactivity=inactivity,
        )
        keymap = PicoEgo.get_keymap(timing=spec, gripper_mode="width", norm_mode=True)
        if expected_count <= 0:
            try:
                PicoEgoDataset(path, keymap, None, **kwargs)
            except ValueError as error:
                require(
                    "full future window needs" in str(error),
                    "short-fragment rejection is unclear",
                )
            else:
                raise AssertionError("short full-window fragment was silently admitted")
            results.append(
                {
                    "trim": trim,
                    "inactivity": inactivity,
                    "skipped": "fewer than 50 rows; explicit rejection verified",
                }
            )
            continue
        raw_loader = PicoEgoDataset(path, keymap, None, **kwargs)
        require(len(raw_loader) == expected_count, "wrong eligible start count")
        equal(
            list(raw_loader.eligible_start_indices),
            np.arange(expected_count),
            "training starts are not contiguous valid rows",
        )
        last = raw_loader[expected_count - 1]
        expected_indices = np.minimum(np.arange(50) + expected_count - 1, n - 1)
        for side in ("left", "right"):
            close(
                last[f"{side}.action_ee_pose"],
                data[f"{side}.obs_ee_pose"][expected_indices],
                "last action window does not use actual stored future poses",
                atol=2e-6,
            )
            close(
                last[f"{side}.action_gripper"],
                data[f"{side}.obs_gripper"][expected_indices],
                "last action width support mismatch",
                atol=2e-6,
            )
        transforms = PicoEgo.get_transform_list(timing=spec, gripper_mode="width")
        norm = RecordingEpisode(
            PicoEgoDataset(path, keymap, transforms, **kwargs), str(path)
        )
        train = RecordingEpisode(
            PicoEgoDataset(
                path,
                PicoEgo.get_keymap(timing=spec, gripper_mode="width"),
                transforms,
                **kwargs,
            ),
            str(path),
        )
        require(
            len(norm) == len(train) == expected_count,
            "normalization/train eligible start counts differ",
        )
        require(
            norm.recording_id == train.recording_id == recording_group(metadata),
            "source group changes between normalization/training",
        )
        for i in sorted({0, expected_count // 2, expected_count - 1}):
            a, b = train[i], norm[i]
            equal(
                a["sample_timestamp_ns"].numpy(),
                b["sample_timestamp_ns"].numpy(),
                "sample provenance differs in normalization",
            )
            close(
                a["actions_cartesian"].numpy(),
                b["actions_cartesian"].numpy(),
                "normalization and training action construction differ",
                atol=1e-6,
            )
            close(
                a["observations.state.ee_pose"].numpy(),
                b["observations.state.ee_pose"].numpy(),
                "normalization and training state construction differ",
                atol=1e-6,
            )
            require(
                tuple(a["actions_cartesian"].shape) == (100, 14),
                "wrong model action dimensions",
            )
        results.append(
            {
                "trim": trim,
                "inactivity": inactivity,
                "full_window": required,
                "starts": expected_count,
                "stored_rows": n,
                "last_start": expected_count - 1,
                "tail_support_preserved": True,
                "normalization_start_policy_identical": True,
            }
        )
    return results


def check_pixels(recording, episodes):
    """Compare representative saved JPEGs with an independent OpenCV remap."""
    import zarr

    wanted = defaultdict(list)
    for path in episodes:
        group = zarr.open_group(str(path), mode="r")
        n = int(group.attrs["total_frames"])
        for row in sorted({0, n - 1}):
            source_index = int(group["source.video_frame_index"][row])
            wanted[source_index].append((path, row))
    results = []
    with av.open(str(recording.video_path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if index not in wanted:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            for path, row in wanted[index]:
                group = zarr.open_group(str(path), mode="r")
                geometry = group.attrs["image_geometry"]["front_1"]
                new_k, _ = cv2.getOptimalNewCameraMatrix(
                    recording.raw_K,
                    recording.raw_D,
                    recording.source_size,
                    float(geometry["alpha"]),
                    recording.source_size,
                )
                maps = cv2.initUndistortRectifyMap(
                    recording.raw_K,
                    recording.raw_D,
                    None,
                    new_k,
                    recording.source_size,
                    cv2.CV_32FC1,
                )
                expected = cv2.remap(rgb, maps[0], maps[1], cv2.INTER_LINEAR)
                if tuple(geometry["output_size"]) != recording.source_size:
                    expected = cv2.resize(
                        expected,
                        tuple(geometry["output_size"]),
                        interpolation=cv2.INTER_AREA,
                    )
                stored = simplejpeg.decode_jpeg(
                    bytes(group["images.front_1"][row : row + 1][0]),
                    colorspace="RGB",
                )
                require(
                    stored.shape == expected.shape, "rectified JPEG dimensions differ"
                )
                # Compare after the same lossy codec round trip, isolating the
                # geometry rather than treating ordinary JPEG loss as an error.
                expected = simplejpeg.decode_jpeg(
                    simplejpeg.encode_jpeg(expected, quality=85, colorspace="RGB"),
                    colorspace="RGB",
                )
                error = np.abs(stored.astype(float) - expected.astype(float))
                require(
                    float(error.mean()) < 0.25 and np.quantile(error, 0.99) <= 2,
                    "saved image differs from independent rectification",
                )
                results.append(
                    {
                        "episode": str(path),
                        "row": row,
                        "source_frame_index": index,
                        "mean_absolute_rgb_error": float(error.mean()),
                        "p99_absolute_rgb_error": float(np.quantile(error, 0.99)),
                    }
                )
    require(
        len(results) == sum(len(v) for v in wanted.values()),
        "not every selected JPEG checked",
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--converted-root", required=True, type=Path)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    import zarr

    from egomimic.rldb.zarr.local_recordings import recording_group

    cv2.setNumThreads(1)
    paths = sorted(args.converted_root.rglob("*.zarr"))
    require(paths, "no converted episodes discovered")
    cache, episodes, errors, image_checks = {}, [], [], []
    image_deltas_ms = []
    groups = defaultdict(list)
    by_source = defaultdict(list)
    for path in paths:
        try:
            group = zarr.open_group(str(path), mode="r")
            metadata = dict(group.attrs)
            name = metadata["pico_ego"]["joint_input"]["joint_session"]["session_id"]
            if name not in cache:
                cache[name] = load_joint_input(args.raw_root / name, args.fit)
            result, arrays = check_episode(path, group, cache[name])
            result["window_policies"] = check_windows(path, metadata, arrays)
            image_deltas_ms.extend(
                (
                    np.abs(arrays["obs_rgb_timestamps_ns"] - arrays["timestamp_ns"])
                    / 1e6
                ).tolist()
            )
            identity = recording_group(metadata)
            require(
                identity == "pico:" + result["source_tracking_sha256"],
                "manifest group not based on original tracking hash",
            )
            groups[identity].append(path.relative_to(args.converted_root).as_posix())
            by_source[name].append(path)
            episodes.append(result)
            print(
                f"PASS {path.relative_to(args.converted_root)}: {result['rows']} rows, end={result['end_reason']}",
                flush=True,
            )
        except Exception as error:
            errors.append(
                {"episode": str(path), "error": f"{type(error).__name__}: {error}"}
            )
            print(f"FAIL {path}: {error}", flush=True)
    for name, source_episodes in by_source.items():
        try:
            image_checks.extend(check_pixels(cache[name], source_episodes))
        except Exception as error:
            errors.append(
                {
                    "source_session": name,
                    "error": f"image check: {type(error).__name__}: {error}",
                }
            )
    raw_sources = sorted(
        p.name
        for p in args.raw_root.iterdir()
        if p.is_dir() and (p / "joint_session.json").is_file()
    )
    discarded_reports = []
    for name in raw_sources:
        path = args.converted_root / name / "preprocessing_report.json"
        if not path.is_file():
            errors.append(
                {"source_session": name, "error": "missing preprocessing report"}
            )
            continue
        report = json.loads(path.read_text())
        expected = len(by_source.get(name, []))
        if report["episodes_written"] != expected:
            errors.append(
                {
                    "source_session": name,
                    "error": "preprocessing report episode count differs from accepted outputs",
                }
            )
        if not expected:
            discarded_reports.append(
                {
                    "source_session": name,
                    "status": report["status"],
                    "candidate_segments": report["candidate_segments"],
                    "discarded_short_segments": report["discarded_short_segments"],
                }
            )
    result = {
        "schema": "pico_uvc_preprocessing_acceptance_v1",
        "passed": not errors,
        "raw_root": str(args.raw_root.resolve()),
        "converted_root": str(args.converted_root.resolve()),
        "fit": str(args.fit.resolve()),
        "sources_total": len(raw_sources),
        "sources_retained": len(by_source),
        "accepted_episodes": len(episodes),
        "accepted_rows": sum(r["rows"] for r in episodes),
        "retained_span_s": sum(r["span_s"] for r in episodes),
        "image_grid_abs_delta_ms": quantiles(image_deltas_ms),
        "end_reasons": dict(Counter(r["end_reason"] for r in episodes)),
        "grouped_episode_manifest": dict(groups),
        "zero_retained_sources": discarded_reports,
        "episodes": episodes,
        "rectification_pixel_checks": image_checks,
        "errors": errors,
        "scope": "offline converted data and dataset construction; no model training, hardware or measured exposure ground truth",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "passed",
                    "sources_total",
                    "sources_retained",
                    "accepted_episodes",
                    "accepted_rows",
                    "retained_span_s",
                    "end_reasons",
                )
            },
            ensure_ascii=False,
        )
    )
    print(f"Report: {args.report}")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
