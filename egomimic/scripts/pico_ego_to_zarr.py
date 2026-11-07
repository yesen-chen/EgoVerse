"""Convert XRoboToolkit ego recordings using a frozen, explicitly supplied fit.

No hardware, MANO fitting, MediaPipe, or training code is run. Poses remain in
the tracking world; obs_head_pose is the LEFT OPTICAL CAMERA, not the headset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import cv2
import numpy as np
import simplejpeg
from scipy.spatial.transform import Rotation

from egomimic.rldb.zarr.zarr_writer import ZarrWriter

SIDES = ("left", "right")
XR_TO_21 = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]


@dataclass(frozen=True)
class Options:
    fps: float = 30.0
    width: int = 640
    height: int = 480
    max_interp_gap_s: float = 0.05
    max_hold_s: float = 4 / 30
    max_image_dt_s: float = 1 / 30
    max_video_gap_s: float = 0.05
    pose_sampling: str = "interpolate"
    max_pose_dt_s: float = 0.025
    gripper_max_m: float = 0.08
    min_segment_frames: int = 2
    max_recovery_step_m: float = 0.1
    max_recovery_rotation_deg: float = 60.0
    axis_convention: str = "piper_x"
    video_layout: str = "stereo-sbs"
    task_name: str = ""
    task_description: str = ""

    def validate(self):
        for key in (
            "fps",
            "max_interp_gap_s",
            "max_image_dt_s",
            "max_video_gap_s",
            "max_pose_dt_s",
            "gripper_max_m",
            "max_recovery_step_m",
            "max_recovery_rotation_deg",
        ):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not np.isfinite(self.max_hold_s) or self.max_hold_s < 0:
            raise ValueError("max_hold_s must be finite and nonnegative")
        if min(self.width, self.height) <= 0 or self.min_segment_frames < 2:
            raise ValueError("Positive image size and min_segment_frames >= 2 required")
        if self.axis_convention not in ("piper_x", "piper"):
            raise ValueError("axis_convention must be piper_x or piper")
        if self.video_layout not in ("stereo-sbs", "left"):
            raise ValueError("video_layout must be stereo-sbs or left")
        if self.pose_sampling not in ("interpolate", "nearest"):
            raise ValueError("pose_sampling must be interpolate or nearest")


def provenance(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def rotation_matrix(value, label):
    r = np.asarray(value, dtype=np.float64)
    if not (
        r.shape == (3, 3)
        and np.isfinite(r).all()
        and np.allclose(r.T @ r, np.eye(3), atol=1e-5)
        and np.isclose(np.linalg.det(r), 1, atol=1e-5)
    ):
        raise ValueError(f"{label}: expected proper 3x3 rotation")
    return r


def load_fit(path):
    info = provenance(path)
    data = json.loads(Path(path).read_text())
    for key in ("post_axis", "rig_rotation"):
        rotation_matrix(data[key], key)
    for key, size in (("rig_translation_m", 3), ("K_fx_fy_cx_cy", 4)):
        value = np.asarray(data[key], dtype=float)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid fit {key}")
    if min(data["K_fx_fy_cx_cy"][:2]) <= 0:
        raise ValueError("Fit focal lengths must be positive")
    if data["extrinsics"] not in ("direct", "inverse"):
        raise ValueError("Fit extrinsics must be direct or inverse")
    if not isinstance(data["eye_swap"], bool):
        raise ValueError("Fit eye_swap must be boolean")
    for key in ("head_offset_ms", "hand_offset_ms"):
        if not np.isfinite(data[key]):
            raise ValueError(f"Invalid fit {key}")
    return {**info, "data": data}


def read_tracking(path):
    with Path(path).open() as handle:
        meta = json.loads(next(handle))
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) < 2:
        raise ValueError("Tracking needs at least two rows")
    times = np.array([r["timeStampNs"] for r in rows], dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("Tracking timestamps must be strictly increasing")
    raw = {
        "timestamp_ns": times,
        "head_pose": np.array(
            [np.fromstring(r["Head"]["pose"], sep=",") for r in rows]
        ),
        "head_status": np.array([r["Head"].get("status", 0) for r in rows], np.uint32),
    }
    if raw["head_pose"].shape != (len(rows), 7):
        raise ValueError("Expected xyz+xyzw head pose")
    for side in SIDES:
        hands = [r["Hand"][side + "Hand"] for r in rows]
        if any(
            int(h["count"]) != 26 or len(h["HandJointLocations"]) != 26 for h in hands
        ):
            raise ValueError(f"{side}: expected 26 XR joints per row")
        raw[side + "_pose"] = np.array(
            [
                [np.fromstring(j["p"], sep=",") for j in h["HandJointLocations"]]
                for h in hands
            ]
        )
        raw[side + "_status"] = np.array(
            [[int(float(j["s"])) for j in h["HandJointLocations"]] for h in hands],
            np.uint32,
        )
        raw[side + "_active"] = np.array([bool(h["isActive"]) for h in hands])
        if raw[side + "_pose"].shape != (len(rows), 26, 7):
            raise ValueError(f"{side}: expected xyz+xyzw for each joint")
    return meta, raw


def probe_video(path):
    ticks, relative = [], []
    size = None
    base = None
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError("Expected exactly one video stream")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Video frame has no PTS")
            current = frame.pts * frame.time_base
            if base is None:
                base, size = current, (frame.width, frame.height)
            if size != (frame.width, frame.height):
                raise ValueError("Video resolution changed")
            ticks.append(int(frame.pts))
            relative.append(round((current - base) * 1_000_000_000))
    ns = np.asarray(relative, dtype=np.int64)
    if len(ns) < 2 or np.any(np.diff(ns) <= 0):
        raise ValueError("Video needs strictly increasing PTS and at least two frames")
    return ns, size


def camera_transform(meta, fit):
    data = fit["data"]
    eyes = np.array(
        [json.loads(v) for v in meta["cameraExtrinsics"].split("|")]
    ).reshape(2, 4, 4)
    for eye in eyes:
        rotation_matrix(eye[:3, :3], "cameraExtrinsics rotation")
        if not np.isfinite(eye).all() or not np.allclose(eye[3], [0, 0, 0, 1]):
            raise ValueError("Invalid cameraExtrinsics SE3")
    if data["extrinsics"] == "inverse":
        eyes = np.linalg.inv(eyes)
    # The LEFT image is always slot 0. eye_swap changes which EXTRINSIC belongs
    # to that slot, never which half of the image is saved.
    eye = eyes[1 if data["eye_swap"] else 0]
    post, rig = np.eye(4), np.eye(4)
    post[:3, :3] = data["post_axis"]
    rig[:3, :3], rig[:3, 3] = data["rig_rotation"], data["rig_translation_m"]
    return post @ eye @ rig


def pose_matrices(pose):
    """Stored pose format: xyz + quaternion wxyz."""
    result = np.broadcast_to(np.eye(4), (len(pose), 4, 4)).copy()
    result[:, :3, :3] = Rotation.from_quat(pose[:, [4, 5, 6, 3]]).as_matrix()
    result[:, :3, 3] = pose[:, :3]
    return result


def matrix_poses(matrices):
    q = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
    for i in range(1, len(q)):
        if q[i - 1] @ q[i] < 0:
            q[i] *= -1
    return np.c_[matrices[:, :3, 3], q]


def hand_geometry(points, side, convention="piper_x", correction=None):
    """World palm centroid and bone frame; never trust raw joint quaternions.

    EgoVerse-style palm geometry using equivalent XR joints, orthogonalized
    while retaining the wrist-to-palm forward direction.
    PiperX axes: X lateral/opening, Y palm normal, Z wrist-to-fingers.
    Piper axes: Y lateral/opening, -X palm normal, Z wrist-to-fingers.
    """
    palm = points[:, [1, 7, 12, 17]].mean(axis=1)
    forward = palm - points[:, 1]
    normal = np.cross(points[:, 7] - points[:, 1], points[:, 22] - points[:, 1])
    normal *= 1 if side == "right" else -1
    forward_norm = np.linalg.norm(forward, axis=1)
    normal_norm = np.linalg.norm(normal, axis=1)
    valid = (
        np.isfinite(points).all(axis=(1, 2))
        & (forward_norm > 1e-8)
        & (normal_norm > 1e-8)
    )
    z = forward / np.maximum(forward_norm[:, None], 1e-12)
    x = np.cross(normal, z)
    xnorm = np.linalg.norm(x, axis=1)
    valid &= xnorm > 1e-8
    x /= np.maximum(xnorm[:, None], 1e-12)
    y = np.cross(z, x)
    rotations = np.stack([x, y, z], axis=-1)
    if convention == "piper":
        rotations = rotations @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    if correction is not None:
        rotations = rotations @ rotation_matrix(correction, side + " hand correction")
    pose = np.full((len(points), 7), np.nan)
    pose[valid, :3] = palm[valid]
    pose[valid, 3:] = (
        Rotation.from_matrix(rotations[valid]).as_quat()[:, [3, 0, 1, 2]]
        if valid.any()
        else np.empty((0, 4))
    )
    return pose, valid


def interpolate_quaternions(q0, q1, alpha):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = q0 @ q1
    if dot < 0:
        q1, dot = -q1, -dot
    dot = np.clip(dot, 0, 1)
    if dot > 0.9995:
        q = (1 - alpha) * q0 + alpha * q1
    else:
        angle = np.arccos(dot)
        q = (np.sin((1 - alpha) * angle) * q0 + np.sin(alpha * angle) * q1) / np.sin(
            angle
        )
    return q / np.linalg.norm(q)


def sample_stream(
    times,
    poses,
    valid,
    query,
    *,
    max_gap_ns,
    hold_ns=0,
    extras=None,
    mode="interpolate",
    max_pose_dt_ns=25_000_000,
):
    """Sample adjacent/nearest valid observations, or hold a SHORT whole gap.

    A long gap is discarded in its entirety (including its first four periods).
    Holds copy world pose AND extras. No extrapolation outside stream coverage.
    """
    output = np.full((len(query), 7), np.nan)
    extra_out = (
        None if extras is None else np.full((len(query), *extras.shape[1:]), np.nan)
    )
    good, filled = np.zeros(len(query), bool), np.zeros(len(query), bool)
    ages = np.full(len(query), np.nan)
    support = np.zeros((len(query), 2), dtype=np.int64)
    valid_idx = np.flatnonzero(valid)
    typical_dt = int(np.median(np.diff(times)))
    for i, q in enumerate(query):
        if q < times[0] or q > times[-1]:
            continue
        hi = int(np.searchsorted(times, q))
        lo = hi if times[hi] == q else hi - 1
        if mode == "nearest":
            chosen = lo if q - times[lo] <= times[hi] - q else hi
            distance = abs(int(times[chosen]) - int(q))
            if distance > max_pose_dt_ns:
                # A timestamp mismatch is not permission to bypass the bound
                # by treating an otherwise valid source row as a lost hand.
                continue
            if valid[chosen]:
                output[i] = poses[chosen]
                output[i, 3:] /= np.linalg.norm(output[i, 3:])
                if extras is not None:
                    extra_out[i] = extras[chosen]
                good[i], ages[i] = True, distance / 1e9
                support[i] = times[chosen]
                continue
        elif valid[lo] and valid[hi] and times[hi] - times[lo] <= max_gap_ns:
            alpha = (
                0.0
                if lo == hi
                else (int(q) - int(times[lo])) / (int(times[hi]) - int(times[lo]))
            )
            output[i, :3] = (1 - alpha) * poses[lo, :3] + alpha * poses[hi, :3]
            output[i, 3:] = interpolate_quaternions(poses[lo, 3:], poses[hi, 3:], alpha)
            if extras is not None:
                extra_out[i] = (1 - alpha) * extras[lo] + alpha * extras[hi]
            good[i], ages[i] = True, (int(q) - int(times[lo])) / 1e9
            support[i] = [times[lo], times[hi]]
            continue
        if hold_ns and len(valid_idx):
            vhi = int(np.searchsorted(times[valid_idx], q))
            if vhi == 0 or vhi == len(valid_idx):
                continue
            prev, nxt = valid_idx[vhi - 1], valid_idx[vhi]
            first_bad = times[prev + 1] if nxt > prev + 1 else times[prev] + typical_dt
            # +1ns accommodates rounding four 30Hz periods onto integer ns.
            age = int(q) - int(times[prev])
            if times[nxt] - first_bad <= hold_ns + 1 and age <= hold_ns + 1:
                output[i], filled[i], ages[i] = poses[prev], True, age / 1e9
                support[i] = times[prev]
                if extras is not None:
                    extra_out[i] = extras[prev]
    return output, good, filled, ages, extra_out, support


def prepare_recording(
    meta, raw, video_ns, fit, options, source_size=(2160, 810), hand_frame=None
):
    options.validate()
    video_ns = np.asarray(video_ns, np.int64)
    if len(video_ns) < 2 or video_ns[0] != 0 or np.any(np.diff(video_ns) <= 0):
        raise ValueError("Expected increasing relative video PTS starting at zero")
    t = np.asarray(raw["timestamp_ns"], np.int64)
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("Tracking timestamps must be strictly increasing")
    epoch = int(meta["timeStampNs"])
    grid = np.rint(
        np.arange(int(np.floor(video_ns[-1] / 1e9 * options.fps)) + 1)
        / options.fps
        * 1e9
    ).astype(np.int64)
    right = np.searchsorted(video_ns, grid).clip(0, len(video_ns) - 1)
    left = (right - 1).clip(0)
    chosen = np.where(grid - video_ns[left] <= video_ns[right] - grid, left, right)
    nominal = np.int64(epoch) + grid
    image_nominal = np.int64(epoch) + video_ns[chosen]
    max_gap = round(options.max_interp_gap_s * 1e9)
    data = fit["data"]
    head_query = nominal + np.int64(round(data["head_offset_ms"] * 1e6))
    hand_query = nominal + np.int64(round(data["hand_offset_ms"] * 1e6))
    hp = np.asarray(raw["head_pose"])[:, [0, 1, 2, 6, 3, 4, 5]]
    head_valid = (
        ((raw["head_status"] & 3) == 3)
        & np.isfinite(hp).all(axis=1)
        & (np.linalg.norm(hp[:, 3:], axis=1) > 1e-8)
    )
    head, head_good, _, _, _, head_support = sample_stream(
        t,
        hp,
        head_valid,
        head_query,
        max_gap_ns=max_gap,
        mode=options.pose_sampling,
        max_pose_dt_ns=round(options.max_pose_dt_s * 1e9),
    )
    camera_from_head = camera_transform(meta, fit)
    camera = np.full_like(head, np.nan)
    if head_good.any():
        camera[head_good] = matrix_poses(
            pose_matrices(head[head_good]) @ np.linalg.inv(camera_from_head)
        )
    valid = head_good & (
        np.abs(video_ns[chosen] - grid) <= round(options.max_image_dt_s * 1e9)
    )
    numeric = {
        "timestamp": nominal.astype(np.float64) / 1e9,
        "timestamp_ns": nominal,
        "obs_rgb_timestamps_ns": image_nominal,
        "source.grid_timestamp_ns": np.int64(epoch) + grid,
        "source.video_frame_index": chosen.astype(np.int64),
        "source.video_pts_ns": video_ns[chosen],
        "source.head_query_timestamp_ns": head_query,
        "source.hand_query_timestamp_ns": hand_query,
        "source.head_pose": head,
        "source.head_support_timestamp_ns": head_support,
        "obs_head_pose": camera,
    }
    for side in SIDES:
        points = np.asarray(raw[side + "_pose"])[..., :3]
        pose, geometry_valid = hand_geometry(
            points,
            side,
            options.axis_convention,
            None if hand_frame is None else hand_frame[side],
        )
        hand_valid = (
            raw[side + "_active"]
            & ((raw[side + "_status"] & 2) != 0).all(axis=1)
            & geometry_valid
        )
        pinch = np.linalg.norm(points[:, 5] - points[:, 10], axis=1)
        extras = np.c_[points.reshape(len(t), -1), pinch]
        sampled, good, held, ages, sampled_extra, support = sample_stream(
            t,
            pose,
            hand_valid,
            hand_query,
            max_gap_ns=max_gap,
            hold_ns=round(options.max_hold_s * 1e9),
            extras=extras,
            mode=options.pose_sampling,
            max_pose_dt_ns=round(options.max_pose_dt_s * 1e9),
        )
        kp = sampled_extra[:, :-1].reshape(-1, 26, 3)
        raw_width = sampled_extra[:, -1:]
        width = np.clip(raw_width, 0, options.gripper_max_m)
        numeric.update(
            {
                f"{side}.obs_ee_pose": sampled,
                f"{side}.obs_keypoints": kp[:, XR_TO_21].reshape(-1, 63),
                f"{side}.obs_xr_keypoints": kp.reshape(-1, 78),
                f"{side}.obs_pinch_distance_m": raw_width,
                f"{side}.obs_gripper_width_m": width,
                f"{side}.obs_gripper": width / options.gripper_max_m,
                f"{side}.gripper_clipped": (raw_width[:, 0] > options.gripper_max_m),
                f"{side}.hand_valid": good,
                f"{side}.hand_filled": held,
                f"{side}.source_age_s": ages,
                f"{side}.source_support_timestamp_ns": support,
            }
        )
        valid &= good | held
    # Also split at video holes; never make an action window jump over a gap.
    boundaries = np.r_[
        True,
        np.diff(image_nominal) > round(options.max_video_gap_s * 1e9),
    ]
    recovery_splits = {s: [] for s in SIDES}
    for side in SIDES:
        pose = numeric[f"{side}.obs_ee_pose"]
        candidates = (
            np.flatnonzero(
                numeric[f"{side}.hand_filled"][:-1] & numeric[f"{side}.hand_valid"][1:]
            )
            + 1
        )
        for i in candidates:
            previous = Rotation.from_quat(pose[i - 1, [4, 5, 6, 3]])
            current = Rotation.from_quat(pose[i, [4, 5, 6, 3]])
            angle = np.degrees((previous.inv() * current).magnitude())
            if (
                np.linalg.norm(pose[i, :3] - pose[i - 1, :3])
                > options.max_recovery_step_m
                or angle > options.max_recovery_rotation_deg
            ):
                boundaries[i] = True
                recovery_splits[side].append(int(i))
    starts, segments = None, []
    for i in range(len(valid) + 1):
        if starts is not None and (i == len(valid) or not valid[i] or boundaries[i]):
            if i - starts >= options.min_segment_frames:
                segments.append((starts, i))
            starts = None
        if i < len(valid) and valid[i] and starts is None:
            starts = i
    w, h = source_size
    if options.video_layout == "stereo-sbs":
        if w % 2:
            raise ValueError("Stereo side-by-side width must be even")
        w //= 2
    fx, fy, cx, cy = data["K_fx_fy_cx_cy"]
    k = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]], float)
    k[0] *= options.width / w
    k[1] *= options.height / h
    kept = np.zeros(len(valid), bool)
    for a, b in segments:
        kept[a:b] = True
    report = {
        "input_tracking_rows": len(t),
        "pose_sampling": options.pose_sampling,
        "input_video_frames": len(video_ns),
        "source_video_fps": (len(video_ns) - 1) * 1e9 / int(video_ns[-1]),
        "candidate_rows": len(nominal),
        "repeated_image_rows": int(np.count_nonzero(np.diff(chosen) == 0)),
        "output_rows": int(kept.sum()),
        "discarded_rows": int((~kept).sum()),
        "segments": [list(s) for s in segments],
        "recovery_jump_splits": recovery_splits,
        "segment_timing": [
            {
                "frames": b - a,
                "effective_fps": (b - a - 1) * 1e9 / int(nominal[b - 1] - nominal[a]),
                "interval_min_ms": float(np.diff(nominal[a:b]).min() / 1e6),
                "interval_max_ms": float(np.diff(nominal[a:b]).max() / 1e6),
            }
            for a, b in segments
        ],
        "head_unsupported_rows": int((~head_good).sum()),
        "max_grid_image_dt_ms": float(np.max(np.abs(video_ns[chosen] - grid)) / 1e6),
        "source_image_size": list(source_size),
        "saved_image_size": [options.width, options.height],
        "hands": {
            s: {
                "held_rows": int(numeric[f"{s}.hand_filled"][kept].sum()),
                "clipped_rows": int(numeric[f"{s}.gripper_clipped"][kept].sum()),
                "max_source_age_ms": float(
                    np.max(numeric[f"{s}.source_age_s"][kept]) * 1000
                )
                if kept.any()
                else None,
            }
            for s in SIDES
        },
    }
    return numeric, segments, report, camera_from_head, k


def encode_images(path, wanted, count_expected, source_size, options):
    encoded = {}
    count = 0
    cv2.setNumThreads(1)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for count, frame in enumerate(container.decode(stream), 1):
            if (frame.width, frame.height) != source_size:
                raise ValueError("Video resolution changed between passes")
            if count - 1 not in wanted:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            if options.video_layout == "stereo-sbs":
                rgb = rgb[:, : frame.width // 2]
            rgb = cv2.resize(
                rgb, (options.width, options.height), interpolation=cv2.INTER_AREA
            )
            encoded[count - 1] = simplejpeg.encode_jpeg(
                np.ascontiguousarray(rgb), quality=85, colorspace="RGB"
            )
    if count != count_expected or set(encoded) != wanted:
        raise ValueError("Decoded frame count/selection mismatch")
    return encoded


def unique_file(source, pattern):
    files = sorted(source.glob(pattern))
    if len(files) != 1:
        raise ValueError(f"Expected one {pattern} in {source}; pass an explicit path")
    return files[0]


def check_fit_reference(meta, fit, size, options):
    """Check camera geometry against the fit's reference recording when present.

    Legacy fits do not embed image size/header. Missing reference is explicitly
    reported, never substituted with an invented reference calibration.
    """
    reference = fit["data"].get("recording")
    if reference is None or not Path(reference).is_dir():
        return {"status": "reference_unavailable", "reference": reference}
    track_files = list(Path(reference).glob("trackingData_*.txt"))
    video_files = list(Path(reference).glob("CameraRecord_*.mp4"))
    if len(track_files) != 1 or len(video_files) != 1:
        return {"status": "reference_unavailable", "reference": reference}
    with track_files[0].open() as handle:
        original = json.loads(next(handle))
    for key in ("cameraIntrinsics", "cameraExtrinsics"):

        def parse(header):
            return np.asarray(
                [json.loads(part) for part in header[key].split("|")], dtype=float
            )

        a, b = parse(meta), parse(original)
        if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-6):
            raise ValueError(
                f"{key} differs from frozen fit recording; supply a matching fit"
            )
    with av.open(str(video_files[0])) as container:
        stream = container.streams.video[0]
        # Original independent fit was defined for stereo side-by-side images.
        expected_eye_size = (stream.width // 2, stream.height)
    current_eye_size = (
        size[0] // 2 if options.video_layout == "stereo-sbs" else size[0],
        size[1],
    )
    if expected_eye_size != current_eye_size:
        raise ValueError(
            f"Per-eye resolution {current_eye_size} differs from frozen fit {expected_eye_size}"
        )
    return {
        "status": "matched",
        "reference": reference,
        "image_size": list(expected_eye_size),
        "checked": ["cameraIntrinsics", "cameraExtrinsics", "per_eye_resolution"],
    }


def convert_dataset(
    source,
    output_dir,
    *,
    fit_path,
    options=None,
    tracking_path=None,
    video_path=None,
    hand_frame_path=None,
):
    options = options or Options()
    options.validate()
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    destination = output_dir / source.name
    if destination == source or source in destination.parents:
        raise ValueError("Output must be separate from raw input")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    tracking = (
        Path(tracking_path)
        if tracking_path
        else unique_file(source, "trackingData_*.txt")
    )
    video = (
        Path(video_path) if video_path else unique_file(source, "CameraRecord_*.mp4")
    )
    fit = load_fit(fit_path)
    hand_frame = None
    hand_frame_info = None
    if hand_frame_path is not None:
        hand_frame = json.loads(Path(hand_frame_path).read_text())
        if set(hand_frame) != set(SIDES):
            raise ValueError("Hand frame JSON must contain left/right 3x3 rotations")
        for side in SIDES:
            rotation_matrix(hand_frame[side], side)
        hand_frame_info = {**provenance(hand_frame_path), "data": hand_frame}
    meta, raw = read_tracking(tracking)
    video_ns, size = probe_video(video)
    reference_check = check_fit_reference(meta, fit, size, options)
    numeric, segments, report, camera_from_head, k = prepare_recording(
        meta, raw, video_ns, fit, options, size, hand_frame
    )
    if not segments:
        raise ValueError(
            "No usable bimanual segments after alignment/validity filtering"
        )
    wanted = {
        int(v) for a, b in segments for v in numeric["source.video_frame_index"][a:b]
    }
    encoded = encode_images(video, wanted, len(video_ns), size, options)
    report.update(
        {
            "source_dataset": str(source),
            "fit": {k: v for k, v in fit.items() if k != "data"},
            "fit_reference_check": reference_check,
        }
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
        "requires_training_transform": "current-camera inverse applied to current obs and entire future world pose chunk; wire measured ego gripper keys explicitly (Human defaults ignore them)",
        "pico_ego": {
            "fit": fit,
            "fit_reference_check": reference_check,
            "raw_header": meta,
            "camera_from_head": camera_from_head.tolist(),
            "source_tracking": provenance(tracking),
            "source_video": str(video.resolve()),
            "options": asdict(options),
            "hand_frame_correction": hand_frame_info,
            "tool_frame": {
                "origin": "mean XR wrist/index/middle/ring MCP [1,7,12,17]; virtual analogue of SDK end frame, no fingertip translation",
                "orientation": "bone-derived proper rotation; not calibrated raw hand quaternion",
                "axis_convention": options.axis_convention,
                "piper_x_axes": "X lateral/opening; Y signed palm normal (left flipped); Z wrist toward palm",
                "piper_rotation_from_piper_x": "right-multiply Rz(-90deg)",
            },
            "gripper_encoding": {
                "raw": "thumb tip 5 to index tip 10 distance in metres",
                "open_m": options.gripper_max_m,
                "normalized": "clip(distance,0,open_m)/open_m; fixed across episodes",
            },
            "time_alignment": {
                "selection": "uniform control grid -> nearest actual video PTS, ties earlier; repeated image frames allowed and reported",
                "fps_semantics": "each episode has a uniform grid at fps; actual image PTS stored separately in obs_rgb_timestamps_ns",
                "pose_query": "uniform grid timestamp + respective head/hand fitted offset; video matching uses real PTS, never frame-number/source-fps",
                "nominal": "header epoch + (video PTS - first PTS); application timestamp, not proven sensor exposure timestamp",
                "pose_sampling": options.pose_sampling,
                "head": "interpolate: linear xyz+SLERP valid adjacent brackets; nearest: closest raw valid row within max_pose_dt_s (ties earlier), never skip invalid rows; no hold/extrapolation",
                "hand": "same sampling choice as head; interpolation uses world xyz/keypoints/width + SLERP bone orientation; nearest keeps a raw sample; short loss holds world values; long whole gaps removed and episodes split",
                "support_timestamps": "two actual source timestamps per query; interpolation brackets or identical nearest/held timestamps",
                "source_age_s": "interpolate: query minus lower bracket; nearest: absolute source-query delta; held: age of last valid row",
                "max_pose_dt_s": options.max_pose_dt_s,
                "max_hold_s": options.max_hold_s,
                "fit_scope": "frozen effective alignment fit; must be rechecked for changed capture/camera configuration",
            },
            "keypoints": {
                "obs_keypoints": "canonical21 from XR, no MANO fitting",
                "obs_xr_keypoints": "26 XR joints xyz in world",
                "xr_indices_for_canonical21": XR_TO_21,
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{source.name}_", suffix=".partial", dir=output_dir)
    )
    paths = []
    try:
        for index, (a, b) in enumerate(segments):
            name = f"episode_{index:06d}.zarr"
            values = {key: value[a:b] for key, value in numeric.items()}
            for key in (
                "source.head_pose",
                "obs_head_pose",
                "left.obs_ee_pose",
                "right.obs_ee_pose",
            ):
                values[key] = values[key].copy()
                for i in range(1, len(values[key])):
                    if values[key][i - 1, 3:] @ values[key][i, 3:] < 0:
                        values[key][i, 3:] *= -1
            if any(
                v.dtype.kind == "f" and not np.isfinite(v).all()
                for v in values.values()
            ):
                raise ValueError("Nonfinite values in an accepted segment")
            frames = np.empty(b - a, dtype=object)
            frames[:] = [encoded[int(i)] for i in values["source.video_frame_index"]]
            segment_report = {
                **report,
                "segment_index": index,
                "segment_candidate_range": [a, b],
                "segment_frames": b - a,
            }
            writer = ZarrWriter(
                temporary / name,
                embodiment="human_bimanual",
                fps=options.fps,
                task_name=options.task_name,
                task_description=options.task_description,
                intrinsics={"front_1": k.tolist()},
                extrinsics=None,
                annotations=[(options.task_description, 0, b - a)]
                if options.task_description
                else [],
            )
            writer.write(
                numeric_data=values,
                pre_encoded_image_data={
                    "images.front_1": (frames, [options.height, options.width, 3])
                },
                metadata_override={**metadata, "conversion": segment_report},
            )
            paths.append(destination / name)
        (temporary / "conversion_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fit", type=Path, required=True, help="Frozen fit.json; never refitted"
    )
    parser.add_argument("--tracking", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument(
        "--hand-frame",
        type=Path,
        help="Optional left/right local 3x3 proper rotation corrections",
    )
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-interp-gap-s", type=float, default=0.05)
    parser.add_argument("--max-hold-s", type=float, default=4 / 30)
    parser.add_argument("--max-image-dt-s", type=float, default=1 / 30)
    parser.add_argument("--max-video-gap-s", type=float, default=0.05)
    parser.add_argument(
        "--pose-sampling", choices=["interpolate", "nearest"], default="interpolate"
    )
    parser.add_argument("--max-pose-dt-s", type=float, default=0.025)
    parser.add_argument("--gripper-max-m", type=float, default=0.08)
    parser.add_argument("--min-segment-frames", type=int, default=2)
    parser.add_argument("--max-recovery-step-m", type=float, default=0.1)
    parser.add_argument("--max-recovery-rotation-deg", type=float, default=60)
    parser.add_argument(
        "--axis-convention", choices=["piper_x", "piper"], default="piper_x"
    )
    parser.add_argument(
        "--video-layout", choices=["stereo-sbs", "left"], default="stereo-sbs"
    )
    parser.add_argument("--task-name", default="")
    parser.add_argument("--task-description", default="")
    args = parser.parse_args()
    options = Options(**{k: getattr(args, k) for k in Options.__dataclass_fields__})
    convert_dataset(
        args.input,
        args.output_dir,
        fit_path=args.fit,
        options=options,
        tracking_path=args.tracking,
        video_path=args.video,
        hand_frame_path=args.hand_frame,
    )


if __name__ == "__main__":
    main()
