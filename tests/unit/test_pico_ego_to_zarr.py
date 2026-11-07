"""Offline PICO ego conversion: timing, world frames, gaps and exported images."""

import hashlib
import json
from dataclasses import replace
from fractions import Fraction

import av
import numpy as np
import pytest
import simplejpeg
import zarr
from scipy.spatial.transform import Rotation

from egomimic.scripts.pico_ego_to_zarr import (
    Options,
    convert_dataset,
    load_fit,
    prepare_recording,
    read_tracking,
)

EPOCH_NS = 1_791_906_094_525_271_040
SIDES = ("left", "right")


def skeleton(side, time_s=0.0, width=0.04):
    """Nondegenerate, known bone frame; motion is in the tracking world."""
    points = np.zeros((26, 3), dtype=np.float64)
    points[:, 2] = 1.0
    points[1] = [0, 0, 1]
    points[7] = [0.03, 0.1, 1]
    points[12] = [0, 0.11, 1]
    points[17] = [-0.02, 0.1, 1]
    points[22] = [-0.04, 0.08, 1]
    points[5] = [0, 0.12, 1]
    points[10] = [width, 0.12, 1]
    points[:, 0] += (-0.1 if side == "left" else 0.1) + 0.2 * time_s
    return points


def make_tracking(times_s=None):
    if times_s is None:
        times_s = np.arange(-20, 101) / 100
    times_s = np.asarray(times_s)
    n = len(times_s)
    poses = np.zeros((n, 7))
    poses[:, 0] = 0.3 * times_s
    poses[:, 3:] = Rotation.from_euler("z", (0.4 * times_s)[:, None]).as_quat()
    raw = {
        "timestamp_ns": EPOCH_NS + np.rint(times_s * 1e9).astype(np.int64),
        "head_pose": poses,
        "head_status": np.full(n, 3, dtype=np.uint32),
    }
    for side in SIDES:
        joints = np.zeros((n, 26, 7))
        joints[:, :, :3] = [skeleton(side, t) for t in times_s]
        joints[:, :, 6] = 1
        raw[f"{side}_pose"] = joints
        raw[f"{side}_active"] = np.ones(n, dtype=bool)
        raw[f"{side}_status"] = np.full((n, 26), 15, dtype=np.uint32)
    eye0, eye1 = np.eye(4), np.eye(4)
    eye0[:3, 3] = [0.03, 0.01, 0]
    eye1[:3, 3] = [-0.03, 0.01, 0]
    meta = {
        "timeStampNs": EPOCH_NS,
        "cameraExtrinsics": "|".join(
            json.dumps(t.ravel().tolist()) for t in (eye0, eye1)
        ),
        "cameraIntrinsics": json.dumps([16, 6, 200, 100]),
    }
    return meta, raw


def write_fit(path, **changes):
    data = {
        "extrinsics": "direct",
        "eye_swap": True,
        "post_axis": np.eye(3).tolist(),
        "rig_rotation": Rotation.from_euler("z", 0.2).as_matrix().tolist(),
        "rig_translation_m": [0.01, -0.02, 0.03],
        "K_fx_fy_cx_cy": [100, 100, 8, 6],
        "head_offset_ms": -40.0,
        "hand_offset_ms": -10.0,
        "reference": "synthetic frozen calibration",
    }
    data.update(changes)
    path.write_text(json.dumps(data))
    return load_fit(path)


def prepare(meta, raw, fit, pts_ns, **options):
    opts = replace(Options(width=16, height=12), **options)
    return prepare_recording(
        meta, raw, np.asarray(pts_ns, dtype=np.int64), fit, opts, source_size=(32, 12)
    )


def pose_matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def valid_rows(segments):
    return np.concatenate([np.arange(start, end) for start, end in segments])


def test_true_pts_and_independent_offsets_keep_epoch_precision(tmp_path):
    meta, raw = make_tracking()
    fit = write_fit(tmp_path / "fit.json")
    pts = np.array([0, 20, 60, 100, 133, 167, 200], dtype=np.int64) * 1_000_000
    numeric, segments, _, camera_from_head, intrinsic = prepare(meta, raw, fit, pts)
    idx = valid_rows(segments)
    np.testing.assert_array_equal(
        numeric["source.video_frame_index"][idx], np.arange(7)
    )
    np.testing.assert_array_equal(numeric["obs_rgb_timestamps_ns"][idx], EPOCH_NS + pts)
    grid = np.rint(np.arange(7) / 30 * 1e9).astype(np.int64)
    np.testing.assert_array_equal(numeric["timestamp_ns"][idx], EPOCH_NS + grid)
    np.testing.assert_array_equal(
        numeric["source.grid_timestamp_ns"][idx], EPOCH_NS + grid
    )
    assert np.any(numeric["timestamp_ns"][idx] != numeric["obs_rgb_timestamps_ns"][idx])
    # Motion uses the strict 30 Hz grid plus each sensor's fitted offset. Images
    # retain their actual PTS, separately from the common sample timestamps.
    np.testing.assert_array_equal(
        numeric["source.head_query_timestamp_ns"][idx], EPOCH_NS + grid - 40_000_000
    )
    np.testing.assert_array_equal(
        numeric["source.hand_query_timestamp_ns"][idx], EPOCH_NS + grid - 10_000_000
    )
    hand_time = grid / 1e9 - 0.01
    for side in SIDES:
        expected = np.array(
            [skeleton(side, t)[[1, 7, 12, 17]].mean(0) for t in hand_time]
        )
        np.testing.assert_allclose(
            numeric[f"{side}.obs_ee_pose"][idx, :3], expected, atol=1e-10
        )
        xr = numeric[f"{side}.obs_xr_keypoints"][idx].reshape(-1, 26, 3)
        canonical = numeric[f"{side}.obs_keypoints"][idx].reshape(-1, 21, 3)
        np.testing.assert_allclose(
            canonical,
            xr[
                :,
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    7,
                    8,
                    9,
                    10,
                    12,
                    13,
                    14,
                    15,
                    17,
                    18,
                    19,
                    20,
                    22,
                    23,
                    24,
                    25,
                ],
            ],
        )
    head_time = grid / 1e9 - 0.04
    np.testing.assert_allclose(
        numeric["source.head_pose"][idx, 0], 0.3 * head_time, atol=1e-10
    )
    # Compose an independent reference for camera<-head, including swapped eye.
    eye = np.array(json.loads(meta["cameraExtrinsics"].split("|")[1])).reshape(4, 4)
    rig = np.eye(4)
    rig[:3, :3] = fit["data"]["rig_rotation"]
    rig[:3, 3] = fit["data"]["rig_translation_m"]
    expected_camera_from_head = eye @ rig
    np.testing.assert_allclose(camera_from_head, expected_camera_from_head, atol=1e-12)
    for row in idx:
        world_head = pose_matrix(numeric["source.head_pose"][row])
        world_camera = pose_matrix(numeric["obs_head_pose"][row])
        np.testing.assert_allclose(
            world_camera,
            world_head @ np.linalg.inv(expected_camera_from_head),
            atol=1e-9,
        )
        world_points = numeric["left.obs_xr_keypoints"][row].reshape(26, 3)
        direct = (
            np.linalg.inv(world_head)[:3, :3] @ (world_points - world_head[:3, 3]).T
        ).T
        direct = (
            expected_camera_from_head[:3, :3] @ direct.T
        ).T + expected_camera_from_head[:3, 3]
        saved = (world_camera[:3, :3].T @ (world_points - world_camera[:3, 3]).T).T
        np.testing.assert_allclose(saved, direct, atol=1e-9)
    assert np.asarray(intrinsic).shape in {(3, 3), (3, 4)}


def test_repeated_image_does_not_delete_regular_grid_samples(tmp_path):
    meta, raw = make_tracking()
    fit = write_fit(tmp_path / "fit.json")
    pts = np.array([0, 70, 130, 200], dtype=np.int64) * 1_000_000
    numeric, segments, *_ = prepare(meta, raw, fit, pts, max_video_gap_s=0.1)
    np.testing.assert_array_equal(valid_rows(segments), np.arange(7))
    np.testing.assert_array_equal(
        numeric["source.video_frame_index"], [0, 0, 1, 1, 2, 3, 3]
    )
    np.testing.assert_array_equal(
        numeric["obs_rgb_timestamps_ns"], EPOCH_NS + pts[[0, 0, 1, 1, 2, 3, 3]]
    )
    np.testing.assert_array_equal(
        numeric["timestamp_ns"],
        EPOCH_NS + np.rint(np.arange(7) / 30 * 1e9).astype(np.int64),
    )
    assert np.all(np.diff(numeric["timestamp_ns"]) > 0)


@pytest.mark.parametrize("offsets_ms", [(-40, -10), (-5, 5)])
def test_nearest_uses_raw_rows_with_separate_offsets_and_earlier_ties(
    tmp_path, offsets_ms
):
    meta, raw = make_tracking()
    fit = write_fit(
        tmp_path / "fit.json",
        head_offset_ms=offsets_ms[0],
        hand_offset_ms=offsets_ms[1],
    )
    pts = np.array([0, 20, 60, 100, 133, 167, 200], dtype=np.int64) * 1_000_000
    nearest, segments, *_ = prepare(meta, raw, fit, pts, pose_sampling="nearest")
    interpolated, *_ = prepare(meta, raw, fit, pts)
    rows = valid_rows(segments)
    np.testing.assert_array_equal(rows, np.arange(7))
    # Independent brute-force reference keeps the first index on an equal tie.
    head_query = nearest["source.head_query_timestamp_ns"]
    hand_query = nearest["source.hand_query_timestamp_ns"]
    head_source = np.abs(raw["timestamp_ns"][:, None] - head_query).argmin(axis=0)
    hand_source = np.abs(raw["timestamp_ns"][:, None] - hand_query).argmin(axis=0)
    assert np.any(head_source != hand_source)
    expected_head = raw["head_pose"][head_source][:, [0, 1, 2, 6, 3, 4, 5]]
    np.testing.assert_allclose(nearest["source.head_pose"], expected_head, atol=1e-12)
    np.testing.assert_array_equal(
        nearest["source.head_support_timestamp_ns"],
        np.repeat(raw["timestamp_ns"][head_source, None], 2, axis=1),
    )
    for side in SIDES:
        expected_points = raw[f"{side}_pose"][hand_source, :, :3]
        np.testing.assert_array_equal(
            nearest[f"{side}.obs_xr_keypoints"].reshape(-1, 26, 3), expected_points
        )
        np.testing.assert_allclose(
            nearest[f"{side}.obs_ee_pose"][:, :3],
            expected_points[:, [1, 7, 12, 17]].mean(axis=1),
        )
        np.testing.assert_allclose(
            nearest[f"{side}.source_age_s"],
            np.abs(raw["timestamp_ns"][hand_source] - hand_query) / 1e9,
        )
        np.testing.assert_array_equal(
            nearest[f"{side}.source_support_timestamp_ns"],
            np.repeat(raw["timestamp_ns"][hand_source, None], 2, axis=1),
        )
        assert np.all(nearest[f"{side}.source_age_s"] >= 0)
        assert nearest[f"{side}.hand_valid"].all()
        assert not nearest[f"{side}.hand_filled"].any()
        assert not np.allclose(
            nearest[f"{side}.obs_ee_pose"][:, :3],
            interpolated[f"{side}.obs_ee_pose"][:, :3],
        )
    for key in (
        "timestamp_ns",
        "source.grid_timestamp_ns",
        "obs_rgb_timestamps_ns",
        "source.video_frame_index",
    ):
        np.testing.assert_array_equal(nearest[key], interpolated[key])
    if offsets_ms == (-5, 5):
        # At t=0, head query=-5ms ties -10/0ms; hand query=+5ms ties 0/10ms.
        assert raw["timestamp_ns"][head_source[0]] == EPOCH_NS - 10_000_000
        assert raw["timestamp_ns"][hand_source[0]] == EPOCH_NS


def test_nearest_invalid_row_holds_prior_world_pose_instead_of_skipping_ahead(tmp_path):
    meta, raw = make_tracking()
    fit = write_fit(tmp_path / "fit.json", head_offset_ms=0, hand_offset_ms=0)
    bad = np.flatnonzero(raw["timestamp_ns"] == EPOCH_NS + 30_000_000).item()
    raw["left_active"][bad] = False
    # The future valid row is closer than the prior valid row to the grid query,
    # so filtering invalid records before nearest-search would produce a false hit.
    raw["left_pose"][bad + 1, 10, 0] += 0.03
    pts = np.rint(np.arange(7) / 30 * 1e9).astype(np.int64)
    numeric, segments, *_ = prepare(meta, raw, fit, pts, pose_sampling="nearest")
    np.testing.assert_array_equal(valid_rows(segments), np.arange(7))
    assert not numeric["left.hand_valid"][1]
    assert numeric["left.hand_filled"][1]
    np.testing.assert_array_equal(
        numeric["left.obs_xr_keypoints"][1].reshape(26, 3),
        raw["left_pose"][bad - 1, :, :3],
    )
    np.testing.assert_allclose(numeric["left.obs_pinch_distance_m"][1], 0.04)
    np.testing.assert_array_equal(
        numeric["left.source_support_timestamp_ns"][1],
        [raw["timestamp_ns"][bad - 1]] * 2,
    )


def test_nearest_rejects_valid_rows_beyond_max_pose_dt(tmp_path):
    meta, raw = make_tracking(np.arange(11) / 10)
    fit = write_fit(tmp_path / "fit.json", head_offset_ms=0, hand_offset_ms=0)
    pts = np.rint(np.arange(31) / 30 * 1e9).astype(np.int64)
    numeric, *_ = prepare(
        meta,
        raw,
        fit,
        pts,
        pose_sampling="nearest",
        max_pose_dt_s=0.01,
        max_hold_s=0,
    )
    assert np.isfinite(numeric["source.head_pose"][[0, 3]]).all()
    assert np.isnan(numeric["source.head_pose"][[1, 2]]).all()
    for side in SIDES:
        assert numeric[f"{side}.hand_valid"][[0, 3]].all()
        assert not numeric[f"{side}.hand_valid"][[1, 2]].any()
        assert not numeric[f"{side}.hand_filled"][[1, 2]].any()
        assert np.isnan(numeric[f"{side}.obs_ee_pose"][[1, 2]]).all()


@pytest.mark.parametrize("jump", ["translation", "rotation"])
def test_recovery_jump_after_fill_starts_new_episode(tmp_path, jump):
    times = np.arange(15) / 30
    meta, raw = make_tracking(times)
    fit = write_fit(tmp_path / "fit.json", head_offset_ms=0, hand_offset_ms=0)
    raw["left_active"][5:7] = False
    if jump == "translation":
        raw["left_pose"][7:, :, 0] += 0.25
    else:
        points = raw["left_pose"][7:, :, :3]
        wrist = points[:, 1:2].copy()
        rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        points[:] = (points - wrist) @ rotation.T + wrist
    numeric, segments, *_ = prepare(meta, raw, fit, raw["timestamp_ns"] - EPOCH_NS)
    assert segments == [(0, 7), (7, 15)]
    np.testing.assert_array_equal(np.flatnonzero(numeric["left.hand_filled"]), [5, 6])
    np.testing.assert_array_equal(valid_rows(segments), np.arange(15))


def test_short_loss_holds_world_hand_while_camera_continues(tmp_path):
    times = np.arange(19) / 30
    meta, raw = make_tracking(times)
    fit = write_fit(tmp_path / "fit.json", head_offset_ms=0, hand_offset_ms=0)
    raw["left_active"][5:9] = False  # Exactly four 30 Hz periods.
    pts = raw["timestamp_ns"] - EPOCH_NS
    numeric, segments, *_ = prepare(meta, raw, fit, pts)
    np.testing.assert_array_equal(valid_rows(segments), np.arange(19))
    filled = numeric["left.hand_filled"]
    np.testing.assert_array_equal(np.flatnonzero(filled), np.arange(5, 9))
    assert not numeric["left.hand_valid"][5:9].any()
    for key in (
        "obs_ee_pose",
        "obs_xr_keypoints",
        "obs_gripper",
        "obs_pinch_distance_m",
    ):
        np.testing.assert_allclose(
            numeric[f"left.{key}"][5:9],
            np.broadcast_to(
                numeric[f"left.{key}"][4], numeric[f"left.{key}"][5:9].shape
            ),
        )
    assert not np.allclose(numeric["obs_head_pose"][5], numeric["obs_head_pose"][8])
    np.testing.assert_allclose(
        numeric["left.source_age_s"][5:9], np.arange(1, 5) / 30, atol=1e-9
    )


def test_long_loss_drops_whole_gap_and_head_gaps_split(tmp_path):
    times = np.arange(25) / 30
    meta, raw = make_tracking(times)
    fit = write_fit(tmp_path / "fit.json", head_offset_ms=0, hand_offset_ms=0)
    raw["left_active"][5:10] = False
    raw["head_status"][15:17] = 0
    numeric, segments, *_ = prepare(meta, raw, fit, raw["timestamp_ns"] - EPOCH_NS)
    np.testing.assert_array_equal(valid_rows(segments), np.r_[0:5, 10:15, 17:25])
    assert len(segments) == 3
    # Do not retain the first four frames of an already-known long outage.
    assert not numeric["left.hand_filled"][5:10].any()


def test_no_initial_or_terminal_hand_extrapolation(tmp_path):
    times = np.arange(15) / 30
    meta, raw = make_tracking(times)
    fit = write_fit(tmp_path / "fit.json", head_offset_ms=0, hand_offset_ms=0)
    raw["left_active"][:2] = False
    raw["right_active"][-2:] = False
    _, segments, *_ = prepare(meta, raw, fit, raw["timestamp_ns"] - EPOCH_NS)
    np.testing.assert_array_equal(valid_rows(segments), np.arange(2, 13))


def test_fixed_metric_gripper_mapping_preserves_raw_width(tmp_path):
    times = np.arange(9) / 30
    meta, raw = make_tracking(times)
    fit = write_fit(tmp_path / "fit.json", head_offset_ms=0, hand_offset_ms=0)
    widths = np.array([0, 0.02, 0.04, 0.08, 0.12, 0.10, 0.04, 0.02, 0])
    for side in SIDES:
        raw[f"{side}_pose"][:, :, :3] = [
            skeleton(side, t, w) for t, w in zip(times, widths)
        ]
    numeric, segments, *_ = prepare(meta, raw, fit, raw["timestamp_ns"] - EPOCH_NS)
    np.testing.assert_array_equal(valid_rows(segments), np.arange(9))
    for side in SIDES:
        np.testing.assert_allclose(
            numeric[f"{side}.obs_pinch_distance_m"].ravel(), widths, atol=1e-10
        )
        np.testing.assert_allclose(
            numeric[f"{side}.obs_gripper_width_m"].ravel(),
            np.clip(widths, 0, 0.08),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            numeric[f"{side}.obs_gripper"].ravel(),
            np.clip(widths / 0.08, 0, 1),
            atol=1e-10,
        )
        clipped = numeric[f"{side}.gripper_clipped"].ravel()
        assert clipped[widths > 0.080001].all()
        assert not clipped[widths < 0.079999].any()


def test_piper_axis_conventions_are_proper_rotations_without_tool_translation(tmp_path):
    meta, raw = make_tracking()
    fit = write_fit(tmp_path / "fit.json")
    pts = np.arange(9, dtype=np.int64) * 33_333_333
    piper_x, *_ = prepare(meta, raw, fit, pts, axis_convention="piper_x")
    piper, *_ = prepare(meta, raw, fit, pts, axis_convention="piper")
    for side in SIDES:
        a = piper_x[f"{side}.obs_ee_pose"]
        b = piper[f"{side}.obs_ee_pose"]
        np.testing.assert_allclose(a[:, :3], b[:, :3])
        ra = Rotation.from_quat(a[:, [4, 5, 6, 3]]).as_matrix()
        rb = Rotation.from_quat(b[:, [4, 5, 6, 3]]).as_matrix()
        np.testing.assert_allclose(np.linalg.det(ra), 1, atol=1e-10)
        np.testing.assert_allclose(
            rb, ra @ Rotation.from_euler("z", -90, degrees=True).as_matrix(), atol=1e-9
        )
        np.testing.assert_allclose(
            ra[:, :, 2],
            np.broadcast_to(
                np.array([0.0025, 0.0775, 0]) / np.linalg.norm([0.0025, 0.0775, 0]),
                (len(ra), 3),
            ),
            atol=1e-9,
        )


def test_degenerate_bones_or_invalid_joint_status_cannot_create_valid_pose(tmp_path):
    times = np.arange(13) / 30
    meta, raw = make_tracking(times)
    fit = write_fit(tmp_path / "fit.json", head_offset_ms=0, hand_offset_ms=0)
    raw["left_pose"][4, :, :3] = [0, 0, 1]
    raw["left_status"][8, 25] = 0
    numeric, segments, *_ = prepare(meta, raw, fit, raw["timestamp_ns"] - EPOCH_NS)
    np.testing.assert_array_equal(valid_rows(segments), np.arange(13))
    np.testing.assert_array_equal(np.flatnonzero(numeric["left.hand_filled"]), [4, 8])
    assert not numeric["left.hand_valid"][[4, 8]].any()
    assert np.isfinite(numeric["left.obs_ee_pose"]).all()
    np.testing.assert_allclose(
        numeric["left.obs_ee_pose"][[4, 8]], numeric["left.obs_ee_pose"][[3, 7]]
    )


def write_tracking(path, meta, raw):
    lines = [json.dumps(meta)]
    for i, timestamp in enumerate(raw["timestamp_ns"]):
        row = {
            "timeStampNs": int(timestamp),
            "Head": {
                "pose": ",".join(map(str, raw["head_pose"][i])),
                "status": int(raw["head_status"][i]),
            },
            "Hand": {},
        }
        for side in SIDES:
            row["Hand"][f"{side}Hand"] = {
                "isActive": int(raw[f"{side}_active"][i]),
                "count": 26,
                "HandJointLocations": [
                    {"p": ",".join(map(str, pose)), "s": int(status), "r": 0.0}
                    for pose, status in zip(
                        raw[f"{side}_pose"][i], raw[f"{side}_status"][i]
                    )
                ],
            }
        lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n")


def write_video(path, pts_ms):
    with av.open(str(path), mode="w") as out:
        stream = out.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 12, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for pts in pts_ms:
            rgb = np.empty((12, 32, 3), dtype=np.uint8)
            rgb[:, :16] = [220, 20, 20]
            rgb[:, 16:] = [20, 20, 220]
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def test_export_reader_left_image_fit_provenance_and_no_overwrite(tmp_path):
    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrEpisode

    source = tmp_path / "capture"
    source.mkdir()
    meta, raw = make_tracking()
    tracking = source / "trackingData_20261013_234134.txt"
    write_tracking(tracking, meta, raw)
    write_video(
        source / "CameraRecord_20261013_234134.mp4", [0, 20, 60, 100, 133, 167, 200]
    )
    fit_path = tmp_path / "fit.json"
    fit = write_fit(fit_path)
    read_meta, read_raw = read_tracking(tracking)
    assert read_meta == meta
    np.testing.assert_array_equal(read_raw["timestamp_ns"], raw["timestamp_ns"])
    options = Options(width=16, height=12)
    output = convert_dataset(
        source, tmp_path / "export", fit_path=fit_path, options=options
    )
    assert len(output) == 1
    episode = ZarrEpisode(output[0])
    assert len(episode) == 7
    sample = episode.read({"images.front_1": (2, None), "left.obs_ee_pose": (0, 7)})
    rgb = simplejpeg.decode_jpeg(sample["images.front_1"])
    assert rgb.shape == (12, 16, 3)
    # eye_swap reorders extrinsics; the first video slot remains the left crop.
    assert rgb[..., 0].mean() > 180 and rgb[..., 2].mean() < 50
    assert sample["left.obs_ee_pose"].shape == (7, 7)
    group = zarr.open_group(str(output[0]), mode="r")
    assert "images.right_wrist" not in group
    assert "left.cmd_ee_pose" not in group
    metadata = json.dumps(dict(group.attrs))
    assert fit["sha256"] == hashlib.sha256(fit_path.read_bytes()).hexdigest()
    assert fit["sha256"] in metadata and "synthetic frozen calibration" in metadata
    with pytest.raises(FileExistsError):
        convert_dataset(source, tmp_path / "export", fit_path=fit_path, options=options)


def test_invalid_fit_transform_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_fit(tmp_path / "fit.json", post_axis=np.diag([1, 1, -1]).tolist())
    with pytest.raises(ValueError):
        write_fit(tmp_path / "fit2.json", rig_rotation=np.ones((3, 3)).tolist())


@pytest.mark.parametrize(
    "changes",
    [
        {"K_fx_fy_cx_cy": [-100, 100, 8, 6]},
        {"head_offset_ms": float("nan")},
        {"eye_swap": "false"},
        {"extrinsics": "unknown"},
    ],
)
def test_invalid_fit_intrinsics_timing_or_conventions_are_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        write_fit(tmp_path / "fit.json", **changes)


@pytest.mark.parametrize("failure", ["decode", "write"])
def test_failed_conversion_leaves_no_partial_dataset(tmp_path, monkeypatch, failure):
    import egomimic.scripts.pico_ego_to_zarr as converter

    source = tmp_path / "capture"
    source.mkdir()
    meta, raw = make_tracking()
    tracking = source / "trackingData_test.txt"
    write_tracking(tracking, meta, raw)
    write_video(source / "CameraRecord_test.mp4", [0, 33, 67, 100])
    before = tracking.read_bytes()
    fit_path = tmp_path / "fit.json"
    write_fit(fit_path)
    output = tmp_path / "export"

    def fail_decode(*args, **kwargs):
        raise RuntimeError("test decode failure")

    def fail_write(self, *args, **kwargs):
        self.episode_path.mkdir()
        (self.episode_path / "partial.txt").write_text("interrupted writer")
        raise RuntimeError("test write failure")

    if failure == "decode":
        monkeypatch.setattr(converter, "encode_images", fail_decode)
    else:
        monkeypatch.setattr(converter.ZarrWriter, "write", fail_write)
    with pytest.raises(RuntimeError, match=f"test {failure} failure"):
        convert_dataset(
            source, output, fit_path=fit_path, options=Options(width=16, height=12)
        )
    assert not output.exists() or not list(output.iterdir())
    assert tracking.read_bytes() == before


def test_output_cannot_replace_or_be_inside_raw_recording(tmp_path):
    source = tmp_path / "capture"
    source.mkdir()
    for output in (source, source.parent):
        with pytest.raises(ValueError, match="separate from raw input"):
            convert_dataset(source, output, fit_path=tmp_path / "unused.json")
