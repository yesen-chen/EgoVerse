"""External conversion connects clock alignment, strict slicing and actual pixels."""

import hashlib
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
import pytest
import simplejpeg
import zarr
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_pico.ego_preprocessing import PreprocessOptions
from egomimic.robot.piper_pico.image_geometry import RgbImageProcessor
from egomimic.scripts import pico_uvc_ego_to_zarr as converter
from egomimic.scripts.pico_ego_to_zarr import Options, probe_video

EPOCH = 1_794_729_600_123_456_789


@pytest.fixture
def recording(tmp_path):
    source = tmp_path / "joint_fixture"
    source.mkdir()
    video = source / "ego.mp4"
    tracking = source / "tracking.jsonl"
    tracking.write_text("synthetic raw input for adapter-independent export test\n")
    size = (64, 48)
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(121):
            image = np.zeros((48, 64, 3), np.uint8)
            image[:, :, 0] = np.arange(64, dtype=np.uint8) * 3
            image[10:30, 15:45, 1:] = 200
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    pts, _ = probe_video(video)
    seconds = np.arange(-5, 411) / 100
    t = EPOCH + np.rint(seconds * 1e9).astype(np.int64)
    head = np.c_[
        0.1 * seconds,
        np.zeros((len(t), 2)),
        Rotation.from_euler("z", (seconds * 0.2)[:, None]).as_quat(),
    ]
    raw = {"timestamp_ns": t, "head_pose": head, "head_status": np.full(len(t), 3)}
    for side in ("left", "right"):
        points = np.zeros((len(t), 26, 7))
        points[..., 2] = 1
        points[..., 6] = 1
        points[:, 7, :2] = [0.03, 0.1]
        points[:, 12, :2] = [0, 0.11]
        points[:, 17, :2] = [-0.02, 0.1]
        points[:, 22, :2] = [-0.04, 0.08]
        points[:, 5, :2] = [0, 0.12]
        points[:, 10, :2] = [0.04, 0.12]
        points[:, :, 0] += 0.2 * seconds[:, None]
        raw.update(
            {
                f"{side}_pose": points,
                f"{side}_active": np.ones(len(t), bool),
                f"{side}_status": np.full((len(t), 26), 15),
            }
        )
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("y", 0.15).as_matrix()
    T[:3, 3] = [0.02, -0.01, 0.04]
    K = np.array([[40.0, 0, 32], [0, 41, 24], [0, 0, 1]])
    D = np.array([-0.2, 0.05, 0.001, -0.001, 0.01])
    fit = {
        "path": "synthetic",
        "sha256": "synthetic",
        "data": {
            "schema": "seeded_external_uvc_spacetime_v1",
            "parameters": {
                "R_CH": T[:3, :3].tolist(),
                "t_CH_m": T[:3, 3].tolist(),
                "delta_head_ms": -20.0,
                "delta_hand_ms": 10.0,
            },
            "intrinsics": {
                "raw_K": K.tolist(),
                "raw_D": D.tolist(),
                "image_size": list(size),
                "mirror": False,
            },
        },
    }
    return SimpleNamespace(
        source_dir=source,
        video_path=video,
        tracking_path=tracking,
        fit=fit,
        metadata={"synthetic": True},
        raw=raw,
        source_size=size,
        raw_K=K,
        raw_D=D,
        T_C_H=T,
        video_pts_ns=pts,
        video_pts_absolute_ns=pts,
        video_frame_ids=np.arange(len(pts)) * 2,
        video_utc_ns=EPOCH + pts + (np.arange(len(pts)) % 2) * 2_000_000,
    )


def test_utc_match_pts_and_fitted_offsets_applied_once(recording):
    options = Options(video_layout="left", max_hold_s=0, width=64, height=48)
    values, segments, _, _, _ = converter.prepare_joint(recording, options=options)
    assert segments
    idx = values["source.video_frame_index"]
    np.testing.assert_array_equal(
        values["source.video_pts_ns"], recording.video_pts_ns[idx]
    )
    np.testing.assert_array_equal(
        values["source.video_frame_id"], recording.video_frame_ids[idx]
    )
    np.testing.assert_array_equal(
        values["obs_rgb_timestamps_ns"], recording.video_utc_ns[idx]
    )
    grid = values["timestamp_ns"]
    np.testing.assert_array_equal(
        values["source.head_query_timestamp_ns"], grid - 20_000_000
    )
    np.testing.assert_array_equal(
        values["source.hand_query_timestamp_ns"], grid + 10_000_000
    )
    a, b = segments[0]
    np.testing.assert_allclose(
        values["source.head_pose"][a:b, 0],
        0.1 * ((grid[a:b] - EPOCH) / 1e9 - 0.02),
        atol=1e-12,
    )
    strict = replace(options, max_image_dt_s=0.001)
    _, strict_segments, _, report, _ = converter.prepare_joint(
        recording, options=strict
    )
    assert report["rows_outside_image_time_bound"] > 0
    assert not strict_segments  # Alternating 2ms jitter leaves no 2s continuous run.


def test_subtick_degenerate_skeleton_never_bridged(recording):
    raw = recording.raw
    index = np.searchsorted(raw["timestamp_ns"], EPOCH + 2_020_000_000)
    raw["left_pose"][index, :, :3] = [0, 0, 1]
    _, segments, boundaries, _, _ = converter.prepare_joint(
        recording, preprocessing=PreprocessOptions(0.1, 0.1, 0.5)
    )
    assert len(segments) == 2
    assert boundaries[0]["end_reason"] == "inactivity"


def test_atomic_export_rectifies_pixels_and_preserves_raw(
    recording, tmp_path, monkeypatch
):
    monkeypatch.setattr(converter, "load_joint_input", lambda *args: recording)
    before = hashlib.sha256(recording.video_path.read_bytes()).hexdigest()
    options = Options(video_layout="left", max_hold_s=0, width=64, height=48)
    output = tmp_path / "converted"
    paths, report = converter.convert_dataset(
        recording.source_dir, output, fit_path="synthetic", options=options
    )
    assert len(paths) == 1 and report["episodes_written"] == 1
    group = zarr.open_group(str(paths[0]), mode="r")
    meta = dict(group.attrs)
    assert meta["preprocessing"]["end_reason"] == "ui_trim"
    assert meta["image_space"] == "processed"
    assert meta["distortion_coefficients"]["front_1"] == [0.0] * 5
    assert meta["pico_ego"]["source_format"] == "uvc_joint_v1"
    index = int(group["source.video_frame_index"][0])
    with av.open(str(recording.video_path)) as video:
        native = next(
            frame.to_ndarray(format="rgb24")
            for i, frame in enumerate(video.decode(video=0))
            if i == index
        )
    expected = RgbImageProcessor(report["image_geometry"]).apply(native)
    # Slice as the production episode reader does. Zarr v3 scalar indexing
    # wraps variable-length bytes in nested 0d arrays on this version.
    stored = simplejpeg.decode_jpeg(
        group["images.front_1"][:1][0], colorspace="RGB"
    )
    assert np.abs(stored.astype(float) - expected).mean() < 6
    assert hashlib.sha256(recording.video_path.read_bytes()).hexdigest() == before
    with pytest.raises(FileExistsError):
        converter.convert_dataset(recording.source_dir, output, fit_path="synthetic")
    assert not list(output.glob("*.partial"))


def test_preprocess_only_does_not_encode_or_write_episodes(
    recording, tmp_path, monkeypatch
):
    monkeypatch.setattr(converter, "load_joint_input", lambda *args: recording)
    monkeypatch.setattr(
        converter, "_encode", lambda *args: pytest.fail("Unexpected encoding")
    )
    output = tmp_path / "preview"
    paths, report = converter.convert_dataset(
        recording.source_dir, output, fit_path="synthetic", preprocess_only=True
    )
    assert not paths and report["kept_segments"] > 0
    assert (output / recording.source_dir.name / "preprocessing_report.json").is_file()
    assert not list(output.rglob("*.zarr"))
