"""The joint adapter must preserve clock precision and reject ambiguous pairing."""

import copy
import csv
import json
from fractions import Fraction

import av
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_pico.ego_joint_input import (
    load_joint_fit,
    load_joint_input,
)

BASE_UTC = 1_794_727_145_486_862_123
BASE_MONO = 1_118_301_568_738


def dump(path, value):
    path.write_text(json.dumps(value))


def update_json(path, key, value):
    data = json.loads(path.read_text())
    data[key] = value
    dump(path, data)


def update_tracking(folder, index, function):
    path = folder / "tracking.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    function(rows[index])
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def update_frames(folder, function):
    path = folder / "frames.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    function(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def recording(tmp_path):
    source = tmp_path / "joint_fixture"
    source.mkdir()
    width, height = 64, 48
    with av.open(str(source / "ego.mp4"), "w") as video:
        stream = video.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.time_base = Fraction(1, 30)
        for i in range(5):
            frame = av.VideoFrame.from_ndarray(
                np.full((height, width, 3), i * 40, np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = i, Fraction(1, 30)
            for packet in stream.encode(frame):
                video.mux(packet)
        for packet in stream.encode():
            video.mux(packet)
    with av.open(str(source / "ego.mp4")) as video:
        pts_ns = [
            round(frame.pts * frame.time_base * 1_000_000_000)
            for frame in video.decode(video=0)
        ]
    fields = [
        "frame_id",
        "receive_time_ns",
        "mux_pts_us",
        "status",
        "receive_utc_ns",
        "utc_source",
        "utc_discontinuity",
        "utc_outside_anchor_tolerance",
    ]
    with (source / "frames.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for i, ns in enumerate(pts_ns):
            writer.writerow(
                dict(
                    frame_id=i,
                    receive_time_ns=BASE_MONO + ns,
                    mux_pts_us=round(ns / 1000),
                    status="saved",
                    receive_utc_ns=BASE_UTC + ns,
                    utc_source="direct_wall_clock",
                    utc_discontinuity=0,
                    utc_outside_anchor_tolerance=0,
                )
            )
    common = dict(
        session_id=source.name,
        clock_id="clock-a",
        complete=True,
        status="complete",
        synthetic=False,
        alignment_clock="unix_utc",
        width=width,
        height=height,
    )
    dump(
        source / "session.json",
        dict(
            common,
            schema="pico_uvc_probe_v2",
            mp4_finalized=True,
            saved_frames=5,
            received_frames=5,
            terminal_csv_rows=5,
        ),
    )
    dump(
        source / "joint_session.json",
        dict(
            common,
            schema="pico_uvc_joint_v1",
            video_complete=True,
            tracking_complete=True,
            tracking_config={"source_sensor_latency_applied": False},
        ),
    )
    dump(source / "tracking_summary.json", {"clock_id": "clock-a"})
    joint = {"p": "0.2,0.3,0.4,0,0,0,1", "s": 15}
    hand = dict(
        count=26,
        isActive=True,
        sdk_success=True,
        HandJointLocations=[copy.deepcopy(joint) for _ in range(26)],
    )
    rows = []
    for i in range(15):
        delta = i * 10_000_000
        rows.append(
            dict(
                clock_id="clock-a",
                sample_id=i,
                receive_utc_ns=BASE_UTC + delta,
                timeStampNs=BASE_UTC + delta,
                receive_time_ns=BASE_MONO + delta,
                sample_end_time_ns=BASE_MONO + delta + 1_000_000,
                utc_source="direct_wall_clock",
                utc_discontinuity=False,
                utc_outside_anchor_tolerance=False,
                status="saved",
                tracking={
                    "appState": {"focus": True},
                    "Head": {"pose": "1,2,3,0,0,0,1", "status": 3, "sdk_result": 0},
                    "Hand": {
                        "leftHand": copy.deepcopy(hand),
                        "rightHand": copy.deepcopy(hand),
                    },
                },
            )
        )
    (source / "tracking.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    fit = tmp_path / "fit.json"
    dump(
        fit,
        {
            "schema": "seeded_external_uvc_spacetime_v1",
            "parameters": {
                "R_CH": Rotation.from_euler("xyz", [15, 30, -20], degrees=True)
                .as_matrix()
                .tolist(),
                "t_CH_m": [0.01, -0.03, 0.1],
                "delta_head_ms": -38.82067185775031,
                "delta_hand_ms": 8.457320987015416,
            },
            "intrinsics": {
                "file": "/does/not/exist/frozen_original.json",
                "raw_K": [[40, 0, 31], [0, 40, 23], [0, 0, 1]],
                "raw_D": [0.01, -0.02, 0, 0, 0.005],
                "image_size": [width, height],
                "mirror": False,
            },
        },
    )
    return source, fit, np.asarray(pts_ns, np.int64)


def test_exact_utc_and_frame_map_and_frozen_intrinsics(recording):
    source, fit, pts = recording
    before = (source / "tracking.jsonl").read_bytes()
    result = load_joint_input(source, fit)
    assert result.video_utc_ns.dtype == np.int64
    assert result.video_utc_ns[0] == BASE_UTC
    np.testing.assert_array_equal(result.video_utc_ns - np.int64(BASE_UTC), pts)
    np.testing.assert_array_equal(result.video_pts_absolute_ns, pts)
    np.testing.assert_array_equal(
        result.raw["timestamp_ns"],
        BASE_UTC + np.arange(15, dtype=np.int64) * 10_000_000,
    )
    assert (
        len(result.raw["timestamp_ns"]) == 15
    )  # First JSONL row is a sample, not a header.
    assert result.source_size == (64, 48)
    assert result.raw["left_pose"].shape == (15, 26, 7)
    assert result.raw["left_active"].all()
    assert result.head_offset_ns == -38_820_672
    assert result.hand_offset_ns == 8_457_321
    assert result.metadata["frame_mapping"]["decoded_frames"] == 5
    assert len(result.metadata["source_files"]["tracking.jsonl"]["sha256"]) == 64
    assert (source / "tracking.jsonl").read_bytes() == before
    assert (
        result.fit["data"]["intrinsics"]["file"]
        == "/does/not/exist/frozen_original.json"
    )


def test_camera_chain_matches_fit_head_relative_formula(recording):
    result = load_joint_input(*recording[:2])
    head_rotation = Rotation.from_euler("xyz", [-22, 13, 48], degrees=True).as_matrix()
    head_translation = np.array([0.4, 0.1, -0.2])
    world_points = np.array([[0.5, 0.7, 1.2], [1.2, 0.9, 0.8]])
    twh = np.eye(4)
    twh[:3, :3], twh[:3, 3] = head_rotation, head_translation
    chain = result.T_C_H @ np.linalg.inv(twh)
    actual = world_points @ chain[:3, :3].T + chain[:3, 3]
    # Independent expression from fit_engine.head_relative + projection.
    relative = (world_points - head_translation) @ head_rotation
    expected = relative @ result.T_C_H[:3, :3].T + result.T_C_H[:3, 3]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_saved_completion_order_may_differ_and_dropped_rows_are_not_frames(recording):
    source, fit, _ = recording

    def change(rows):
        for row in rows[2:]:
            row["frame_id"] = str(int(row["frame_id"]) + 1)
        dropped = dict(rows[1], frame_id="2", status="dropped", mux_pts_us="")
        rows.insert(2, dropped)
        rows.reverse()

    update_frames(source, change)
    update_json(source / "session.json", "terminal_csv_rows", 6)
    update_json(source / "session.json", "received_frames", 6)
    result = load_joint_input(source, fit)
    assert result.video_frame_ids.tolist() == [0, 1, 3, 4, 5]
    assert result.metadata["frame_mapping"]["dropped_csv_rows"] == 1


def test_small_mux_quantization_tolerated(recording):
    source, fit, _ = recording

    def change(rows):
        rows[3]["mux_pts_us"] = str(int(rows[3]["mux_pts_us"]) + 145)

    update_frames(source, change)
    result = load_joint_input(source, fit)
    assert (
        144_000 <= result.metadata["frame_mapping"]["max_abs_pts_error_ns"] <= 146_000
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda rows: rows[2].update(mux_pts_us=rows[1]["mux_pts_us"]),
            "strictly increasing",
        ),
        (
            lambda rows: rows[2].update(
                mux_pts_us=str(int(rows[2]["mux_pts_us"]) + 1000)
            ),
            "unique CSV",
        ),
        (lambda rows: rows[2].update(frame_id=rows[1]["frame_id"]), "duplicate"),
        (
            lambda rows: rows[2].update(receive_utc_ns=rows[1]["receive_utc_ns"]),
            "strictly increasing",
        ),
        (lambda rows: rows[2].update(utc_discontinuity="1"), "utc_discontinuity"),
        (
            lambda rows: rows[2].update(utc_source="anchor_estimate"),
            "directly recorded",
        ),
    ],
)
def test_bad_frame_mapping_fails(recording, mutation, match):
    source, fit, _ = recording
    update_frames(source, mutation)
    with pytest.raises(ValueError, match=match):
        load_joint_input(source, fit)


def test_count_mismatch_is_not_silently_zipped(recording):
    source, fit, _ = recording
    update_frames(source, lambda rows: rows.pop())
    update_json(source / "session.json", "saved_frames", 4)
    update_json(source / "session.json", "received_frames", 4)
    update_json(source / "session.json", "terminal_csv_rows", 4)
    with pytest.raises(ValueError, match="count mismatch"):
        load_joint_input(source, fit)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda row: row.update(clock_id="wrong"), "clock_id"),
        (lambda row: row.update(timeStampNs=row["timeStampNs"] + 1), "disagrees"),
        (
            lambda row: row.update(receive_utc_ns=float(row["receive_utc_ns"])),
            "exact integer",
        ),
        (lambda row: row.update(utc_outside_anchor_tolerance=True), "utc_outside"),
        (lambda row: row.update(utc_step_ns=12_000_000), "clock threshold"),
        (
            lambda row: row.update(sample_end_time_ns=row["receive_time_ns"] - 1),
            "negative query",
        ),
    ],
)
def test_bad_tracking_clocks_fail(recording, mutation, match):
    source, fit, _ = recording
    update_tracking(source, 4, mutation)
    with pytest.raises(ValueError, match=match):
        load_joint_input(source, fit)


def test_unflagged_clock_step_fails_independent_monotonic_check(recording):
    source, fit, _ = recording
    rows = [
        json.loads(line)
        for line in (source / "tracking.jsonl").read_text().splitlines()
    ]
    for row in rows[7:]:
        row["receive_utc_ns"] += 15_000_000
        row["timeStampNs"] += 15_000_000
    (source / "tracking.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    with pytest.raises(ValueError, match="drift/step"):
        load_joint_input(source, fit)


def test_inactive_and_bad_sdk_samples_retained_as_invalid(recording):
    source, fit, _ = recording
    update_tracking(
        source,
        3,
        lambda row: row["tracking"]["Hand"]["leftHand"].update(isActive=False),
    )
    update_tracking(
        source, 4, lambda row: row["tracking"]["Head"].update(sdk_result=-1)
    )
    update_tracking(
        source,
        5,
        lambda row: row["tracking"]["Hand"]["rightHand"].update(sdk_success=False),
    )
    update_tracking(
        source,
        6,
        lambda row: row.update(sample_end_time_ns=row["receive_time_ns"] + 21_000_000),
    )
    result = load_joint_input(source, fit)
    assert len(result.raw["timestamp_ns"]) == 15
    assert not result.raw["left_active"][3]
    assert result.raw["head_status"][4] == 0
    assert result.raw["head_source_status"][4] == 3
    assert not result.raw["right_active"][5]
    assert np.all(result.raw["right_status"][5] == 0)
    assert result.raw["head_status"][6] == 0
    assert not result.raw["left_active"][6]
    assert result.metadata["tracking_validation"]["query_duration_over_20ms"] == 1


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda d: d["parameters"].update(R_CH=np.diag([-1, 1, 1]).tolist()),
            "proper rotation",
        ),
        (lambda d: d["parameters"].update(t_CH_m=[1, 2]), "shape"),
        (lambda d: d["parameters"].update(delta_hand_ms=float("nan")), "finite"),
        (lambda d: d["intrinsics"].update(mirror=True), "unmirrored"),
        (lambda d: d["intrinsics"].update(raw_D=[0, 0, 0, 0]), "Brown5"),
        (
            lambda d: d["intrinsics"].update(
                raw_K=[[40, 1, 32], [0, 40, 24], [0, 0, 1]]
            ),
            "skewed",
        ),
    ],
)
def test_invalid_frozen_fit_rejected(recording, mutation, match):
    _, fit, _ = recording
    data = json.loads(fit.read_text())
    mutation(data)
    dump(fit, data)
    with pytest.raises(ValueError, match=match):
        load_joint_fit(fit)


def test_intrinsics_size_cannot_be_silently_scaled(recording):
    source, fit, _ = recording
    data = json.loads(fit.read_text())
    data["intrinsics"]["image_size"] = [128, 96]
    dump(fit, data)
    with pytest.raises(ValueError, match="frozen fit intrinsics"):
        load_joint_input(source, fit)


def test_metadata_clock_and_prior_compensation_rejected(recording):
    source, fit, _ = recording
    data = json.loads((source / "joint_session.json").read_text())
    data["tracking_config"]["source_sensor_latency_applied"] = True
    dump(source / "joint_session.json", data)
    with pytest.raises(ValueError, match="already applied"):
        load_joint_input(source, fit)


def test_tracking_summary_count_not_silently_ignored(recording):
    source, fit, _ = recording
    update_json(source / "tracking_summary.json", "saved_samples", 14)
    with pytest.raises(ValueError, match="saved_samples disagrees"):
        load_joint_input(source, fit)


def test_matching_streams_still_must_match_session_clock_anchor(recording):
    source, fit, _ = recording
    update_json(source / "session.json", "clock_anchor_utc_ns", BASE_UTC + 30_000_000)
    update_json(source / "session.json", "clock_anchor_monotonic_ns", BASE_MONO)
    with pytest.raises(ValueError, match="session anchor"):
        load_joint_input(source, fit)
