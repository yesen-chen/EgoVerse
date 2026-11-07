"""CPU-only source freshness and CSV diagnostics, without a hardware SDK."""

from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.robot.piper_pico.pico_process import (
    SourceTimestampMonitor,
    pico_examples,
    read_pico_sample,
)
from egomimic.robot.piper_pico.teleop_log import CAMERA_COLUMNS, TeleopRecorder


def test_source_stall_counts_intervals_and_keeps_current_gap():
    monitor = SourceTimestampMonitor()
    stamp = 1_760_000_000_123_456_789
    assert np.isnan(monitor.update(0, 50.0)["source_age_s"])
    first = monitor.update(stamp, 100.0)
    assert first["source_timestamp_ns"] == stamp
    assert first["source_age_s"] == 0.0
    assert monitor.update(stamp, 100.05)["source_stall_count"] == 0
    stalled = monitor.update(stamp, 100.125)
    assert stalled["source_stall_count"] == 1
    assert stalled["source_age_s"] == pytest.approx(0.125)
    ongoing = monitor.update(stamp, 100.375)
    assert ongoing["source_stall_count"] == 1
    assert ongoing["source_max_gap_s"] == pytest.approx(0.375)
    recovered = monitor.update(stamp + 1, 100.5)
    assert recovered["source_age_s"] == 0.0
    assert recovered["source_max_gap_s"] == pytest.approx(0.5)
    assert monitor.update(stamp + 1, 100.625)["source_stall_count"] == 2


def test_source_reset_and_missing_timestamp_do_not_fabricate_a_stall():
    monitor = SourceTimestampMonitor()
    monitor.update(1000, 10.0)
    reset = monitor.update(500, 10.02)
    assert reset["source_age_s"] == 0.0
    assert reset["source_stall_count"] == 0
    missing = monitor.update(0, 100.0)
    assert np.isnan(missing["source_age_s"])
    assert missing["source_stall_count"] == 0
    restored = monitor.update(500, 200.0)
    assert restored["source_age_s"] == 0.0
    assert restored["source_max_gap_s"] == pytest.approx(0.02)


def _fake_sdk(**extra):
    def forbidden_init():
        raise AssertionError("a diagnostic read must not initialize hardware")

    return SimpleNamespace(
        init=forbidden_init,
        get_left_controller_pose=lambda: [0.2, 0.3, -0.4, 0, 0, 0, 1],
        get_right_controller_pose=lambda: [-0.2, 0.3, -0.4, 0, 0, 0, 1],
        get_left_grip=lambda: 0.8,
        get_right_grip=lambda: 1.0,
        get_left_trigger=lambda: 0.2,
        get_right_trigger=lambda: 0.4,
        get_A_button=lambda: True,
        get_B_button=lambda: False,
        get_X_button=lambda: False,
        get_Y_button=lambda: True,
        **extra,
    )


def test_diagnostic_read_preserves_controls_and_int64_source_stamp():
    stamp = 1_760_000_000_123_456_789
    sample = read_pico_sample(_fake_sdk(
        get_headset_pose=lambda: [0.1, 1.5, -0.1, 0, 0, 0, 1],
        get_time_stamp_ns=lambda: stamp,
    ))
    sample.update(SourceTimestampMonitor().update(int(sample["source_timestamp_ns"]), 0.0))
    assert sample["source_timestamp_ns"] == stamp
    assert sample["source_timestamp_ns"].dtype == np.dtype("int64")
    np.testing.assert_array_equal(sample["head_pos"], [0.1, 1.5, -0.1])
    np.testing.assert_array_equal(sample["left_pos"], [0.2, 0.3, -0.4])
    assert sample["left_grip"] == 0.8
    assert sample["button_a"] == 1.0
    examples = pico_examples()
    assert sample.keys() == examples.keys()
    for name, example in examples.items():
        assert np.asarray(sample[name]).shape == np.asarray(example).shape
        assert np.asarray(sample[name]).dtype == np.asarray(example).dtype


@pytest.mark.parametrize("extra", [
    {},
    {"get_headset_pose": lambda: [0], "get_time_stamp_ns": lambda: float("nan")},
])
def test_unavailable_optional_diagnostics_do_not_break_controller_reads(extra):
    sample = read_pico_sample(_fake_sdk(**extra))
    assert sample["source_timestamp_ns"] == 0
    assert np.isnan(sample["head_pos"]).all()
    assert np.isnan(sample["head_quat_xyzw"]).all()
    assert sample["right_grip"] == 1.0
    assert sample["right_trigger"] == 0.4


def test_csv_distinguishes_loss_hand_samples_ticks_events_and_logs_diagnostics(tmp_path, capsys):
    env = SimpleNamespace(
        left_clutch=SimpleNamespace(last_event="lost", enabled=True),
        right_clutch=SimpleNamespace(last_event="lost", enabled=True),
        last_action=None,
        measured_pose=lambda index: (np.array([0.2, 0.0, 0.2]), None),
    )
    path = tmp_path / "diagnostics.csv"
    recorder = TeleopRecorder(path)
    recorder.start()
    sample = read_pico_sample(_fake_sdk())
    stamp = 1_760_000_000_123_456_789
    sample.update(SourceTimestampMonitor().update(stamp, 0.0))
    sample["head_pos"] = np.array([0.1, 1.5, -0.1])
    sample["head_quat_xyzw"] = np.array([0.0, 0.0, 0.0, 1.0])
    camera = dict(zip(CAMERA_COLUMNS, [30, 29, 100.1, 100.09, 0.01, 0.02, 0.01, 0.002]))
    camera["camera_mode"] = "stereo"
    camera["camera_left_count_mono"] = 42.0  # monitor-only extras are ignored
    # Two bilateral ticks, one left-only tick, recovery, then a right-only loss.
    for index, (left_lost, right_lost) in enumerate([(1, 1), (1, 1), (1, 0), (0, 0), (0, 1)]):
        sample["receive_timestamp"] = 100.0 + index / 30.0
        sample["left_pos"] = np.zeros(3) if left_lost else np.array([0.2, 0.3, -0.4])
        sample["right_pos"] = np.zeros(3) if right_lost else np.array([-0.2, 0.3, -0.4])
        recorder.write(env, sample, diagnostics=camera if index == 0 else None)
    recorder.close()
    assert recorder.lost_rows == 6
    assert recorder.lost_by_side == {"left": 3, "right": 3}
    assert recorder.lost_ticks == 4
    assert recorder.lost_events == 2
    assert recorder.both_lost_ticks == 2
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["lost_event_start"] for row in rows] == ["1", "0", "0", "0", "1"]
    assert rows[0]["source_timestamp_ns"] == str(stamp)
    assert float(rows[0]["head_raw_y"]) == 1.5
    assert rows[0]["camera_left_count"] == "30"
    assert rows[0]["camera_mode"] == "stereo"
    assert rows[1]["camera_left_count"] == ""
    output = capsys.readouterr().out
    assert "lost-origin ticks (either hand): 4" in output
    assert "lost-origin hand samples: 6" in output


def test_old_sample_without_diagnostics_still_records(tmp_path):
    env = SimpleNamespace(
        left_clutch=SimpleNamespace(last_event="idle", enabled=False),
        right_clutch=SimpleNamespace(last_event="idle", enabled=False),
        measured_pose=lambda index: (np.zeros(3), None),
    )
    sample = read_pico_sample(_fake_sdk())
    for name in ("source_timestamp_ns", "head_pos", "head_quat_xyzw"):
        sample.pop(name)
    path = tmp_path / "legacy.csv"
    recorder = TeleopRecorder(path)
    recorder.start()
    recorder.write(env, sample)
    recorder.close()
    with path.open(newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["source_timestamp_ns"] == ""
    assert row["head_raw_x"] == ""
    assert row["left_grip"] == "0.800000"


@pytest.mark.parametrize("mode", ["left", "right"])
def test_mono_csv_records_mode_and_leaves_disabled_eye_blank(tmp_path, mode):
    env = SimpleNamespace(
        left_clutch=SimpleNamespace(last_event="accepted", enabled=True),
        right_clutch=SimpleNamespace(last_event="accepted", enabled=True),
        measured_pose=lambda index: (np.zeros(3), None),
    )
    path = tmp_path / "mono.csv"
    recorder = TeleopRecorder(path)
    recorder.start()
    recorder.write(env, read_pico_sample(_fake_sdk()), diagnostics={
        "camera_mode": mode, f"camera_{mode}_count": 42,
        f"camera_{mode}_receive_timestamp": 100.0,
        f"camera_{mode}_age_s": 0.01, "camera_copy_s": 0.001,
    })
    recorder.close()
    with path.open(newline="") as fh:
        row = next(csv.DictReader(fh))
    inactive = "right" if mode == "left" else "left"
    assert row["camera_mode"] == mode
    assert row[f"camera_{mode}_count"] == "42"
    assert row[f"camera_{inactive}_count"] == ""
    assert row[f"camera_{inactive}_receive_timestamp"] == ""
    assert row["camera_read_phase_skew_s"] == ""
    assert row["left_raw_x"] and row["right_raw_x"]
