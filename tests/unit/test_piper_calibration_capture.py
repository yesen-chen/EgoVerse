"""Calibration snapshots remain paired while control and vision advance."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.robot.piper_pico.calibration_capture import (
    CalibrationControls,
    CaptureLimits,
    CaptureWorker,
    SessionStore,
    check_stability,
    freeze_snapshot,
)

NOW = 1700000000.0


def history(end=NOW, duration=1.0):
    times = end + np.linspace(-duration, 0.0, int(round(duration * 100)) + 1)
    count = len(times)
    return {
        "robot_receive_timestamp": times.copy(),
        "robot_timestamp": times - 0.01,
        "robot_pose_source_timestamp": times - 0.003,
        "robot_gripper_source_timestamp": times - 0.005,
        "robot_joint_source_timestamp": times - 0.004,
        "robot_source_timestamps_simulated": np.zeros(count, dtype=bool),
        "ee_pose": np.tile([0.2, -0.1, 0.3, 0.0, 0.0, 0.0], (count, 1)),
        "joint_pos": np.zeros((count, 6)),
        "joint_vel": np.zeros((count, 6)),
        "gripper_position": np.full(count, 0.019),
    }


def camera_data():
    return {
        "color": np.full((1, 8, 10, 3), [10, 20, 30], dtype=np.uint8),
        "timestamp": np.array([NOW - 0.08]),
        "camera_receive_timestamp": np.array([NOW - 0.01]),
    }


def sample(**buttons):
    return {f"button_{key}": buttons.get(key, 0.0) for key in "xyab"}


def test_snapshot_copies_the_image_pose_and_entire_feedback_history():
    frames, states = camera_data(), history()
    snapshot = freeze_snapshot(frames, states, NOW)
    frames["color"][:] = 255
    frames["timestamp"][:] = NOW + 50
    states["ee_pose"][:] = 12
    states["robot_pose_source_timestamp"][:] = 0

    np.testing.assert_array_equal(snapshot["image"][0, 0], [10, 20, 30])
    assert not snapshot["image"].flags.writeable
    np.testing.assert_allclose(snapshot["state"]["ee_pose"][:3], [0.2, -0.1, 0.3])
    np.testing.assert_allclose(snapshot["T_base_end"][:3, 3], [0.2, -0.1, 0.3])
    assert snapshot["camera_timestamp"] == pytest.approx(NOW - 0.08)
    assert snapshot["feedback"]["robot_pose_source_timestamp"][-1] > 0


def test_stability_accepts_wrapped_euler_angles_for_the_same_orientation():
    states = history()
    states["ee_pose"][:, 5] = np.pi - 0.001
    states["ee_pose"][::2, 5] = -np.pi + 0.001
    result = check_stability(states, NOW - 0.7, NOW, CaptureLimits())
    assert result["ok"], result
    assert result["rotation_from_latest_deg"] < 0.12


def test_opposite_rotation_excursions_count_the_full_range():
    states = history()
    states["ee_pose"][-30, 3] = np.deg2rad(-0.29)
    states["ee_pose"][-20, 3] = np.deg2rad(0.29)
    result = check_stability(states, NOW - 0.7, NOW, CaptureLimits())
    assert not result["ok"]
    assert result["rotation_range_deg"] == pytest.approx(0.58)


@pytest.mark.parametrize(
    "source", ["robot_pose_source_timestamp", "robot_gripper_source_timestamp"]
)
def test_fresh_host_polling_cannot_hide_stale_sdk_feedback(source):
    states = history()
    states[source][:] = NOW - 0.75
    result = check_stability(states, NOW - 0.7, NOW, CaptureLimits())
    assert not result["ok"]
    assert "实际反馈过期" in result["reason"]


def test_constant_sdk_timestamp_is_rejected_even_within_age_limit():
    states = history()
    states["robot_pose_source_timestamp"][:] = NOW - 0.10
    result = check_stability(states, NOW - 0.05, NOW, CaptureLimits())
    assert not result["ok"]
    assert "未更新" in result["reason"]


@pytest.mark.parametrize(
    "kind", ["translation", "rotation", "gripper", "missing", "simulated", "nan"]
)
def test_unusable_stationary_windows_are_rejected(kind):
    states = history()
    if kind == "translation":
        states["ee_pose"][-20:, 0] += 0.002
    elif kind == "rotation":
        states["ee_pose"][-20:, 3] += np.deg2rad(0.5)
    elif kind == "gripper":
        states["gripper_position"][-20:] += 0.001
    elif kind == "missing":
        del states["robot_pose_source_timestamp"]
    elif kind == "simulated":
        states["robot_source_timestamps_simulated"][:] = True
    elif kind == "nan":
        states["ee_pose"][-1, 0] = np.nan
    assert not check_stability(states, NOW - 0.7, NOW, CaptureLimits())["ok"]


def test_too_short_or_interrupted_history_cannot_pass_stability():
    short = history(duration=0.2)
    assert not check_stability(short, NOW - 0.7, NOW, CaptureLimits())["ok"]
    states = history()
    keep = np.r_[np.arange(45), np.arange(65, 101)]
    states = {key: value[keep] for key, value in states.items()}
    result = check_stability(states, NOW - 0.7, NOW, CaptureLimits())
    assert not result["ok"]
    assert "中断" in result["reason"]


def store_args():
    snapshot = freeze_snapshot(camera_data(), history(), NOW)
    return (
        snapshot,
        {"accepted": True, "corner_count": 25, "reprojection_rms_px": 0.2},
        snapshot["image"].copy(),
        {"split": "train", "timestamp": NOW - 1.0},
        {"ok": True},
    )


def test_session_save_and_undo_preserve_records_without_reusing_ids(tmp_path):
    store = SessionStore(tmp_path / "session", {"arm": "left"})
    first = store.save(*store_args())
    assert first.name == "000001"
    record = json.loads((first / "sample.json").read_text())
    assert record["arm"] == "left"
    np.testing.assert_allclose(record["T_base_end"], store_args()[0]["T_base_end"])
    assert (first / "raw.png").is_file()
    assert (first / "feedback.npz").is_file()
    assert store.invalidate_last() == first
    assert (first / "invalidated.json").is_file()
    assert (first / "raw.png").is_file()
    assert store.saved == []
    assert store.invalidate_last() is None
    second = store.save(*store_args())
    assert second.name == "000002"
    assert store.saved == [second]


def test_atomic_publish_failure_keeps_hidden_pending_and_does_not_count(
    tmp_path, monkeypatch
):
    store = SessionStore(tmp_path / "session", {"arm": "left"})

    def fail_publish(*_):
        raise OSError("disk publish failed")

    monkeypatch.setattr(
        "egomimic.robot.piper_pico.calibration_capture.os.rename", fail_publish
    )
    with pytest.raises(OSError, match="publish failed"):
        store.save(*store_args())
    assert store.saved == []
    assert store.next_id == 1
    assert not (store.path / "samples" / "000001").exists()
    pending = store.path / "samples" / ".000001.pending"
    assert pending.is_dir()
    assert (pending / "sample.json").is_file()


class FakeControlWorker:
    def __init__(self):
        self.enabled = True
        self.requests = []
        self.enable_calls = []
        self.undo_calls = 0

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.enable_calls.append(enabled)

    def request(self, split):
        if not self.enabled:
            return False
        self.requests.append(split)
        return True

    def undo(self):
        self.undo_calls += 1
        return True


class FakeEnv:
    def __init__(self):
        self.home_calls = []
        self.teleop_calls = []
        self.executed = []
        self.seed_count = 0
        self.last_action = np.arange(14, dtype=float) / 1000
        self.action = np.arange(14, dtype=float) + 1
        self.arms = [
            SimpleNamespace(
                go_home=lambda gripper_m, i=i: self.home_calls.append((i, gripper_m))
            )
            for i in range(2)
        ]

    def seed_clutches_from_measured(self):
        self.seed_count += 1

    def teleop_action(self, current_sample, now):
        self.teleop_calls.append((current_sample, now))
        return self.action

    def exec_actions(self, action, deadlines):
        self.executed.append((action, deadlines.copy()))


@pytest.mark.parametrize("key,index", [("x", 0), ("a", 1)])
def test_home_moves_only_named_arm_preserves_clamp_and_pauses_both(key, index):
    controls = CalibrationControls("left", home_seconds=3.0)
    env, worker = FakeEnv(), FakeControlWorker()
    current = sample()
    assert controls.tick(env, worker, current, {key}, NOW, 0.03)
    assert env.home_calls == [(index, env.last_action[index * 7 + 6])]
    assert env.teleop_calls == []
    assert env.executed == []
    assert worker.enable_calls == [False]
    controls.tick(env, worker, current, {"y"}, NOW + 1.0, 0.03)
    assert worker.requests == []
    assert env.teleop_calls == []

    controls.tick(env, worker, current, set(), NOW + 3.0, 0.03)
    assert env.seed_count == 1
    assert worker.enable_calls == [False, True]
    assert env.teleop_calls == [(current, NOW + 3.0)]
    assert env.executed[0][0] is env.action
    controls.tick(env, worker, current, set(), NOW + 3.1, 0.03)
    assert env.seed_count == 1


def test_selected_arm_filters_capture_only_and_preserves_teleop_call():
    controls = CalibrationControls("right", 3.0)
    env, worker = FakeEnv(), FakeControlWorker()
    current = sample()
    controls.tick(env, worker, current, {"y"}, NOW, 0.03)
    assert worker.requests == []
    controls.tick(env, worker, current, {"v", "b"}, NOW + 0.03, 0.03)
    assert worker.requests == ["validation"]
    assert len(env.teleop_calls) == 2
    assert all(action is env.action for action, _ in env.executed)


def test_button_edges_debounce_and_quit_performs_no_actions():
    controls = CalibrationControls("left", 3.0)
    held = sample(x=1, y=1)
    assert controls.edges(held) == {"x", "y"}
    assert controls.edges(held) == set()
    assert controls.edges(sample(), {"v"}) == {"v"}
    assert controls.edges(held) == {"x", "y"}
    env, worker = FakeEnv(), FakeControlWorker()
    assert not controls.tick(env, worker, held, {"q", "x", "y"}, NOW, 0.03)
    assert env.home_calls == []
    assert env.executed == []
    assert worker.requests == []


class MemoryStore:
    def __init__(self):
        self.records = []
        self.events = []

    def event(self, kind, **fields):
        self.events.append((kind, fields))

    def save(self, snapshot, detection, overlay, request, stability):
        self.records.append((snapshot, detection, overlay, request, stability))
        return Path("000001")


def make_worker(monkeypatch, accepted=True):
    monkeypatch.setattr(
        "egomimic.robot.piper_pico.calibration_capture.time.time", lambda: NOW
    )
    monkeypatch.setattr(
        "egomimic.robot.piper_pico.calibration_capture.time.monotonic", lambda: 50.0
    )
    frames, states = camera_data(), history()
    camera = SimpleNamespace(get_last=lambda count: frames)
    arm = SimpleNamespace(get_all_state=lambda: states)
    detection = {
        "accepted": accepted,
        "corner_count": 25 if accepted else 3,
        "reprojection_rms_px": 0.2,
        "reason": "通过" if accepted else "角点不足",
    }
    estimator = SimpleNamespace(
        detect=lambda image: detection, annotate=lambda image, result: image.copy()
    )
    store = MemoryStore()
    worker = CaptureWorker(camera, arm, estimator, store, CaptureLimits())
    assert worker.request("train")
    worker.pending["timestamp"] = NOW - 1.0
    worker.pending["monotonic"] = 49.0
    worker._post_check = lambda snapshot: {"ok": True, "reason": "稳定"}
    return worker, frames, states


def test_worker_keeps_the_frozen_pose_when_detection_changes_live_buffers(monkeypatch):
    worker, frames, states = make_worker(monkeypatch)
    original_detect = worker.estimator.detect

    def detect_then_change_live_data(image):
        states["ee_pose"][-1, 0] = 0.9
        frames["color"][:] = 255
        return original_detect(image)

    worker.estimator.detect = detect_then_change_live_data
    worker._process()
    snapshot = worker.store.records[0][0]
    assert snapshot["state"]["ee_pose"][0] == pytest.approx(0.2)
    assert snapshot["T_base_end"][0, 3] == pytest.approx(0.2)
    np.testing.assert_array_equal(snapshot["image"][0, 0], [10, 20, 30])
    assert worker.saved_count == 1
    assert worker.counts == {"train": 1, "validation": 0}
    assert worker.pending is None


def test_worker_rejects_bad_board_without_saving_or_counting(monkeypatch):
    worker, _, _ = make_worker(monkeypatch, accepted=False)
    worker._process()
    assert worker.store.records == []
    assert worker.saved_count == 0
    assert worker.counts == {"train": 0, "validation": 0}
    assert worker.pending is None
    assert "角点不足" in worker.status


def test_old_image_has_explicit_timeout_reason_even_when_arm_stable(monkeypatch):
    worker, frames, _ = make_worker(monkeypatch)
    frames["timestamp"][:] = NOW - 2
    worker.pending["monotonic"] = 40.0
    worker._process()
    assert worker.saved_count == 0
    assert "新图像" in worker.status


@pytest.mark.parametrize("phase", ["detect", "post"])
def test_cancellation_before_commit_discards_the_pending_pair(monkeypatch, phase):
    worker, _, _ = make_worker(monkeypatch)
    if phase == "detect":
        original_detect = worker.estimator.detect

        def cancel_during_detection(image):
            worker.cancel("操作员取消")
            return original_detect(image)

        worker.estimator.detect = cancel_during_detection
    else:

        def cancel_during_postcheck(snapshot):
            worker.cancel("操作员取消")
            return {"ok": True}

        worker._post_check = cancel_during_postcheck
    worker._process()
    assert worker.store.records == []
    assert worker.saved_count == 0
    assert worker.pending is None


def test_worker_rejects_busy_or_home_requests_without_queueing(monkeypatch):
    worker, _, _ = make_worker(monkeypatch)
    first = worker.pending
    assert not worker.request("validation")
    assert worker.pending is first
    assert not worker.undo()
    worker.set_enabled(False)
    assert worker.pending is None
    assert not worker.request("train")
    worker.set_enabled(True)
    assert worker.request("validation")
    assert worker.pending["split"] == "validation"


def test_saving_runs_outside_the_control_lock(monkeypatch):
    worker, _, _ = make_worker(monkeypatch)
    original_save = worker.store.save

    def save_with_control_activity(*args):
        # These methods take the worker lock. Saving under that lock would
        # deadlock the control thread's home/cancel path.
        acquired = worker.lock.acquire(blocking=False)
        assert acquired
        worker.lock.release()
        worker.set_enabled(False)
        return original_save(*args)

    worker.store.save = save_with_control_activity
    worker._process()
    assert worker.saved_count == 1
    assert len(worker.store.records) == 1
    assert worker.pending is None
    assert any(
        kind == "capture_saved_before_cancellation" for kind, _ in worker.store.events
    )
