"""Frame/feedback identity stays attached to the selected sample, without devices."""

from dataclasses import replace
from queue import Empty, Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from egomimic.robot.piper_pico import arm_process, camera_process, env, pico_camera
from egomimic.robot.piper_pico.env import PiperPicoEnv
from egomimic.robot.piper_pico.live_policy_devices import LiveObservationProvider
from egomimic.robot.piper_pico.policy_inference import CAMERA_KEYS, ObservationFrame


def _environment(*, identities=True):
    result = PiperPicoEnv.__new__(PiperPicoEnv)
    result.is_ready = True
    result.frequency, result.horizon = 10.0, 3
    result.obs_accumulator = None
    result.arms, result.cameras = [], []
    result.camera_specs = [
        {"name": name} for name in ("left_wrist", "right_wrist", "front_1")
    ]
    for i in range(3):
        stamps = np.array([9.75, 9.85, 9.85, 9.95, 10.02])
        ids = np.arange(10, 15, dtype=np.int64) + 20 * i
        data = {
            "timestamp": stamps,
            "color": np.broadcast_to(ids[:, None, None, None], (5, 2, 3, 3))
            .astype(np.uint8)
            .copy(),
        }
        if identities:
            data["frame_id"] = ids
            data["camera_receive_timestamp"] = np.array(
                [9.79, 9.89, 9.891, 9.99, 10.06]
            )
        result.cameras.append(
            SimpleNamespace(
                process=SimpleNamespace(exitcode=None),
                get_last=Mock(return_value=data),
            )
        )
    for _ in range(2):
        data = {
            "robot_timestamp": np.array([9.7, 9.99]),
            "ee_pose": np.zeros((2, 6)),
            "joint_pos": np.zeros((2, 6)),
            "gripper_position": np.array([0.04, 0.05]),
            **{
                f"robot_{field}_source_timestamp": np.array([9.7, 9.98])
                for field in ("joint", "pose", "gripper")
            },
        }
        result.arms.append(
            SimpleNamespace(
                process=SimpleNamespace(exitcode=None),
                get_all_state=Mock(return_value=data),
            )
        )
    return result


def _aligned(monkeypatch, *, identities=True):
    monkeypatch.setattr(env.time, "time", lambda: 10.0)
    rig = _environment(identities=identities)
    obs = rig.get_obs()
    frame = ObservationFrame.from_aligned_obs(
        obs, camera_names=[spec["name"] for spec in rig.camera_specs]
    )
    return rig, obs, frame


def test_selected_ids_and_receive_times_follow_exact_image_indices(monkeypatch):
    rig, obs, frame = _aligned(monkeypatch)
    for i, spec in enumerate(rig.camera_specs):
        name = spec["name"]
        # The duplicate 9.85 timestamp selects index 2, not the first match.
        expected = np.array([10, 12, 13]) + 20 * i
        np.testing.assert_array_equal(obs[f"camera{i}_frame_id"], expected)
        np.testing.assert_array_equal(obs[f"camera{i}_rgb"][:, 0, 0, 0], expected)
        np.testing.assert_allclose(
            obs[f"camera{i}_receive_timestamp"], [9.79, 9.891, 9.99]
        )
        assert frame.image_frame_ids[name] == expected[-1]
        assert frame.image_receive_timestamps[name] == 9.99
        assert frame.source_timestamps[name] == 9.95
        rig.cameras[i].get_last.assert_called_once()
        # Copies remain a fixed observation after source arrays are reused.
        source = rig.cameras[i].get_last.return_value
        source["frame_id"][:] = 999
        source["camera_receive_timestamp"][:] = 999
        source["color"][:] = 0
        assert frame.image_frame_ids[name] == expected[-1]
        assert frame.images[name][0, 0, 0] == expected[-1]
        assert obs[f"camera{i}_frame_id"][-1] == expected[-1]
        assert obs[f"camera{i}_receive_timestamp"][-1] == 9.99


def test_legacy_observation_without_identity_still_valid(monkeypatch):
    _, obs, frame = _aligned(monkeypatch, identities=False)
    assert not any(key.endswith("frame_id") for key in obs)
    assert frame.image_frame_ids == frame.image_receive_timestamps == {}
    frame.validate()


def test_provider_source_timestamp_replacement_preserves_frame_identity(monkeypatch):
    rig, _, expected = _aligned(monkeypatch)
    provider = LiveObservationProvider(rig)
    provider.check_health = Mock()
    actual = provider.latest()
    assert actual.source_timestamps["left"] == 9.98
    assert expected.source_timestamps["left"] == 9.99
    assert actual.image_frame_ids == expected.image_frame_ids
    assert actual.image_receive_timestamps == expected.image_receive_timestamps
    for name in CAMERA_KEYS:
        np.testing.assert_array_equal(actual.images[name], expected.images[name])


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("image_frame_ids", {"front_1": 1}, "all three cameras"),
        ("image_receive_timestamps", {"front_1": 1.0}, "all three cameras"),
        ("image_frame_ids", None, "mapping"),
        ("image_receive_timestamps", [], "mapping"),
        *[
            ("image_frame_ids", dict.fromkeys(CAMERA_KEYS, value), "int64")
            for value in (-1, 1.5, True, np.bool_(False), 2**63, "1")
        ],
        *[
            (
                "image_receive_timestamps",
                dict.fromkeys(CAMERA_KEYS, value),
                "finite numbers",
            )
            for value in (np.nan, np.inf, True, "9.9")
        ],
    ],
)
def test_optional_identity_requires_complete_valid_mapping(
    monkeypatch, field, value, match
):
    _, _, frame = _aligned(monkeypatch)
    with pytest.raises(ValueError, match=match):
        replace(frame, **{field: value}).validate()


def test_partial_aligned_identity_is_not_silently_discarded(monkeypatch):
    rig, obs, _ = _aligned(monkeypatch)
    del obs["camera1_frame_id"]
    with pytest.raises(ValueError, match="image_frame_ids must identify all three"):
        ObservationFrame.from_aligned_obs(
            obs, camera_names=[spec["name"] for spec in rig.camera_specs]
        )


def test_camera_publishes_monotonic_ids_despite_read_failure_and_equal_stamps(
    monkeypatch,
):
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.side_effect = {
        cv2.CAP_PROP_FRAME_WIDTH: 3,
        cv2.CAP_PROP_FRAME_HEIGHT: 2,
        cv2.CAP_PROP_FPS: 30,
        cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
    }.__getitem__
    bgr = np.zeros((2, 3, 3), dtype=np.uint8)
    bgr[..., 0], bgr[..., 2] = 7, 19
    capture.read.side_effect = [(False, None), (True, bgr), (True, bgr + 1)]
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: capture)
    monkeypatch.setattr(camera_process, "verify_identity", lambda *_a, **_k: {})
    monkeypatch.setattr(camera_process, "verify_controls", lambda *_a, **_k: {})
    monkeypatch.setattr(camera_process.time, "time", lambda: 10.0)
    monkeypatch.setattr(camera_process.time, "sleep", lambda _: None)
    stop, ready, published = Event(), Event(), []

    def put(row):
        published.append(row)
        if len(published) == 2:
            stop.set()

    camera_process.camera_process_loop(
        "/dev/fake",
        3,
        2,
        30,
        0.08,
        "MJPG",
        SimpleNamespace(put=put),
        Queue(),
        Queue(),
        stop,
        ready,
    )
    assert [row["frame_id"] for row in published] == [0, 1]
    assert all(isinstance(row["frame_id"], np.int64) for row in published)
    assert [row["color"][0, 0, 0] for row in published] == [19, 20]
    assert [row["camera_receive_timestamp"] for row in published] == [10.0, 10.0]
    assert capture.read.call_count == 3
    assert ready.is_set()


def test_arm_ids_advance_only_on_success_and_preserve_cached_can_source_times(
    monkeypatch,
):
    arm = Mock()
    arm.feedback_snapshot.return_value = {
        "joint_pos": np.zeros(6),
        "ee_pose": np.zeros(6),
        "gripper_position": 0.05,
        **{
            f"robot_{field}_source_timestamp": 9.5
            for field in ("joint", "pose", "gripper")
        },
    }
    arm.joint_velocities.return_value = np.zeros(6)
    monkeypatch.setattr(arm_process, "PiperArm", lambda *_a, **_k: arm)
    monkeypatch.setattr(arm_process.time, "time", lambda: 10.0)
    monkeypatch.setattr(arm_process.time, "sleep", lambda _: None)
    stop, ready, attempts, published = Event(), Event(), [], []

    def put(row):
        attempts.append(int(row["feedback_id"]))
        if len(attempts) == 2:
            raise TimeoutError("test publication failure")
        published.append(row)
        if len(published) == 3:
            stop.set()

    arm_process.arm_process_loop(
        "fake",
        50,
        "left",
        SimpleNamespace(put=put),
        SimpleNamespace(get_all=Mock(side_effect=Empty)),
        0.08,
        100,
        ready,
        stop,
        enable_motion=False,
    )
    assert attempts == [0, 1, 1, 2]
    assert [row["feedback_id"] for row in published] == [0, 1, 2]
    for row in published:
        assert isinstance(row["feedback_id"], np.int64)
        assert row["robot_timestamp"] == pytest.approx(9.92)
        assert row["robot_receive_timestamp"] == 10.0
        assert not row["robot_source_timestamps_simulated"]
        for field in ("joint", "pose", "gripper"):
            assert row[f"robot_{field}_source_timestamp"] == 9.5
    arm.send_ee_pose.assert_not_called()
    arm.send_joints.assert_not_called()
    arm.disconnect.assert_called_once_with(stop_motion=False)


def test_pico_eye_ids_keep_shared_schema_and_do_not_advance_on_failed_put():
    rows = []
    ring = SimpleNamespace(put=rows.append)
    bgr = np.zeros((2, 3, 3), dtype=np.uint8)
    pico_camera._put_eye(ring, bgr, 10.0, 0.1, 3, 2)
    ring.put = Mock(side_effect=TimeoutError("test publication failure"))
    with pytest.raises(TimeoutError):
        pico_camera._put_eye(ring, bgr, 10.0, 0.1, 3, 2)
    ring.put = rows.append
    pico_camera._put_eye(ring, bgr, 10.0, 0.1, 3, 2)
    assert [row["frame_id"] for row in rows] == [0, 1]
    assert all(set(row) == set(camera_process.camera_examples(2, 3)) for row in rows)
    assert all(isinstance(row["frame_id"], np.int64) for row in rows)


class _ConcurrentCopyRing:
    """One controlled concurrent writer advance during get_last_k's copy."""

    get_max_k = 4
    buffer_size = 8
    get_time_budget = 0.2

    def __init__(self, *, before=10, after=10):
        self.count, self.after = before, after
        self.requests = []

    def get_last_k(self, k):
        self.requests.append(k)
        self.count = self.after
        ids = np.arange(self.after - k, self.after, dtype=np.int64)
        return {
            "frame_id": ids,
            "color": np.broadcast_to(ids[:, None, None, None], (k, 2, 3, 3))
            .astype(np.uint8)
            .copy(),
        }


def _camera_reader(kind, ring):
    if kind == "uvc":
        camera = camera_process.CameraProcess.__new__(camera_process.CameraProcess)
        camera.name, camera.ring = "front_1", ring
        return camera
    eye = pico_camera.EyeView("pico_left", ring)
    if kind == "pico_eye":
        return eye
    camera = pico_camera.PicoCameraProcess.__new__(pico_camera.PicoCameraProcess)
    camera.view = eye
    return camera


@pytest.mark.parametrize("kind", ["uvc", "pico_eye", "pico_camera"])
@pytest.mark.parametrize("advance", [0, 1, 4])
def test_camera_copy_accepts_normal_concurrent_publications(monkeypatch, kind, advance):
    ring = _ConcurrentCopyRing(after=10 + advance)
    times = iter([100.0, 100.01])
    monkeypatch.setattr(camera_process.time, "monotonic", lambda: next(times))
    snapshot = _camera_reader(kind, ring).get_last(3)
    assert ring.requests == [3]
    np.testing.assert_array_equal(
        snapshot["frame_id"], np.arange(7 + advance, 10 + advance)
    )
    np.testing.assert_array_equal(snapshot["color"][:, 0, 0, 0], snapshot["frame_id"])


@pytest.mark.parametrize("kind", ["uvc", "pico_eye", "pico_camera"])
@pytest.mark.parametrize(
    "after,elapsed,message",
    [(10, 0.21, "read budget"), (15, 0.01, "overwritten"), (9, 0.01, "reset")],
)
def test_camera_copy_rejects_stalls_inflight_overwrite_and_counter_reset(
    monkeypatch, kind, after, elapsed, message
):
    ring = _ConcurrentCopyRing(after=after)
    times = iter([100.0, 100.0 + elapsed])
    monkeypatch.setattr(camera_process.time, "monotonic", lambda: next(times))
    # Before=10 and k=3: oldest possible ID=7. Count=15 means the writer
    # can already be overwriting slot 7 even before publishing count=16.
    with pytest.raises(RuntimeError, match=message):
        _camera_reader(kind, ring).get_last(3)
    assert ring.requests == [3]  # No retry, lock or wait on the producer.


@pytest.mark.parametrize("kind", ["uvc", "pico_eye", "pico_camera"])
def test_camera_copy_limits_to_available_history(monkeypatch, kind):
    ring = _ConcurrentCopyRing(before=2, after=3)
    monkeypatch.setattr(camera_process.time, "monotonic", lambda: 100.0)
    snapshot = _camera_reader(kind, ring).get_last(99)
    assert ring.requests == [2]
    np.testing.assert_array_equal(snapshot["frame_id"], [1, 2])


@pytest.mark.parametrize("kind", ["uvc", "pico_eye", "pico_camera"])
def test_camera_copy_empty_ring_fails_without_reading(monkeypatch, kind):
    ring = _ConcurrentCopyRing(before=0, after=0)
    monkeypatch.setattr(camera_process.time, "monotonic", lambda: 100.0)
    with pytest.raises(RuntimeError, match="has no frames"):
        _camera_reader(kind, ring).get_last(3)
    assert ring.requests == []
