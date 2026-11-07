"""Dataset playback acceptance: robot-only, units, time, arrival and failure paths."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_pico.dataset_replay import (
    PiperReplayDevice,
    ReplayOptions,
    SimulatedDevice,
    VirtualClock,
    approach_path,
    discover_episodes,
    load_episode,
    pose_errors,
    replay_episode,
)
from egomimic.scripts.replay_piper_dataset import main


def write_episode(path, *, embodiment="piper_bimanual", x=0.2):
    store = zarr.open_group(str(path), mode="w")
    count, padded = 5, 8
    store.attrs.update(
        {
            "embodiment": embodiment,
            "schema": "piper_base_v3",
            "total_frames": count,
            "pose_coordinates": {s: f"piper_{s}_base" for s in ("left", "right")},
            "pose_representation": "xyz_m+quat_wxyz",
            "tool_frame": "piper_sdk_end_pose; no additional TCP offset applied",
            "gripper_encoding": {
                "unit": "normalized_width",
                "closed_m": 0,
                "open_m": 0.08,
            },
            "calibration_status": "missing",
            "calibration": None,
            "source_capture_config": {"replay_attributes": {"move_spd_rate": 50}},
        }
    )

    def put(name, values):
        values = np.asarray(values, dtype=float)
        result = np.full((padded, *values.shape[1:]), np.nan)
        result[:count] = values
        store.create_array(name, data=result)

    tau = 1000 + np.array([0, 0.03, 0.065, 0.1, 0.132])
    put("timestamp", tau)
    put("source.decision_timestamp", tau + 0.002)
    for i, side in enumerate(("left", "right")):
        rotation = Rotation.from_euler("xyz", [0.2, -0.4, 0.6]).as_quat()[[3, 0, 1, 2]]
        pose = np.tile(np.r_[x + i * 0.05, 0, 0.3, rotation], (count, 1))
        pose[:, 0] += np.arange(count) * 0.002
        for kind in ("cmd", "obs"):
            put(f"{side}.{kind}_ee_pose", pose)
            width = np.linspace(0.2, 0.8, count)[:, None]
            put(f"{side}.{kind}_gripper", width)
            put(f"{side}.{kind}_gripper_width_m", width * 0.08)
    return path


@pytest.fixture
def recorded(tmp_path):
    return write_episode(tmp_path / "batch" / "episode_000000.zarr")


def test_load_uncalibrated_robot_quaternions_grippers_time_and_padding(recorded):
    ep = load_episode(recorded, ReplayOptions())
    assert ep.actions.shape == (5, 14)  # Storage contains eight rows.
    np.testing.assert_allclose(ep.actions[:, 6], np.linspace(0.016, 0.064, 5))
    np.testing.assert_allclose(ep.offsets, [0, 0.03, 0.065, 0.1, 0.132])
    expected = Rotation.from_euler("xyz", [0.2, -0.4, 0.6]).as_matrix()
    np.testing.assert_allclose(
        Rotation.from_rotvec(ep.actions[0, 3:6]).as_matrix(), expected
    )
    np.testing.assert_allclose(ep.actions[:, 0], 0.2 + np.arange(5) * 0.002)
    assert ep.move_spd_rate == 50


@pytest.mark.parametrize(
    "key,value",
    [
        ("embodiment", "human_bimanual"),
        ("embodiment", "eva_bimanual"),
        ("schema", "pico_ego_v1"),
        ("pose_representation", "xyz+rotvec"),
        ("pose_coordinates", {"left": "camera", "right": "camera"}),
        ("tool_frame", "virtual_grasp_center"),
        ("total_frames", True),
        ("gripper_encoding", {"unit": "normalized_width", "closed_m": 0, "open_m": 80}),
    ],
)
def test_incompatible_metadata_rejected(recorded, key, value):
    zarr.open_group(str(recorded), mode="a").attrs[key] = value
    with pytest.raises(ValueError):
        load_episode(recorded, ReplayOptions())


@pytest.mark.parametrize(
    "key,index,value",
    [
        ("source.decision_timestamp", 2, 1000.02),
        ("source.decision_timestamp", 4, 1001.0),
        ("left.cmd_ee_pose", (1, 3), 0),
        ("left.cmd_ee_pose", (1, 0), np.nan),
        ("right.cmd_ee_pose", (2, 0), 0.8),
        ("right.cmd_gripper", (2, 0), 1.1),
        ("left.cmd_gripper_width_m", (2, 0), 0.079),
    ],
)
def test_invalid_trajectory_rejected_before_motion(recorded, key, index, value):
    zarr.open_group(str(recorded), mode="a")[key][index] = value
    with pytest.raises(ValueError):
        load_episode(recorded, ReplayOptions())


def test_rotation_jump_rejected(recorded):
    g = zarr.open_group(str(recorded), mode="a")
    g["left.cmd_ee_pose"][2, 3:] = Rotation.from_euler("y", 2).as_quat()[[3, 0, 1, 2]]
    with pytest.raises(ValueError, match="rotation step"):
        load_episode(recorded, ReplayOptions())


def test_discover_nested_batches_sorted_unique_and_never_skip_ego(recorded, tmp_path):
    second = write_episode(
        tmp_path / "batch2" / "episode_000000.zarr", embodiment="human_bimanual"
    )
    assert discover_episodes([tmp_path]) == [recorded, second]
    with pytest.raises(ValueError, match="duplicates"):
        discover_episodes([recorded.parent, recorded])
    with pytest.raises(ValueError, match="Ego"):
        load_episode(second, ReplayOptions())


def test_all_preflight_before_hardware_even_if_later_episode_is_ego(
    recorded, tmp_path, monkeypatch
):
    other = write_episode(tmp_path / "ego.zarr", embodiment="human_bimanual")
    called = []
    monkeypatch.setattr(PiperReplayDevice, "open", lambda self: called.append(True))
    with pytest.raises(ValueError, match="Ego"):
        main(["--input", str(recorded), str(other), "--execute"])
    assert not called


def test_approach_shortest_rotation_and_all_speed_bounds():
    opts = ReplayOptions(approach_duration=0.1)
    current, target = np.zeros(14), np.zeros(14)
    current[[5, 12]], target[[5, 12]] = np.deg2rad(179), np.deg2rad(-179)
    target[[0, 7]], target[[6, 13]] = 0.3, 0.08
    rows, times = approach_path(current, target, opts)
    np.testing.assert_allclose(pose_errors(rows[-1], target), 0, atol=1e-12)
    assert times[-1] == pytest.approx(9.0)
    steps = np.array([pose_errors(b, a) for a, b in zip(rows, rows[1:])])
    speeds = steps / np.diff(times)[:, None, None]
    assert np.max(speeds[:, :, 0]) <= opts.approach_linear_speed + 1e-10
    assert np.max(speeds[:, :, 1]) <= opts.approach_angular_speed + 1e-10
    assert np.max(speeds[:, :, 2]) <= opts.approach_gripper_speed + 1e-10
    assert np.sum(steps[:, 0, 1]) == pytest.approx(np.deg2rad(2))


def execute(ep, *, options=None, device=None, clock=None, report=None):
    options = options or ReplayOptions(approach_duration=0.1, settle_duration=0.04)
    device = device or SimulatedDevice(ep.initial_state)
    clock = clock or VirtualClock()
    report = {} if report is None else report
    replay_episode(ep, device, options, clock, report, notify=lambda x: None)
    return report, device, clock


def test_each_episode_returns_to_its_own_first_command_and_slows_time_only(recorded):
    ep = load_episode(recorded, ReplayOptions())
    opts = ReplayOptions(speed=0.5, approach_duration=0.1, settle_duration=0.04)
    report, device, clock = execute(ep, options=opts)
    second_actions = ep.actions.copy()
    second_actions[:, [0, 7]] += 0.2
    second = replace(ep, path=Path("second.zarr"), actions=second_actions)
    before = len(device.sent)
    report2, _, _ = execute(second, device=device, clock=clock, options=opts)
    for r in (report, report2):
        assert r["status"] == "completed"
        assert r["playback"]["count"] == 5
        assert r["playback"]["actual_span_s"] == pytest.approx(0.264)
        assert r["start_arrival"]["wait_s"] >= 0.04
    np.testing.assert_allclose(device.sent[before], ep.actions[-1])
    playback_start = before + report2["approach"]["count"]
    np.testing.assert_allclose(device.sent[playback_start], second.actions[0])
    np.testing.assert_allclose(device.sent[-5:], second.actions)
    assert device.holds == 2


def test_never_replay_until_both_arms_arrive(recorded):
    ep = load_episode(recorded, ReplayOptions())
    initial = ep.initial_state.copy()
    initial[7] -= 0.1

    class StuckRight(SimulatedDevice):
        def send(self, row):
            old = self.state[7:].copy()
            super().send(row)
            self.state[7:] = old

    device, report = StuckRight(initial), {}
    with pytest.raises(RuntimeError, match="Arrival timed out"):
        execute(
            ep,
            device=device,
            report=report,
            options=ReplayOptions(settle_timeout=0.15, settle_duration=0.04),
        )
    assert "playback" not in report
    assert device.holds == 1


@pytest.mark.parametrize("error", [RuntimeError("CAN failed"), KeyboardInterrupt()])
def test_send_failure_and_interrupt_hold_and_never_continue(recorded, error):
    ep = load_episode(recorded, ReplayOptions())

    class FailingDevice(SimulatedDevice):
        def send(self, row):
            if len(self.sent) == 2:
                raise error
            super().send(row)

    device, report = FailingDevice(ep.initial_state), {}
    with pytest.raises(type(error)):
        execute(ep, device=device, report=report)
    assert device.holds == 1
    assert report["status"] in ("failed", "interrupted")
    assert "playback" not in report


def test_scheduler_stall_aborts_without_catchup(recorded):
    ep = load_episode(recorded, ReplayOptions())

    class StallClock(VirtualClock):
        def sleep(self, seconds):
            self.now += seconds + 0.5

    report, device = {}, SimulatedDevice(ep.initial_state)
    with pytest.raises(RuntimeError, match="schedule slip"):
        execute(ep, report=report, device=device, clock=StallClock())
    assert len(device.sent) == 1
    assert device.holds == 1


def test_cli_default_no_can_completes_and_writes_report(
    recorded, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        PiperReplayDevice, "open", lambda self: pytest.fail("CAN opened")
    )
    output = tmp_path / "report.json"
    assert main(["--input", str(recorded.parent), "--report", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["mode"] == "dry_run_no_CAN"
    assert report["episodes"][0]["playback"]["count"] == 5
    with pytest.raises(FileExistsError):
        main(["--input", str(recorded), "--report", str(output)])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speed": 2},
        {"speed": 0},
        {"approach_duration": np.nan},
        {"settle_timeout": 0.1},
    ],
)
def test_bad_options_rejected(kwargs):
    with pytest.raises(ValueError):
        ReplayOptions(**kwargs)


def test_hardware_send_uses_continuous_euler_and_physical_width():
    sends = []
    d = PiperReplayDevice("test0", "test1")
    d.references = [np.array([0, 0, np.deg2rad(179)]) for _ in range(2)]
    d.arms = [
        SimpleNamespace(send_ee_pose=lambda p, g: sends.append((p, g)))
        for _ in range(2)
    ]
    row = np.zeros(14)
    row[[5, 12]], row[[6, 13]] = np.deg2rad(-179), 0.04
    d.send(row)
    assert len(sends) == 2
    for p, g in sends:
        assert p[5] == pytest.approx(np.deg2rad(181))
        assert g == 0.04


def test_hold_clears_both_and_never_sends_stale_feedback(monkeypatch):
    calls = []
    d = PiperReplayDevice("test0", "test1")
    d.arms = [
        SimpleNamespace(
            connected=True,
            name=side,
            clear_trajectory=lambda s=side: calls.append((s, "clear")),
            send_ee_pose=lambda p, g, s=side: calls.append((s, "send")),
            disconnect=lambda stop_motion, s=side: calls.append((s, "disconnect")),
        )
        for side in ("left", "right")
    ]

    def stale(arm):
        raise RuntimeError("stale")

    monkeypatch.setattr(d, "_read_arm", stale)
    with pytest.raises(RuntimeError, match="stale"):
        d.close()
    assert calls == [
        ("left", "clear"),
        ("right", "clear"),
        ("left", "disconnect"),
        ("right", "disconnect"),
    ]


def test_small_sleep_jitter_does_not_accumulate_over_long_recording(recorded):
    ep = load_episode(recorded, ReplayOptions())
    ep = replace(
        ep, actions=np.tile(ep.actions[0], (2000, 1)), offsets=np.arange(2000) / 30
    )

    class JitterClock(VirtualClock):
        def sleep(self, seconds):
            self.now += seconds + 0.0001

    report, _, _ = execute(ep, clock=JitterClock())
    assert report["playback"]["count"] == 2000
    assert report["playback"]["max_slip_s"] < 0.00011
    assert report["playback"]["actual_span_s"] == pytest.approx(1999 / 30, abs=0.00011)


def fake_sdk(stamp=100, estop=False):
    pose = SimpleNamespace(
        time_stamp=stamp,
        end_pose=SimpleNamespace(
            X_axis=200000,
            Y_axis=-10000,
            Z_axis=300000,
            RX_axis=0,
            RY_axis=0,
            RZ_axis=90000,
        ),
    )
    grip = SimpleNamespace(
        time_stamp=stamp, gripper_state=SimpleNamespace(grippers_angle=40000)
    )
    status = SimpleNamespace(
        time_stamp=stamp, arm_status=SimpleNamespace(arm_status=int(estop))
    )
    return SimpleNamespace(
        GetArmEndPoseMsgs=lambda: pose,
        GetArmGripperMsgs=lambda: grip,
        GetArmStatus=lambda: status,
    )


@pytest.mark.parametrize(
    "stamp,estop", [(0, False), (99, False), (101, False), (np.nan, False), (100, True)]
)
def test_stale_missing_future_feedback_or_estop_refused(monkeypatch, stamp, estop):
    monkeypatch.setattr(
        "egomimic.robot.piper_pico.dataset_replay.time.time", lambda: 100
    )
    arm = SimpleNamespace(_require=lambda: fake_sdk(stamp, estop), name="left")
    with pytest.raises(RuntimeError):
        PiperReplayDevice("test0", "test1")._read_arm(arm)


def test_feedback_sdk_units_are_converted_to_metres_and_rotation_vector(monkeypatch):
    monkeypatch.setattr(
        "egomimic.robot.piper_pico.dataset_replay.time.time", lambda: 100
    )
    device = PiperReplayDevice("test0", "test1")
    device.arms = [
        SimpleNamespace(_require=fake_sdk, name=s) for s in ("left", "right")
    ]
    actual = device.read_state()
    np.testing.assert_allclose(actual[:7], [0.2, -0.01, 0.3, 0, 0, np.pi / 2, 0.04])


def test_second_port_open_failure_closes_first_port_and_partial_second(monkeypatch):
    calls = []

    class FakeArm:
        def __init__(self, port, name):
            self.port, self.name, self.connected = port, name, False

        def connect(self, enable):
            self.connected = True
            calls.append((self.port, "connect"))
            if self.port == "test1":
                raise RuntimeError("second CAN open failed")

        def clear_trajectory(self):
            calls.append((self.port, "clear"))

        def disconnect(self, stop_motion):
            calls.append((self.port, "disconnect"))
            self.connected = False

    monkeypatch.setattr("egomimic.robot.piper_pico.arm.PiperArm", FakeArm)
    with pytest.raises(RuntimeError, match="second CAN open failed"):
        PiperReplayDevice("test0", "test1").open()
    assert ("test0", "disconnect") in calls
    assert ("test1", "disconnect") in calls


def test_end_arrival_timeout_is_failure_not_silent_success(recorded):
    ep = load_episode(recorded, ReplayOptions())

    class StuckAtEnd(SimulatedDevice):
        def send(self, action):
            if np.allclose(action, ep.actions[-1]):
                self.sent.append(action.copy())
            else:
                super().send(action)

    device, report = StuckAtEnd(ep.initial_state), {}
    with pytest.raises(RuntimeError, match="Arrival timed out"):
        execute(
            ep,
            device=device,
            report=report,
            options=ReplayOptions(settle_timeout=0.15, settle_duration=0.04),
        )
    assert report["playback"]["count"] == len(ep.actions)
    assert report["status"] == "failed"
    assert device.holds == 1
