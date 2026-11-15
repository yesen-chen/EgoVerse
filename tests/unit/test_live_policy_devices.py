"""Live device contracts tested without SDK, CAN, camera, GUI, or subprocesses."""

import json
import sys
import threading
import time
from queue import Empty
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from egomimic.robot.piper_pico import arm_process as worker
from egomimic.robot.piper_pico.arm import PiperArm
from egomimic.robot.piper_pico.env import PiperPicoEnv
from egomimic.robot.piper_pico.live_policy_devices import (
    LiveActionSink,
    LiveObservationProvider,
)


def feedback():
    now = time.time()
    return {
        "joint_pos": worker.HOME_QPOS.copy(),
        "ee_pose": np.arange(6) / 10,
        "gripper_position": 0.08,
        "robot_receive_timestamp": now,
        **{
            f"robot_{field}_source_timestamp": now
            for field in ("joint", "pose", "gripper")
        },
        "last_command_id": -1,
        "last_command_send_timestamp": np.nan,
    }


def fake_device(name):
    device = SimpleNamespace(name=name, process=SimpleNamespace(pid=123, exitcode=None))
    device.get_state = Mock(side_effect=feedback)
    device.held_event = threading.Event()
    device.request_hold = Mock(side_effect=device.held_event.set)
    return device


def fake_env(execute=True):
    arms = [fake_device("left"), fake_device("right")]
    return SimpleNamespace(
        enable_motion=execute,
        arms=arms,
        exec_actions=Mock(),
        home=Mock(),
        seed_clutches_from_measured=Mock(),
    )


def test_rtc_health_readers_have_thread_owned_camera_buffers():
    from concurrent.futures import ThreadPoolExecutor

    barrier = threading.Barrier(2)
    per_thread = {}

    class Ring:
        count = 1
        get_time_budget = 5.0
        buffer_size = 32

        def get(self, out=None):
            value = out if out is not None else {"timestamp": np.array(time.time())}
            value["timestamp"][...] = time.time()
            per_thread.setdefault(threading.get_ident(), []).append(value)
            return value

    camera = fake_device("front_1")
    camera.ring = Ring()
    env = fake_env(False)
    env.cameras, env.camera_specs = [camera], [{"name": "front_1"}]
    provider = LiveObservationProvider(env)

    def read_twice():
        provider.check_health()
        barrier.wait(timeout=5)
        provider.check_health()

    with ThreadPoolExecutor(2) as pool:
        a, b = pool.submit(read_twice), pool.submit(read_twice)
        a.result(timeout=10)
        b.result(timeout=10)
    copies = list(per_thread.values())
    assert len(copies) == 2
    assert all(values[0] is values[1] for values in copies)
    assert copies[0][0] is not copies[1][0]


@pytest.mark.parametrize("enable", [False, True])
def test_sdk_read_only_connect_skips_initialization_queries(monkeypatch, enable):
    sdk = Mock()
    monkeypatch.setitem(
        sys.modules,
        "piper_sdk",
        SimpleNamespace(C_PiperInterface_V2=Mock(return_value=sdk)),
    )
    arm = PiperArm("unused")
    arm.ensure_enabled = Mock()
    arm.connect(enable=enable)
    if enable:
        sdk.ConnectPort.assert_called_once_with()
        arm.ensure_enabled.assert_called_once()
    else:
        sdk.ConnectPort.assert_called_once_with(piper_init=False)
        arm.ensure_enabled.assert_not_called()
    arm.disconnect(stop_motion=False)
    assert [call[0] for call in sdk.mock_calls] == ["ConnectPort", "DisconnectPort"]


def test_connect_forwards_preserve_gripper_flag(monkeypatch):
    sdk = Mock()
    monkeypatch.setitem(
        sys.modules,
        "piper_sdk",
        SimpleNamespace(C_PiperInterface_V2=Mock(return_value=sdk)),
    )
    arm = PiperArm("unused")
    arm.ensure_enabled = Mock()
    arm.connect(enable=True, preserve_gripper_on_enable=True)
    arm.ensure_enabled.assert_called_once_with(
        timeout=10.0, preserve_gripper_on_enable=True
    )


@pytest.mark.parametrize("preserve, expected", [(False, 0), (True, 42000)])
def test_enable_preserves_measured_gripper_only_when_requested(
    monkeypatch, preserve, expected
):
    from egomimic.robot.piper_pico import arm as arm_module

    sdk = Mock()
    sdk.GetArmGripperMsgs.return_value = SimpleNamespace(
        time_stamp=time.time(),
        gripper_state=SimpleNamespace(grippers_angle=42000),
    )
    arm = PiperArm("unused").bind(sdk)
    arm.is_estopped = lambda: False
    arm.all_motors_enabled = Mock(side_effect=[False, True])
    monkeypatch.setattr(arm_module.time, "sleep", lambda _: None)
    arm.ensure_enabled(preserve_gripper_on_enable=preserve)
    sdk.EnableArm.assert_called_once_with(7)
    assert sdk.GripperCtrl.call_args.args[0] == expected
    if not preserve:
        sdk.GetArmGripperMsgs.assert_not_called()


@pytest.mark.parametrize("stamp_offset", [float("nan"), -10.0])
def test_enable_waits_then_fails_without_closing_on_invalid_gripper_feedback(
    monkeypatch, stamp_offset
):
    from egomimic.robot.piper_pico import arm as arm_module

    sdk = Mock()
    sdk.GetArmGripperMsgs.return_value = SimpleNamespace(
        time_stamp=time.time() + stamp_offset,
        gripper_state=SimpleNamespace(grippers_angle=42000),
    )
    arm = PiperArm("unused").bind(sdk)
    arm.is_estopped, arm.all_motors_enabled = lambda: False, lambda: False
    ticks = iter([0.0, 0.0, 0.1, 0.3])
    monkeypatch.setattr(arm_module.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(arm_module.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="did not enable"):
        arm.ensure_enabled(timeout=0.2, preserve_gripper_on_enable=True)
    sdk.EnableArm.assert_not_called()
    sdk.GripperCtrl.assert_not_called()


class WorkerArm:
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.connected = False

    def connect(self, *, enable):
        self.connected = True
        self.calls.append(("connect", enable))

    def disconnect(self, *, stop_motion):
        self.calls.append(("disconnect", stop_motion))

    def feedback_snapshot(self):
        return feedback()

    def joint_velocities(self):
        return np.zeros(6)

    def clear_trajectory(self):
        self.calls.append(("clear",))

    def send_ee_pose(self, pose, gripper):
        self.calls.append(("pose", np.asarray(pose).copy(), gripper))

    def send_joints(self, joints, gripper):
        self.calls.append(("joints", np.asarray(joints).copy(), gripper))


@pytest.mark.parametrize("command", [worker.CMD_WAYPOINT, worker.CMD_HOME])
@pytest.mark.parametrize("read_fails", [False, True])
def test_read_only_worker_never_writes_even_with_injected_commands(
    monkeypatch, command, read_fails
):
    arm, stop = WorkerArm(), threading.Event()
    monkeypatch.setattr(worker, "PiperArm", lambda *a, **kw: arm)
    batch = {
        key: np.asarray([value]) for key, value in worker.command_examples().items()
    }
    batch["cmd"][0] = command
    batch["target_time"][0] = time.time() + 10
    commands = SimpleNamespace(get_all=lambda: batch)

    def publish(state):
        stop.set()
        if read_fails:
            raise RuntimeError("injected read failure")

    worker.arm_process_loop(
        "unused",
        50,
        "test",
        SimpleNamespace(put=publish),
        commands,
        0,
        100,
        threading.Event(),
        stop,
        enable_motion=False,
    )
    assert arm.calls == [("connect", False), ("disconnect", False)]


def test_hold_clears_pending_targets_and_holds_measured_pose(monkeypatch):
    arm, stop, hold, held = (
        WorkerArm(),
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    hold.set()
    monkeypatch.setattr(worker, "PiperArm", lambda *a, **kw: arm)
    commands = SimpleNamespace(clear=Mock(), get_all=Mock(side_effect=Empty))
    worker.arm_process_loop(
        "unused",
        50,
        "test",
        SimpleNamespace(put=lambda state: stop.set()),
        commands,
        0,
        100,
        threading.Event(),
        stop,
        hold_event=hold,
        held_event=held,
    )
    commands.clear.assert_called_once()
    assert held.is_set() and not hold.is_set()
    assert not any(call[0] == "joints" for call in arm.calls)
    pose_calls = [call for call in arm.calls if call[0] == "pose"]
    assert pose_calls
    np.testing.assert_allclose(pose_calls[0][1], feedback()["ee_pose"])
    assert pose_calls[0][2] == 0.08
    assert arm.calls[1][0] == "clear"


def test_stale_feedback_is_cleared_but_never_sent_as_hold():
    arm = WorkerArm()
    stale = feedback()
    stale["robot_pose_source_timestamp"] = time.time() - 10
    arm.feedback_snapshot = lambda: stale
    with pytest.raises(RuntimeError, match="stale"):
        worker.hold_fresh_measured_pose(arm)
    assert arm.calls == [("clear",)]


def test_simulated_worker_can_hold_and_shutdown_without_hardware():
    stop, ready = threading.Event(), threading.Event()
    samples = []

    def publish(state):
        samples.append(state)
        stop.set()

    worker.arm_process_loop(
        "unused",
        50,
        "sim",
        SimpleNamespace(put=publish),
        SimpleNamespace(get_all=Mock(side_effect=Empty)),
        0,
        100,
        ready,
        stop,
        simulate=True,
    )
    assert ready.is_set()
    assert samples[0]["robot_source_timestamps_simulated"] == 1
    np.testing.assert_array_equal(samples[0]["joint_pos"], worker.HOME_QPOS)


def test_parent_motion_guards_reject_home_and_actions():
    arm = worker.PiperArmProcess.__new__(worker.PiperArmProcess)
    arm.enable_motion, arm.name, arm.commands = False, "left", Mock()
    for operation in (
        lambda: arm.schedule_waypoint(np.zeros(6), 0, time.time() + 1),
        arm.go_home,
        arm.request_hold,
    ):
        with pytest.raises(RuntimeError, match="disabled"):
            operation()
    arm.commands.put.assert_not_called()
    env = PiperPicoEnv.__new__(PiperPicoEnv)
    env.enable_motion, env.is_ready = False, True
    with pytest.raises(RuntimeError, match="disabled"):
        env.exec_actions(np.zeros(14), np.array([time.time() + 1]))
    with pytest.raises(RuntimeError, match="disabled"):
        env.home()


def test_arm_process_uses_spawn_for_events_and_worker(monkeypatch):
    import multiprocessing as mp

    context = SimpleNamespace(Event=Mock(side_effect=threading.Event), Process=Mock())
    get_context = Mock(return_value=context)
    monkeypatch.setattr(mp, "get_context", get_context)
    monkeypatch.setattr(worker.SharedMemoryRingBuffer, "create_from_examples", Mock())
    monkeypatch.setattr(worker.SharedMemoryQueue, "create_from_examples", Mock())
    arm = worker.PiperArmProcess(object(), "unused", "left", enable_motion=False)
    get_context.assert_called_once_with("spawn")
    assert context.Event.call_count == 4
    kwargs = context.Process.call_args.kwargs["kwargs"]
    assert kwargs["enable_motion"] is False
    assert kwargs["hold_event"] is arm.hold_event


def test_no_pico_constructor_or_start_when_disabled(monkeypatch, tmp_path):
    from egomimic.robot.piper_pico import env as env_module

    pico = Mock(side_effect=AssertionError("PICO must not be constructed"))
    monkeypatch.setattr(env_module, "PicoProcess", pico)
    monkeypatch.setattr(
        env_module,
        "ReplayBuffer",
        lambda *args: SimpleNamespace(root=SimpleNamespace(attrs={})),
    )
    constructors = []

    def make_arm(*args, **kwargs):
        constructors.append(kwargs)
        device = fake_device(kwargs["name"])
        device.start, device.stop = Mock(), Mock()
        return device

    monkeypatch.setattr(env_module, "PiperArmProcess", make_arm)
    env = PiperPicoEnv(
        object(),
        tmp_path,
        simulate=True,
        enable_pico=False,
        enable_motion=False,
        preserve_gripper_on_enable=True,
    )
    env._wait_ready = Mock()
    env.seed_clutches_from_measured = Mock()
    env.start()
    env.stop()
    assert env.pico is None
    assert all(kwargs["enable_motion"] is False for kwargs in constructors)
    assert all(kwargs["preserve_gripper_on_enable"] is True for kwargs in constructors)
    pico.assert_not_called()


def test_dry_sink_never_touches_devices_including_hold():
    env = fake_env(execute=False)
    sink = LiveActionSink(env)
    sink.send_action(np.zeros(14), time.time() + 1)
    sink.hold("pause")
    with pytest.raises(RuntimeError, match="observation mode"):
        sink.home()
    env.exec_actions.assert_not_called()
    env.home.assert_not_called()
    for arm in env.arms:
        arm.get_state.assert_not_called()
        arm.request_hold.assert_not_called()
    assert len(sink.records) == 1
    assert sink.last_record is sink.records[-1]
    for _ in range(130):
        sink.send_action(np.zeros(14), time.time() + 1)
        sink.hold("pause")
    assert len(sink.records) == len(sink.hold_reasons) == 128


def test_live_send_requires_actual_worker_receipts():
    env = fake_env()

    def execute(row, deadlines):
        sent = time.time()
        for arm in env.arms:
            arm.get_state.side_effect = lambda: {
                **feedback(),
                "last_command_id": 7,
                "last_command_send_timestamp": sent,
            }
        return np.array([7])

    env.exec_actions.side_effect = execute
    sink = LiveActionSink(env, execute=True)
    sink.send_action(np.zeros(14), time.time() + 1)
    assert len(sink.records[0]["receipts"]) == 2
    assert all(receipt["command_id"] == 7 for receipt in sink.records[0]["receipts"])
    assert all(
        receipt["measured_feedback"]["joint_vel_rad_s"] is None
        for receipt in sink.records[0]["receipts"]
    )


def test_receipts_copy_existing_feedback_with_units_and_original_timestamps():
    env = fake_env()
    queued = time.time()
    states = []
    for index, arm in enumerate(env.arms):
        state = feedback()
        state.update(
            ee_pose=np.arange(6, dtype=np.float64) + index,
            joint_vel=np.arange(6, dtype=np.float64) / 100,
            gripper_position=np.float64(0.04 + index * 0.01),
            last_command_id=7,
            last_command_send_timestamp=queued + 0.001,
            robot_pose_source_timestamp=queued - 0.01,
        )
        states.append(state)
        arm.get_state.side_effect = None
        arm.get_state.return_value = state
    sink = LiveActionSink(env, execute=True)
    receipts = sink._wait_receipts([7, 7], queued_at=queued)

    # Logging reuses the poll's snapshot: it must not introduce another read.
    for arm, state, receipt in zip(env.arms, states, receipts):
        arm.get_state.assert_called_once_with()
        measured = receipt["measured_feedback"]
        assert measured["schema"] == "piper_measured_feedback_v1"
        assert measured["frame"] == "arm_base"
        assert measured["rotation_convention"] == "extrinsic_xyz_rpy"
        np.testing.assert_array_equal(
            measured["ee_pose_xyz_rpy_m_rad"], state["ee_pose"]
        )
        np.testing.assert_array_equal(measured["joint_pos_rad"], state["joint_pos"])
        np.testing.assert_array_equal(measured["joint_vel_rad_s"], state["joint_vel"])
        assert measured["gripper_position_m"] == state["gripper_position"]
        for key in (
            "robot_receive_timestamp",
            "robot_joint_source_timestamp",
            "robot_pose_source_timestamp",
            "robot_gripper_source_timestamp",
        ):
            assert measured[key] == state[key]
        assert measured["robot_pose_source_timestamp"] < receipt["host_send_timestamp"]

    original = json.dumps(receipts, allow_nan=False, sort_keys=True)
    for state in states:
        for key in ("ee_pose", "joint_pos", "joint_vel"):
            state[key][...] = 999
        state["robot_pose_source_timestamp"] = -1
    assert json.dumps(receipts, allow_nan=False, sort_keys=True) == original


@pytest.mark.parametrize("velocity", [np.full(6, np.nan), np.zeros(5)])
def test_receipt_optional_invalid_velocity_is_null(velocity):
    env = fake_env()
    queued = time.time()
    for arm in env.arms:
        arm.get_state.side_effect = None
        arm.get_state.return_value = {
            **feedback(),
            "joint_vel": velocity,
            "last_command_id": 7,
            "last_command_send_timestamp": queued + 0.001,
        }
    receipts = LiveActionSink(env, execute=True)._wait_receipts(
        [7, 7], queued_at=queued
    )
    assert all(r["measured_feedback"]["joint_vel_rad_s"] is None for r in receipts)
    json.dumps(receipts, allow_nan=False)


def test_queue_acceptance_is_not_execution_confirmation():
    env = fake_env()
    env.exec_actions.return_value = np.array([7])
    sink = LiveActionSink(env, execute=True, command_timeout_s=0.005)
    with pytest.raises(TimeoutError, match="receipt"):
        sink.send_action(np.zeros(14), time.time() + 1)
    assert not sink.records


def test_hold_attempts_other_arm_when_one_process_is_dead():
    env = fake_env()
    env.arms[0].process.exitcode = 1
    sink = LiveActionSink(env, execute=True)
    with pytest.raises(RuntimeError, match="left"):
        sink.hold("failure")
    env.arms[1].request_hold.assert_called_once()


def test_home_is_cancelled_and_held():
    env = fake_env()
    env.home.return_value = np.array([2, 2])
    sink = LiveActionSink(env, execute=True)
    with pytest.raises(InterruptedError, match="cancelled"):
        sink.home(should_stop=lambda: True)
    for arm in env.arms:
        arm.request_hold.assert_called_once()


def test_home_waits_for_measured_home_then_holds(monkeypatch):
    from egomimic.robot.piper_pico import live_policy_devices as live

    env = fake_env()
    env.home.return_value = np.array([2, 2])
    sink = LiveActionSink(env, execute=True)
    sink._wait_receipts = Mock()
    monotonic = iter(np.arange(0, 5, 0.11))
    monkeypatch.setattr(live.time, "monotonic", lambda: float(next(monotonic)))
    monkeypatch.setattr(live.time, "sleep", lambda _: None)
    report = sink.home(timeout_s=3)
    assert report["status"] == "completed"
    np.testing.assert_array_equal(report["home_joints_rad"], worker.HOME_QPOS)
    assert list(sink.hold_reasons) == ["home complete"]
    env.seed_clutches_from_measured.assert_called_once()


def test_home_timeout_holds_both_arms(monkeypatch):
    from egomimic.robot.piper_pico import live_policy_devices as live

    env = fake_env()
    for arm in env.arms:
        arm.get_state.side_effect = lambda: {**feedback(), "joint_pos": np.full(6, 1.0)}
    env.home.return_value = np.array([2, 2])
    sink = LiveActionSink(env, execute=True)
    sink._wait_receipts = Mock()
    ticks = iter(np.arange(0, 5, 0.11))
    monkeypatch.setattr(live.time, "monotonic", lambda: float(next(ticks)))
    monkeypatch.setattr(live.time, "sleep", lambda _: None)
    with pytest.raises(TimeoutError, match="home"):
        sink.home(timeout_s=0.2)
    for arm in env.arms:
        arm.request_hold.assert_called_once()


def test_provider_uses_real_can_stamps_and_rejects_stale_cache():
    env = fake_env()
    names = ["left_wrist", "front_1", "right_wrist"]
    env.camera_specs = [{"name": name} for name in names]
    env.cameras = [fake_device(name) for name in names]
    now = time.time()
    for camera in env.cameras:
        camera.ring = SimpleNamespace(
            count=1,
            buffer_size=100,
            get_time_budget=0.01,
            get=lambda out=None: {"timestamp": np.asarray(now)},
        )
    obs = {"timestamp": np.array([now])}
    for i in range(3):
        obs[f"camera{i}_timestamp"] = np.array([now])
        obs[f"camera{i}_rgb"] = np.full((1, 2, 2, 3), i, dtype=np.uint8)
    for i in range(2):
        for key, value in (
            ("eef_pos", np.zeros((1, 3))),
            ("eef_rot_axis_angle", np.zeros((1, 3))),
            ("gripper_width", np.zeros((1, 1))),
            ("source_timestamp", now),
        ):
            obs[f"robot{i}_{key}"] = value
        for field in ("joint", "pose", "gripper"):
            obs[f"robot{i}_{field}_source_timestamp"] = now - 0.01
    env.get_obs = lambda: obs
    provider = LiveObservationProvider(env)
    frame = provider.latest()
    assert frame.source_timestamps["left"] == now - 0.01
    assert frame.images["front_1"][0, 0, 0] == 1
    stale = feedback()
    stale["robot_pose_source_timestamp"] = now - 10
    env.arms[0].get_state.side_effect = lambda: stale
    with pytest.raises(RuntimeError, match="CAN.*stale"):
        provider.latest()
