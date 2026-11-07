"""Clutch, camera-time alignment, stale waypoints, and the stop path."""

from __future__ import annotations

import time
from multiprocessing.managers import SharedMemoryManager
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_pico.arm import MOVE_P, PiperArm
from egomimic.robot.piper_pico.arm_process import HOME_QPOS, reduce_arm_commands
from egomimic.robot.piper_pico.clutch import (
    HandClutch,
    looks_like_tracking_origin,
    transform_pico_to_robot,
)
from egomimic.robot.piper_pico.env import PiperPicoEnv
from egomimic.robot.piper_pico.interpolate import align_bimanual_obs
from egomimic.robot.piper_pico.record import ReplayBuffer
from egomimic.robot.piper_pico.shm import SharedMemoryRingBuffer
from egomimic.robot.piper_pico.sim_arm import HOME_EE_POSE, SimPiperArm
from egomimic.robot.piper_pico.units import continuous_euler_xyz, pose_to_endpose_sdk


def _identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def test_grip_edge_holds_the_arm_and_ten_centimetres_stay_ten_centimetres():
    clutch = HandClutch()
    start = np.array([0.20, 0.10, -0.30], dtype=np.float64)
    clutch.seed(np.zeros(3), _identity())
    held = clutch.update(
        start,
        _identity(),
        grip=0.6,
        trigger=0.0,
        measured_pos=np.zeros(3),
        measured_quat_wxyz=_identity(),
        now=0.2,
    )
    assert held.engaged
    np.testing.assert_allclose(held.position, np.zeros(3), atol=1e-8)
    moved = clutch.update(
        start + np.array([0.08, 0.0, 0.0]),
        _identity(),
        grip=0.6,
        trigger=0.0,
        measured_pos=np.zeros(3),
        measured_quat_wxyz=_identity(),
        now=0.4,
    )
    np.testing.assert_allclose(moved.position, np.array([0.04 / 1.2, 0.0, 0.0]), atol=1e-8)
    assert moved.gripper_m == pytest.approx(0.08)


def _engaged(clutch: HandClutch, pos: np.ndarray, now: float):
    return clutch.update(
        pos,
        _identity(),
        grip=0.6,
        trigger=0.0,
        measured_pos=np.zeros(3),
        measured_quat_wxyz=_identity(),
        now=now,
    )


def test_one_origin_frame_does_not_move_the_arm():
    clutch = HandClutch(name="left")
    clutch.seed(np.zeros(3), _identity())
    home = np.array([0.27, 0.19, -0.26])
    held = _engaged(clutch, home, 0.0)
    origin = np.zeros(3)
    for step in range(10):
        target = _engaged(clutch, origin, 0.03 * (step + 1))
        np.testing.assert_allclose(target.position, held.position, atol=1e-8)
        assert target.event == "lost"
    assert clutch.lost_count == 10
    assert clutch.last_event == "lost"
    assert looks_like_tracking_origin(origin, _identity())


def test_origin_dropout_then_real_pose_reanchors_without_moving():
    clutch = HandClutch()
    clutch.seed(np.zeros(3), _identity())
    home = np.array([0.27, 0.19, -0.26])
    held = _engaged(clutch, home, 0.0)
    for step in range(10):
        _engaged(clutch, np.zeros(3), 0.03 * (step + 1))
    recovered = np.array([0.19, 0.15, -0.35])
    for step in range(4):
        target = _engaged(clutch, recovered, 0.4 + 0.03 * step)
        np.testing.assert_allclose(target.position, held.position, atol=1e-8)
    nudged = _engaged(clutch, recovered + np.array([0.02, 0.0, 0.0]), 0.6)
    np.testing.assert_allclose(nudged.position - held.position, [0.01 / 1.2, 0.0, 0.0], atol=1e-8)


def test_confirmed_teleport_reanchors_instead_of_chasing():
    clutch = HandClutch()
    clutch.seed(np.zeros(3), _identity())
    start = np.array([0.04, -0.05, -0.01])
    held = _engaged(clutch, start, 0.0)
    jumped = start + np.array([0.52, 0.39, -0.47])
    for step in range(4):
        target = _engaged(clutch, jumped, 0.03 * (step + 1))
        np.testing.assert_allclose(target.position, held.position, atol=1e-8)
    nudged = _engaged(clutch, jumped + np.array([0.02, 0.0, 0.0]), 0.2)
    np.testing.assert_allclose(nudged.position - held.position, [0.01 / 1.2, 0.0, 0.0], atol=1e-8)


def test_ema_and_pos_ratio_settle_to_the_scaled_hand_motion():
    clutch = HandClutch()
    clutch.seed(np.zeros(3), _identity())
    start = np.array([0.20, 0.0, 0.10])
    _engaged(clutch, start, 0.0)
    goal = start + np.array([0.06, 0.0, 0.0])
    target = None
    for step in range(40):
        target = clutch.update(
            goal,
            _identity(),
            grip=0.6,
            trigger=1.0,
            measured_pos=np.zeros(3),
            measured_quat_wxyz=_identity(),
            now=0.03 * (step + 1),
        )
    assert target is not None
    np.testing.assert_allclose(target.position, np.array([0.06 / 1.2, 0.0, 0.0]), atol=1e-4)
    assert target.gripper_m == pytest.approx(0.0, abs=1e-3)


def test_sim_arm_shows_a_command_jump_without_filtering_it():
    arm = SimPiperArm("sim", name="left")
    arm.connect()
    np.testing.assert_allclose(arm.ee_pose(), HOME_EE_POSE)
    np.testing.assert_allclose(arm.joint_positions(), HOME_QPOS)
    jumped = HOME_EE_POSE.copy()
    jumped[2] -= 0.47
    arm.send_ee_pose(jumped, gripper_m=0.02)
    np.testing.assert_allclose(arm.ee_pose(), jumped)
    assert arm.gripper() == pytest.approx(0.02)
    assert arm.command_jumps == 1
    arm.send_joints(HOME_QPOS, 0.0)
    assert arm.ee_pose()[2] > jumped[2]
    assert float(np.linalg.norm(arm.ee_pose()[:3] - jumped[:3])) == pytest.approx(0.04)
    arm.disconnect()
    assert not arm.connected


def test_pico_forward_axis_maps_to_robot_x():
    pos, quat = transform_pico_to_robot(np.array([0.0, 0.0, -0.10]), _identity())
    np.testing.assert_allclose(pos, np.array([0.10, 0.0, 0.0]), atol=1e-8)
    np.testing.assert_allclose(np.abs(quat), np.abs(_identity()), atol=1e-8)


def _aligned(query_time, camera_stamps, robot_stamps):
    poses = np.zeros((len(robot_stamps), 6))
    poses[:, 0] = np.arange(len(robot_stamps)) * 0.02 + 0.10
    return align_bimanual_obs(
        camera_timestamps=[np.asarray(stamps) for stamps in camera_stamps],
        camera_colors=[
            np.arange(len(stamps), dtype=np.uint8)[:, None, None, None].repeat(3, axis=3)
            for stamps in camera_stamps
        ],
        robot_timestamps=[np.asarray(robot_stamps)],
        robot_poses_rotvec=[poses],
        gripper_widths=[np.linspace(0.01, 0.05, len(robot_stamps))],
        frequency=20.0,
        horizon=2,
        query_time=query_time,
        robot_joint_positions=[np.repeat(poses[:, :1], 6, axis=1)],
    )


def test_get_obs_takes_latest_frames_and_arm_state_at_the_query_time():
    obs = _aligned(
        1.0,
        camera_stamps=[[0.90, 0.94, 0.98], [0.93, 0.96, 0.999]],
        robot_stamps=[0.94, 0.96, 0.98, 0.99],
    )
    np.testing.assert_allclose(obs["timestamp"], [0.95, 1.0])
    np.testing.assert_allclose(obs["camera0_timestamp"], [0.94, 0.98])
    np.testing.assert_allclose(obs["camera1_timestamp"], [0.93, 0.999])
    np.testing.assert_array_equal(obs["camera0_rgb"][:, 0, 0, 0], [1, 2])
    # 0.95 interpolates between 0.94 and 0.96; 1.0 holds the newest sample.
    np.testing.assert_allclose(obs["robot0_eef_pos"][:, 0], [0.11, 0.16], atol=1e-8)
    np.testing.assert_allclose(obs["robot0_joint_pos"][:, 3], [0.11, 0.16], atol=1e-8)
    assert obs["robot0_source_timestamp"] == pytest.approx(0.99)


def test_get_obs_never_uses_a_frame_or_arm_sample_after_the_query_time():
    obs = _aligned(
        0.97,
        camera_stamps=[[0.90, 0.94, 0.98]],
        robot_stamps=[0.94, 0.96, 0.98, 0.99],
    )
    np.testing.assert_allclose(obs["camera0_timestamp"], [0.90, 0.94])
    np.testing.assert_allclose(obs["robot0_eef_pos"][:, 0], [0.10, 0.12], atol=1e-8)
    assert obs["robot0_source_timestamp"] == pytest.approx(0.96)
    with pytest.raises(ValueError, match="camera 0"):
        _aligned(0.89, camera_stamps=[[0.90]], robot_stamps=[0.80])
    with pytest.raises(ValueError, match="robot 0"):
        _aligned(0.93, camera_stamps=[[0.90]], robot_stamps=[0.94])


def test_queue_time_drops_stale_waypoints_and_keeps_the_latest_fresh_target():
    batch = {
        "cmd": np.array([1, 1, 1]),
        "target_time": np.array([0.0, 1.0, 1.5]),
        "pose": np.array(
            [
                [1.0, 0, 0, 0, 0, 0],
                [2.0, 0, 0, 0, 0, 0],
                [3.0, 0, 0, 0, 0, 0],
            ]
        ),
        "gripper": np.zeros(3),
        "joints": np.zeros((3, 6)),
    }
    chosen = reduce_arm_commands(batch, now=1.2)
    assert chosen is not None
    assert chosen.kind == "waypoint"
    assert chosen.pose[0] == pytest.approx(3.0)
    assert chosen.command_id == -1
    batch["command_id"] = np.array([7, 8, 9])
    assert reduce_arm_commands(batch, now=1.2).command_id == 9
    assert (
        reduce_arm_commands(
            {
                "cmd": np.array([1]),
                "target_time": np.array([1.0]),
                "pose": np.zeros((1, 6)),
                "gripper": np.zeros(1),
                "joints": np.zeros((1, 6)),
            },
            now=1.2,
        )
        is None
    )


def _bare_env():
    env = PiperPicoEnv.__new__(PiperPicoEnv)
    sent = []

    class _Arm:
        def schedule_waypoint(self, pose, gripper_m, target_time, command_id):
            sent.append((pose.copy(), gripper_m, target_time, command_id))

    env.arms = [_Arm(), _Arm()]
    env.cameras = [object()]
    env._last_rpy = [np.zeros(3), np.zeros(3)]
    env._next_command_id = 0
    env.tick_accumulator = None
    env.is_ready = True
    return env, sent


def test_exec_actions_does_not_queue_a_waypoint_that_is_already_due():
    env, sent = _bare_env()
    now = time.time()
    actions = np.zeros((2, 14))
    actions[0, 0] = 1.0
    actions[1, 0] = 0.2
    actions[1, 7] = 0.3
    ids = env.exec_actions(actions, np.array([now - 1.0, now + 1.0]))
    np.testing.assert_array_equal(ids, [-1, 0])
    assert len(sent) == 2
    assert sent[0][0][0] == pytest.approx(0.2)
    assert sent[1][0][0] == pytest.approx(0.3)
    assert [item[3] for item in sent] == [0, 0]
    assert env.exec_actions(actions[1], np.array([now + 1.0]))[0] == 1


def _tick_obs(stamp):
    obs = {"timestamp": np.array([stamp - 0.05, stamp]), "camera0_timestamp": np.array([stamp - 0.06, stamp - 0.01])}
    for i in range(2):
        obs[f"robot{i}_eef_pos"] = np.full((2, 3), i + stamp)
        obs[f"robot{i}_eef_rot_axis_angle"] = np.zeros((2, 3))
        obs[f"robot{i}_gripper_width"] = np.full((2, 1), 0.04)
        obs[f"robot{i}_joint_pos"] = np.zeros((2, 6))
        obs[f"robot{i}_source_timestamp"] = np.float64(stamp - 0.003)
    return obs


def test_record_tick_pairs_each_observation_with_the_command_decided_after_it():
    from egomimic.robot.piper_pico.record import TickAccumulator, episode_from_ticks

    env, _ = _bare_env()
    env.record_tick(_tick_obs(1.0), np.zeros(14), 1.002, 1.035, 0)  # not recording
    env.tick_accumulator = TickAccumulator()
    for k in range(3):
        stamp = 1.0 + k / 30
        env.record_tick(_tick_obs(stamp), np.full(14, k), stamp + 0.002, stamp + 0.035, k)
    streams = [
        {"last_command_id": np.array([-1, 0, 0, 1, 2]),
         "last_command_send_timestamp": np.array([np.nan, 1.004, 1.004, 1.037, 1.07])},
        {"last_command_id": np.array([-1, 0, 2]),
         "last_command_send_timestamp": np.array([np.nan, 1.005, 1.071])},
    ]
    episode = episode_from_ticks(env.tick_accumulator, streams)
    np.testing.assert_allclose(episode["timestamp"], 1.0 + np.arange(3) / 30)
    np.testing.assert_allclose(episode["action"][:, 0], [0, 1, 2])
    np.testing.assert_allclose(episode["camera0_timestamp"], episode["timestamp"] - 0.01)
    np.testing.assert_allclose(episode["robot1_eef_pos"][:, 0], 1 + episode["timestamp"])
    np.testing.assert_allclose(episode["robot0_send_timestamp"], [1.004, 1.037, 1.07])
    # Command 1 was superseded before the right arm process read it.
    np.testing.assert_allclose(episode["robot1_send_timestamp"], [1.005, np.nan, 1.071])
    with pytest.raises(ValueError, match="increase"):
        env.record_tick(_tick_obs(1.0), np.zeros(14), 1.002, 1.035, 3)


class _FakeSdk:
    def __init__(self):
        self.calls = []
        self.end = SimpleNamespace(
            X_axis=100_000.0,
            Y_axis=0.0,
            Z_axis=200_000.0,
            RX_axis=0.0,
            RY_axis=0.0,
            RZ_axis=0.0,
        )

    def MotionCtrl_1(self, *args):
        self.calls.append(("MotionCtrl_1", args))

    def MotionCtrl_2(self, *args):
        self.calls.append(("MotionCtrl_2", args))

    def EndPoseCtrl(self, *args):
        self.calls.append(("EndPoseCtrl", args))

    def DisconnectPort(self):
        self.calls.append(("DisconnectPort", ()))

    def EnableArm(self, *args):
        self.calls.append(("EnableArm", args))

    def DisableArm(self, *args):
        self.calls.append(("DisableArm", args))

    def GetArmEndPoseMsgs(self):
        return SimpleNamespace(end_pose=self.end)


def test_disconnect_clears_the_plan_and_does_not_drop_torque():
    sdk = _FakeSdk()
    arm = PiperArm("can0", move_spd_rate=50, name="left").bind(sdk)
    arm.disconnect(stop_motion=True)
    names = [name for name, _ in sdk.calls]
    assert "DisableArm" not in names
    assert "EnableArm" not in names
    assert names[-1] == "DisconnectPort"
    clear_codes = [args[1] for name, args in sdk.calls if name == "MotionCtrl_1"]
    assert clear_codes[:3] == [0x03, 0x04, 0x00]
    move = [args for name, args in sdk.calls if name == "MotionCtrl_2"][0]
    assert move[1] == MOVE_P
    assert move[2] == 50
    expected = pose_to_endpose_sdk(np.array([0.1, 0.0, 0.2, 0.0, 0.0, 0.0]))
    sent = [args for name, args in sdk.calls if name == "EndPoseCtrl"][0]
    assert sent[0] == expected[0]
    assert sent[2] == expected[2]


def test_continuous_euler_keeps_the_previous_branch():
    turned = Rotation.from_euler("xyz", [0.0, 0.0, 0.0])
    rpy = continuous_euler_xyz(turned, np.array([0.0, 0.0, 2.0 * np.pi]))
    np.testing.assert_allclose(rpy, np.array([0.0, 0.0, 2.0 * np.pi]), atol=1e-8)


def test_replay_buffer_appends_and_drops_one_episode(tmp_path):
    buffer = ReplayBuffer(tmp_path / "replay_buffer.zarr")
    episode = {
        "timestamp": np.array([0.0, 0.1]),
        "action": np.zeros((2, 14)),
    }
    buffer.add_episode(episode)
    buffer.add_episode(episode)
    assert buffer.n_episodes == 2
    assert buffer.n_steps == 4
    buffer.drop_episode()
    assert buffer.n_episodes == 1
    assert buffer.n_steps == 2


def test_ring_returns_the_latest_sample():
    with SharedMemoryManager() as shm:
        ring = SharedMemoryRingBuffer.create_from_examples(
            shm,
            {"value": np.float64(0.0)},
            get_max_k=4,
            get_time_budget=0.01,
            put_desired_frequency=10,
        )
        ring.put({"value": np.float64(1.0)})
        ring.put({"value": np.float64(2.0)})
        assert float(ring.get()["value"]) == pytest.approx(2.0)
        latest = ring.get_last_k(2)
        np.testing.assert_allclose(latest["value"], np.array([1.0, 2.0]))
