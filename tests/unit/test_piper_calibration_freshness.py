"""Feedback source time must not advance when the CAN cache stops updating."""

from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.robot.piper_pico.arm import PiperArm
from egomimic.robot.piper_pico.arm_process import read_arm_state, state_examples
from egomimic.robot.piper_pico.shm import _specs_from_examples
from egomimic.robot.piper_pico.sim_arm import SimPiperArm


class FeedbackSdk:
    def __init__(self):
        self.pose_reads = 0
        self.gripper_reads = 0
        self.joint_reads = 0
        self.pose = SimpleNamespace(
            time_stamp=1700000000.25,
            end_pose=SimpleNamespace(
                X_axis=100000, Y_axis=-200000, Z_axis=300000,
                RX_axis=90000, RY_axis=0, RZ_axis=-90000,
            ),
        )
        self.gripper = SimpleNamespace(
            time_stamp=1700000000.20,
            gripper_state=SimpleNamespace(grippers_angle=19000),
        )
        self.joints = SimpleNamespace(
            time_stamp=1700000000.15,
            joint_state=SimpleNamespace(**{f"joint_{i}": 0 for i in range(1, 7)}),
        )

    def GetArmEndPoseMsgs(self):
        self.pose_reads += 1
        return self.pose

    def GetArmGripperMsgs(self):
        self.gripper_reads += 1
        return self.gripper

    def GetArmJointMsgs(self):
        self.joint_reads += 1
        return self.joints

    def GetArmHighSpdInfoMsgs(self):
        return SimpleNamespace(
            **{f"motor_{i}": SimpleNamespace(motor_speed=0) for i in range(1, 7)}
        )


def test_snapshot_copies_values_and_stamp_from_one_getter_result():
    sdk = FeedbackSdk()
    arm = PiperArm("unused").bind(sdk)
    snapshot = arm.feedback_snapshot()
    assert (sdk.pose_reads, sdk.gripper_reads, sdk.joint_reads) == (1, 1, 1)
    np.testing.assert_allclose(
        snapshot["ee_pose"], [0.1, -0.2, 0.3, np.pi / 2, 0.0, -np.pi / 2]
    )
    assert snapshot["gripper_position"] == pytest.approx(0.019)
    assert snapshot["robot_pose_source_timestamp"] == 1700000000.25
    assert snapshot["robot_gripper_source_timestamp"] == 1700000000.20
    assert snapshot["robot_joint_source_timestamp"] == 1700000000.15

    sdk.pose.end_pose.X_axis = 999999
    sdk.pose.time_stamp += 2
    sdk.gripper.gripper_state.grippers_angle = 80000
    assert snapshot["ee_pose"][0] == pytest.approx(0.1)
    assert snapshot["gripper_position"] == pytest.approx(0.019)
    assert snapshot["robot_pose_source_timestamp"] == 1700000000.25


def test_repolling_cached_feedback_does_not_refresh_source_time(monkeypatch):
    arm = PiperArm("unused").bind(FeedbackSdk())
    monkeypatch.setattr("egomimic.robot.piper_pico.arm_process.time.time", lambda: 1700000001.0)
    first = read_arm_state(arm, 0.1, 9, 1700000000.9)
    monkeypatch.setattr("egomimic.robot.piper_pico.arm_process.time.time", lambda: 1700000003.0)
    second = read_arm_state(arm, 0.1)
    assert second["robot_receive_timestamp"] - first["robot_receive_timestamp"] == 2.0
    for key in (
        "robot_pose_source_timestamp", "robot_gripper_source_timestamp",
        "robot_joint_source_timestamp",
    ):
        assert second[key] == first[key]
    assert first["robot_timestamp"] == pytest.approx(1700000000.9)
    assert first["last_command_id"] == 9
    assert first["last_command_send_timestamp"] == 1700000000.9
    assert not first["robot_source_timestamps_simulated"]
    assert set(first) == set(state_examples())


def test_host_receive_timestamp_is_taken_after_feedback_read(monkeypatch):
    sdk = FeedbackSdk()
    arm = PiperArm("unused").bind(sdk)
    clock = {"now": 1700000001.0}
    original = sdk.GetArmEndPoseMsgs

    def read_new_feedback():
        clock["now"] += 0.01
        sdk.pose.time_stamp = clock["now"]
        return original()

    sdk.GetArmEndPoseMsgs = read_new_feedback
    monkeypatch.setattr(
        "egomimic.robot.piper_pico.arm_process.time.time", lambda: clock["now"]
    )
    state = read_arm_state(arm, 0.0)
    assert state["robot_receive_timestamp"] >= state["robot_pose_source_timestamp"]


@pytest.mark.parametrize("stamp", [None, 0, -1, np.nan, np.inf, "unknown"])
def test_unavailable_source_timestamps_stay_unknown(stamp):
    sdk = FeedbackSdk()
    sdk.pose.time_stamp = stamp
    sdk.gripper.time_stamp = stamp
    sdk.joints.time_stamp = stamp
    snapshot = PiperArm("unused").bind(sdk).feedback_snapshot()
    assert np.isnan(snapshot["robot_pose_source_timestamp"])
    assert np.isnan(snapshot["robot_gripper_source_timestamp"])
    assert np.isnan(snapshot["robot_joint_source_timestamp"])


def test_legacy_getters_without_source_timestamps_are_supported():
    sdk = FeedbackSdk()
    del sdk.pose.time_stamp
    del sdk.gripper.time_stamp
    del sdk.joints.time_stamp
    snapshot = PiperArm("unused").bind(sdk).feedback_snapshot()
    assert np.isnan(snapshot["robot_pose_source_timestamp"])
    assert np.isnan(snapshot["robot_gripper_source_timestamp"])
    assert np.isnan(snapshot["robot_joint_source_timestamp"])


def test_timestamp_change_while_copying_invalidates_source_freshness():
    sdk = FeedbackSdk()
    original = sdk.pose.end_pose

    class ChangingPose:
        def __getattr__(self, key):
            if key == "X_axis":
                sdk.pose.time_stamp += 0.01
            return getattr(original, key)

    sdk.pose.end_pose = ChangingPose()
    snapshot = PiperArm("unused").bind(sdk).feedback_snapshot()
    assert np.isnan(snapshot["robot_pose_source_timestamp"])
    assert snapshot["robot_gripper_source_timestamp"] == 1700000000.20


def test_fake_arm_fallback_has_unknown_freshness_unless_explicitly_simulated(monkeypatch):
    arm = SimPiperArm("unused")
    monkeypatch.setattr("egomimic.robot.piper_pico.arm_process.time.time", lambda: 1700000005.0)
    unknown = read_arm_state(arm, 0.02)
    assert np.isnan(unknown["robot_pose_source_timestamp"])
    assert np.isnan(unknown["robot_gripper_source_timestamp"])
    assert np.isnan(unknown["robot_joint_source_timestamp"])
    assert not unknown["robot_source_timestamps_simulated"]

    simulated = read_arm_state(arm, 0.02, simulated=True)
    assert simulated["robot_pose_source_timestamp"] == 1700000005.0
    assert simulated["robot_gripper_source_timestamp"] == 1700000005.0
    assert simulated["robot_joint_source_timestamp"] == 1700000005.0
    assert simulated["robot_source_timestamps_simulated"]


def test_freshness_metadata_has_shared_memory_compatible_scalar_types():
    examples = state_examples()
    state = read_arm_state(SimPiperArm("unused"), 0.0, simulated=True)
    specs = {spec.name: spec for spec in _specs_from_examples(examples)}
    actual = {spec.name: spec for spec in _specs_from_examples(state)}
    assert specs == actual
    assert specs["robot_source_timestamps_simulated"].dtype == np.dtype(np.uint8)
    for name in (
        "robot_pose_source_timestamp", "robot_gripper_source_timestamp",
        "robot_joint_source_timestamp",
    ):
        assert specs[name].shape == ()
        assert specs[name].dtype == np.dtype(np.float64)
