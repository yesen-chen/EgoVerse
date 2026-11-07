"""The left-arm joint reader listens only and reports radians."""

from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.scripts.read_left_arm_joints import (
    format_home_qpos,
    read_joint_radians,
    write_home_qpos,
)
from egomimic.robot.piper_pico.units import JOINT_FACTOR


class _JointInterface:
    def __init__(self, can_port: str):
        self.can_port = can_port
        self.calls = []
        self.reads = 0

    def ConnectPort(self, can_init=False, piper_init=True, start_thread=True):
        self.calls.append(("ConnectPort", can_init, piper_init, start_thread))

    def GetArmJointMsgs(self):
        self.reads += 1
        ready = self.reads >= 2
        counts = [1000, -2000, 0, 3000, -4000, 5000]
        return SimpleNamespace(
            Hz=20.0 if ready else 0.0,
            time_stamp=1_700_000_000.0 if ready else 0.0,
            joint_state=SimpleNamespace(**{f"joint_{i + 1}": counts[i] for i in range(6)}),
        )

    def DisconnectPort(self):
        self.calls.append(("DisconnectPort",))

    def EnableArm(self, *_args):
        raise AssertionError("reader must not enable the arm")

    def JointCtrl(self, *_args):
        raise AssertionError("reader must not send joint commands")

    def GripperCtrl(self, *_args):
        raise AssertionError("reader must not send gripper commands")


def test_read_returns_radians_and_sends_no_control(monkeypatch):
    monkeypatch.setattr(
        "egomimic.scripts.read_left_arm_joints.time.sleep", lambda _seconds: None
    )
    created = []

    def factory(can_port):
        arm = _JointInterface(can_port)
        created.append(arm)
        return arm

    radians = read_joint_radians("can0", 1.0, factory)
    expected = np.array([1000, -2000, 0, 3000, -4000, 5000], dtype=np.float64) / JOINT_FACTOR
    np.testing.assert_allclose(radians, expected)
    arm = created[0]
    assert arm.calls == [
        ("ConnectPort", False, False, True),
        ("DisconnectPort",),
    ]


def test_read_times_out_without_feedback(monkeypatch):
    monkeypatch.setattr(
        "egomimic.scripts.read_left_arm_joints.time.sleep", lambda _seconds: None
    )
    monkeypatch.setattr(
        "egomimic.scripts.read_left_arm_joints.time.monotonic",
        iter([0.0, 0.0, 2.0]).__next__,
    )

    class Silent(_JointInterface):
        def GetArmJointMsgs(self):
            return SimpleNamespace(Hz=0.0, time_stamp=0.0, joint_state=SimpleNamespace())

    with pytest.raises(TimeoutError, match="will not enable"):
        read_joint_radians("can0", 1.0, lambda _can: Silent("can0"))


def test_write_home_replaces_only_the_assignment(tmp_path):
    path = tmp_path / "arm_process.py"
    path.write_text(
        "CMD_HOME = 3\n"
        "HOME_QPOS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)\n"
        "def go_home():\n"
        "    return HOME_QPOS\n",
        encoding="utf-8",
    )
    sample = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.float64)
    write_home_qpos(path, sample)
    text = path.read_text(encoding="utf-8")
    assert format_home_qpos(sample) in text
    assert "def go_home():" in text
    assert text.count("HOME_QPOS = np.array(") == 1
