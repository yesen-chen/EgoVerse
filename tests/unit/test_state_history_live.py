"""History acquisition and remote input parity without CAN, camera or sockets."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from test_live_policy_contract import fixture as metadata_files  # noqa: F401
from test_pi05_remote import FakeRPC, frame, runtime, server_metadata  # noqa: F401

from egomimic.robot.piper_pico import env as env_module
from egomimic.robot.piper_pico.env import PiperPicoEnv
from egomimic.robot.piper_pico.pi05_data import _data_identity, canonical_hash
from egomimic.robot.piper_pico.pi05_remote import (
    Pi05RemotePolicy,
    pack_message,
    unpack_message,
)
from egomimic.robot.piper_pico.policy_inference import ObservationFrame
from egomimic.robot.piper_pico.state_history import (
    HistoryCoverageError,
    make_history_contract,
)


def fake_shm_env(offsets):
    env = PiperPicoEnv.__new__(PiperPicoEnv)
    env.is_ready = True
    env.state_history_offsets_s = offsets
    env.horizon, env.frequency = 2, 30.0
    env.obs_accumulator = None
    stamp = 99.8 + np.arange(20) * 0.01
    env.arms, env.cameras = [], []
    for side in range(2):
        poses = np.zeros((len(stamp), 6))
        poses[:, 0] = 2 * (stamp - 100) + side
        data = {
            "robot_timestamp": stamp,
            "ee_pose": poses,
            "joint_pos": np.zeros((len(stamp), 6)),
            "gripper_position": 0.04 + 0.02 * (stamp - 100) + side * 0.01,
        }
        arm = SimpleNamespace(process=SimpleNamespace(exitcode=None))
        arm.get_all_state = Mock(return_value=data)
        env.arms.append(arm)
    for _ in range(3):
        camera = SimpleNamespace(process=SimpleNamespace(exitcode=None))
        camera.get_last = Mock(
            return_value={
                "timestamp": np.array([99.9, 99.94]),
                "color": np.zeros((2, 8, 8, 3), dtype=np.uint8),
            }
        )
        env.cameras.append(camera)
    return env


def test_get_obs_uses_same_feedback_snapshot_for_arbitrary_history(monkeypatch):
    monkeypatch.setattr(env_module.time, "time", lambda: 100.0)
    offsets = (-0.075, -0.025)
    env = fake_shm_env(offsets)
    obs = env.get_obs()
    for side, k in ((0, 0), (1, 7)):
        np.testing.assert_allclose(
            obs["state_history_base"][:, k], 2 * np.array(offsets) + side
        )
        np.testing.assert_allclose(
            obs["state_history_base"][:, k + 6],
            0.04 + 0.02 * np.array(offsets) + side * 0.01,
        )
        env.arms[side].get_all_state.assert_called_once()
    current = obs["robot0_eef_pos"][-1].copy()
    # The recorded/current state is held at the actual newest sample, not
    # extrapolated forward to the observation query time.
    assert current[0] < -0.019
    env.state_history_offsets_s = ()
    without = env.get_obs()
    np.testing.assert_array_equal(current, without["robot0_eef_pos"][-1])
    assert "state_history_base" not in without


@pytest.mark.parametrize("offsets", [(-0.5,), (-0.001,)])
def test_get_obs_rejects_unavailable_history_instead_of_clipping(monkeypatch, offsets):
    monkeypatch.setattr(env_module.time, "time", lambda: 100.0)
    with pytest.raises(HistoryCoverageError):
        fake_shm_env(offsets).get_obs()


def test_observation_frame_copies_shared_history(monkeypatch):
    monkeypatch.setattr(env_module.time, "time", lambda: 100.0)
    obs = fake_shm_env((-0.075, -0.025)).get_obs()
    captured = ObservationFrame.from_aligned_obs(
        obs, camera_names=("front_1", "left_wrist", "right_wrist")
    )
    before = captured.base_state_history.copy()
    obs["state_history_base"][:] = 999
    np.testing.assert_array_equal(before, captured.base_state_history)
    assert captured.state_history_offsets_s == (-0.075, -0.025)


def history_server(metadata, offsets):
    server = server_metadata(metadata)
    contract = make_history_contract(offsets)
    server["state_history"] = contract
    server["manifest"]["state_history"] = contract
    # This fixture tests remote validation; the source archive is not opened.
    for record in server["manifest"]["episodes"]:
        record["eligible_indices"] = list(range(record["total_frames"]))
        record["feedback_history"] = {"status": "available"}
        record["feedback_history_sha256"] = canonical_hash(record["feedback_history"])
        record["feedback_available_until_sha256"] = "0" * 64
    server["manifest"]["data_identity_sha256"] = _data_identity(
        server["manifest"]["episodes"]
    )
    server["manifest"]["manifest_sha256"] = canonical_hash(
        {k: v for k, v in server["manifest"].items() if k != "manifest_sha256"}
    )
    return server


@pytest.mark.parametrize("offsets", [(-0.025,), (-0.15, -0.075, -0.025)])
def test_remote_history_geometry(runtime, offsets):  # noqa: F811
    rpc = FakeRPC(history_server(runtime["metadata"], offsets))
    policy = Pi05RemotePolicy(rpc, runtime["metadata"])
    current = frame()
    history = np.repeat(current.base_state[None], len(offsets), axis=0)
    history[:, 0] += np.array(offsets)
    history[:, 7] -= np.array(offsets)
    history[:, 6] = np.linspace(0.01, 0.07, len(offsets))
    current = replace(
        current, base_state_history=history, state_history_offsets_s=offsets
    )
    prediction = policy.predict(current)
    sent = unpack_message(pack_message(rpc.requests[-1]))
    expected = np.stack([policy.adapter.base_to_camera_state(row) for row in history])
    np.testing.assert_allclose(sent["observation.state_history"], expected, atol=1e-6)
    np.testing.assert_allclose(
        sent["observation.state_history"][:, 6],
        history[:, 6] / policy.adapter.gripper_open_m,
    )
    np.testing.assert_allclose(
        sent["observation.state"],
        policy.adapter.base_to_camera_state(current.base_state),
        atol=1e-6,
    )
    assert prediction["base_actions"].shape == (100, 14)


def test_remote_rejects_missing_or_mismatched_history_before_rpc(runtime):  # noqa: F811
    rpc = FakeRPC(history_server(runtime["metadata"], (-0.05,)))
    policy = Pi05RemotePolicy(rpc, runtime["metadata"])
    with pytest.raises(ValueError, match="history offsets"):
        policy.predict(frame())
    assert not rpc.requests
    assert rpc.closed
    rpc = FakeRPC(history_server(runtime["metadata"], (-0.05,)))
    policy = Pi05RemotePolicy(rpc, runtime["metadata"])
    wrong = replace(
        frame(),
        base_state_history=frame().base_state[None],
        state_history_offsets_s=(-0.025,),
    )
    with pytest.raises(ValueError, match="history offsets"):
        policy.predict(wrong)
    assert not rpc.requests


def test_remote_rejects_tampered_history_handshake(runtime):  # noqa: F811
    server = history_server(runtime["metadata"], (-0.05,))
    server["state_history"] = make_history_contract((-0.025,))
    rpc = FakeRPC(server)
    with pytest.raises(ValueError, match="history"):
        Pi05RemotePolicy(rpc, runtime["metadata"])
    assert rpc.closed


def test_old_policy_does_not_silently_ignore_history(runtime):  # noqa: F811
    rpc = FakeRPC(server_metadata(runtime["metadata"]))
    policy = Pi05RemotePolicy(rpc, runtime["metadata"])
    current = frame()
    with pytest.raises(ValueError, match="history offsets"):
        policy.predict(
            replace(
                current,
                base_state_history=current.base_state[None],
                state_history_offsets_s=(-0.05,),
            )
        )
    assert not rpc.requests
