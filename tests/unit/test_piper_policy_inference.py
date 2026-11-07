"""Offline deployment parity with the actual Piper loader and saved HPT."""

from __future__ import annotations

import copy
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
import zarr
from fixtures.piper_pico_training import write_recording, write_training_roots
from fixtures.train_harness import hermetic_env
from scipy.spatial.transform import Rotation
from test_piper_pico_training import compose_local

from egomimic.rldb.action_timing import ActionTiming
from egomimic.rldb.embodiment.piper import Piper
from egomimic.rldb.zarr.piper_dataset import PiperDataset
from egomimic.robot.piper_pico.policy_inference import (
    ACTION,
    CAMERA_KEYS,
    STATE,
    ObservationFrame,
    PiperObservationAdapter,
    PiperPolicy,
    check_capture_config,
    checkpoint_contract,
)


def episode_frame(path, index=3):
    import simplejpeg

    store = zarr.open_group(str(path), mode="r")
    metadata = dict(store.attrs)
    state = []
    for side in ("left", "right"):
        pose = store[f"{side}.obs_ee_pose"][index]
        rotvec = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_rotvec()
        state.extend(
            [
                *pose[:3],
                *rotvec,
                float(store[f"{side}.obs_gripper"][index, 0])
                * metadata["gripper_encoding"]["open_m"],
            ]
        )
    stamp = float(store["timestamp"][index])
    frame = ObservationFrame(
        stamp,
        {
            name: simplejpeg.decode_jpeg(
                store[f"images.{name}"][index : index + 1][0], colorspace="RGB"
            )
            for name in CAMERA_KEYS
        },
        np.asarray(state),
        {name: stamp for name in (*CAMERA_KEYS, "left", "right")},
    )
    return metadata, frame


@pytest.fixture(scope="module")
def saved_policy(tmp_path_factory):
    import egomimic.trainHydra as train_hydra

    root = tmp_path_factory.mktemp("policy_checkpoint")
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    with pytest.MonkeyPatch.context() as patch:
        hermetic_env(patch)
        roots = write_training_roots(root)
        cfg = compose_local("piper_pico_local", "zero", roots, root / "train")
        _, objects = train_hydra.train(cfg)
        path = root / "model.ckpt"
        objects["trainer"].save_checkpoint(path)
        episode = next(roots["piper"].glob("*/episode_*.zarr"))
        yield path, episode, objects["model"]
    torch.set_num_threads(threads)


@pytest.mark.parametrize("index", [0, 3, 13, 23])
def test_observation_adapter_matches_training_loader(tmp_path, index):
    path = write_recording(tmp_path, "piper", "one")
    metadata, frame = episode_frame(path, index)
    timing = ActionTiming(30, 9, 2, 11)
    loader = PiperDataset(
        path,
        Piper.get_keymap(timing=timing),
        Piper.get_transform_list(timing=timing),
        allow_synthetic_calibration=True,
    )
    adapter = PiperObservationAdapter(metadata, allow_synthetic_calibration=True)
    batch = adapter.prepare(frame)[7]
    expected = loader[index]
    np.testing.assert_allclose(batch[STATE][0], expected[STATE], rtol=2e-6, atol=2e-6)
    for key in CAMERA_KEYS.values():
        np.testing.assert_allclose(
            batch[key][0].numpy() / 255, expected[key].numpy(), atol=6e-8
        )
    # Inverse must recover physical SDK endpoint and metres, without any Eva axis correction.
    back = adapter.camera_to_base_actions(batch[STATE].numpy())[0]
    np.testing.assert_allclose(back, frame.base_state, atol=2e-7)


def test_per_rig_calibration_is_independent_and_rpy_is_not_rotvec(tmp_path):
    path = write_recording(tmp_path, "piper", "rigs")
    metadata, frame = episode_frame(path)
    other = copy.deepcopy(metadata)
    for side, angle in (("left", 0.4), ("right", -0.3)):
        mat = np.eye(4)
        mat[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.5, angle]).as_matrix()
        mat[:3, 3] = [0.2, -0.5, 0.1]
        other["extrinsics"][side] = mat.tolist()
        other["calibration"]["data"]["extrinsics"][side] = mat.tolist()
    first = PiperObservationAdapter(metadata, allow_synthetic_calibration=True)
    second = PiperObservationAdapter(other, allow_synthetic_calibration=True)
    state = frame.base_state.copy()
    state[3:6] = [1, 0.7, -0.8]
    state[10:13] = [-0.4, 0.9, 1.3]
    before = first.base_to_camera_state(state)
    changed = second.base_to_camera_state(state)
    assert not np.allclose(before, changed)
    for adapter in (first, second, first):
        np.testing.assert_allclose(
            adapter.camera_to_base_actions(adapter.base_to_camera_state(state)[None])[
                0
            ],
            state,
            atol=2e-15,
        )
    np.testing.assert_array_equal(first.base_to_camera_state(state), before)


def test_freshness_checks_every_source_and_requires_explicit_camera_mapping(tmp_path):
    path = write_recording(tmp_path, "piper", "freshness")
    _, frame = episode_frame(path)
    frame.check_freshness(frame.timestamp + 0.1, 0.5)
    for source in frame.source_timestamps:
        stamps = dict(frame.source_timestamps)
        stamps[source] -= 1
        stale = ObservationFrame(
            frame.timestamp, frame.images, frame.base_state, stamps
        )
        with pytest.raises(ValueError, match=source):
            stale.check_freshness(frame.timestamp + 0.1, 0.5)
    with pytest.raises(ValueError, match="future"):
        frame.check_freshness(frame.timestamp - 1, 0.5)
    obs = {"timestamp": np.array([frame.timestamp])}
    order = ["right_wrist", "front_1", "left_wrist"]
    for i, name in enumerate(order):
        obs[f"camera{i}_rgb"] = frame.images[name][None]
        obs[f"camera{i}_timestamp"] = np.array([frame.timestamp - 0.01 * (i + 1)])
    for i in range(2):
        obs[f"robot{i}_eef_pos"] = frame.base_state[i * 7 : i * 7 + 3][None]
        obs[f"robot{i}_eef_rot_axis_angle"] = frame.base_state[i * 7 + 3 : i * 7 + 6][
            None
        ]
        obs[f"robot{i}_gripper_width"] = frame.base_state[i * 7 + 6 : i * 7 + 7][None]
        obs[f"robot{i}_source_timestamp"] = np.float64(frame.timestamp - 0.004 * (i + 1))
    restored = ObservationFrame.from_aligned_obs(obs, camera_names=order)
    np.testing.assert_array_equal(restored.base_state, frame.base_state)
    for name in order:
        np.testing.assert_array_equal(restored.images[name], frame.images[name])
    # Freshness comes from the device stamps get_obs selected, per source.
    assert restored.source_timestamps == pytest.approx(
        {
            "right_wrist": frame.timestamp - 0.01,
            "front_1": frame.timestamp - 0.02,
            "left_wrist": frame.timestamp - 0.03,
            "left": frame.timestamp - 0.004,
            "right": frame.timestamp - 0.008,
        }
    )
    obs["camera1_timestamp"] = np.array([frame.timestamp - 2])
    with pytest.raises(ValueError, match="front_1"):
        ObservationFrame.from_aligned_obs(obs, camera_names=order).check_freshness(
            frame.timestamp, 0.5
        )


def test_live_cameras_must_match_recorded_capture_and_latency():
    recorded = [
        {"name": "front_1", "type": "pico", "fps": 60, "width": 1080, "height": 810,
         "latency": 0.0, "eye": "left", "headset_ip": "192.0.2.1"},
        {"name": "left_wrist", "type": "uvc", "fps": 30.0, "width": 640, "height": 480,
         "latency": 0.0, "device": "/dev/video0"},
        {"name": "right_wrist", "type": "uvc", "fps": 30.0, "width": 640, "height": 480,
         "latency": 0.0, "device": "/dev/video2"},
    ]
    metadata = {"source_capture_config": {"cameras": recorded}}
    live = copy.deepcopy(recorded)
    live[1]["device"] = "/dev/video4"
    check_capture_config(metadata, live)
    live[0]["latency"] = 0.13
    with pytest.raises(ValueError, match="front_1.latency"):
        check_capture_config(metadata, live)
    with pytest.raises(ValueError, match="must both provide"):
        check_capture_config(metadata, recorded[:2])
    with pytest.raises(ValueError, match="source_capture_config"):
        check_capture_config({}, recorded)


def test_adapter_exposes_recorded_observation_to_send_delay(tmp_path):
    path = write_recording(tmp_path, "piper", "delay")
    metadata, _ = episode_frame(path)
    adapter = PiperObservationAdapter(metadata, allow_synthetic_calibration=True)
    assert adapter.observation_to_send_s == pytest.approx(0.004)
    del metadata["time_alignment"]
    with pytest.raises(ValueError, match="observation_to_send_s"):
        PiperObservationAdapter(metadata, allow_synthetic_calibration=True)


def test_physical_gripper_feedback_is_not_clipped(tmp_path):
    path = write_recording(tmp_path, "piper", "gripper")
    metadata, frame = episode_frame(path)
    adapter = PiperObservationAdapter(metadata, allow_synthetic_calibration=True)
    state = frame.base_state.copy()
    state[[6, 13]] = [0.081, -0.001]
    camera = adapter.base_to_camera_state(state)
    np.testing.assert_allclose(camera[[6, 13]], [1.0125, -0.0125])
    np.testing.assert_allclose(
        adapter.camera_to_base_actions(camera[None])[0, [6, 13]], state[[6, 13]]
    )


def test_checkpoint_restore_predicts_without_targets_or_downloads(
    saved_policy, monkeypatch
):
    path, episode, original = saved_policy
    metadata, frame = episode_frame(episode)
    monkeypatch.setattr(
        torch.hub,
        "load_state_dict_from_url",
        lambda *a, **k: pytest.fail("weights download"),
    )
    monkeypatch.setattr(
        "torchvision.models._api.load_state_dict_from_url",
        lambda *a, **k: pytest.fail("weights download"),
    )
    policy = PiperPolicy.from_checkpoint(
        path, metadata, inference_steps=10, allow_synthetic_calibration=True
    )
    batch = policy.adapter.prepare(frame)
    assert ACTION not in batch[7]
    torch.manual_seed(152)
    expected = original.model.predict_from_observations(batch, num_inference_steps=10)[
        "piper_bimanual_actions_cartesian"
    ]
    torch.manual_seed(152)
    result = policy.predict(frame)
    np.testing.assert_allclose(result["camera_actions"], expected[0].numpy(), atol=1e-6)
    assert result["base_actions"].shape == (policy.timing.output_chunk_length, 14)
    assert result["inference_steps"] == 10
    assert result["inference_elapsed_s"] > 0
    assert policy.model.nets["policy"].heads["shared"].num_inference_steps == 2
    assert not policy.model.nets.training


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("contract", "contract"),
        ("timing", "timing"),
        ("shape", "shapes"),
        ("nan", "normalization"),
        ("range", "range"),
        ("head", "FM head"),
        ("aliases", "aliases"),
        ("language", "non-language"),
        ("language_stem", "MLP observation stems"),
    ],
)
def test_checkpoint_rejects_semantic_mismatches(saved_policy, mutation, match):
    path, _, _ = saved_policy
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    cfg = hp["config_tree"]
    if mutation == "contract":
        cfg["data_contracts"]["piper_bimanual"]["pose_output"] = "xyz_rotvec"
    elif mutation == "timing":
        cfg["action_timing"]["piper_bimanual"]["output_offsets_s"][-1] += 0.1
    elif mutation == "shape":
        hp["norm_stats_state"]["shapes"][7][STATE] = (12,)
    elif mutation == "nan":
        hp["norm_stats_state"]["norm_stats"][7][STATE]["quantile_1"][0] = np.nan
    elif mutation == "range":
        hp["norm_stats_state"]["norm_stats"][7][STATE]["quantile_99"][0] = -999
    elif mutation == "head":
        cfg["model"]["robomimic_model"]["head_specs"]["shared"]["model"]["act_dim"] = 12
    elif mutation == "aliases":
        hp["norm_stats_state"]["zarr_keys"][7][STATE] = "wrong_observation"
    elif mutation == "language":
        cfg["model"]["robomimic_model"]["multitask"] = True
    elif mutation == "language_stem":
        cfg["model"]["robomimic_model"]["shared_stem_specs"]["language"] = {
            "_target_": "egomimic.models.hpt_nets.LanguageStem"
        }
    with pytest.raises(ValueError, match=match):
        checkpoint_contract(ckpt)


def test_legacy_checkpoint_and_unlabelled_mock_are_rejected(saved_policy):
    with pytest.raises(ValueError, match="contract"):
        checkpoint_contract({})
    path, episode, _ = saved_policy
    metadata, _ = episode_frame(episode)
    with pytest.raises(ValueError, match="synthetic"):
        PiperPolicy.from_checkpoint(path, metadata)


def test_offline_import_does_not_load_device_modules():
    code = "import sys,json; import egomimic.robot.piper_pico.policy_inference; print(json.dumps([k for k in sys.modules if k in ('piper_sdk','egomimic.robot.piper_pico.env','egomimic.robot.piper_pico.arm_process','egomimic.robot.piper_pico.pico_process')]))"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert json.loads(proc.stdout.splitlines()[-1]) == []
