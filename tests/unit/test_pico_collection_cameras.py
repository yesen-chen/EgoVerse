"""Mixed PICO/UVC recording contracts; no SDK, CAN, USB or network access."""

import json
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from egomimic.robot.piper_pico import collect, pico_camera, video_recording
from egomimic.robot.piper_pico import env as env_module


def test_cli_keeps_headset_and_uvc_rates_separate_and_orders_front_first():
    args = collect.build_parser().parse_args([
        "--headset-ip", "192.0.2.1", "--pc-ip", "192.0.2.2",
        "--camera", "left_wrist=/dev/video0",
        "--camera", "right_wrist=/dev/video2",
        "--frequency", "30", "--camera-fps", "30", "--pico-camera-fps", "60",
        "--camera-latency", "0.08", "--pico-camera-latency", "0.13",
    ])
    specs = collect.camera_specs_from_args(args)
    assert [x["name"] for x in specs] == ["front_1", "left_wrist", "right_wrist"]
    assert [x["type"] for x in specs] == ["pico", "uvc", "uvc"]
    assert [x["fps"] for x in specs] == [60, 30, 30]
    assert [x["latency"] for x in specs] == [0.13, 0.08, 0.08]
    assert specs[0]["eye"] == "left"
    assert (specs[0]["width"], specs[0]["height"]) == (1080, 810)
    assert (specs[1]["width"], specs[1]["height"]) == (640, 480)
    assert args.frequency == 30


def test_uvc_only_configuration_is_still_supported():
    specs = collect.camera_specs_from_args(
        collect.build_parser().parse_args(["--camera", "test_uvc=0"])
    )
    assert len(specs) == 1
    assert specs[0]["device"] == "0"
    assert specs[0]["type"] == "uvc"


def test_no_camera_fails_before_starting_hardware(monkeypatch):
    manager = Mock(side_effect=AssertionError("must not allocate hardware"))
    monkeypatch.setattr(collect, "SharedMemoryManager", manager)
    with pytest.raises(SystemExit):
        collect.main([])
    manager.assert_not_called()


@pytest.fixture
def fake_env_resources(monkeypatch):
    created = []

    def camera(manager, **kwargs):
        obj = SimpleNamespace(**kwargs)
        obj.start = Mock()
        obj.stop = Mock()
        obj.start_recording = Mock()
        obj.stop_recording = Mock()
        obj.process = SimpleNamespace(exitcode=None)
        created.append(obj)
        return obj

    monkeypatch.setattr(env_module, "CameraProcess", camera)
    monkeypatch.setattr(env_module, "PicoCameraProcess", camera)
    monkeypatch.setattr(env_module, "PiperArmProcess", lambda *args, **kwargs: object())
    monkeypatch.setattr(env_module, "PicoProcess", lambda *args, **kwargs: object())
    monkeypatch.setattr(env_module, "ReplayBuffer", lambda *args: SimpleNamespace(
        root=SimpleNamespace(attrs={}), n_episodes=0, add_episode=Mock()
    ))
    return created


def mixed_specs():
    return [
        {"type": "pico", "name": "front_1", "headset_ip": "192.0.2.1"},
        {"name": "left_wrist", "device": "/dev/video0"},
        {"name": "right_wrist", "device": "/dev/video2"},
    ]


def test_mixed_camera_factory_and_metadata(tmp_path, fake_env_resources):
    env = env_module.PiperPicoEnv(object(), tmp_path, cameras=mixed_specs())
    assert [camera.name for camera in env.cameras] == ["front_1", "left_wrist", "right_wrist"]
    assert env.cameras[0].fps == 60
    assert env.cameras[0].eye == "left"
    assert env.cameras[1].fps == env.cameras[2].fps == 30
    assert json.loads((tmp_path / "videos/cameras.json").read_text()) == [
        "front_1", "left_wrist", "right_wrist"
    ]
    assert json.loads((tmp_path / "videos/camera_config.json").read_text()) == env.camera_specs
    changed = mixed_specs()
    changed[0]["fps"] = 30
    with pytest.raises(ValueError, match="新的 output"):
        env_module.PiperPicoEnv(object(), tmp_path, cameras=changed)


def test_integer_zero_device_is_valid(tmp_path, fake_env_resources):
    env = env_module.PiperPicoEnv(object(), tmp_path, cameras=[{"name": "front_1", "device": 0}])
    assert env.cameras[0].device == 0


def test_uvc_settings_and_calibration_are_forwarded_without_changing_requested_metadata(
    tmp_path, fake_env_resources,
):
    spec = {"name": "left_wrist", "device": "/dev/video0",
            "controls": {"focus_auto": 0, "focus_absolute": 120},
            "expected_identity": {"ID_SERIAL": "camera-a"},
            "calibration": {"schema": "uvc_intrinsics_calibration_v1", "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            "latency_status": "unmeasured"}
    env = env_module.PiperPicoEnv(object(), tmp_path, cameras=[spec])
    for key in ("controls", "expected_identity", "calibration", "latency_status"):
        expected = {"brightness": 48, **spec[key]} if key == "controls" else spec[key]
        assert getattr(env.cameras[0], key) == expected
        assert env.camera_specs[0][key] == expected
    assert "brightness" not in spec["controls"]
    assert json.loads((tmp_path / "videos/camera_config.json").read_text()) == env.camera_specs


def test_three_uvc_brightness_reaches_workers_and_saved_metadata(tmp_path, fake_env_resources):
    specs = [
        {"name": "front_1", "device": "0", "controls": {"focus_auto": 0}},
        {"name": "left_wrist", "device": "2", "controls": {"brightness": 0}},
        {"name": "right_wrist", "device": "4", "controls": {"brightness": 40}},
    ]
    env = env_module.PiperPicoEnv(object(), tmp_path, cameras=specs)
    assert [camera.controls["brightness"] for camera in env.cameras] == [48, 0, 40]
    saved = json.loads((tmp_path / "videos/camera_config.json").read_text())
    assert [spec["controls"]["brightness"] for spec in saved] == [48, 0, 40]
    assert env.cameras[0].controls["focus_auto"] == 0
    assert specs[0]["controls"] == {"focus_auto": 0}
    # Do not silently pretend the new default was used for earlier recordings.
    saved[0]["controls"].pop("brightness")
    (tmp_path / "videos/camera_config.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="新的 output"):
        env_module.PiperPicoEnv(object(), tmp_path, cameras=specs)


def test_legacy_camera_config_without_latency_status_can_resume(tmp_path, fake_env_resources):
    env = env_module.PiperPicoEnv(object(), tmp_path, cameras=mixed_specs())
    path = tmp_path / "videos/camera_config.json"
    legacy = json.loads(path.read_text())
    for row in legacy:
        row.pop("latency_status")
    path.write_text(json.dumps(legacy))
    resumed = env_module.PiperPicoEnv(object(), tmp_path, cameras=mixed_specs())
    assert resumed.camera_specs == env.camera_specs


@pytest.mark.parametrize("specs,match", [
    ([{"name": "same", "device": "0"}, {"name": "same", "device": "2"}], "unique"),
    ([{"name": "a", "device": "0"}, {"name": "b", "device": "/dev/video0"}], "opened once"),
    ([{"name": "a", "type": "pico", "headset_ip": "192.0.2.1"},
      {"name": "b", "type": "pico", "headset_ip": "192.0.2.1"}], "one PICO"),
])
def test_invalid_mix_fails_before_devices_are_constructed(tmp_path, fake_env_resources, specs, match):
    with pytest.raises(ValueError, match=match):
        env_module.PiperPicoEnv(object(), tmp_path, cameras=specs)
    assert fake_env_resources == []


def test_episode_start_save_and_drop_reach_all_three_streams(tmp_path, fake_env_resources, monkeypatch):
    env = env_module.PiperPicoEnv(object(), tmp_path, cameras=mixed_specs())
    env.start_episode()
    for index, camera in enumerate(env.cameras):
        camera.start_recording.assert_called_once_with(str(tmp_path / f"videos/0/{index}.mp4"))
    episode = {"timestamp": np.array([1.0]), "action": np.zeros((1, 14))}
    monkeypatch.setattr(env_module, "episode_from_ticks", lambda *args, **kwargs: episode)
    env.end_episode()
    assert (tmp_path / "videos/0/robot0_state.npz").exists()
    for camera in env.cameras:
        camera.stop_recording.assert_called_once()
    env.replay_buffer.add_episode.assert_called_once_with(episode)
    assert not env._recording
    env.replay_buffer.n_episodes = 1
    env.start_episode()
    for index in range(3):
        (tmp_path / f"videos/1/{index}.mp4").write_bytes(b"partial video")
    env.drop_episode()
    assert not (tmp_path / "videos/1").exists()
    assert (tmp_path / "videos/0").exists()
    assert all(x.stop_recording.call_count == 2 for x in env.cameras)


def test_start_failure_closes_already_started_camera_and_retains_partial_files(
    tmp_path, fake_env_resources
):
    env = env_module.PiperPicoEnv(object(), tmp_path, cameras=mixed_specs())
    env.cameras[1].start_recording.side_effect = RuntimeError("open failed")
    with pytest.raises(RuntimeError, match="open failed"):
        env.start_episode()
    assert not env._recording
    env.cameras[0].stop_recording.assert_called_once()
    env.cameras[1].stop_recording.assert_called_once()
    env.cameras[2].start_recording.assert_not_called()
    with pytest.raises(RuntimeError, match="未提交"):
        env.start_episode()


def test_stop_failure_closes_every_stream_and_does_not_commit_episode(tmp_path, fake_env_resources):
    env = env_module.PiperPicoEnv(object(), tmp_path, cameras=mixed_specs())
    env.start_episode()
    env.cameras[0].stop_recording.side_effect = RuntimeError("disk error")
    with pytest.raises(RuntimeError, match="disk error"):
        env.end_episode()
    assert all(x.stop_recording.call_count == 1 for x in env.cameras)
    env.replay_buffer.add_episode.assert_not_called()
    assert not env._recording
    assert (tmp_path / "videos/0").exists()


def test_dead_camera_is_not_reused_as_frozen_observations(tmp_path, fake_env_resources):
    env = env_module.PiperPicoEnv(object(), tmp_path, cameras=mixed_specs())
    env.is_ready = True
    env.cameras[0].process.exitcode = 1
    with pytest.raises(RuntimeError, match="front_1"):
        env.get_obs()


def test_decoded_pico_frames_record_with_matching_timestamps_and_no_control_poll(
    tmp_path, monkeypatch
):
    av = pytest.importorskip("av")
    commands, acks = Queue(), Queue()
    session = video_recording.CameraRecordingSession(commands, acks, 64, 48, 60)
    video = tmp_path / "front.mp4"
    commands.put({"id": "start", "cmd": "start", "path": str(video)})
    frames = []
    ring = SimpleNamespace(put=frames.append)
    ready = Event()
    try:
        for index in range(16):
            stamp = 100.0 + index / 60
            monkeypatch.setattr(pico_camera.time, "time", lambda t=stamp: t)
            bgr = np.full((48, 64, 3), (10, 30, 200), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
            pico_camera._publish_frame(
                frame, ready, 64, 48, 0.13, 0, ring, None, "left", recording=session
            )
        assert acks.get_nowait() == {"id": "start", "error": None}
        commands.put({"id": "stop", "cmd": "stop_record"})
        session.poll()
        assert acks.get_nowait() == {"id": "stop", "error": None}
    finally:
        session.close()
    assert ready.is_set()
    assert len(frames) == 16
    with av.open(str(video)) as container:
        decoded = list(container.decode(video=0))
    assert len(decoded) == 16
    assert decoded[0].width == 64 and decoded[0].height == 48
    rgb = decoded[0].to_ndarray(format="rgb24")
    assert rgb[:, :, 0].mean() > 180 and rgb[:, :, 2].mean() < 20
    np.testing.assert_allclose(
        np.load(video.with_suffix(".npy")), [x["timestamp"] for x in frames]
    )
    np.testing.assert_allclose(
        np.load(video.with_suffix(".receive_timestamps.npy")),
        [x["camera_receive_timestamp"] for x in frames],
    )
    assert (len(frames) - 1) / (frames[-1]["timestamp"] - frames[0]["timestamp"]) == pytest.approx(60)


def test_record_open_failure_returns_error_ack(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(video_recording, "_open_writer", fail)
    commands, acks = Queue(), Queue()
    session = video_recording.CameraRecordingSession(commands, acks, 64, 48, 60)
    commands.put({"id": "start", "cmd": "start", "path": str(tmp_path / "x.mp4")})
    with pytest.raises(RuntimeError, match="disk full"):
        session.poll()
    assert "disk full" in acks.get_nowait()["error"]


def test_recording_queue_overflow_is_explicit_and_accepted_frames_drain(tmp_path, monkeypatch):
    entered, release = Event(), Event()

    def encode(frame=None):
        if frame is not None:
            entered.set()
            assert release.wait(2)
        return []

    monkeypatch.setattr(video_recording, "_open_writer", lambda *args, **kwargs: (
        SimpleNamespace(close=lambda: None, mux=lambda packet: None),
        SimpleNamespace(encode=encode),
    ))
    video = tmp_path / "bounded.mp4"
    writer = video_recording.AsyncVideoRecorder(video, 4, 4, 60, queue_size=1)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    try:
        writer.write(rgb, 1.0, 1.1)
        assert entered.wait(2)
        writer.write(rgb, 2.0, 2.1)
        with pytest.raises(RuntimeError, match="队列已满"):
            writer.write(rgb, 3.0, 3.1)
    finally:
        release.set()
        writer.close()
    np.testing.assert_allclose(np.load(video.with_suffix(".npy")), [1.0, 2.0])
