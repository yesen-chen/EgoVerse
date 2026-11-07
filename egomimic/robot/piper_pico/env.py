"""Bimanual Piper + Pico collection environment.

``get_obs`` reads the device buffers, then takes the current wall time as the
observation time: each camera contributes its latest frame stamped at or
before it, and arm state is taken at it. Live inference calls the same method.
``exec_actions`` forwards absolute targets with a freshness deadline and a
command id; the arm process drops expired ones and sends the rest with MOVE P.
``record_tick`` logs one row per control tick while an episode is recording.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_pico.arm_process import PiperArmProcess
from egomimic.robot.piper_pico.camera_config import uvc_controls_with_defaults
from egomimic.robot.piper_pico.camera_process import CameraProcess
from egomimic.robot.piper_pico.clutch import (
    HandClutch,
    action_from_sample,
    xyzw_to_wxyz,
)
from egomimic.robot.piper_pico.interpolate import align_bimanual_obs, rpy_to_rotvec_pose
from egomimic.robot.piper_pico.pico_camera import PicoCameraProcess
from egomimic.robot.piper_pico.pico_process import PicoProcess
from egomimic.robot.piper_pico.record import (
    ObsAccumulator,
    ReplayBuffer,
    TickAccumulator,
    episode_from_ticks,
    save_robot_streams,
)
from egomimic.robot.piper_pico.units import continuous_euler_xyz


class PiperPicoEnv:
    def __init__(
        self,
        shm_manager,
        output_dir: str | Path,
        cameras: list[dict] | None = None,
        left_can: str = "can0",
        right_can: str = "can1",
        frequency: float = 30.0,
        horizon: int = 2,
        move_spd_rate: int = 50,
        robot_obs_latency: float = 0.0,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: float = 30.0,
        simulate: bool = False,
        enable_pico: bool = True,
        enable_motion: bool = True,
        preserve_gripper_on_enable: bool = False,
    ):
        if cameras is None:
            cameras = []
        self.simulate = bool(simulate)
        self.enable_pico = bool(enable_pico)
        self.enable_motion = bool(enable_motion)
        self.preserve_gripper_on_enable = bool(preserve_gripper_on_enable)
        if not self.simulate and len(cameras) < 1:
            raise ValueError("at least one camera is required")
        self.frequency = float(frequency)
        self.horizon = int(horizon)
        self.move_spd_rate = int(move_spd_rate)
        self.output_dir = Path(output_dir)
        self.video_dir = self.output_dir / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.camera_specs = []
        for spec in cameras:
            source = spec.get("type", "uvc")
            if source not in ("uvc", "pico"):
                raise ValueError(f"unknown camera type: {source}")
            defaults = (
                {"width": camera_width, "height": camera_height, "fps": camera_fps,
                 "fourcc": "MJPG"}
                if source == "uvc" else
                {"width": 1080, "height": 810, "fps": 60, "eye": "left",
                 "bitrate": 5_000_000, "pc_ip": None, "startup_timeout": 25.0}
            )
            normalized = {"latency": 0.0, "latency_status": "unmeasured",
                          **defaults, **spec, "type": source}
            if source == "uvc":
                normalized["controls"] = uvc_controls_with_defaults(spec.get("controls"))
            if not normalized.get("name"):
                raise ValueError("camera name cannot be empty")
            for key in ("width", "height", "fps"):
                if not np.isfinite(normalized[key]) or normalized[key] <= 0:
                    raise ValueError(f"camera {key} must be positive")
            if not np.isfinite(normalized["latency"]):
                raise ValueError("camera latency must be finite")
            if source == "uvc" and normalized.get("device") in (None, ""):
                raise ValueError("UVC camera requires a device")
            if source == "pico" and not normalized.get("headset_ip"):
                raise ValueError("PICO camera requires a headset IP")
            self.camera_specs.append(normalized)
        names = [spec["name"] for spec in self.camera_specs]
        if len(set(names)) != len(names):
            raise ValueError("camera names must be unique")
        uvc_devices = [
            str(Path("/dev/video" + str(int(spec["device"])) if str(spec["device"]).isdigit()
                     else spec["device"]).resolve())
            for spec in self.camera_specs if spec["type"] == "uvc"
        ]
        if len(set(uvc_devices)) != len(uvc_devices):
            raise ValueError("each UVC device may only be opened once")
        if sum(spec["type"] == "pico" for spec in self.camera_specs) > 1:
            raise ValueError("only one PICO image stream can use the video port")
        names_path = self.video_dir / "cameras.json"
        config_path = self.video_dir / "camera_config.json"
        for path, expected in ((names_path, names), (config_path, self.camera_specs)):
            if path.exists():
                existing = json.loads(path.read_text())
                if path == config_path:
                    # Legacy collections predate explicit latency provenance.
                    existing = [{"latency_status": "unmeasured", **row} for row in existing]
                if existing != expected:
                    raise ValueError(f"相机配置与已有数据不同：{path}；请使用新的 output 目录")
        self.replay_buffer = ReplayBuffer(self.output_dir / "replay_buffer.zarr")
        self.replay_buffer.root.attrs["frequency"] = self.frequency
        self.replay_buffer.root.attrs["move_spd_rate"] = self.move_spd_rate
        self.cameras = []
        for spec in self.camera_specs:
            common = dict(
                name=spec["name"], width=int(spec["width"]), height=int(spec["height"]),
                fps=float(spec["fps"]), latency=float(spec["latency"]),
            )
            if spec["type"] == "uvc":
                camera = CameraProcess(
                    shm_manager, device=spec["device"], fourcc=spec["fourcc"],
                    controls=spec.get("controls"), expected_identity=spec.get("expected_identity"),
                    calibration=spec.get("calibration"), latency_status=spec["latency_status"],
                    **common,
                )
            else:
                camera = PicoCameraProcess(
                    shm_manager, headset_ip=spec["headset_ip"], pc_ip=spec["pc_ip"],
                    eye=spec["eye"], bitrate=spec["bitrate"],
                    startup_timeout=spec["startup_timeout"], **common,
                )
            self.cameras.append(camera)
        names_path.write_text(json.dumps(names, indent=2))
        config_path.write_text(json.dumps(self.camera_specs, indent=2))
        self.arms = [
            PiperArmProcess(
                shm_manager,
                can_port=left_can,
                name="left",
                move_spd_rate=self.move_spd_rate,
                receive_latency=robot_obs_latency,
                simulate=self.simulate,
                enable_motion=self.enable_motion,
                preserve_gripper_on_enable=self.preserve_gripper_on_enable,
            ),
            PiperArmProcess(
                shm_manager,
                can_port=right_can,
                name="right",
                move_spd_rate=self.move_spd_rate,
                receive_latency=robot_obs_latency,
                simulate=self.simulate,
                enable_motion=self.enable_motion,
                preserve_gripper_on_enable=self.preserve_gripper_on_enable,
            ),
        ]
        self.pico = PicoProcess(shm_manager) if self.enable_pico else None
        self.left_clutch = HandClutch(name="left")
        self.right_clutch = HandClutch(name="right")
        self._last_rpy = [np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)]
        self.obs_accumulator = None
        self.tick_accumulator = None
        self._episode_id = None
        self._recording = False
        self._next_command_id = 0
        self.is_ready = False
        self.last_action = None

    def __enter__(self) -> "PiperPicoEnv":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self, timeout: float = 30.0) -> None:
        try:
            for camera in self.cameras:
                camera.start()
            for arm in self.arms:
                arm.start()
            if self.pico is not None:
                self.pico.start()
            self._wait_ready(timeout)
            self.seed_clutches_from_measured()
            self.is_ready = True
        except BaseException:
            # __exit__ is not called when __enter__ fails. Stop even the device
            # whose start raised: it may have launched a child before failing.
            try:
                self.stop()
            except BaseException as exc:
                print(f"[piper_pico] failed to clean up startup: {exc}")
            raise

    def stop(self) -> None:
        if self._recording:
            try:
                self.drop_episode()
            except Exception as exc:
                print(f"[piper_pico] failed to close the open episode: {exc}")
        for device in (self.pico, *self.arms, *self.cameras):
            if device is None:
                continue
            try:
                device.stop()
            except Exception as exc:
                print(f"[piper_pico] failed to stop a device: {exc}")
        self.is_ready = False

    def _wait_ready(self, timeout: float) -> None:
        devices = [device for device in (self.pico, *self.cameras, *self.arms) if device is not None]
        deadline = time.time() + timeout
        while time.time() < deadline:
            if all(device.ready_event.is_set() for device in devices):
                return
            for device in devices:
                if device.process.exitcode is not None:
                    raise RuntimeError(
                        f"{device.name} exited with code {device.process.exitcode}"
                    )
            time.sleep(0.05)
        missing = [device.name for device in devices if not device.ready_event.is_set()]
        raise TimeoutError(f"still waiting for {missing}")

    def seed_clutches_from_measured(self) -> None:
        for index, clutch in enumerate((self.left_clutch, self.right_clutch)):
            position, quat, rpy = self.measured_pose(index)
            clutch.seed(position, quat)
            self._last_rpy[index] = rpy.copy()

    def measured_pose(self, robot_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pose = np.asarray(self.arms[robot_idx].get_state()["ee_pose"], dtype=np.float64).reshape(6)
        quat = xyzw_to_wxyz(Rotation.from_euler("xyz", pose[3:]).as_quat())
        return pose[:3].copy(), quat, pose[3:].copy()

    def get_pico(self) -> dict:
        if self.pico is None:
            raise RuntimeError("PICO is disabled for this environment")
        return self.pico.get_latest()

    def teleop_action(self, sample: dict, now: float) -> np.ndarray:
        measured = [self.measured_pose(0), self.measured_pose(1)]
        action = action_from_sample(
            self.left_clutch,
            self.right_clutch,
            sample,
            (measured[0][0], measured[0][1]),
            (measured[1][0], measured[1][1]),
            now,
        )
        self.last_action = action
        return action

    def get_obs(self) -> dict:
        if not self.is_ready:
            raise RuntimeError("environment is not started")
        for camera in self.cameras:
            if camera.process.exitcode is not None:
                raise RuntimeError(
                    f"相机 {camera.name} 进程退出，code {camera.process.exitcode}"
                )
        for arm in self.arms:
            if arm.process.exitcode is not None:
                raise RuntimeError(f"机械臂 {arm.name} 进程退出，code {arm.process.exitcode}")
        horizon_scale = max(self.horizon * (60.0 / self.frequency) + 2.0, self.horizon)
        k = int(np.ceil(horizon_scale))
        camera_data = [camera.get_last(k) for camera in self.cameras]
        robot_data = [arm.get_all_state() for arm in self.arms]
        # Taken after the reads, so every buffered sample is already available.
        query_time = time.time()
        poses = [rpy_to_rotvec_pose(data["ee_pose"]) for data in robot_data]
        obs = align_bimanual_obs(
            camera_timestamps=[data["timestamp"] for data in camera_data],
            camera_colors=[data["color"] for data in camera_data],
            robot_timestamps=[data["robot_timestamp"] for data in robot_data],
            robot_poses_rotvec=poses,
            gripper_widths=[data["gripper_position"] for data in robot_data],
            frequency=self.frequency,
            horizon=self.horizon,
            query_time=query_time,
            robot_joint_positions=[data["joint_pos"] for data in robot_data],
        )
        # Preserve the collector's host-time alignment. Inference additionally
        # checks SDK receive stamps, which do not advance on a stale CAN cache.
        for index, data in enumerate(robot_data):
            for field in ("joint", "pose", "gripper"):
                key = f"robot_{field}_source_timestamp"
                if key in data:
                    obs[f"robot{index}_{field}_source_timestamp"] = float(data[key][-1])
        if self.obs_accumulator is not None:
            for robot_idx, data in enumerate(robot_data):
                grip = np.asarray(data["gripper_position"], dtype=np.float64).reshape(-1, 1)
                self.obs_accumulator.put(
                    {
                        f"robot{robot_idx}_eef_pose": poses[robot_idx],
                        f"robot{robot_idx}_joint_pos": data["joint_pos"],
                        f"robot{robot_idx}_joint_vel": data["joint_vel"],
                        f"robot{robot_idx}_gripper_width": grip,
                        f"robot{robot_idx}_last_command_id": data["last_command_id"],
                        f"robot{robot_idx}_last_command_send_timestamp": data[
                            "last_command_send_timestamp"
                        ],
                    },
                    timestamps=data["robot_timestamp"],
                )
        return obs

    def exec_actions(self, actions: np.ndarray, deadlines: np.ndarray) -> np.ndarray:
        """Queue targets that are still fresh; return their command ids, -1 if dropped.

        A deadline only expires a stale target. The arm sends a fresh one at once.
        """
        if not self.is_ready:
            raise RuntimeError("environment is not started")
        if not getattr(self, "enable_motion", True):
            raise RuntimeError("motion is disabled in observation mode")
        actions = np.asarray(actions, dtype=np.float64)
        deadlines = np.asarray(deadlines, dtype=np.float64).reshape(-1)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.shape[0] != deadlines.shape[0] or actions.shape[1] != 14:
            raise ValueError(f"expected actions (N, 14) and N deadlines, got {actions.shape}")
        command_ids = np.full(deadlines.shape[0], -1, dtype=np.int64)
        now = time.time()
        for index, (row, deadline) in enumerate(zip(actions, deadlines)):
            if not deadline > now:
                continue
            command_ids[index] = self._next_command_id
            self._next_command_id += 1
            for robot_idx, arm in enumerate(self.arms):
                start = robot_idx * 7
                pose_rpy, self._last_rpy[robot_idx] = _rotvec_to_continuous_rpy(
                    row[start : start + 6],
                    self._last_rpy[robot_idx],
                )
                arm.schedule_waypoint(
                    pose_rpy, float(row[start + 6]), float(deadline), int(command_ids[index])
                )
        return command_ids

    def record_tick(
        self,
        obs: dict,
        action: np.ndarray,
        decision_timestamp: float,
        deadline: float,
        command_id: int,
    ) -> None:
        """Log the observation of this tick and the command decided after it."""
        if self.tick_accumulator is None:
            return
        row = {
            "timestamp": np.float64(np.asarray(obs["timestamp"])[-1]),
            "decision_timestamp": np.float64(decision_timestamp),
            "deadline_timestamp": np.float64(deadline),
            "command_id": np.int64(command_id),
            "action": np.asarray(action, dtype=np.float64).reshape(14),
        }
        for robot_idx in range(len(self.arms)):
            for key in ("eef_pos", "eef_rot_axis_angle", "gripper_width", "joint_pos"):
                name = f"robot{robot_idx}_{key}"
                row[name] = np.asarray(obs[name], dtype=np.float64)[-1]
            name = f"robot{robot_idx}_source_timestamp"
            row[name] = np.float64(obs[name])
        for camera_idx in range(len(self.cameras)):
            name = f"camera{camera_idx}_timestamp"
            row[name] = np.float64(np.asarray(obs[name])[-1])
        self.tick_accumulator.put(row)

    def home(self) -> np.ndarray:
        if not getattr(self, "enable_motion", True):
            raise RuntimeError("motion is disabled in observation mode")
        command_id = self._next_command_id
        self._next_command_id += 1
        for arm in self.arms:
            arm.go_home(command_id=command_id)
        return np.full(len(self.arms), command_id, dtype=np.int64)

    def start_episode(self) -> int:
        if self._recording:
            raise RuntimeError("an episode is already recording")
        episode_id = self.replay_buffer.n_episodes
        episode_dir = self.video_dir / str(episode_id)
        if episode_dir.exists():
            raise RuntimeError(f"已有未提交的视频目录 {episode_dir}；请保留检查并使用新的 output")
        episode_dir.mkdir(parents=True)
        attempted = []
        try:
            for index, camera in enumerate(self.cameras):
                attempted.append(camera)
                camera.start_recording(str(episode_dir / f"{index}.mp4"))
        except BaseException:
            for camera in attempted:
                try:
                    camera.stop_recording()
                except Exception as exc:
                    print(f"[piper_pico] failed to roll back recording: {exc}")
            # Keep partial files for diagnosis, never silently overwrite them.
            raise
        self.obs_accumulator = ObsAccumulator()
        self.tick_accumulator = TickAccumulator()
        self._episode_id = episode_id
        self._recording = True
        print(f"Episode {episode_id} started")
        return episode_id

    def end_episode(self) -> None:
        if not self._recording:
            return
        errors = []
        for camera in self.cameras:
            try:
                camera.stop_recording()
            except Exception as exc:
                errors.append(f"{camera.name}: {exc}")
        if errors:
            self.obs_accumulator = None
            self.tick_accumulator = None
            self._recording = False
            raise RuntimeError("未保存不完整的数据段：" + "; ".join(errors))
        episode = None
        if self.obs_accumulator is not None and self.tick_accumulator is not None:
            streams = [self.obs_accumulator.stream(f"robot{i}_") for i in range(len(self.arms))]
            save_robot_streams(self.video_dir / str(self._episode_id), streams)
            episode = episode_from_ticks(self.tick_accumulator, streams)
        if episode is not None:
            self.replay_buffer.add_episode(episode)
            print(f"Episode {self.replay_buffer.n_episodes - 1} saved")
        else:
            print(f"Episode {self._episode_id} had no recorded ticks")
        self.obs_accumulator = None
        self.tick_accumulator = None
        self._recording = False

    def drop_episode(self) -> None:
        episode_id = self._episode_id
        if self._recording:
            for camera in self.cameras:
                try:
                    camera.stop_recording()
                except Exception as exc:
                    print(f"[piper_pico] failed to stop recording: {exc}")
            self.obs_accumulator = None
            self.tick_accumulator = None
            self._recording = False
        if episode_id is None:
            return
        video_dir = self.video_dir / str(episode_id)
        if video_dir.exists():
            shutil.rmtree(video_dir)
        print(f"Episode {episode_id} dropped")
        self._episode_id = None


def _rotvec_to_continuous_rpy(pose6: np.ndarray, reference_rpy: np.ndarray):
    pose = np.asarray(pose6, dtype=np.float64).reshape(6)
    rpy = continuous_euler_xyz(Rotation.from_rotvec(pose[3:]), reference_rpy)
    return np.concatenate([pose[:3], rpy]), rpy
