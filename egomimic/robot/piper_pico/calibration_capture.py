"""Stationary calibration snapshots, independent of control-time observations.

The worker only reads device buffers. The control thread alone sends commands.
Image and measured pose are copied before detection; detection never selects a
replacement pose. Source CAN stamps are freshness evidence, not exposure sync.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def detection_feedback(result):
    reasons = {
        "no_board": "没有识别到标定板",
        "image_size_or_format_mismatch": "图像尺寸与内参配置不一致",
        "insufficient_corners": f"角点不足（识别到 {result.get('corner_count', 0)} 个）",
        "invalid_corners": "角点数据无效",
        "insufficient_spatial_spread": "角点分布过窄或标定板在图中太小",
        "pnp_failed": "无法求出标定板位姿",
        "pnp_no_positive_depth": "标定板位姿的深度无效",
        "pnp_invalid_refinement": "标定板位姿求解无效",
        "high_reprojection_error": "重投影误差超过门限",
        "opencv_detection_error": "图像检测异常",
    }
    return reasons.get(result["reason"], result["reason"])


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )


@dataclass(frozen=True)
class CaptureLimits:
    stable_seconds: float = 0.7
    post_stable_seconds: float = 0.15
    capture_timeout: float = 3.0
    max_pose_age: float = 0.15
    max_camera_age: float = 0.3
    position_tolerance_mm: float = 1.0
    rotation_tolerance_deg: float = 0.3
    gripper_tolerance_mm: float = 0.5

    def validate(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("所有采样阈值必须为有限正数")
        if self.stable_seconds + self.post_stable_seconds > 1.0:
            raise ValueError("前后稳定窗口总长最多 1 秒（机械臂缓存约 1.28 秒）")
        if self.capture_timeout <= self.stable_seconds + self.post_stable_seconds:
            raise ValueError("采样等待时间必须大于前后稳定窗口")


def pose_matrix(pose) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    if not np.isfinite(pose).all():
        raise ValueError("机械臂 pose 含无效值")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def check_stability(
    history: dict, start: float, end: float, limits: CaptureLimits
) -> dict:
    """Check measured movement AND source freshness throughout a closed interval."""
    result = {"ok": False, "start": start, "end": end}

    def fail(reason):
        return {**result, "reason": reason}

    try:
        times = np.asarray(history["robot_receive_timestamp"], dtype=float).reshape(-1)
        if (
            times.size < 2
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
        ):
            return fail("机械臂反馈时间戳缺失或不连续")
        # Include the observation immediately before the interval boundary.
        first = max(0, int(np.searchsorted(times, start, side="right")) - 1)
        last = int(np.searchsorted(times, end, side="right"))
        selection = slice(first, last)
        stamps = times[selection]
        if stamps.size < 2 or stamps[0] > start or stamps[-1] < end - 0.04:
            return fail("等待稳定窗口的完整反馈")
        if np.max(np.diff(stamps)) > limits.max_pose_age:
            return fail("机械臂反馈中断")
        if np.asarray(history["robot_source_timestamps_simulated"])[selection].any():
            return fail("模拟反馈不能用于真机标定")
        for key in ("robot_pose_source_timestamp", "robot_gripper_source_timestamp"):
            source = np.asarray(history[key], dtype=float)[selection]
            if not np.isfinite(source).all() or np.any(source <= 0):
                return fail("SDK 来源时间戳缺失或读取期间变化")
            age = stamps - source
            if np.any(age < -0.02) or np.any(age > limits.max_pose_age):
                return fail("SDK 实际反馈过期")
            if np.any(np.diff(source) < 0) or np.unique(source).size < 2:
                return fail("SDK 实际反馈未更新")
            if np.max(np.diff(np.unique(source))) > limits.max_pose_age:
                return fail("SDK 实际反馈有时间缺口")
        poses = np.asarray(history["ee_pose"], dtype=float)[selection]
        grip = np.asarray(history["gripper_position"], dtype=float)[selection]
        if (
            poses.shape != (len(stamps), 6)
            or not np.isfinite(poses).all()
            or not np.isfinite(grip).all()
        ):
            return fail("机械臂实际位姿无效")
        # Full translation range prevents opposite excursions hiding in an average.
        delta = poses[:, None, :3] - poses[None, :, :3]
        pos_mm = float(np.linalg.norm(delta, axis=-1).max() * 1000)
        rotations = Rotation.from_euler("xyz", poses[:, 3:])
        quaternions = rotations.as_quat()
        similarities = np.clip(np.abs(quaternions @ quaternions.T), 0, 1)
        rotation_range = float(np.rad2deg(2 * np.arccos(similarities)).max())
        rot_deg = float(np.rad2deg((rotations[-1].inv() * rotations).magnitude()).max())
        grip_mm = float(np.ptp(grip) * 1000)
        result.update(
            position_range_mm=pos_mm,
            rotation_from_latest_deg=rot_deg,
            rotation_range_deg=rotation_range,
            gripper_range_mm=grip_mm,
            feedback_count=len(stamps),
        )
        if (
            pos_mm > limits.position_tolerance_mm
            or rotation_range > limits.rotation_tolerance_deg
        ):
            return fail("机械臂仍在移动")
        if grip_mm > limits.gripper_tolerance_mm:
            return fail("夹爪仍在变化")
        return {**result, "ok": True, "reason": "稳定"}
    except (KeyError, ValueError, IndexError, TypeError) as exc:
        return fail(f"反馈数据不完整：{exc}")


def freeze_snapshot(camera_data: dict, history: dict, now: float) -> dict:
    """Copy a camera frame and the latest actual arm feedback before detection."""
    image = np.asarray(camera_data["color"])[-1].copy()
    image.setflags(write=False)
    state = {key: np.asarray(value)[-1].copy() for key, value in history.items()}
    copied_history = {key: np.asarray(value).copy() for key, value in history.items()}
    return {
        "image": image,
        "state": state,
        "feedback": copied_history,
        "snapshot_timestamp": now,
        "camera_timestamp": float(np.asarray(camera_data["timestamp"])[-1]),
        "camera_receive_timestamp": float(
            np.asarray(camera_data["camera_receive_timestamp"])[-1]
        ),
        "T_base_end": pose_matrix(state["ee_pose"]),
    }


class SessionStore:
    """One board mounting, one selected base. Never append to an old session."""

    def __init__(self, path: Path, metadata: dict):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "samples").mkdir()
        self.metadata = metadata
        self.saved: list[Path] = []
        self.next_id = 1
        write_json(self.path / "session.json", metadata)

    def event(self, kind: str, **fields):
        with (self.path / "events.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    json_value({"timestamp": time.time(), "event": kind, **fields}),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )

    def save(self, snapshot, detection, overlay, request, stability) -> Path:
        import cv2

        identifier = f"{self.next_id:06d}"
        temporary = self.path / "samples" / f".{identifier}.pending"
        final = self.path / "samples" / identifier
        temporary.mkdir(exist_ok=False)
        for name, rgb in (("raw.png", snapshot["image"]), ("overlay.jpg", overlay)):
            if not cv2.imwrite(
                str(temporary / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ):
                raise OSError(f"图像写入失败：{name}")
        np.savez_compressed(temporary / "feedback.npz", **snapshot["feedback"])
        record = {
            "schema_version": 1,
            "sample_id": identifier,
            "arm": self.metadata["arm"],
            "split": request["split"],
            "request_timestamp": request["timestamp"],
            "snapshot_timestamp": snapshot["snapshot_timestamp"],
            "camera_timestamp": snapshot["camera_timestamp"],
            "camera_receive_timestamp": snapshot["camera_receive_timestamp"],
            "state": snapshot["state"],
            "T_base_end": snapshot["T_base_end"],
            "detection": detection,
            "stability": stability,
            "image": "raw.png",
            "overlay": "overlay.jpg",
            "feedback": "feedback.npz",
            "pairing": "latest copied image + measured SDK pose; stationary; no interpolation",
            "transform_convention": "T_A_B maps coordinates from B to A; metres; right handed",
        }
        write_json(temporary / "sample.json", record)
        # A numeric directory with sample.json is a committed sample. Incomplete
        # writes retain their hidden .pending directory and are never counted.
        os.rename(temporary, final)
        self.next_id += 1
        self.saved.append(final)
        return final

    def invalidate_last(self) -> Path | None:
        if not self.saved:
            return None
        path = self.saved[-1]
        write_json(
            path / "invalidated.json",
            {"timestamp": time.time(), "reason": "operator undo"},
        )
        self.saved.pop()
        return path


class CaptureWorker:
    """Bounded capture state and vision/I/O worker; never sends arm commands."""

    def __init__(
        self, camera, arm, estimator, store: SessionStore, limits: CaptureLimits
    ):
        self.camera, self.arm, self.estimator = camera, arm, estimator
        self.store, self.limits = store, limits
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="calibration-capture", daemon=False
        )
        self.pending = None
        self.undo_requested = False
        self.enabled = True
        self.generation = 0
        self.preview = None
        self.status = "等待图像"
        self.feedback_label = "READY"
        self.fatal_error = None
        self.saved_count = 0
        self.counts = {"train": 0, "validation": 0}

    def start(self):
        self.thread.start()

    def close(self):
        self.cancel("退出，取消未完成的采样")
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError("标定后台未及时退出，已提交样本仍保留")

    def say(self, message, label=None):
        with self.lock:
            changed = message != self.status
            self.status = message
            if label is not None:
                self.feedback_label = label
        if changed:
            print(f"[标定] {message}", flush=True)

    def request(self, split: str) -> bool:
        with self.lock:
            if self.pending is not None or self.undo_requested or not self.enabled:
                return False
            self.pending = {
                "timestamp": time.time(),
                "monotonic": time.monotonic(),
                "split": split,
                "generation": self.generation,
            }
        self.say(
            f"等待 {self.limits.stable_seconds:g} 秒静止：{split}",
            "WAITING FOR STILLNESS",
        )
        return True

    def undo(self) -> bool:
        with self.lock:
            if self.pending is not None or self.undo_requested:
                return False
            self.undo_requested = True
        return True

    def cancel(self, reason):
        with self.lock:
            self.generation += 1
            pending = self.pending
            self.pending = None
        if pending is not None:
            self.say(reason, "CANCELLED")

    def set_enabled(self, enabled):
        with self.lock:
            self.enabled = enabled
        if not enabled:
            self.cancel("回 home，取消未完成的采样")

    def display(self):
        with self.lock:
            return self.preview, self.status, dict(self.counts)

    def _active(self, request):
        with self.lock:
            return (
                self.pending is request
                and request["generation"] == self.generation
                and not self.stop_event.is_set()
            )

    def _finish(self, request, message, label="NOT SAVED"):
        with self.lock:
            if self.pending is not request:
                return
            self.pending = None
        self.store.event("capture_result", message=message, request=request)
        self.say(message, label)

    def _post_check(self, snapshot):
        end = snapshot["snapshot_timestamp"] + self.limits.post_stable_seconds
        # Wait in this worker only, never in the teleoperation loop.
        if self.stop_event.wait(max(0, end - time.time()) + 0.02):
            return {"ok": False, "reason": "退出"}
        history = self.arm.get_all_state()
        result = check_stability(
            history,
            snapshot["snapshot_timestamp"] - self.limits.stable_seconds,
            end,
            self.limits,
        )
        if result["ok"]:
            # Retain original frozen state, plus the whole pre/post audit window.
            snapshot["feedback"] = {
                key: np.asarray(value).copy() for key, value in history.items()
            }
        return result

    def _process(self):
        with self.lock:
            request = self.pending
            undo = self.undo_requested
        if undo:
            path = self.store.invalidate_last()
            if path is not None:
                split = json.loads((path / "sample.json").read_text())["split"]
                with self.lock:
                    self.saved_count -= 1
                    self.counts[split] -= 1
            with self.lock:
                self.undo_requested = False
            self.say(
                f"已作废 {path.name}，原文件保留" if path else "没有可作废的样本",
                f"INVALIDATED {path.name}" if path else "NO SAMPLES TO UNDO",
            )
        camera_data = self.camera.get_last(1)
        history = self.arm.get_all_state()
        now = time.time()
        snapshot = freeze_snapshot(camera_data, history, now)
        image_stamp = snapshot["camera_timestamp"]
        received = snapshot["camera_receive_timestamp"]
        frame_fresh = (
            np.isfinite([image_stamp, received]).all()
            and -0.02 <= now - received <= self.limits.max_camera_age
            and image_stamp <= received + 0.001
        )
        stable = check_stability(
            history, now - self.limits.stable_seconds, now, self.limits
        )
        ready = (
            request is not None
            and self._active(request)
            and frame_fresh
            and stable["ok"]
            and now - request["timestamp"] >= self.limits.stable_seconds
            and image_stamp
            >= max(now - self.limits.stable_seconds, request["timestamp"])
        )
        # The pair is already copied. No arm read below can replace its pose.
        detection = self.estimator.detect(snapshot["image"])
        overlay = self.estimator.annotate(snapshot["image"], detection)
        with self.lock:
            self.preview = (overlay, detection, received, stable)
        if request is None or not self._active(request):
            return
        if time.monotonic() - request["monotonic"] > self.limits.capture_timeout:
            reason = "图像过期" if not frame_fresh else stable["reason"]
            if frame_fresh and stable["ok"] and not ready:
                reason = "等待本次静止窗口内的新图像"
            if ready:
                reason = (
                    detection_feedback(detection)
                    if not detection["accepted"]
                    else "检测耗时超出采样等待时间"
                )
            self._finish(request, f"未保存：等待超时（{reason}）")
            return
        if not ready:
            return
        if not detection["accepted"]:
            self._finish(request, f"未保存：{detection_feedback(detection)}")
            return
        self.say("快照已固定，检查图像与采样后的稳定性", "CHECKING FROZEN SNAPSHOT")
        post = self._post_check(snapshot)
        if not self._active(request):
            return
        if not post["ok"]:
            self._finish(request, f"未保存：{post['reason']}")
            return
        # Approve this already stable snapshot before I/O. No control-thread
        # reader waits for disk writes; later movement cannot alter the pair.
        with self.lock:
            if self.pending is not request or self.generation != request["generation"]:
                return
        path = self.store.save(snapshot, detection, overlay, request, post)
        with self.lock:
            self.saved_count += 1
            self.counts[request["split"]] += 1
        message = (
            f"已保存 {path.name} [{request['split']}]，"
            f"{detection['corner_count']} 角点，RMS {detection['reprojection_rms_px']:.3f} px"
        )
        if self._active(request):
            self._finish(request, message, f"SAVED {path.name}")
        else:
            self.store.event("capture_saved_before_cancellation", sample=path.name)
            self.say(message, f"SAVED {path.name}")

    def _run(self):
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                self._process()
                self.stop_event.wait(max(0, 0.1 - (time.monotonic() - started)))
        except Exception as exc:
            self.fatal_error = exc
            self.say(f"后台采样失败：{exc}")


class CalibrationControls:
    """Button routing only; normal targets still come from the unchanged env."""

    def __init__(self, arm: str, home_seconds: float):
        self.arm = arm
        self.home_seconds = home_seconds
        self.home_until = 0.0
        self.split = "train"
        self.held = {key: False for key in "xyab"}

    def edges(self, sample, keyboard=()):
        pressed = set(keyboard)
        for key in self.held:
            current = float(sample[f"button_{key}"]) > 0.5
            if current and not self.held[key]:
                pressed.add(key)
            self.held[key] = current
        return pressed

    def tick(self, env, worker, sample, pressed, now, dt):
        if pressed & {"q", "\x1b"}:
            return False
        if "v" in pressed:
            self.split = "validation" if self.split == "train" else "train"
            print(f"[标定] 下一次采样：{self.split}", flush=True)
        if "u" in pressed and not worker.undo():
            print("[标定] 采样处理中，暂不能作废", flush=True)
        homes = [index for index, key in enumerate(("x", "a")) if key in pressed]
        if homes:
            worker.set_enabled(False)
            for index in homes:
                # Preserve the effective clamp command while moving home; measured
                # width alone would release gripping force against the handle.
                action = env.last_action
                grip = float(action[index * 7 + 6]) if action is not None else 0.0
                env.arms[index].go_home(gripper_m=grip)
                print(f"[标定] {('左', '右')[index]}臂回 home", flush=True)
            self.home_until = now + self.home_seconds
        elif now >= self.home_until:
            if self.home_until > 0:
                env.seed_clutches_from_measured()
                self.home_until = 0.0
                worker.set_enabled(True)
            action = env.teleop_action(sample, now)
            env.exec_actions(action, np.array([time.time() + dt]))
        for key, side in (("y", "left"), ("b", "right")):
            if key not in pressed:
                continue
            if side != self.arm:
                print(f"[标定] 当前只记录 {self.arm}，忽略 {key.upper()}", flush=True)
            elif not worker.request(self.split):
                print("[标定] 正在回 home 或处理已有请求，本次按键未排队", flush=True)
        return True
