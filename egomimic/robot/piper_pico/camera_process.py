"""One UVC camera process.

Frames go into a ring at the camera rate. ``camera_receive_timestamp`` is the
host time immediately after ``VideoCapture.read``. Native frames must match
the requested size; collection never resizes them.
``timestamp`` is that receive time minus the configured latency. While
recording, the same frames are written to mp4 and their calibrated timestamps
are stored beside the video.

The latency script reads this process, so a measured ``--camera-latency``
matches collection. On 2026-10-18, MJPG 640x480 at 30 fps measured 77.3 ms.
"""

from __future__ import annotations

import json
import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from queue import Empty

import numpy as np

from egomimic.robot.piper_pico.shm import SharedMemoryRingBuffer
from egomimic.robot.piper_pico.uvc_controls import (
    validate_controls,
    verify_controls,
    verify_identity,
)


def stop_camera_process(process, stop_event, timeout: float, name: str) -> None:
    """Do not mistake a forced termination (or a stuck driver) for clean shutdown."""
    stop_event.set()
    if process.pid is None:
        return
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            raise TimeoutError(
                f"相机 {name} 退出超时，PID {process.pid} 在终止请求后仍未退出；"
                "可能阻塞在设备驱动，请勿立即重开设备"
            )
        raise TimeoutError(
            f"相机 {name} 未在 {timeout:g}s 内正常退出，已强制终止；"
            "设备释放和录像收尾未获确认"
        )
    if process.exitcode != 0:
        raise RuntimeError(f"相机 {name} 异常退出，code {process.exitcode}")


def camera_examples(height: int, width: int) -> dict:
    return {
        "color": np.zeros((height, width, 3), dtype=np.uint8),
        "timestamp": np.float64(0.0),
        "camera_receive_timestamp": np.float64(0.0),
    }


def _open_writer(
    path: Path, width: int, height: int, fps: float, *, fast: bool = False
):
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=int(round(fps)))
    except Exception:
        stream = container.add_stream("mpeg4", rate=int(round(fps)))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    if fast:
        if stream.codec_context.name == "libx264":
            stream.options = {"preset": "ultrafast", "tune": "zerolatency", "crf": "18"}
        stream.codec_context.thread_count = 2
    return container, stream


def apply_uvc_format(capture, width: int, height: int, fps: float, fourcc: str) -> None:
    """Set the mode before the first read. Fourcc comes before size."""
    import cv2

    if fourcc:
        if len(fourcc) != 4:
            raise ValueError(f"fourcc must be 4 characters, got {fourcc!r}")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)


def camera_process_loop(
    device: str,
    width: int,
    height: int,
    fps: float,
    latency: float,
    fourcc: str,
    ring: SharedMemoryRingBuffer,
    command_queue,
    ack_queue,
    stop_event,
    ready_event,
    controls: dict | None = None,
    expected_identity: dict | None = None,
    calibration: dict | None = None,
    latency_status: str = "unmeasured",
    name: str | None = None,
) -> None:
    import cv2

    identity = verify_identity(device, expected_identity, calibration=calibration)
    # Apply focus/exposure before VideoCapture claims the node. Several UVC
    # cameras reject SET_CUR with Protocol error once streaming has started.
    settings = verify_controls(device, controls, calibration, apply=True)
    source = int(device) if str(device).isdigit() else device
    # CAP_ANY may select FFmpeg for /dev/videoN and ignore the mode setters.
    capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"打不开相机 {device}")
    announced = False
    container = None
    stream = None
    timestamps: list[float] = []
    receive_timestamps: list[float] = []
    video_path: Path | None = None
    recording_metadata = None

    def close_recording(container, stream, path, stamps, received_stamps, metadata):
        if container is None:
            return
        try:
            metadata["controls_at_end"] = verify_controls(device, controls, calibration)
            metadata["verified_at_end_unix_s"] = time.time()
        except Exception as exc:
            metadata["end_verification_error"] = str(exc)
            # Flush accepted frames even when a changed lens setting invalidates
            # the episode, but never acknowledge that recording as successful.
            try:
                _close_writer(
                    container, stream, path, stamps, received_stamps, metadata
                )
            finally:
                raise
        _close_writer(container, stream, path, stamps, received_stamps, metadata)

    try:
        apply_uvc_format(capture, width, height, fps, fourcc)
        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        if (
            (actual_width, actual_height) != (width, height)
            or (fourcc and actual_fourcc != cv2.VideoWriter_fourcc(*fourcc))
            or not np.isfinite(actual_fps)
            or abs(actual_fps - fps) > 0.5
        ):
            raise RuntimeError(
                f"UVC 模式未生效：请求 {fourcc} {width}x{height}@{fps:g}，"
                f"读回 {actual_width}x{actual_height}@{actual_fps:g}；"
                "请用 v4l2-ctl --list-formats-ext 检查支持的模式"
            )
        runtime_metadata = {
            "schema": "uvc_stream_metadata_v1",
            "name": name,
            "type": "uvc",
            "identity": identity,
            "requested_mode": {
                "width": width,
                "height": height,
                "fps": fps,
                "fourcc": fourcc,
            },
            "actual_mode": {
                "width": actual_width,
                "height": actual_height,
                "fps": float(actual_fps),
                "fourcc": "".join(
                    chr((actual_fourcc >> (8 * i)) & 255) for i in range(4)
                ),
            },
            "preprocessing": {"crop": None, "resize": None, "mirror": False},
            "latency": float(latency),
            "latency_status": latency_status,
            "timestamp_convention": "host_receive_time_minus_configured_latency",
            "receive_timestamp_convention": "host_time_immediately_after_VideoCapture.read",
            "controls_at_startup": settings,
            "calibration": calibration,
            "verified_at_startup_unix_s": time.time(),
            "verification_scope": "startup_and_recording_boundaries; no_continuous_control_monitor",
        }
        while not stop_event.is_set():
            while True:
                try:
                    command = command_queue.get_nowait()
                except Empty:
                    break
                kind = command.get("cmd")
                if kind == "start":
                    if container is not None:
                        raise RuntimeError("camera is already recording")
                    recording_metadata = {
                        **runtime_metadata,
                        "controls_at_recording_start": verify_controls(
                            device, controls, calibration
                        ),
                        "verified_at_recording_start_unix_s": time.time(),
                    }
                    video_path = Path(command["path"])
                    container, stream = _open_writer(video_path, width, height, fps)
                    timestamps, receive_timestamps = [], []
                    ack_queue.put("started")
                elif kind == "stop_record":
                    closing = (
                        container,
                        stream,
                        video_path,
                        timestamps,
                        receive_timestamps,
                        recording_metadata,
                    )
                    container, stream, timestamps, video_path = None, None, [], None
                    receive_timestamps, recording_metadata = [], None
                    close_recording(*closing)
                    ack_queue.put("stopped")
                elif kind == "stop":
                    stop_event.set()
            if stop_event.is_set():
                break
            ok, bgr = capture.read()
            received = time.time()
            if not ok:
                time.sleep(0.005)
                continue
            if (
                not isinstance(bgr, np.ndarray)
                or bgr.shape != (height, width, 3)
                or bgr.dtype != np.uint8
            ):
                raise RuntimeError(
                    f"UVC 原始图像尺寸/格式变化：{getattr(bgr, 'shape', None)} "
                    f"{getattr(bgr, 'dtype', None)}；预期 {(height, width, 3)} uint8，拒绝缩放"
                )
            if not announced:
                print(
                    f"[uvc] {device} V4L2 请求 {fourcc or '默认'} {width}x{height}@{fps:g}，"
                    f"驱动 {int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                    f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                    f"{capture.get(cv2.CAP_PROP_FPS):.1f} fps",
                    flush=True,
                )
                announced = True
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            calibrated = received - float(latency)
            ring.put(
                {
                    "color": rgb,
                    "timestamp": np.float64(calibrated),
                    "camera_receive_timestamp": np.float64(received),
                }
            )
            if not ready_event.is_set():
                ready_event.set()
            if container is not None and stream is not None:
                import av

                video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
                timestamps.append(calibrated)
                receive_timestamps.append(received)
    finally:
        try:
            # Stop USB streaming before potentially slow encoder/disk flushing.
            # The capture is owned and released by this worker only.
            capture.release()
        finally:
            close_recording(
                container,
                stream,
                video_path,
                timestamps,
                receive_timestamps,
                recording_metadata,
            )


def _close_writer(
    container,
    stream,
    video_path: Path | None,
    timestamps: list[float],
    receive_timestamps: list[float] | None = None,
    metadata: dict | None = None,
) -> None:
    if container is None:
        return
    try:
        if stream is not None:
            for packet in stream.encode():
                container.mux(packet)
    finally:
        container.close()
    if video_path is not None:
        if not timestamps:
            raise RuntimeError(f"本段录像没有收到图像：{video_path}")
        if receive_timestamps is not None and len(receive_timestamps) != len(
            timestamps
        ):
            raise RuntimeError(
                "UVC recording receive/calibrated timestamp counts differ"
            )
        np.save(
            video_path.with_suffix(".npy"), np.asarray(timestamps, dtype=np.float64)
        )
        if receive_timestamps is not None:
            np.save(
                video_path.with_suffix(".receive_timestamps.npy"),
                np.asarray(receive_timestamps, dtype=np.float64),
            )
        if metadata is not None:
            metadata = {
                **metadata,
                "frame_count": len(timestamps),
                "native_frame_shape": [
                    metadata["actual_mode"]["height"],
                    metadata["actual_mode"]["width"],
                    3,
                ],
            }
            path = video_path.with_suffix(".stream_metadata.json")
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
                + "\n"
            )
            temporary.replace(path)


class CameraProcess:
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        device: str,
        name: str,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        latency: float = 0.0,
        fourcc: str = "MJPG",
        get_max_k: int = 32,
        controls: dict | None = None,
        expected_identity: dict | None = None,
        calibration: dict | None = None,
        latency_status: str = "unmeasured",
    ):
        import multiprocessing as mp

        # A collector/preview may already have OpenCV or FFmpeg threads.
        # Forking that state can leave the new worker stuck inside encoding;
        # start a fresh interpreter and create its IPC with the same context.
        context = mp.get_context("spawn")
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.latency = float(latency)
        self.controls = validate_controls(controls)
        self.ring = SharedMemoryRingBuffer.create_from_examples(
            shm_manager,
            camera_examples(self.height, self.width),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=fps,
        )
        self.commands = context.Queue()
        self.acks = context.Queue()
        self.stop_event = context.Event()
        self.ready_event = context.Event()
        self.process = context.Process(
            target=camera_process_loop,
            args=(
                str(device),
                self.width,
                self.height,
                float(fps),
                self.latency,
                str(fourcc),
                self.ring,
                self.commands,
                self.acks,
                self.stop_event,
                self.ready_event,
                self.controls,
                expected_identity,
                calibration,
                latency_status,
                name,
            ),
            daemon=False,
        )

    def start(self) -> None:
        self.process.start()

    def get_last(self, k: int) -> dict:
        available = min(k, self.ring.count, self.ring.get_max_k)
        if available <= 0:
            raise RuntimeError(f"camera {self.name} has no frames")
        return self.ring.get_last_k(available)

    def start_recording(self, path: str, timeout: float = 5.0) -> None:
        self.commands.put({"cmd": "start", "path": path})
        self._wait_record_ack("started", timeout)

    def stop_recording(self, timeout: float = 10.0) -> None:
        self.commands.put({"cmd": "stop_record"})
        self._wait_record_ack("stopped", timeout)

    def _wait_record_ack(self, expected: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                message = self.acks.get(
                    timeout=min(0.1, max(0.001, deadline - time.monotonic()))
                )
            except Empty:
                if self.process.pid is None or self.process.exitcode is not None:
                    raise RuntimeError(
                        f"相机 {self.name} 进程未运行/已退出，code {self.process.exitcode}"
                    )
                continue
            if message != expected:
                raise RuntimeError(
                    f"camera {self.name}: expected {expected}, got {message}"
                )
            return
        raise TimeoutError(f"camera {self.name} timed out waiting for {expected}")

    def stop(self, timeout: float = 15.0) -> None:
        try:
            stop_camera_process(self.process, self.stop_event, timeout, self.name)
        finally:
            if self.process.pid is None or not self.process.is_alive():
                # All useful recording commands are acknowledged before stop.
                # Do not leave a parent queue feeder waiting on a dead consumer.
                for queue in (self.commands, self.acks):
                    queue.cancel_join_thread()
                    queue.close()
