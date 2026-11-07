"""Pico headset camera, one process, shared memory for the selected eyes.

The headset sends one H.264 stream, either side-by-side or one selected eye.
This process decodes it and writes each enabled eye into its own ring using the same
fields as ``camera_process.py``. Inference can read the eyes together with a
UVC camera. ``camera_receive_timestamp`` is the host time when the decoded
frame becomes an array. ``timestamp`` subtracts that eye's latency.

Measured 2026-10-18 on the left eye, 2160x810 request at 30 fps: mean 130.4 ms.
The right eye has not been measured.
"""

from __future__ import annotations

import ipaddress
import socket
import struct
import time
import uuid
from multiprocessing.managers import SharedMemoryManager
from queue import Empty

import numpy as np

from egomimic.robot.piper_pico.camera_process import camera_examples, stop_camera_process
from egomimic.robot.piper_pico.shm import SharedMemoryRingBuffer

MAX_PACKET = 16 * 1024 * 1024
CAMERA_RENDER_MODES = {"stereo": 2, "left": 0, "right": 1}


def _enabled_eyes(mode: str) -> tuple[str, ...]:
    if mode not in CAMERA_RENDER_MODES:
        raise ValueError(f"camera mode must be stereo, left or right, got {mode!r}")
    return ("left", "right") if mode == "stereo" else (mode,)


def control_packet(command: str, payload: bytes = b"") -> bytes:
    command_bytes = command.encode("utf-8")
    body = (
        struct.pack("<I", len(command_bytes))
        + command_bytes
        + struct.pack("<I", len(payload))
        + payload
    )
    return struct.pack(">I", len(body)) + body


def camera_request(width: int, height: int, fps: int, bitrate: int, pc_ip: str, port: int, mode: str = "stereo") -> bytes:
    _enabled_eyes(mode)
    def compact(text: str) -> bytes:
        raw = text.encode("utf-8")
        if len(raw) > 255:
            raise ValueError("camera request string is too long")
        return bytes([len(raw)]) + raw

    # CameraRequestSerializer v1. PXRCaptureRenderMode: left=0, right=1, stereo=2.
    return (
        b"\xca\xfe\x01"
        + struct.pack("<7i", width, height, fps, bitrate, 0, CAMERA_RENDER_MODES[mode], port)
        + compact("VR")
        + compact(pc_ip)
    )


def route_ip(headset_ip: str) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect((headset_ip, 13579))
        return probe.getsockname()[0]


def release_pico_camera(headset_ip: str, control_port: int = 13579) -> None:
    """Drop a camera session left open by an interrupted run."""
    sock = socket.create_connection((headset_ip, control_port), timeout=5)
    try:
        sock.sendall(control_packet("CLOSE_CAMERA"))
        time.sleep(0.3)
    finally:
        sock.close()


def split_eyes(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    width = int(image.shape[1])
    if width % 2:
        raise ValueError(f"stereo width {width} is odd")
    half = width // 2
    return image[:, :half], image[:, half:]


class _StreamStopped(Exception):
    pass


class _StreamEOF(Exception):
    def __init__(self, received: int, expected: int):
        super().__init__(f"视频连接已断开：当前字段收到 {received}/{expected} 字节")


def _recv_exact(sock, length: int, stop_event, check_timeout) -> bytes:
    data = bytearray()
    while len(data) < length:
        if stop_event.is_set():
            raise _StreamStopped()
        check_timeout()
        try:
            part = sock.recv(length - len(data))
        except socket.timeout:
            continue
        if not part:
            raise _StreamEOF(len(data), length)
        data.extend(part)
    return bytes(data)


def _put_eye(ring: SharedMemoryRingBuffer, bgr: np.ndarray, received: float, latency: float, width: int, height: int):
    import cv2

    if bgr.shape[1] != width or bgr.shape[0] != height:
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ring.put(
        {
            "color": rgb,
            "timestamp": np.float64(received - float(latency)),
            "camera_receive_timestamp": np.float64(received),
        }
    )
    return rgb


def pico_camera_loop(
    headset_ip: str,
    pc_ip: str | None,
    stereo_width: int,
    stereo_height: int,
    fps: int,
    bitrate: int,
    latency_left: float,
    latency_right: float,
    left_ring: SharedMemoryRingBuffer | None,
    right_ring: SharedMemoryRingBuffer | None,
    stop_event,
    ready_event,
    startup_timeout: float,
    control_port: int = 13579,
    video_port: int = 12345,
    mode: str = "stereo",
    recording_commands=None,
    recording_acks=None,
    recording_eye: str | None = None,
) -> None:
    import av

    _enabled_eyes(mode)
    eye_width = stereo_width // 2
    eye_height = stereo_height
    stream_width = stereo_width if mode == "stereo" else eye_width
    host = socket.gethostbyname(headset_ip)
    local_ip = str(ipaddress.IPv4Address(pc_ip)) if pc_ip else route_ip(host)
    print("先关闭头显上一次可能未退出的相机", flush=True)
    try:
        release_pico_camera(host, control_port)
    except OSError as exc:
        print(f"[pico camera] 关闭残留会话失败：{exc}", flush=True)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    control = None
    open_sent = False
    recording = None
    try:
        if recording_eye is not None:
            from egomimic.robot.piper_pico.video_recording import CameraRecordingSession

            if mode != recording_eye:
                raise ValueError("recording requires one selected eye")
            recording = CameraRecordingSession(
                recording_commands, recording_acks, eye_width, eye_height, fps
            )
        server.bind(("0.0.0.0", video_port))
        server.listen(2)
        server.settimeout(0.2)
        bound_port = server.getsockname()[1]
        control = socket.create_connection((host, control_port), timeout=5)
        control.settimeout(1)
        control.sendall(
            control_packet(
                "OPEN_CAMERA",
                camera_request(stream_width, stereo_height, fps, bitrate, local_ip, bound_port, mode),
            )
        )
        open_sent = True
        print(f"已请求 {mode} {stream_width}×{stereo_height}，{fps} fps，等待头显把视频连到本机 {bound_port}", flush=True)
        try:
            _receive(
                server,
                host,
                stop_event,
                ready_event,
                startup_timeout,
                eye_width,
                eye_height,
                latency_left,
                latency_right,
                left_ring,
                right_ring,
                av,
                mode,
                recording=recording,
            )
        except _StreamStopped:
            pass
    finally:
        if control is not None:
            try:
                if open_sent:
                    control.sendall(control_packet("CLOSE_CAMERA"))
            except OSError as exc:
                print(f"[pico camera] 停止相机命令发送失败：{exc}", flush=True)
            control.close()
        try:
            server.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        server.close()
        if recording is not None:
            recording.close()


def _receive(
    server,
    headset_ip: str,
    stop_event,
    ready_event,
    startup_timeout: float,
    eye_width: int,
    eye_height: int,
    latency_left: float,
    latency_right: float,
    left_ring,
    right_ring,
    av,
    mode: str = "stereo",
    *,
    recording=None,
) -> None:
    _enabled_eyes(mode)
    codec = av.CodecContext.create("h264", "r")
    codec.thread_type = "SLICE"
    codec.thread_count = 2
    started = time.monotonic()
    connection = None

    def check_timeout() -> None:
        if stop_event.is_set():
            raise _StreamStopped()
        if not ready_event.is_set() and time.monotonic() - started > startup_timeout:
            raise TimeoutError("未解码到图像；请检查头显应用、相机权限、IP 和端口")
        if recording is not None:
            recording.poll()

    while not stop_event.is_set() and connection is None:
        check_timeout()
        try:
            conn, peer = server.accept()
        except socket.timeout:
            continue
        if peer[0] != headset_ip:
            conn.close()
            print(f"[pico camera] 拒绝非目标设备的视频连接：{peer[0]}", flush=True)
            continue
        conn.settimeout(0.2)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection = conn
    if connection is None:
        return
    try:
        while not stop_event.is_set():
            try:
                prefix = _recv_exact(connection, 4, stop_event, check_timeout)
            except _StreamEOF as exc:
                if "0/" in str(exc):
                    for packet in codec.parse(b""):
                        _publish_packet(
                            packet, codec, ready_event, eye_width, eye_height,
                            latency_left, latency_right, left_ring, right_ring,
                            mode,
                            recording=recording,
                        )
                    for frame in codec.decode(None):
                        _publish_frame(
                            frame, ready_event, eye_width, eye_height,
                            latency_left, latency_right, left_ring, right_ring,
                            mode,
                            recording=recording,
                        )
                raise
            size = struct.unpack(">I", prefix)[0]
            if not 0 < size <= MAX_PACKET:
                raise ValueError(f"无效视频包长度：{size}")
            payload = _recv_exact(connection, size, stop_event, check_timeout)
            for packet in codec.parse(payload):
                _publish_packet(
                    packet, codec, ready_event, eye_width, eye_height,
                    latency_left, latency_right, left_ring, right_ring,
                    mode,
                    recording=recording,
                )
    finally:
        connection.close()


def _publish_packet(packet, codec, ready_event, eye_width, eye_height, latency_left, latency_right, left_ring, right_ring, mode: str = "stereo", *, recording=None) -> None:
    for frame in codec.decode(packet):
        _publish_frame(frame, ready_event, eye_width, eye_height, latency_left, latency_right, left_ring, right_ring, mode, recording=recording)


def _publish_frame(frame, ready_event, eye_width, eye_height, latency_left, latency_right, left_ring, right_ring, mode: str = "stereo", *, recording=None) -> None:
    if recording is not None:
        recording.poll()
    enabled_eyes = _enabled_eyes(mode)
    bgr = frame.to_ndarray(format="bgr24")
    if bgr.ndim != 3 or bgr.shape[2] != 3 or min(bgr.shape[:2]) <= 0:
        raise ValueError(f"无效相机图像尺寸：{bgr.shape}，需要 H×W×3 BGR 图像")
    if mode != "stereo" and bgr.shape[:2] != (eye_height, eye_width):
        raise ValueError(
            f"相机 {mode} 模式返回 {bgr.shape[1]}×{bgr.shape[0]}，"
            f"请求的是 {eye_width}×{eye_height} 单目图像；"
            "请检查头显应用是否支持并采用 renderMode，不能把双目拼接帧标记成单目"
        )
    received = time.time()
    eyes = dict(zip(("left", "right"), split_eyes(bgr))) if mode == "stereo" else {mode: bgr}
    rings = {"left": left_ring, "right": right_ring}
    latencies = {"left": latency_left, "right": latency_right}
    for eye in enabled_eyes:
        if rings[eye] is None:
            raise ValueError(f"相机 {mode} 模式缺少 {eye} 眼共享内存")
        rgb = _put_eye(rings[eye], eyes[eye], received, latencies[eye], eye_width, eye_height)
        if recording is not None:
            recording.write(rgb, received - latencies[eye], received)
    if not ready_event.is_set():
        ready_event.set()


class EyeView:
    """One eye of the headset stream. ``get_last`` matches ``CameraProcess``."""

    def __init__(self, name: str, ring: SharedMemoryRingBuffer):
        self.name = name
        self.ring = ring

    def get_last(self, k: int) -> dict:
        available = min(k, self.ring.count, self.ring.get_max_k)
        if available <= 0:
            raise RuntimeError(f"camera {self.name} has no frames")
        return self.ring.get_last_k(available)


class PicoStereoProcess:
    """Headset camera with stereo, left or right output; defaults to stereo.

    ``stereo_width`` always describes the reference stereo width. Mono requests
    use half that width, preserving the per-eye resolution. Disabled eyes are
    ``None`` and have no shared-memory allocation.
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        headset_ip: str,
        pc_ip: str | None = None,
        stereo_width: int = 2160,
        stereo_height: int = 810,
        fps: int = 30,
        bitrate: int = 5_000_000,
        latency_left: float = 0.0,
        latency_right: float = 0.0,
        get_max_k: int = 32,
        startup_timeout: float = 25.0,
        mode: str = "stereo",
        recording_eye: str | None = None,
    ):
        import multiprocessing as mp

        self.enabled_eyes = _enabled_eyes(mode)
        if recording_eye is not None and (mode == "stereo" or recording_eye != mode):
            raise ValueError("recording requires one selected eye")
        self.mode = mode
        self.name = f"pico_{mode}"
        if stereo_width <= 0 or stereo_height <= 0 or stereo_width % 2 or stereo_height % 2:
            raise ValueError("H.264 请求的基准双目宽、高必须为正偶数")
        self.headset_ip = headset_ip
        self.pc_ip = pc_ip
        self.stereo_width = int(stereo_width)
        self.stereo_height = int(stereo_height)
        self.eye_width = self.stereo_width // 2
        self.eye_height = self.stereo_height
        self.stream_width = self.stereo_width if mode == "stereo" else self.eye_width
        if self.stream_width % 2:
            raise ValueError("H.264 单目请求的宽必须为偶数，请使用能被 4 整除的基准双目宽")
        self.fps = int(fps)
        if self.fps <= 0:
            raise ValueError("camera fps must be positive")
        self.startup_timeout = float(startup_timeout)
        example = camera_examples(self.eye_height, self.eye_width)
        self.left: EyeView | None = None
        self.right: EyeView | None = None
        for eye in self.enabled_eyes:
            ring = SharedMemoryRingBuffer.create_from_examples(
                shm_manager,
                example,
                get_max_k=get_max_k,
                get_time_budget=0.2,
                put_desired_frequency=self.fps,
            )
            setattr(self, eye, EyeView(f"pico_{eye}", ring))
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.recording_commands = mp.Queue() if recording_eye is not None else None
        self.recording_acks = mp.Queue() if recording_eye is not None else None
        self.process = mp.Process(
            target=pico_camera_loop,
            kwargs={
                "headset_ip": headset_ip,
                "pc_ip": pc_ip,
                "stereo_width": self.stereo_width,
                "stereo_height": self.stereo_height,
                "fps": self.fps,
                "bitrate": int(bitrate),
                "latency_left": float(latency_left),
                "latency_right": float(latency_right),
                "left_ring": self.left.ring if self.left else None,
                "right_ring": self.right.ring if self.right else None,
                "stop_event": self.stop_event,
                "ready_event": self.ready_event,
                "startup_timeout": self.startup_timeout,
                "mode": self.mode,
                "recording_eye": recording_eye,
                "recording_commands": self.recording_commands,
                "recording_acks": self.recording_acks,
            },
            daemon=False,
        )

    def start(self) -> None:
        self.process.start()

    def eye(self, view: str) -> EyeView:
        if view not in ("left", "right"):
            raise ValueError(f"view must be left or right, got {view}")
        if view not in self.enabled_eyes:
            raise ValueError(f"{view} eye is disabled in camera mode {self.mode}")
        return getattr(self, view)

    def stop(self, timeout: float = 15.0) -> None:
        # The recording thread alone may need up to 10s to drain and close.
        try:
            stop_camera_process(self.process, self.stop_event, timeout, self.name)
        finally:
            if self.process.pid is None or not self.process.is_alive():
                for queue in (self.recording_commands, self.recording_acks):
                    if queue is not None:
                        queue.cancel_join_thread()
                        queue.close()


class PicoCameraProcess(PicoStereoProcess):
    """One recordable eye with the same lifecycle/API as a UVC CameraProcess."""

    def __init__(
        self, shm_manager, headset_ip, name="front_1", pc_ip=None,
        width=1080, height=810, fps=60, latency=0.0, bitrate=5_000_000,
        startup_timeout=25.0, eye="left", get_max_k=32,
    ):
        if eye not in ("left", "right"):
            raise ValueError("recordable Pico camera must select left or right")
        super().__init__(
            shm_manager, headset_ip=headset_ip, pc_ip=pc_ip,
            stereo_width=int(width) * 2, stereo_height=int(height), fps=int(fps),
            bitrate=int(bitrate), latency_left=float(latency), latency_right=float(latency),
            startup_timeout=startup_timeout, mode=eye, recording_eye=eye,
            get_max_k=get_max_k,
        )
        self.name = name
        self.width, self.height, self.latency = int(width), int(height), float(latency)
        self.view = self.eye(eye)
        self.ring = self.view.ring

    def get_last(self, k):
        return self.view.get_last(k)

    def _record_command(self, kind, timeout, **fields):
        request_id = uuid.uuid4().hex
        if self.process.pid is None or self.process.exitcode is not None:
            raise RuntimeError(f"相机 {self.name} 没有运行")
        self.recording_commands.put({"id": request_id, "cmd": kind, **fields})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                response = self.recording_acks.get(
                    timeout=min(0.1, max(0.001, deadline - time.monotonic()))
                )
            except Empty:
                if self.process.exitcode is not None:
                    raise RuntimeError(
                        f"相机 {self.name} 进程退出，code {self.process.exitcode}"
                    )
                continue
            if response["id"] != request_id:
                continue
            if response["error"] is not None:
                raise RuntimeError(f"{self.name}: {response['error']}")
            return
        raise TimeoutError(f"相机 {self.name} 的 {kind} 命令超时")

    def start_recording(self, path: str, timeout: float = 10.0):
        self._record_command("start", timeout, path=path)

    def stop_recording(self, timeout: float = 15.0):
        self._record_command("stop_record", timeout)
