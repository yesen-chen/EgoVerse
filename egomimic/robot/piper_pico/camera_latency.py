"""Estimate visual-event-to-camera-receive latency with timestamp QR codes.

Each QR has its own generation and imshow-return timestamp. Physical display
latency is unknown unless explicitly supplied; refresh Hz is metadata only.
The result still includes camera sampling/exposure/first-success QR effects,
so it is not an exact hardware exposure timestamp calibration.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import threading
import time
from collections import deque
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import cv2
import numpy as np

from egomimic.robot.piper_pico.camera_config import load_camera_config
from egomimic.robot.piper_pico.camera_process import CameraProcess
from egomimic.robot.piper_pico.latency_analysis import analyze_decoded
from egomimic.robot.piper_pico.pico_camera import PicoCameraProcess, PicoStereoProcess


def make_timestamp_qr(size: int, encoder=None) -> tuple[np.ndarray, float, str]:
    """BGR QR whose payload is ``str(t_sample)``."""
    if encoder is None:
        encoder = _qr_encoder()
    t_sample = time.time()
    text = str(t_sample)
    gray = encoder.encode(text)
    gray = cv2.copyMakeBorder(gray, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    # Preserve equal-sized modules; arbitrary nearest-neighbor resizing gives
    # adjacent modules different widths when size is not a matrix multiple.
    scale = size // gray.shape[0]
    if scale < 1:
        raise ValueError("qr-size must fit at least one pixel per QR module")
    # Keep modules integral in decode_qr's half-size fallback as well.
    scale = max(1, scale - scale % 2)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    padding = size - gray.shape[0]
    gray = cv2.copyMakeBorder(
        gray,
        padding // 2,
        padding - padding // 2,
        padding // 2,
        padding - padding // 2,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return image, t_sample, text


def decode_qr(
    detector: cv2.QRCodeDetector, image: np.ndarray
) -> tuple[str, np.ndarray | None]:
    """Read one QR. A bad frame must not abort the measurement.

    OpenCV's QR detector sometimes raises ``floodFill`` on a real camera image.
    That frame is skipped. If the detector itself is left unusable, one fresh
    detector scans the same image. A half-size retry also handles the detector's
    scale-sensitive misses on otherwise clean QR codes. Corners always refer to
    the original image.
    """
    prepared = _qr_image(image)
    if prepared is None:
        return "", None
    code, corners, failed = _scan_qr(detector, prepared)
    if code:
        return code, corners
    if failed:
        code, corners, _failed = _scan_qr(cv2.QRCodeDetector(), prepared)
        if code:
            return code, corners
    height, width = prepared.shape[:2]
    if min(height, width) < 42:
        return "", None
    smaller = cv2.resize(
        prepared, (width // 2, height // 2), interpolation=cv2.INTER_AREA
    )
    code, corners, _failed = _scan_qr(cv2.QRCodeDetector(), smaller)
    if code and corners is not None:
        corners = corners * np.array(
            [width / smaller.shape[1], height / smaller.shape[0]], dtype=np.float32
        )
    return code, corners


def _qr_image(image: np.ndarray) -> np.ndarray | None:
    array = np.ascontiguousarray(image)
    if array.size == 0:
        return None
    if array.dtype != np.uint8:
        array = np.ascontiguousarray(array, dtype=np.uint8)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 21:
        return None
    return array


def _scan_qr(detector, image: np.ndarray) -> tuple[str, np.ndarray | None, bool]:
    for name in ("detectAndDecodeCurved", "detectAndDecode"):
        try:
            code, corners, _ = getattr(detector, name)(image)
        except cv2.error:
            return "", None, True
        if code:
            return str(code), corners, False
    return "", None, False


def pair_first_receive(frames: list[tuple[float, str]]) -> dict[str, float]:
    """Map each QR payload to the earliest receive time that decoded it."""
    found: dict[str, float] = {}
    for received, code in frames:
        if not code or code in found:
            continue
        try:
            float(code)
        except ValueError:
            continue
        found[code] = float(received)
    return found


def latency_offsets(pairs: dict[str, float], display_latency: float) -> np.ndarray:
    offsets = [
        received - float(code) - float(display_latency)
        for code, received in pairs.items()
    ]
    return np.asarray(offsets, dtype=np.float64)


def summarize_offsets(offsets: np.ndarray) -> dict[str, float | int]:
    if offsets.size == 0:
        raise ValueError("no latency samples")
    return {
        "count": int(offsets.size),
        "mean_sec": float(np.mean(offsets)),
        "std_sec": float(np.std(offsets)),
        "median_sec": float(np.median(offsets)),
        "p95_sec": float(np.percentile(offsets, 95)),
        "min_sec": float(np.min(offsets)),
        "max_sec": float(np.max(offsets)),
    }


class FrameBuffer:
    def __init__(self, maxlen: int):
        self._lock = threading.Lock()
        self._frames: deque[tuple[float, np.ndarray]] = deque(maxlen=maxlen)
        self.pushes = 0

    def push(self, received: float, image: np.ndarray) -> None:
        stored = np.ascontiguousarray(image)
        with self._lock:
            self._frames.append((float(received), stored))
            self.pushes += 1

    def latest(self) -> tuple[float, np.ndarray] | None:
        with self._lock:
            if not self._frames:
                return None
            return self._frames[-1]

    def snapshot(self) -> list[tuple[float, np.ndarray]]:
        with self._lock:
            return list(self._frames)


class _CaptureThread:
    def __init__(self, buffer: FrameBuffer, startup_timeout: float):
        self.buffer = buffer
        self.startup_timeout = float(startup_timeout)
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.error: str | None = None
        self.actual: dict = {}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._guard, daemon=True)
        self._thread.start()
        try:
            self._wait_until_ready()
        except BaseException:
            self.close()
            raise

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        last_print = time.monotonic()
        while True:
            if self.ready.wait(0.2):
                if self.error:
                    raise RuntimeError(self.error)
                return
            if self.error or not self._thread.is_alive():
                raise RuntimeError(self.error or "采集线程退出，没有图像")
            now = time.monotonic()
            if now - last_print >= 2.0:
                print(self.status(), flush=True)
                last_print = now
            if now >= deadline:
                raise TimeoutError(self.status())

    def status(self) -> str:
        return self.error or "等待图像超时"

    def close(self) -> None:
        self.stop.set()
        if self._thread is not None:
            self._thread.join(5)

    def _guard(self) -> None:
        try:
            self._run()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def _run(self) -> None:
        raise NotImplementedError


class _RingFeed:
    """Parent-side view of one shared-memory camera ring."""

    def __init__(self, startup_timeout: float):
        self.startup_timeout = float(startup_timeout)
        self.error: str | None = None
        self.actual: dict = {}
        self._shm: SharedMemoryManager | None = None
        self._process = None
        self._ring = None
        self._recording = False

    def start(self) -> None:
        self._shm = SharedMemoryManager()
        self._shm.start()
        try:
            self._open()
            self._wait_until_ready()
            sample = self.latest()
            if sample is None:
                raise RuntimeError("相机已就绪但没有帧")
            _received, image = sample
            self.actual.update(
                stored_width=int(image.shape[1]), stored_height=int(image.shape[0])
            )
        except BaseException:
            self.close()
            raise

    def _open(self) -> None:
        raise NotImplementedError

    def _wait_until_ready(self) -> None:
        ready = self._process.ready_event
        process = self._process.process
        deadline = time.monotonic() + self.startup_timeout
        last_print = time.monotonic()
        while True:
            if ready.wait(0.2):
                return
            if not process.is_alive():
                raise RuntimeError(self.error or "相机进程退出，没有图像")
            now = time.monotonic()
            if now - last_print >= 2.0:
                print(self.status(), flush=True)
                last_print = now
            if now >= deadline:
                raise TimeoutError(self.status())

    def status(self) -> str:
        return self.error or "等待图像超时"

    def latest(self) -> tuple[float, np.ndarray] | None:
        if self._ring is None or self._ring.count <= 0:
            return None
        sample = self._ring.get()
        return float(sample["camera_receive_timestamp"]), sample["color"]

    def snapshot(self) -> list[tuple[float, np.ndarray]]:
        count = 0 if self._ring is None else min(self._ring.count, self._ring.get_max_k)
        if count <= 0:
            return []
        sample = self._ring.get_last_k(count)
        stamps = np.asarray(
            sample["camera_receive_timestamp"], dtype=np.float64
        ).reshape(-1)
        return [
            (float(stamps[index]), sample["color"][index]) for index in range(count)
        ]

    @property
    def frame_count(self) -> int:
        if self._ring is None:
            return 0
        return int(self._ring.count)

    def close(self) -> None:
        try:
            if self._process is not None:
                try:
                    if self._recording:
                        self._process.stop_recording()
                finally:
                    self._recording = False
                    self._process.stop()
                    self._process = None
        finally:
            if self._shm is not None:
                self._shm.shutdown()
                self._shm = None

    def start_recording(self, path: Path) -> None:
        self._process.start_recording(str(path))
        self._recording = True


class UvcFeed(_RingFeed):
    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        fps: float,
        fourcc: str,
        n_frames: int,
        startup_timeout: float,
        camera_spec: dict | None = None,
    ):
        super().__init__(startup_timeout)
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.fourcc = fourcc
        self.n_frames = int(n_frames)
        self.camera_spec = copy.deepcopy(camera_spec)

    def _open(self) -> None:
        settings = {}
        if self.camera_spec is not None:
            settings = {
                key: self.camera_spec[key]
                for key in ("controls", "expected_identity", "calibration")
                if key in self.camera_spec
            }
        self._process = CameraProcess(
            self._shm,
            device=self.device,
            name="uvc",
            width=self.width,
            height=self.height,
            fps=self.fps,
            latency=0.0,
            fourcc=self.fourcc,
            get_max_k=self.n_frames,
            **settings,
        )
        self._ring = self._process.ring
        self.actual = {
            "device": self.device,
            "requested_fps": self.fps,
            "requested_fourcc": self.fourcc,
        }
        self._process.start()


class PicoFeed(_RingFeed):
    def __init__(
        self,
        headset_ip: str,
        pc_ip: str | None,
        view: str,
        width: int,
        height: int,
        fps: int,
        bitrate: int,
        n_frames: int,
        startup_timeout: float,
        mode: str = "left",
    ):
        super().__init__(startup_timeout)
        if view not in ("left", "right"):
            raise ValueError("view must be left or right")
        if mode not in ("left", "right", "stereo") or (
            mode != "stereo" and view != mode
        ):
            raise ValueError("view must select an enabled eye in camera mode")
        self.mode = mode
        self.headset_ip = headset_ip
        self.pc_ip = pc_ip
        self.view = view
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.bitrate = int(bitrate)
        self.n_frames = int(n_frames)

    def _open(self) -> None:
        common = dict(
            headset_ip=self.headset_ip,
            pc_ip=self.pc_ip,
            fps=self.fps,
            bitrate=self.bitrate,
            get_max_k=self.n_frames,
            startup_timeout=self.startup_timeout,
        )
        if self.mode == "stereo":
            self._process = PicoStereoProcess(
                self._shm,
                stereo_width=self.width,
                stereo_height=self.height,
                mode="stereo",
                **common,
            )
        else:
            self._process = PicoCameraProcess(
                self._shm,
                width=self.width,
                height=self.height,
                eye=self.mode,
                latency=0.0,
                **common,
            )
        self._ring = self._process.eye(self.view).ring
        self.actual = {
            "mode": self.mode,
            "view": self.view,
            "headset_ip": self.headset_ip,
            "pc_ip": self.pc_ip,
            "requested_stream_width": self.width,
            "requested_stream_height": self.height,
            "requested_fps": self.fps,
            "requested_bitrate": self.bitrate,
        }
        self._process.start()


def analyze_buffer(
    frames: list[tuple[float, np.ndarray]],
    submissions: dict,
    *,
    display_latency_sec: float = 0.0,
    min_samples: int = 30,
    detector=None,
) -> dict:
    """Decode all buffered producer frames, then match each QR's own submission."""
    detector = detector or cv2.QRCodeDetector()
    decoded = []
    for index, (received, image) in enumerate(frames):
        code, _ = decode_qr(detector, image)
        decoded.append((received, code))
        if index % 10 == 0:
            print(f"回放检测 {index + 1}/{len(frames)}", flush=True)
    stamps = [float(row[0]) for row in frames]
    report = analyze_decoded(
        decoded,
        submissions,
        display_latency_sec=display_latency_sec,
        min_samples=min_samples,
        buffer_start=min(stamps) if stamps else None,
        buffer_end=max(stamps) if stamps else None,
    )
    report["frames_in_buffer"] = len(frames)
    report["measured_fps"] = (
        (len(stamps) - 1) / (stamps[-1] - stamps[0])
        if len(stamps) >= 2 and stamps[-1] > stamps[0]
        else None
    )
    if len(stamps) >= 2 and np.any(np.diff(stamps) <= 0):
        report["quality_warnings"].append(
            "相机接收时间戳不严格递增，需检查时钟或帧重复"
        )
        report["recommendation_allowed"] = False
    return report


def save_report(path: Path, report: dict) -> None:
    """Save matched rows and all three summaries, never overwrite a prior result."""
    path.mkdir(parents=True, exist_ok=True)
    with (path / "latency_report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    fields = [
        "qr_text",
        "qr_timestamp_unix",
        "generated_unix",
        "submitted_unix",
        "receive_unix",
        "software_sec",
        "raw_sec",
        "submitted_to_receive_sec",
        "corrected_sec",
        "valid",
    ]
    with (path / "samples.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: sample[key] for key in fields} for sample in report["samples"]
        )
    if report["samples"]:
        _save_plot(path / "latency.png", report["samples"])


def _save_plot(path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    receive = np.asarray([r["receive_unix"] for r in rows])
    before = np.asarray([r["submitted_to_receive_sec"] for r in rows]) * 1000
    after = np.asarray([r["corrected_sec"] for r in rows]) * 1000
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(receive - receive[0], before, ".-", label="Before display correction")
    axes[0].plot(receive - receive[0], after, ".-", label="After explicit correction")
    axes[0].set_xlabel("Receive time relative to first sample (s)")
    axes[0].set_ylabel("Estimated latency (ms)")
    axes[0].legend()
    axes[1].hist(after, bins=min(30, max(5, len(rows) // 2)))
    axes[1].set_xlabel("After explicit display correction (ms)")
    axes[1].set_ylabel("Distinct QR samples")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def run_measurement(
    capture: _RingFeed, qr_size: int, n_frames: int, output: Path, meta: dict
) -> dict:
    cv2.setNumThreads(1)
    print(
        "正在打开相机。把相机对准完整清晰的二维码，保持约 10 秒，再按 c 分析；q 退出。",
        flush=True,
    )
    print("缓冲只保留最近一段；样本数以不同二维码计，不是相机帧数。", flush=True)
    encoder = _qr_encoder()
    submissions = {}
    next_qr = 0.0
    next_preview = 0.0
    last_print = 0.0
    last_preview = None
    measured = None
    # Small seeded timing variation prevents locking 10 Hz transitions to camera cadence.
    rng = np.random.default_rng(0)
    period = 1.0 / meta["qr_hz"]
    try:
        capture.start()
        if meta["record_video"]:
            capture.start_recording(output / "load_test.mp4")
        print(f"已收到图像：{capture.actual}", flush=True)
        while measured is None:
            if capture.error:
                raise RuntimeError(capture.error)
            now = time.monotonic()
            # No QR detector in the GUI loop: slow recognition must not hold up screen updates.
            if now >= next_preview:
                sample = capture.latest()
                next_preview = now + 1.0 / 30.0
                if sample is not None:
                    received, image = sample
                    if received != last_preview:
                        cv2.imshow("Camera", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                        last_preview = received
            if now >= next_qr:
                qr, generated, text = make_timestamp_qr(qr_size, encoder)
                cv2.imshow("Timestamp QRCode", qr)
                submitted = time.time()
                submissions[text] = {
                    "generated_unix": generated,
                    "submitted_unix": submitted,
                }
                next_qr = time.monotonic() + period * rng.uniform(0.8, 1.2)
            key = cv2.waitKey(1) & 0xFF
            if now - last_print >= 1.0:
                print(
                    f"缓冲 {min(capture.frame_count, n_frames)}/{n_frames} 帧，"
                    f"已提交 {len(submissions)} 个二维码（不等于已识别样本）。按 c 分析。",
                    flush=True,
                )
                last_print = now
            if key in (ord("c"), ord("C")):
                measured = capture.snapshot()
            elif key in (ord("q"), ord("Q"), 27):
                raise SystemExit(0)
    finally:
        try:
            capture.close()
        finally:
            cv2.destroyAllWindows()
    analysis = analyze_buffer(
        measured,
        submissions,
        display_latency_sec=meta["display_latency_sec"],
        min_samples=meta["min_samples"],
    )
    report = dict(meta, **analysis, camera=capture.actual)
    requested, measured_fps = meta["requested_fps"], report["measured_fps"]
    if measured_fps is None or abs(measured_fps - requested) > max(
        1.0, 0.1 * requested
    ):
        report["quality_warnings"].append(
            f"请求 {requested:g} fps，缓冲实测 {measured_fps}；当前测量不代表请求的采集模式"
        )
        report["recommendation_allowed"] = False
    # Save exactly the measurements used in the report, not a UI-history average.
    report["submissions"] = submissions
    save_report(output, report)
    _print_summary(report, output)
    return report


def _print_summary(report: dict, output: Path) -> None:
    for key, label in (
        ("raw_latency", "二维码生成到图像接收"),
        ("latency_before_display_correction", "逐二维码扣除软件耗时后，未扣屏幕延迟"),
        ("latency", "扣除显式屏幕补偿后的估计"),
    ):
        summary = report[key]
        if summary is None:
            print(f"{label}：无有效配对", flush=True)
            continue
        print(
            f"{label}：平均 {summary['mean_sec'] * 1000:.2f} ms，"
            f"中位数 {summary['median_sec'] * 1000:.2f} ms，"
            f"P95 {summary['p95_sec'] * 1000:.2f} ms（{summary['count']} 个二维码）",
            flush=True,
        )
    print(f"缓冲实测帧率：{report['measured_fps']} fps", flush=True)
    for warning in report["quality_warnings"]:
        print(f"注意：{warning}", flush=True)
    if report["recommendation_allowed"]:
        flag = (
            "--pico-camera-latency"
            if report["source"] == "pico"
            else "--camera-latency"
        )
        print(
            f"估计补偿参数：{flag} {report['latency']['mean_sec']:.6f}（秒）",
            flush=True,
        )
        if report["source"] == "uvc":
            print(
                "此数值只属于本次 UVC；使用相机配置时只更新该相机的 latency，"
                "并保留本报告作为测量来源。旧 CLI 的 --camera-latency 会作用于所有 UVC。",
                flush=True,
            )
        print(
            "这是视觉事件到接收的估计，仍含相机采样/曝光/首次识别误差；三流并发需复核。",
            flush=True,
        )
    else:
        print("本次质量检查未通过，不输出可直接套用的补偿参数。", flush=True)
    print(f"报告：{output / 'latency_report.json'}", flush=True)


def _qr_encoder():
    params = cv2.QRCodeEncoder_Params()
    params.correction_level = cv2.QRCODE_ENCODER_CORRECT_LEVEL_H
    return cv2.QRCodeEncoder_create(params)


def _default_output(source: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
    return Path(f"camera_latency_{source}_{stamp}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用屏幕二维码估计视觉事件到相机接收的延迟"
    )
    parser.add_argument("--qr-size", type=int, default=720)
    parser.add_argument("--n-frames", type=int, default=240)
    parser.add_argument("--refresh-hz", type=float, default=120.0)
    parser.add_argument(
        "--display-latency-ms",
        type=float,
        default=0.0,
        help="实测 imshow 返回到对应像素可见的剩余延迟；毫秒，默认不扣",
    )
    parser.add_argument(
        "--qr-hz",
        type=float,
        default=10.0,
        help="二维码平均更新频率；周期有±20%%变化，避免锁定相机相位",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="至少多少个有效不同二维码才输出候选补偿参数",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="同时保存本路 load_test.mp4，测量包含该路录像负载；不等于三路并发",
    )
    parser.add_argument("--output", type=Path, default=None)
    sub = parser.add_subparsers(dest="source", required=True)

    uvc = sub.add_parser("uvc", help="测量 /dev/video 上的 UVC 相机")
    uvc.add_argument("--device", default="/dev/video0")
    uvc.add_argument("--width", type=int, default=640)
    uvc.add_argument("--height", type=int, default=480)
    uvc.add_argument("--fps", type=float, default=30.0)
    uvc.add_argument(
        "--fourcc", default="MJPG", help="采集和测量都用 MJPG。留空表示不设置"
    )
    uvc.add_argument("--startup-timeout", type=float, default=10.0)
    uvc.add_argument(
        "--camera-config", type=Path, help="与采集共用的 camera_rig_v1 JSON"
    )
    uvc.add_argument("--camera-name", help="配置中本次要测的相机，例如 front_1")

    pico = sub.add_parser("pico", help="测量头显相机，一次测一只眼")
    pico.add_argument("--headset-ip", required=True)
    pico.add_argument("--pc-ip", default=None)
    pico.add_argument(
        "--mode",
        choices=("left", "right", "stereo"),
        default="left",
        help="头显实际输出模式，默认与正式采集一致：左目",
    )
    pico.add_argument(
        "--view",
        choices=("left", "right"),
        default=None,
        help="双目模式选哪只眼测量；单目默认选启用眼",
    )
    pico.add_argument(
        "--width",
        type=int,
        default=None,
        help="实际输出宽度；单目默认1080，双目默认2160",
    )
    pico.add_argument("--height", type=int, default=810)
    pico.add_argument("--fps", type=int, default=60)
    pico.add_argument("--bitrate", type=int, default=5_000_000)
    pico.add_argument("--startup-timeout", type=float, default=25.0)
    return parser


def uvc_spec_from_args(args):
    """Select the same native stream without applying its existing latency twice."""
    if args.camera_config is None:
        if args.camera_name is not None:
            raise ValueError("--camera-name requires --camera-config")
        return None
    if not args.camera_name:
        raise ValueError("--camera-config requires --camera-name")
    defaults = build_parser().parse_args(["uvc"])
    conflicts = [
        key
        for key in ("device", "width", "height", "fps", "fourcc")
        if getattr(args, key) != getattr(defaults, key)
    ]
    if conflicts:
        raise ValueError(
            "camera config conflicts with legacy mode options: " + ", ".join(conflicts)
        )
    specs = load_camera_config(args.camera_config)
    matches = [spec for spec in specs if spec["name"] == args.camera_name]
    if not matches or matches[0]["type"] != "uvc":
        raise ValueError(f"camera {args.camera_name!r} must be a UVC entry in the rig")
    return matches[0]


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    camera_spec = None
    if args.source == "uvc":
        try:
            camera_spec = uvc_spec_from_args(args)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        if camera_spec is not None:
            for key in ("device", "width", "height", "fps", "fourcc"):
                setattr(args, key, camera_spec[key])
    if args.n_frames < 2 or args.qr_size < 64 or args.min_samples < 2:
        parser.error("n-frames/min-samples 至少为 2，qr-size 至少为 64")
    if args.min_samples > args.n_frames:
        parser.error("min-samples 不能超过 n-frames")
    for name in ("refresh_hz", "qr_hz", "fps", "startup_timeout"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"{name.replace('_', '-')} 必须是有限正数")
    if args.qr_hz > args.refresh_hz:
        parser.error("qr-hz 不能超过声明的屏幕 refresh-hz")
    if not np.isfinite(args.display_latency_ms) or args.display_latency_ms < 0:
        parser.error("display-latency-ms 必须是有限非负数")
    if args.source == "pico":
        args.view = args.view or ("left" if args.mode == "stereo" else args.mode)
        if args.mode != "stereo" and args.view != args.mode:
            parser.error("view 必须是 mode 中启用的眼")
        if args.mode == "stereo" and args.record_video:
            parser.error("双目模式暂不支持 --record-video；正式采集使用单目")
        args.width = (
            args.width
            if args.width is not None
            else (2160 if args.mode == "stereo" else 1080)
        )
        if args.width % (4 if args.mode == "stereo" else 2) or args.height % 2:
            parser.error("PICO单目宽高须为偶数，双目总宽须为4的倍数")
        if args.bitrate <= 0:
            parser.error("bitrate 必须为正数")
    if args.width <= 0 or args.height <= 0:
        parser.error("width/height 必须为正数")
    output = (args.output or _default_output(args.source)).expanduser().resolve()
    if output.exists():
        parser.error(f"输出目录已存在，避免覆盖请使用新目录：{output}")
    if args.source == "uvc":
        extra = {"camera_spec": camera_spec} if camera_spec is not None else {}
        capture = UvcFeed(
            device=args.device,
            width=args.width,
            height=args.height,
            fps=args.fps,
            fourcc=args.fourcc,
            n_frames=args.n_frames,
            startup_timeout=args.startup_timeout,
            **extra,
        )
        meta = {
            "source": "uvc",
            "device": args.device,
            "requested_width": args.width,
            "requested_height": args.height,
            "requested_fps": args.fps,
            "fourcc": args.fourcc,
            "reader": "egomimic.robot.piper_pico.camera_process.CameraProcess",
            "timestamp": "camera_receive_timestamp from the UVC shared-memory process",
            "capture_camera_spec": camera_spec,
            "measurement_receive_latency_s": 0.0,
        }
    else:
        capture = PicoFeed(
            headset_ip=args.headset_ip,
            pc_ip=args.pc_ip,
            view=args.view,
            width=args.width,
            height=args.height,
            fps=args.fps,
            bitrate=args.bitrate,
            n_frames=args.n_frames,
            startup_timeout=args.startup_timeout,
            mode=args.mode,
        )
        meta = {
            "source": "pico",
            "headset_ip": args.headset_ip,
            "pc_ip": args.pc_ip,
            "view": args.view,
            "mode": args.mode,
            "requested_width": args.width,
            "requested_height": args.height,
            "requested_fps": args.fps,
            "bitrate": args.bitrate,
            "reader": "egomimic.robot.piper_pico.pico_camera."
            + ("PicoStereoProcess" if args.mode == "stereo" else "PicoCameraProcess"),
            "timestamp": "camera_receive_timestamp when the decoded frame becomes an array",
        }
        print(
            "测量 Pico 时保持 XRoboToolkit 运行，停止头显本地 Record，不要同时开预览。",
            flush=True,
        )
    meta["schema"] = "camera_latency_v2"
    meta["display_refresh_hz"] = args.refresh_hz
    meta["display_refresh_period_sec"] = 1.0 / args.refresh_hz
    meta["display_latency_sec"] = args.display_latency_ms / 1000.0
    meta["display_correction_source"] = (
        "user_supplied" if args.display_latency_ms else "unmeasured_not_subtracted"
    )
    meta["qr_hz"] = args.qr_hz
    meta["qr_period_jitter_fraction"] = 0.2
    meta["min_samples"] = args.min_samples
    meta["record_video"] = args.record_video
    meta["workload"] = (
        "one_camera_with_recording"
        if args.record_video
        else "one_camera_without_recording"
    )
    meta["display_refresh_note"] = (
        "Refresh rate is informational only. imshow return is not photon onset; "
        "compositor, scanout and panel delay are unmeasured. No automatic half/full refresh subtraction."
    )
    meta["definition"] = (
        "first decoded receive time per emitted QR minus its own imshow-return time "
        "minus explicitly supplied residual display latency"
    )
    meta["limitations"] = [
        "Not an exact exposure-to-receive calibration; includes camera sampling phase, exposure and first-success decode effects.",
        "Single-camera workload; simultaneous three-camera recording must be checked separately.",
        "Producer receive timestamps precede RGB/ring publication, not exact main-loop availability times.",
    ]
    output.mkdir(parents=True, exist_ok=False)
    print(
        f"相机请求 {args.fps:g} fps。声明屏幕刷新率 {args.refresh_hz:g} Hz，"
        f"周期 {1000.0 / args.refresh_hz:.2f} ms；不会自动扣除刷新周期。"
        f"显式屏幕补偿 {args.display_latency_ms:g} ms。",
        flush=True,
    )
    if args.display_latency_ms == 0:
        print("未测屏幕实际显示延迟，结果仍含这部分延迟。", flush=True)
    run_measurement(capture, args.qr_size, args.n_frames, output, meta)


if __name__ == "__main__":
    main()
