"""Best-effort camera preview: a separate, read-only consumer of camera rings.

Only ring handles and names cross into this process. Camera ownership, encoding,
observation alignment and teleoperation remain in their existing processes.
No image queues, reader acknowledgements or per-control-tick callbacks are used.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import signal
import sys
import time
from queue import Empty

import numpy as np

WINDOW_NAME = "Collect cameras"
TILE_WIDTH = 320
TILE_HEIGHT = 240
HEADER_HEIGHT = 40
FOOTER_HEIGHT = 24


def preview_fps(value):
    try:
        fps = float(value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "preview FPS must be between 1 and 30"
        ) from exc
    if not math.isfinite(fps) or not 1 <= fps <= 30:
        raise argparse.ArgumentTypeError("preview FPS must be between 1 and 30")
    return fps


class FrameRate:
    """Count publications, including frames deliberately skipped by the viewer."""

    def __init__(self):
        self.previous = None
        self.fps = None

    def update(self, count, now):
        if self.previous is None or count < self.previous[0]:
            self.previous = (count, now)
            self.fps = None
        elif now - self.previous[1] >= 1.0:
            self.fps = (count - self.previous[0]) / (now - self.previous[1])
            self.previous = (count, now)
        return self.fps


class LatestFrame:
    """Copy one latest frame into reusable private memory; never wait for data."""

    def __init__(self, name, ring):
        self.name = name
        self.ring = ring
        self.out = None
        self.last_count = -1
        self.image = None
        self.received = None
        self.rate = FrameRate()

    def read(self):
        count = self.ring.count
        self.rate.update(count, time.monotonic())
        if count <= 0 or count == self.last_count:
            return
        started = time.monotonic()
        try:
            self.out = self.ring.get(out=self.out)
        except Empty:
            return
        # A process suspended during copying could outlive the ring's read
        # budget. Drop that preview sample instead of using a torn image.
        if (
            time.monotonic() - started > self.ring.get_time_budget
            or self.ring.count - count >= self.ring.buffer_size - 1
        ):
            return
        self.last_count = count
        # Convert/resize only after the SHM copy. Never draw into shared memory.
        import cv2

        rgb = self.out["color"]
        height, width = rgb.shape[:2]
        scale = min(TILE_WIDTH / width, TILE_HEIGHT / height)
        resized = cv2.resize(
            rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        self.image = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        self.received = float(self.out["camera_receive_timestamp"])


def compose_preview(readers, preview_rate, target_fps, now):
    """Render private thumbnails in at most three columns, preserving aspect."""
    import cv2

    columns = min(3, len(readers))
    rows = math.ceil(len(readers) / columns)
    row_height = HEADER_HEIGHT + TILE_HEIGHT
    canvas = np.full(
        (rows * row_height + FOOTER_HEIGHT, columns * TILE_WIDTH, 3), 24, np.uint8
    )
    for index, reader in enumerate(readers):
        x, y = (index % columns) * TILE_WIDTH, (index // columns) * row_height
        age = None if reader.received is None else max(0.0, now - reader.received)
        stale = age is None or age > 0.5
        color = (80, 100, 255) if stale else (100, 230, 140)
        fps = "--" if reader.rate.fps is None else f"{reader.rate.fps:.1f}"
        cv2.putText(
            canvas,
            f"{reader.name}  |  camera {fps} fps",
            (x + 6, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            color,
            1,
            cv2.LINE_AA,
        )
        status = (
            "WAITING FOR FRAME"
            if age is None
            else f"{'STALE' if stale else 'LIVE'}  |  received {age * 1000:.0f} ms ago"
        )
        cv2.putText(
            canvas,
            status,
            (x + 6, y + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
        if reader.image is not None:
            h, w = reader.image.shape[:2]
            top = y + HEADER_HEIGHT + (TILE_HEIGHT - h) // 2
            left = x + (TILE_WIDTH - w) // 2
            canvas[top : top + h, left : left + w] = reader.image
    rate = "--" if preview_rate is None else f"{preview_rate:.1f}"
    # The short footer also fits the single-camera case.
    footer = f"View {rate}/{target_fps:g} fps | Q/Esc: close view"
    cv2.putText(
        canvas,
        footer,
        (6, canvas.shape[0] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return canvas


def preview_loop(sources, stop_event, ready_event, fps):
    """Own the GUI in the spawned child's main thread, isolated from collect."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        os.nice(5)
    except (AttributeError, OSError):
        pass
    import cv2

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    readers = [LatestFrame(name, ring) for name, ring in sources]
    rate, frames = FrameRate(), 0
    parent = mp.parent_process()
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        while not stop_event.is_set():
            if parent is not None and not parent.is_alive():
                break
            started = time.monotonic()
            for reader in readers:
                reader.read()
            canvas = compose_preview(
                readers, rate.update(frames, started), fps, time.time()
            )
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if (
                key in (27, ord("q"), ord("Q"))
                or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            ):
                break
            ready_event.set()
            frames += 1
            # A slow refresh drops preview frames; it never tries to catch up.
            stop_event.wait(max(0.0, 1.0 / fps - (time.monotonic() - started)))
    except Exception as exc:
        print(f"[预览] 窗口已停止：{exc}；采集继续。", flush=True)
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


class CameraPreview:
    """Optional sidecar; start/stop are outside the 30 Hz control loop."""

    def __init__(self, cameras, fps=15.0):
        self.fps = preview_fps(fps)
        order = {"left_wrist": 0, "front_1": 1, "right_wrist": 2}
        self.sources = sorted(
            [(camera.name, camera.ring) for camera in cameras],
            key=lambda item: order.get(item[0], 3),
        )
        self.process = None
        self.stop_event = None
        self.ready_event = None

    def start(self):
        if self.process is not None or not self.sources:
            return
        if sys.platform.startswith("linux") and not (
            os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")
        ):
            print("[预览] 未检测到桌面显示，跳过窗口；采集继续。", flush=True)
            return
        try:
            context = mp.get_context("spawn")
            self.stop_event = context.Event()
            self.ready_event = context.Event()
            self.process = context.Process(
                name="camera-preview",
                target=preview_loop,
                args=(self.sources, self.stop_event, self.ready_event, self.fps),
                daemon=True,
            )
            self.process.start()
            print(
                f"[预览] 独立窗口 {self.fps:g} fps；Q/Esc 仅关闭预览。采集按键请使用 PICO 或终端。",
                flush=True,
            )
        except Exception as exc:
            print(f"[预览] 无法启动：{exc}；采集继续。", flush=True)

    def stop(self):
        if self.stop_event is not None:
            self.stop_event.set()
        process = self.process
        if process is None or process.pid is None:
            return
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=0.5)
        if process.is_alive():
            process.kill()
            process.join(timeout=0.5)
        if process.is_alive():
            print("[预览] 窗口进程未退出，继续清理采集设备。", flush=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
