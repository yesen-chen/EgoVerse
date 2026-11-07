"""Bounded recording of decoded frames, independent of the control loop.

The camera producer submits every decoded RGB frame. Encoding and disk writes
run on a worker thread in that camera process. Overflow or an encoder failure
is an error, never an implicit reduction to the control loop's frame rate.
"""

from __future__ import annotations

from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread

import numpy as np

from egomimic.robot.piper_pico.camera_process import _open_writer


class AsyncVideoRecorder:
    def __init__(self, path, width, height, fps, *, queue_size=32):
        self.path = Path(path)
        self.width, self.height = int(width), int(height)
        self.fps = float(fps)
        self.frames = Queue(maxsize=queue_size)
        self.closing = Event()
        self.ready = Event()
        self.error = None
        self.thread = Thread(target=self._run, name="camera-video-writer", daemon=True)
        self.thread.start()
        if not self.ready.wait(timeout=5):
            self.closing.set()
            raise TimeoutError(f"打开录像超时：{self.path}")
        self.check()

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"录像失败 {self.path}: {self.error}") from self.error

    def write(self, rgb, timestamp, received):
        self.check()
        if self.closing.is_set():
            raise RuntimeError("recorder is already closing")
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError(f"unexpected recording frame shape: {rgb.shape}")
        # The caller passes a newly allocated RGB array and never mutates it.
        try:
            self.frames.put_nowait((rgb, float(timestamp), float(received)))
        except Full as exc:
            raise RuntimeError(
                f"录像编码/磁盘跟不上相机，队列已满：{self.path}；停止本段以免静默丢帧"
            ) from exc

    def close(self, timeout=10):
        self.closing.set()
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise TimeoutError(f"录像收尾超时：{self.path}")
        self.check()

    def _run(self):
        import av

        container = None
        timestamps, received_times = [], []
        try:
            container, stream = _open_writer(
                self.path, self.width, self.height, self.fps, fast=True
            )
            self.ready.set()
            while not self.closing.is_set() or not self.frames.empty():
                try:
                    rgb, stamp, received = self.frames.get(timeout=0.05)
                except Empty:
                    continue
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
                timestamps.append(stamp)
                received_times.append(received)
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            container = None
            if not timestamps:
                raise RuntimeError("本段录像没有收到图像")
            np.save(self.path.with_suffix(".npy"), np.asarray(timestamps, dtype=np.float64))
            np.save(
                self.path.with_suffix(".receive_timestamps.npy"),
                np.asarray(received_times, dtype=np.float64),
            )
        except BaseException as exc:
            self.error = exc
        finally:
            self.ready.set()
            if container is not None:
                try:
                    container.close()
                except Exception as exc:
                    if self.error is None:
                        self.error = exc


class CameraRecordingSession:
    """Commands are consumed by the camera producer, even between frames."""

    def __init__(self, commands, acks, width, height, fps):
        self.commands, self.acks = commands, acks
        self.width, self.height, self.fps = width, height, fps
        self.writer = None

    def poll(self):
        if self.writer is not None:
            self.writer.check()
        while True:
            try:
                command = self.commands.get_nowait()
            except Empty:
                return
            try:
                if command["cmd"] == "start":
                    if self.writer is not None:
                        raise RuntimeError("camera is already recording")
                    self.writer = AsyncVideoRecorder(
                        command["path"], self.width, self.height, self.fps
                    )
                elif command["cmd"] == "stop_record":
                    self.close()
                else:
                    raise ValueError(f"unknown recording command: {command['cmd']}")
            except Exception as exc:
                self.acks.put({"id": command["id"], "error": str(exc)})
                raise
            self.acks.put({"id": command["id"], "error": None})

    def write(self, rgb, timestamp, received):
        if self.writer is not None:
            self.writer.write(rgb, timestamp, received)

    def close(self):
        writer, self.writer = self.writer, None
        if writer is not None:
            writer.close()
