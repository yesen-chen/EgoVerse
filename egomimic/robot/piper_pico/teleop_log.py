"""Snapshot each control tick; a bounded background writer owns CSV file I/O."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
import time

import numpy as np

from egomimic.robot.piper_pico.clutch import (
    looks_like_tracking_origin,
    transform_pico_to_robot,
    xyzw_to_wxyz,
)
from egomimic.robot.piper_pico.loop_profile import LoopProfiler, profile_span

COLUMNS = [
    "t_wall",
    "dt",
    "phase",
    "source_timestamp_ns",
    "source_age_s",
    "source_stall_count",
    "source_max_gap_s",
    "head_raw_x",
    "head_raw_y",
    "head_raw_z",
    "head_raw_qx",
    "head_raw_qy",
    "head_raw_qz",
    "head_raw_qw",
    "lost_any",
    "lost_both",
    "lost_event_start",
]
CAMERA_COLUMNS = [
    "camera_left_count",
    "camera_right_count",
    "camera_left_receive_timestamp",
    "camera_right_receive_timestamp",
    "camera_left_age_s",
    "camera_right_age_s",
    "camera_read_phase_skew_s",
    "camera_copy_s",
    "camera_mode",
]
COLUMNS.extend(CAMERA_COLUMNS)
for _side in ("left", "right"):
    COLUMNS.extend(
        [
            f"{_side}_raw_x",
            f"{_side}_raw_y",
            f"{_side}_raw_z",
            f"{_side}_raw_qx",
            f"{_side}_raw_qy",
            f"{_side}_raw_qz",
            f"{_side}_raw_qw",
            f"{_side}_grip",
            f"{_side}_trigger",
            f"{_side}_robot_x",
            f"{_side}_robot_y",
            f"{_side}_robot_z",
            f"{_side}_lost",
            f"{_side}_event",
            f"{_side}_engaged",
            f"{_side}_target_x",
            f"{_side}_target_y",
            f"{_side}_target_z",
            f"{_side}_measured_x",
            f"{_side}_measured_y",
            f"{_side}_measured_z",
            f"{_side}_cmd_step",
        ]
    )


def default_log_path(prefix: str = "piper_pico_teleop") -> Path:
    return Path(f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")


class TeleopRecorder:
    def __init__(self, path: Path, *, profiler=None, queue_capacity: int = 1024,
                 close_timeout: float = 10.0):
        if not isinstance(queue_capacity, int) or queue_capacity < 1:
            raise ValueError("queue_capacity must be a positive integer")
        if not np.isfinite(close_timeout) or close_timeout <= 0:
            raise ValueError("close_timeout must be positive and finite")
        self.path = Path(path)
        self._profiler = profiler
        # LoopProfiler has mutable nesting/tick state; never share it between
        # the control thread and the file writer.
        self._io_profiler = (LoopProfiler(slow_s=profiler.slow_s, max_events=profiler.max_events)
                             if profiler is not None else None)
        self.queue_capacity = queue_capacity
        self.close_timeout = float(close_timeout)
        self._queue = Queue(maxsize=queue_capacity)
        self._closing = Event()
        self._ready = Event()
        self._thread = None
        self._worker_error = None
        self._reported_close = False
        self.written_rows = 0
        self.rejected_rows = 0
        self.max_queue_depth = 0
        self.queue_wait_total_s = 0.0
        self.queue_wait_max_s = 0.0
        self.rows = 0
        # Retain the historical per-hand-sample total for callers. It is not
        # the number of CSV ticks: simultaneous loss contributes two here.
        self.lost_rows = 0
        self.lost_by_side = {"left": 0, "right": 0}
        self.lost_ticks = 0
        self.lost_events = 0
        self.both_lost_ticks = 0
        self._previous_any_lost = False
        self.held_rows = 0
        self.reanchor_rows = 0
        self.cmd_jumps = 0
        self.max_cmd_step = 0.0
        self._prev_cmd = {"left": None, "right": None}
        self._prev_t = None
        print(f"Recording every tick to {self.path}")

    def start(self) -> None:
        """Start after device processes have launched, before control ticks.

        Construction does not start a thread: the simulation still has to
        start multiprocessing workers, which may use fork.
        """
        if self._closing.is_set():
            raise RuntimeError("CSV recorder is closing or closed")
        if self._thread is None:
            self._thread = Thread(target=self._write_loop, name="teleop-csv-writer", daemon=True)
            self._thread.start()
        if not self._ready.wait(self.close_timeout):
            self._closing.set()
            raise TimeoutError(f"CSV writer startup timed out: {self.path}")
        self._raise_worker_error()

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError(
                f"CSV 后台写入失败：{self.path}；已提交 {self.rows} 行，"
                f"完成写入和 flush {self.written_rows} 行"
            ) from self._worker_error

    def _write_loop(self) -> None:
        stream = None
        try:
            with profile_span(self._io_profiler, "csv.open"):
                stream = self.path.open("w", newline="")
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            with profile_span(self._io_profiler, "csv.header"):
                writer.writeheader()
                stream.flush()
            self._ready.set()
            while True:
                try:
                    row_index, queued_at, row = self._queue.get(timeout=0.05)
                except Empty:
                    if self._closing.is_set():
                        break
                    continue
                wait = max(0.0, time.monotonic() - queued_at)
                self.queue_wait_total_s += wait
                self.queue_wait_max_s = max(self.queue_wait_max_s, wait)
                metadata = dict(row_index=row_index, sample_timestamp_s=float(row["t_wall"]),
                                queue_wait_s=wait)
                with profile_span(self._io_profiler, "csv.writerow", **metadata):
                    writer.writerow(row)
                with profile_span(self._io_profiler, "csv.flush", **metadata):
                    stream.flush()
                # This means the Python file buffer was flushed, not fsync.
                self.written_rows += 1
        except BaseException as exc:
            self._worker_error = exc
        finally:
            if stream is not None:
                try:
                    with profile_span(self._io_profiler, "csv.close"):
                        stream.close()
                except BaseException as exc:
                    if self._worker_error is None:
                        self._worker_error = exc
            self._ready.set()

    def write(
        self, env, sample: dict, phase: str = "teleop", diagnostics: dict | None = None
    ) -> None:
        if self._thread is None or not self._ready.is_set():
            raise RuntimeError("CSV recorder.start() must finish before control ticks")
        if self._closing.is_set():
            raise RuntimeError("CSV recorder is closing or closed")
        self._raise_worker_error()
        # Only one control thread submits rows. The writer can only free slots,
        # so this early check avoids updating counters for a rejected row.
        if self._queue.full():
            self.rejected_rows += 1
            raise RuntimeError("CSV 后台队列已满；停止记录，未静默丢弃数据")
        with profile_span(self._profiler, "sim.recorder_prepare"):
            row = self._prepare_row(env, sample, phase, diagnostics)
        with profile_span(self._profiler, "sim.recorder_enqueue"):
            depth = self._queue.qsize() + 1
            try:
                self._queue.put_nowait((self.rows, time.monotonic(), row))
            except Full as exc:
                self.rejected_rows += 1
                raise RuntimeError("CSV 后台队列已满；停止记录，未静默丢弃数据") from exc
            self.rows += 1
            # Sampled just before enqueue; the consumer can concurrently drain.
            self.max_queue_depth = max(self.max_queue_depth, depth)

    def _prepare_row(self, env, sample, phase, diagnostics):
        """Log observed samples; optional diagnostic columns stay blank if absent.

        A loss event is a contiguous interval of ticks with either controller
        at the tracking origin. Its end requires a tick with neither lost.
        Camera read phase skew is a host read-time difference, not exposure
        synchronization. SDK getters do not provide an atomic pose snapshot.
        """
        now = float(sample.get("receive_timestamp", 0.0))
        dt = 0.0 if self._prev_t is None else now - self._prev_t
        self._prev_t = now
        row = {"t_wall": f"{now:.6f}", "dt": f"{dt:.6f}", "phase": phase}
        for name in (
            "source_timestamp_ns", "source_age_s", "source_stall_count", "source_max_gap_s"
        ):
            if name in sample:
                row[name] = sample[name]
        for name in CAMERA_COLUMNS:
            if diagnostics is not None and name in diagnostics:
                row[name] = diagnostics[name]
        if "head_pos" in sample and "head_quat_xyzw" in sample:
            head = np.concatenate(
                [
                    np.asarray(sample["head_pos"]).reshape(3),
                    np.asarray(sample["head_quat_xyzw"]).reshape(4),
                ]
            )
            for axis, value in zip(("x", "y", "z", "qx", "qy", "qz", "qw"), head):
                row[f"head_raw_{axis}"] = f"{float(value):.8f}"
        action = getattr(env, "last_action", None)
        lost_count = 0
        for index, side in enumerate(("left", "right")):
            clutch = env.left_clutch if side == "left" else env.right_clutch
            raw = np.concatenate(
                [
                    np.asarray(sample[f"{side}_pos"], dtype=np.float64).reshape(3),
                    np.asarray(sample[f"{side}_quat_xyzw"], dtype=np.float64).reshape(4),
                ]
            )
            robot_pos, _ = transform_pico_to_robot(raw[:3], xyzw_to_wxyz(raw[3:]))
            measured = np.asarray(env.measured_pose(index)[0], dtype=np.float64).reshape(3)
            target = (
                np.asarray(action[index * 7 : index * 7 + 3], dtype=np.float64)
                if action is not None
                else measured
            )
            prev = self._prev_cmd[side]
            step = 0.0 if prev is None else float(np.linalg.norm(target - prev))
            self._prev_cmd[side] = target.copy()
            self.max_cmd_step = max(self.max_cmd_step, step)
            if step > 0.1:
                self.cmd_jumps += 1
            lost = looks_like_tracking_origin(robot_pos, xyzw_to_wxyz(raw[3:]))
            event = clutch.last_event
            if lost:
                self.lost_rows += 1
                self.lost_by_side[side] += 1
                lost_count += 1
            if event == "held":
                self.held_rows += 1
            if event == "reanchor":
                self.reanchor_rows += 1
            row.update(
                {
                    f"{side}_raw_x": f"{raw[0]:.8f}",
                    f"{side}_raw_y": f"{raw[1]:.8f}",
                    f"{side}_raw_z": f"{raw[2]:.8f}",
                    f"{side}_raw_qx": f"{raw[3]:.8f}",
                    f"{side}_raw_qy": f"{raw[4]:.8f}",
                    f"{side}_raw_qz": f"{raw[5]:.8f}",
                    f"{side}_raw_qw": f"{raw[6]:.8f}",
                    f"{side}_grip": f"{float(sample[f'{side}_grip']):.6f}",
                    f"{side}_trigger": f"{float(sample[f'{side}_trigger']):.6f}",
                    f"{side}_robot_x": f"{robot_pos[0]:.8f}",
                    f"{side}_robot_y": f"{robot_pos[1]:.8f}",
                    f"{side}_robot_z": f"{robot_pos[2]:.8f}",
                    f"{side}_lost": int(lost),
                    f"{side}_event": event,
                    f"{side}_engaged": int(clutch.enabled),
                    f"{side}_target_x": f"{target[0]:.8f}",
                    f"{side}_target_y": f"{target[1]:.8f}",
                    f"{side}_target_z": f"{target[2]:.8f}",
                    f"{side}_measured_x": f"{measured[0]:.8f}",
                    f"{side}_measured_y": f"{measured[1]:.8f}",
                    f"{side}_measured_z": f"{measured[2]:.8f}",
                    f"{side}_cmd_step": f"{step:.8f}",
                }
            )
        any_lost = lost_count > 0
        event_start = any_lost and not self._previous_any_lost
        self.lost_ticks += int(any_lost)
        self.lost_events += int(event_start)
        self.both_lost_ticks += int(lost_count == 2)
        self._previous_any_lost = any_lost
        row.update(
            lost_any=int(any_lost), lost_both=int(lost_count == 2),
            lost_event_start=int(event_start),
        )
        # Source and camera metadata may be mutable zero-dimensional numpy
        # views. Convert every value to a detached scalar before queuing.
        return {key: value.item() if isinstance(value, (np.ndarray, np.generic)) else value
                for key, value in row.items()}

    def timing_report(self) -> dict:
        alive = self._thread is not None and self._thread.is_alive()
        return {
            "queue_capacity": self.queue_capacity,
            "queued_rows": self.rows,
            "written_rows": self.written_rows,
            "pending_rows": self.rows - self.written_rows,
            "rejected_rows": self.rejected_rows,
            "max_queue_depth": self.max_queue_depth,
            "queue_wait_total_s": self.queue_wait_total_s,
            "queue_wait_max_s": self.queue_wait_max_s,
            "writer_alive": alive,
            "writer_error": None if self._worker_error is None else repr(self._worker_error),
            "io_profile": (self._io_profiler.to_dict(thread="teleop-csv-writer")
                           if self._io_profiler is not None and not alive else None),
        }

    def close(self) -> None:
        self._closing.set()
        if self._thread is not None:
            # No queue sentinel or queue.join(): both can hang after a failed
            # writer. Only the writer owns/ closes the file, even on timeout.
            self._thread.join(timeout=self.close_timeout)
            if self._thread.is_alive():
                raise TimeoutError(
                    f"CSV 后台写入尚未结束：{self.path}；已提交 {self.rows} 行，"
                    f"完成写入和 flush {self.written_rows} 行，文件可能不完整"
                )
        self._raise_worker_error()
        if self._reported_close:
            return
        self._reported_close = True
        if self._thread is None:
            print(f"CSV recording was not started: {self.path}")
            return
        print(
            f"Recorded {self.written_rows} ticks to {self.path}\n"
            f"  lost-origin ticks (either hand): {self.lost_ticks}; "
            f"both hands: {self.both_lost_ticks}; events: {self.lost_events}\n"
            f"  lost-origin hand samples: {self.lost_rows} "
            f"(left={self.lost_by_side['left']}, right={self.lost_by_side['right']})\n"
            f"  jump holds (hand samples): {self.held_rows}\n"
            f"  re-anchors (hand samples): {self.reanchor_rows}\n"
            f"  command steps > 0.1 m: {self.cmd_jumps}\n"
            f"  max command step: {self.max_cmd_step:.3f} m"
        )
