"""Best-effort inference evidence recording, without device/control ownership.

Only small request descriptors cross the parent queue. Spawned readers copy
existing shared-memory publications; bounded thread queues isolate them from
encoding/disk stalls. Readers never acknowledge or lock the producers. A full
queue drops evidence, never delays cameras, inference or robot commands.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import signal
import threading
import time
from collections import OrderedDict
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np


def _jsonable(value):
    """Keep unknown CAN timestamps explicit; write standards-compliant JSON."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _line(stream, value):
    stream.write(json.dumps(_jsonable(value), allow_nan=False) + "\n")


def _stats():
    return {
        "read": 0,
        "written": 0,
        "requests_written": 0,
        "requests_missing": 0,
        "requests_mismatched": 0,
        "queue_dropped": 0,
        "request_queue_dropped": 0,
        "frame_queue_dropped": 0,
        "source_missing": 0,
        "unsafe_read_dropped": 0,
        "read_timeouts": 0,
        "missing_ranges": [],
        "range_overflow": 0,
        "errors": [],
        "drained": False,
        "incomplete": False,
        "first_receive_timestamp": None,
        "last_receive_timestamp": None,
        "receive_interval_count": 0,
        "receive_interval_sum_s": 0.0,
        "receive_interval_max_s": 0.0,
    }


def _error(stats, message):
    stats["incomplete"] = True
    if len(stats["errors"]) < 20:
        stats["errors"].append(str(message)[:1000])


def _missing(stats, first, end, kind="source_missing"):
    if end <= first:
        return
    stats[kind] += end - first
    stats["incomplete"] = True
    ranges = stats["missing_ranges"]
    if ranges and ranges[-1]["kind"] == kind and ranges[-1]["end"] == first:
        ranges[-1]["end"] = end
    elif len(ranges) < 128:
        ranges.append({"first": first, "end": end, "kind": kind})
    else:
        stats["range_overflow"] += 1


class _RingReader:
    """Copy a bounded batch at one publication count, rejecting unsafe copies.

    The source ring protects its last get_max_k slots for get_time_budget.
    Reading arrays directly avoids get_last_k's second count load. Publication
    IDs must equal the 0-based write count; an ID check and copy-time/overwrite
    checks prevent mixing different publications after scheduling stalls.
    """

    def __init__(self, ring, id_key, stats, *, start_count=None, batch_size=8):
        if id_key not in ring.shared_arrays:
            raise ValueError(f"recording requires source field {id_key}")
        self.ring, self.id_key, self.stats = ring, id_key, stats
        self.cursor = (
            max(0, ring.count - ring.get_max_k) if start_count is None else start_count
        )
        self.batch_size = batch_size

    def read(self, limit=None):
        ring = self.ring
        started = time.monotonic()
        published = int(ring.count)
        if published < self.cursor:
            raise RuntimeError("source publication counter reset during recording")
        first = max(self.cursor, published - ring.get_max_k)
        if limit is not None:
            first = min(first, limit)
        _missing(self.stats, self.cursor, first)
        end = min(published, first + self.batch_size)
        if limit is not None:
            end = min(end, limit)
        if end <= first:
            self.cursor = max(
                self.cursor, min(first, limit) if limit is not None else first
            )
            return []
        indices = np.arange(first, end) % ring.buffer_size
        data = {
            key: value.get()[indices].copy()
            for key, value in ring.shared_arrays.items()
        }
        elapsed = time.monotonic() - started
        after = int(ring.count)
        self.cursor = end
        expected = np.arange(first, end)
        if (
            elapsed > ring.get_time_budget
            or after >= first + ring.buffer_size
            or not np.array_equal(data[self.id_key], expected)
        ):
            self.stats["read_timeouts"] += int(elapsed > ring.get_time_budget)
            _missing(self.stats, first, end, "unsafe_read_dropped")
            return []
        rows = []
        for index in range(end - first):
            row = {key: value[index].copy() for key, value in data.items()}
            for value in row.values():
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)
            rows.append(row)
            stamp = float(
                row.get(
                    "camera_receive_timestamp",
                    row.get("robot_receive_timestamp", np.nan),
                )
            )
            if np.isfinite(stamp):
                previous = self.stats["last_receive_timestamp"]
                if previous is not None:
                    dt = stamp - previous
                    self.stats["receive_interval_count"] += 1
                    self.stats["receive_interval_sum_s"] += dt
                    self.stats["receive_interval_max_s"] = max(
                        self.stats["receive_interval_max_s"], dt
                    )
                else:
                    self.stats["first_receive_timestamp"] = stamp
                self.stats["last_receive_timestamp"] = stamp
        self.stats["read"] += len(rows)
        return rows


class _WritePump:
    """A stalled sink can consume only these bounded queues plus one item."""

    def __init__(self, consume, finish, stats, capacity):
        self.normal, self.priority = Queue(capacity), Queue(capacity)
        self.consume, self.finish, self.stats = consume, finish, stats
        self.stopping, self.failed = threading.Event(), threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="evidence_writer", daemon=True
        )
        self.thread.start()

    def submit(self, item, *, priority=False):
        if self.failed.is_set():
            self.stats["request_queue_dropped" if priority else "queue_dropped"] += 1
            self.stats["incomplete"] = True
            return False
        try:
            (self.priority if priority else self.normal).put_nowait(item)
            return True
        except Full:
            self.stats["request_queue_dropped" if priority else "queue_dropped"] += 1
            self.stats["incomplete"] = True
            return False

    def _run(self):
        try:
            while (
                not self.stopping.is_set()
                or not self.priority.empty()
                or not self.normal.empty()
            ):
                try:
                    item = self.priority.get_nowait()
                except Empty:
                    try:
                        item = self.normal.get(timeout=0.01)
                    except Empty:
                        continue
                self.consume(item)
            self.finish()
        except BaseException as exc:
            _error(self.stats, f"writer {type(exc).__name__}: {exc}")
            self.failed.set()

    def close(self, timeout):
        self.stopping.set()
        self.thread.join(timeout)
        self.stats["pending_normal"] = self.normal.qsize()
        self.stats["pending_requests"] = self.priority.qsize()
        self.stats["writer_busy_on_close"] = self.thread.is_alive()
        self.stats["drained"] = not self.thread.is_alive() and not self.failed.is_set()
        if not self.stats["drained"]:
            _error(self.stats, "writer did not drain successfully")


def _worker_setup():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        os.nice(5)
    except OSError:
        pass


class _CameraSink:
    def __init__(self, directory, video_format, fps, segment_seconds, stats):
        import cv2

        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        self.cv2 = cv2
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "requests").mkdir()
        (self.directory / "frames").mkdir()
        self.frames = (self.directory / "frames.jsonl").open("x", buffering=1)
        self.requests = (self.directory / "requests.jsonl").open("x", buffering=1)
        self.video_format, self.fps = video_format, fps
        self.segment_seconds, self.stats = segment_seconds, stats
        self.container = self.stream = None
        self.segment, self.segment_start, self.video_index = -1, None, 0

    def _close_video(self):
        if self.container is not None:
            for packet in self.stream.encode():
                self.container.mux(packet)
            self.container.close()
            self.container = self.stream = None

    def _png(self, path, rgb):
        if not self.cv2.imwrite(
            str(path), self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
        ):
            raise OSError(f"failed to write {path}")

    def __call__(self, item):
        kind, row, request = item
        if kind == "missing":
            _line(self.requests, request)
            return
        rgb = row["color"]
        metadata = {key: value for key, value in row.items() if key != "color"}
        if kind == "request":
            # The request ID is metadata, never an untrusted filesystem path.
            filename = f"requests/frame_{int(row['frame_id']):012d}.png"
            path = self.directory / filename
            if not path.exists():
                self._png(path, rgb)
            _line(
                self.requests,
                {
                    **metadata,
                    **request,
                    "status": "saved",
                    "file": filename,
                    "sha256_rgb": hashlib.sha256(
                        memoryview(np.ascontiguousarray(rgb))
                    ).hexdigest(),
                    "shape": list(rgb.shape),
                    "dtype": str(rgb.dtype),
                    "color_order": "RGB",
                },
            )
            self.stats["requests_written"] += 1
            return
        if self.video_format == "lossless":
            filename = f"frames/{int(row['frame_id']):012d}.png"
            self._png(self.directory / filename, rgb)
            metadata.update(file=filename, lossless=True)
        else:
            import av

            stamp = float(row["camera_receive_timestamp"])
            if (
                self.container is None
                or stamp - self.segment_start >= self.segment_seconds
            ):
                self._close_video()
                self.segment += 1
                self.segment_start, self.video_index = stamp, 0
                path = self.directory / f"frames/segment_{self.segment:06d}.mp4"
                self.container = av.open(str(path), mode="w")
                self.stream = self.container.add_stream(
                    "libx264", rate=int(round(self.fps))
                )
                self.stream.width, self.stream.height = rgb.shape[1], rgb.shape[0]
                self.stream.pix_fmt = "yuv420p"
                self.stream.codec_context.thread_count = 1
                self.stream.options = {
                    "preset": "ultrafast",
                    "crf": "18",
                    "tune": "zerolatency",
                }
            for packet in self.stream.encode(
                av.VideoFrame.from_ndarray(rgb, format="rgb24")
            ):
                self.container.mux(packet)
            metadata.update(
                file=f"frames/segment_{self.segment:06d}.mp4",
                video_frame=self.video_index,
                lossless=False,
                timing="use source timestamps; video playback uses nominal fps",
            )
            self.video_index += 1
        _line(self.frames, metadata)
        self.stats["written"] += 1

    def close(self):
        self._close_video()
        self.frames.close()
        self.requests.close()


def _camera_worker(directory, ring, commands, status, config):
    _worker_setup()
    stats, cache, pending = _stats(), OrderedDict(), []
    pump = None
    try:
        reader = _RingReader(ring, "frame_id", stats)
        sink = _CameraSink(
            directory,
            config["video_format"],
            ring.put_desired_frequency,
            config["segment_seconds"],
            stats,
        )
        pump = _WritePump(sink, sink.close, stats, config["queue_capacity"])
        status.send(("ready", {"pid": os.getpid(), "start_count": reader.cursor}))
        stop_count = None
        while stop_count is None or reader.cursor < stop_count or pending:
            while stop_count is None:
                try:
                    request = commands.get_nowait()
                except Empty:
                    break
                if request is None:
                    stop_count = int(ring.count)
                    break
                if len(pending) >= config["queue_capacity"]:
                    stats["request_queue_dropped"] += 1
                    stats["incomplete"] = True
                else:
                    pending.append(request)
            rows = reader.read(limit=stop_count)
            for row in rows:
                frame_id = int(row["frame_id"])
                cache[frame_id] = row
                while len(cache) > config["cache_frames"]:
                    cache.popitem(last=False)
                if config["image_mode"] == "all":
                    if not pump.submit(("frame", row, None)):
                        _missing(stats, frame_id, frame_id + 1, "frame_queue_dropped")
            remaining = []
            for request in pending:
                frame_id = request["frame_id"]
                row = cache.get(frame_id)
                if row is not None:
                    if abs(float(row["timestamp"]) - request["timestamp"]) > 1e-8:
                        stats["requests_mismatched"] += 1
                        stats["incomplete"] = True
                        pump.submit(
                            (
                                "missing",
                                None,
                                {**request, "status": "timestamp_mismatch"},
                            ),
                            priority=True,
                        )
                    else:
                        pump.submit(("request", row, request), priority=True)
                elif frame_id < reader.cursor or (
                    stop_count is not None
                    and (frame_id >= stop_count or reader.cursor >= stop_count)
                ):
                    stats["requests_missing"] += 1
                    stats["incomplete"] = True
                    pump.submit(
                        ("missing", None, {**request, "status": "frame_unavailable"}),
                        priority=True,
                    )
                else:
                    remaining.append(request)
            pending = remaining
            if pump.failed.is_set():
                break
            if not rows:
                time.sleep(0.002)
        pump.close(config["close_timeout"])
    except BaseException as exc:
        _error(stats, f"{type(exc).__name__}: {exc}")
        if pump is not None:
            pump.close(config["close_timeout"])
    finally:
        status.send(("closed", stats))
        status.close()


def _feedback_worker(directory, arms, stop, status, config):
    _worker_setup()
    stats, streams, pump = _stats(), {}, None
    per_arm = {}
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        readers = {}
        for name, ring in arms.items():
            per_arm[name] = _stats()
            readers[name] = _RingReader(
                ring, "feedback_id", per_arm[name], batch_size=32
            )
            streams[name] = (Path(directory) / f"{name}.jsonl").open("x", buffering=1)

        def consume(item):
            name, row = item
            _line(streams[name], row)
            stats["written"] += 1
            per_arm[name]["written"] += 1

        def finish():
            for stream in streams.values():
                stream.close()

        pump = _WritePump(consume, finish, stats, config["queue_capacity"])
        status.send(
            (
                "ready",
                {
                    "pid": os.getpid(),
                    "start_counts": {name: r.cursor for name, r in readers.items()},
                },
            )
        )
        limits = None
        while True:
            if stop.is_set() and limits is None:
                limits = {
                    name: int(reader.ring.count) for name, reader in readers.items()
                }
            count = 0
            for name, reader in readers.items():
                rows = reader.read(limit=None if limits is None else limits[name])
                count += len(rows)
                stats["read"] += len(rows)
                for row in rows:
                    if not pump.submit((name, row)):
                        row_id = int(row["feedback_id"])
                        _missing(per_arm[name], row_id, row_id + 1, "queue_dropped")
            if pump.failed.is_set() or (
                limits is not None
                and all(r.cursor >= limits[name] for name, r in readers.items())
            ):
                break
            if not count:
                time.sleep(0.002)
        pump.close(config["close_timeout"])
    except BaseException as exc:
        _error(stats, f"{type(exc).__name__}: {exc}")
        if pump is not None:
            pump.close(config["close_timeout"])
    finally:
        # Both arm streams share this writer pump; they have no independent
        # queues whose default false drain flag should survive a clean close.
        for item in per_arm.values():
            item["drained"] = stats["drained"]
        stats["arms"] = per_arm
        stats["incomplete"] |= any(item["incomplete"] for item in per_arm.values())
        status.send(("closed", stats))
        status.close()


def _rings(devices):
    if isinstance(devices, dict):
        result = {
            name: getattr(value, "ring", value) for name, value in devices.items()
        }
    else:
        result = {device.name: device.ring for device in devices}
    for name in result:
        if (
            not isinstance(name, str)
            or name in ("", ".", "..")
            or Path(name).name != name
        ):
            raise ValueError("recording device names must be simple directory names")
    return result


class PolicyRecorder:
    """Lifecycle waits are outside control. request() never waits for capacity."""

    def __init__(
        self,
        directory,
        cameras,
        arms,
        *,
        image_mode="off",
        video_format="mp4",
        record_feedback=False,
        queue_capacity=64,
        cache_frames=90,
        segment_seconds=60,
        close_timeout=5,
        startup_timeout=20,
        _camera_target=_camera_worker,
        _feedback_target=_feedback_worker,
    ):
        if image_mode not in ("off", "requests", "all") or video_format not in (
            "mp4",
            "lossless",
        ):
            raise ValueError("invalid evidence image mode or format")
        if (
            queue_capacity < 1
            or cache_frames < 1
            or min(segment_seconds, close_timeout, startup_timeout) <= 0
        ):
            raise ValueError("recording capacities and timeouts must be positive")
        self.directory = Path(directory)
        self.cameras, self.arms = _rings(cameras), _rings(arms)
        self.config = dict(
            image_mode=image_mode,
            video_format=video_format,
            queue_capacity=int(queue_capacity),
            cache_frames=int(cache_frames),
            segment_seconds=float(segment_seconds),
            close_timeout=float(close_timeout),
        )
        self.record_feedback, self.startup_timeout = record_feedback, startup_timeout
        self._camera_target, self._feedback_target = _camera_target, _feedback_target
        self._workers, self._started, self._closed = {}, False, False
        self.stats = dict(
            enabled=image_mode != "off" or record_feedback,
            config=dict(self.config),
            record_feedback=bool(record_feedback),
            workers={},
            requests_submitted=0,
            requests_dropped=0,
            errors=[],
            incomplete=False,
            drained=False,
        )

    def start(self):
        if self._started or self._closed:
            return
        self._started = True
        if not self.stats["enabled"]:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        ctx = mp.get_context("spawn")
        specs = []
        if self.config["image_mode"] != "off":
            for name, ring in self.cameras.items():
                queue = ctx.Queue(self.config["queue_capacity"])
                specs.append(
                    (
                        f"camera:{name}",
                        self._camera_target,
                        (str(self.directory / "cameras" / name), ring, queue),
                        queue,
                        None,
                    )
                )
        if self.record_feedback:
            stop = ctx.Event()
            specs.append(
                (
                    "feedback",
                    self._feedback_target,
                    (str(self.directory / "feedback"), self.arms, stop),
                    None,
                    stop,
                )
            )
        try:
            for key, target, args, queue, stop in specs:
                parent, child = ctx.Pipe(duplex=False)
                process = ctx.Process(
                    target=target,
                    args=(*args, child, self.config),
                    daemon=True,
                    name=f"policy_record_{key}",
                )
                self._workers[key] = dict(
                    process=process,
                    status=parent,
                    queue=queue,
                    stop=stop,
                    ready=False,
                    closed=False,
                    eof=False,
                )
                self.stats["workers"][key] = {"drained": False}
                process.start()
                child.close()
            deadline = time.monotonic() + self.startup_timeout
            while not all(w["ready"] for w in self._workers.values()):
                self.poll()
                if self.stats["errors"] or time.monotonic() > deadline:
                    raise RuntimeError(
                        "; ".join(self.stats["errors"]) or "recorder startup timed out"
                    )
                time.sleep(0.01)
        except BaseException:
            self.close()
            raise

    def request(self, request_id, image_refs):
        result = {}
        if self.config["image_mode"] == "off":
            return result
        for name, reference in image_refs.items():
            key = f"camera:{name}"
            worker = self._workers.get(key)
            accepted = False
            if worker is not None and not self._closed and not worker["closed"]:
                # Reconstruct a tiny immutable descriptor; never serialize image/state arrays.
                try:
                    item = dict(
                        request_id=str(request_id),
                        frame_id=int(reference["frame_id"]),
                        timestamp=float(reference["timestamp"]),
                    )
                    if not np.isfinite(item["timestamp"]) or item["frame_id"] < 0:
                        raise ValueError("invalid frame reference")
                    worker["queue"].put_nowait(item)
                    accepted = True
                except (Full, OSError, ValueError, TypeError, KeyError):
                    pass
            result[name] = accepted
            self.stats["requests_submitted" if accepted else "requests_dropped"] += 1
            if not accepted:
                self.stats["incomplete"] = True
        return result

    def poll(self):
        if self._closed:
            return
        for key, worker in self._workers.items():
            while not worker["eof"] and worker["status"].poll():
                try:
                    kind, value = worker["status"].recv()
                except (EOFError, OSError):
                    worker["eof"] = True
                    break
                if kind == "ready":
                    worker["ready"] = True
                    self.stats["workers"][key].update(value)
                else:
                    worker["closed"] = True
                    self.stats["workers"][key].update(value)
                    self.stats["incomplete"] |= value["incomplete"]
                    for error in value["errors"]:
                        _error(self.stats, f"{key}: {error}")
            if (
                worker["process"].pid
                and not worker["process"].is_alive()
                and not worker["closed"]
            ):
                worker["closed"] = True
                _error(self.stats, f"{key} exited without recording summary")

    def close(self):
        if self._closed:
            return
        deadline = time.monotonic() + self.config["close_timeout"]
        for key, worker in self._workers.items():
            if worker["stop"] is not None:
                worker["stop"].set()
            if worker["queue"] is not None and worker["process"].is_alive():
                try:
                    worker["queue"].put(
                        None, timeout=max(0, deadline - time.monotonic())
                    )
                except (Full, OSError, ValueError):
                    _error(
                        self.stats, f"{key}: shutdown descriptor queue did not drain"
                    )
        # Drain status pipes while joining: a bounded summary may exceed the
        # OS pipe buffer, and workers must not wait for a reader until timeout.
        while time.monotonic() < deadline:
            self.poll()
            if all(
                worker["process"].pid is None or not worker["process"].is_alive()
                for worker in self._workers.values()
            ):
                break
            time.sleep(0.005)
        for key, worker in self._workers.items():
            process = worker["process"]
            if process.pid is not None:
                process.join(max(0, deadline - time.monotonic()))
                self.poll()
                if process.is_alive():
                    _error(
                        self.stats,
                        f"{key}: forced termination; queued/in-flight records may be missing",
                    )
                    process.terminate()
                    process.join(0.5)
                    if process.is_alive():
                        process.kill()
                        process.join(0.5)
                self.stats["workers"][key]["exitcode"] = process.exitcode
                if process.exitcode != 0:
                    self.stats["workers"][key]["drained"] = False
                    _error(self.stats, f"{key}: worker exit code {process.exitcode}")
            worker["status"].close()
            worker["eof"] = True
            if worker["queue"] is not None:
                worker["queue"].cancel_join_thread()
                worker["queue"].close()
        self.stats["drained"] = all(
            value.get("drained", False) for value in self.stats["workers"].values()
        )
        self.stats["incomplete"] |= not self.stats["drained"]
        self._closed = True
        if self.stats["enabled"] and self.directory.exists():
            try:
                (self.directory / "summary.json").write_text(
                    json.dumps(_jsonable(self.stats), indent=2, allow_nan=False)
                )
            except OSError as exc:
                _error(self.stats, f"failed to write final recording summary: {exc}")
