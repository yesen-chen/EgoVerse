"""Bounded, best-effort JSONL logging in a dedicated spawned process.

The producer never waits for disk or queue space. Encoding and file writes live
in the child; multiprocessing's feeder transports owned snapshots. Only startup
and shutdown wait for the child, outside action execution.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from contextlib import ExitStack
from pathlib import Path
from queue import Full

import numpy as np

KINDS = ("commands", "cycles", "events")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot log {type(value).__name__}")


def _write_loop(directory, records, status):
    """No model/device imports and no access to robot command channels."""
    written = dict.fromkeys(KINDS, 0)
    try:
        with ExitStack() as stack:
            streams = {
                kind: stack.enter_context(
                    (Path(directory) / f"{kind}.jsonl").open("x", buffering=1)
                )
                for kind in KINDS
            }
            status.send(("ready", {"pid": os.getpid()}))
            while True:
                item = records.get()
                if item is None:
                    break
                kind, payload = item
                streams[kind].write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        allow_nan=False,
                        default=_json_default,
                    )
                    + "\n"
                )
                written[kind] += 1
        # Acknowledgement comes after flushing/closing every stream.
        status.send(("closed", {"written": written}))
    except BaseException as exc:
        status.send(
            (
                "error",
                {
                    "type": type(exc).__name__,
                    "message": str(exc)[:2000],
                    "written": written,
                },
            )
        )
    finally:
        status.close()


class AsyncLogWriter:
    """submit() only enqueues; poll() belongs at a chunk boundary or while idle."""

    def __init__(
        self,
        directory,
        *,
        capacity=256,
        startup_timeout=20.0,
        close_timeout=5.0,
        _worker_target=_write_loop,
    ):
        if capacity < 1 or startup_timeout <= 0 or close_timeout <= 0:
            raise ValueError("log capacity and timeouts must be positive")
        self.close_timeout = close_timeout
        self.stats = {
            "enabled": True,
            "mode": "process",
            "queue_capacity": capacity,
            "submitted": dict.fromkeys(KINDS, 0),
            "dropped": dict.fromkeys(KINDS, 0),
            "written": dict.fromkeys(KINDS, 0),
            "errors": [],
            "incomplete": False,
            "drained": False,
        }
        self._ready = self._closed = self._failed = False
        self._status_eof = False
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue(maxsize=capacity)
        self._status, child_status = ctx.Pipe(duplex=False)
        self._process = ctx.Process(
            target=_worker_target,
            args=(str(directory), self._queue, child_status),
            name="piper_policy_log",
            daemon=True,
        )
        try:
            self._process.start()
            child_status.close()
            deadline = time.monotonic() + startup_timeout
            while not self._ready and not self._failed:
                if time.monotonic() >= deadline:
                    self._fail("logger startup timed out")
                    break
                self._status.poll(min(0.05, max(0.0, deadline - time.monotonic())))
                self.poll()
            if self._failed:
                raise RuntimeError(self.stats["errors"][-1])
        except BaseException:
            child_status.close()
            self.close()
            raise

    def _fail(self, message):
        self._failed = True
        self.stats["incomplete"] = True
        if message not in self.stats["errors"]:
            self.stats["errors"].append(message)

    def submit(self, kind, payload):
        """Payload must be an owned snapshot, never subsequently mutated."""
        if self._closed or self._failed:
            self.stats["dropped"][kind] += 1
            self.stats["incomplete"] = True
            return False
        try:
            self._queue.put_nowait((kind, payload))
        except Full:
            self.stats["dropped"][kind] += 1
            self.stats["incomplete"] = True
            return False
        except (OSError, ValueError) as exc:
            self._fail(f"log queue failed: {exc}")
            self.stats["dropped"][kind] += 1
            return False
        self.stats["submitted"][kind] += 1
        return True

    def poll(self):
        if self._closed:
            return
        while not self._status_eof and self._status.poll():
            try:
                kind, value = self._status.recv()
            except (EOFError, OSError) as exc:
                self._status_eof = True
                if isinstance(exc, OSError):
                    self._fail(f"logger status channel failed: {exc}")
                break
            if kind == "ready":
                self._ready = True
                self.stats["writer_pid"] = value["pid"]
            elif kind == "closed":
                self.stats["written"] = value["written"]
                self.stats["drained"] = True
            elif kind == "error":
                self.stats["written"] = value["written"]
                self._fail(f"{value['type']}: {value['message']}")
        if (
            not self._process.is_alive()
            and not self.stats["drained"]
            and not self._failed
        ):
            self._fail(f"logger exited before drain (code {self._process.exitcode})")

    def close(self):
        if self._closed:
            return
        deadline = time.monotonic() + self.close_timeout
        try:
            if self._process.pid is not None:
                self.poll()
                if not self._failed and self._process.is_alive():
                    try:
                        self._queue.put(
                            None, timeout=max(0.0, deadline - time.monotonic())
                        )
                    except Full:
                        self._fail("logger shutdown queue timed out")
                    except (OSError, ValueError) as exc:
                        self._fail(f"logger shutdown queue failed: {exc}")
                self._process.join(timeout=max(0.0, deadline - time.monotonic()))
                self.poll()
                if self._process.is_alive():
                    self._fail("logger did not drain before shutdown timeout")
                    self._process.terminate()
                    self._process.join(timeout=1.0)
                    if self._process.is_alive():
                        self._process.kill()
                        self._process.join(timeout=1.0)
                self.stats["writer_exitcode"] = self._process.exitcode
                if self.stats["writer_exitcode"] != 0:
                    self._fail(
                        f"logger exited with code {self.stats['writer_exitcode']}"
                    )
        finally:
            self._closed = True
            self.stats["unwritten"] = {
                kind: self.stats["submitted"][kind] - self.stats["written"][kind]
                for kind in KINDS
            }
            if any(self.stats["unwritten"].values()):
                self.stats["incomplete"] = True
            # A dead/stuck reader may leave the feeder blocked on a full pipe.
            # Never join that feeder during hardware-session cleanup.
            self._queue.cancel_join_thread()
            self._queue.close()
            self._status.close()
            if self._process.pid is not None and not self._process.is_alive():
                self._process.close()
