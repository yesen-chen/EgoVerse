"""Bounded, opt-in loop timing; the only file write is at shutdown.

Durations use a monotonic clock. Thread CPU time excludes other threads and
processes: wall/CPU differences do not by themselves identify scheduling, I/O,
locks, or GC. Nested spans (notably on_tick) must not be added together.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import datetime
import json
from pathlib import Path
import time


def profile_span(profiler, name, **metadata):
    return nullcontext() if profiler is None else profiler.span(name, **metadata)


class LoopProfiler:
    def __init__(self, slow_s=0.050, max_events=256, *, monotonic=None, wall=None, cpu=None):
        self.slow_s = slow_s
        self.max_events = max_events
        self._mono = monotonic or time.monotonic
        self._wall = wall or time.time
        self._cpu = cpu or time.thread_time
        self.stats = {}
        self.events = []
        self.dropped_events = 0
        self._active = None
        self._previous = None
        self._depth = 0

    def _point(self):
        return {"monotonic_s": self._mono(), "wall_timestamp_s": self._wall(),
                "thread_cpu_timestamp_s": self._cpu()}

    def _finish(self, name, start, *, end=None, **metadata):
        end = self._point() if end is None else end
        record = {"stage": name, **start, **metadata,
                  "end_monotonic_s": end["monotonic_s"],
                  "wall_s": max(0.0, end["monotonic_s"] - start["monotonic_s"]),
                  "thread_cpu_s": max(0.0, end["thread_cpu_timestamp_s"] - start["thread_cpu_timestamp_s"])}
        if "deadline_monotonic_s" in record:
            record["deadline_lateness_s"] = max(0.0, end["monotonic_s"] - record["deadline_monotonic_s"])
        stats = self.stats.setdefault(name, {
            "count": 0, "wall_total_s": 0.0, "wall_max_s": 0.0,
            "thread_cpu_total_s": 0.0, "thread_cpu_max_s": 0.0,
        })
        stats["count"] += 1
        for field in ("wall", "thread_cpu"):
            stats[f"{field}_total_s"] += record[f"{field}_s"]
            stats[f"{field}_max_s"] = max(stats[f"{field}_max_s"], record[f"{field}_s"])
        if "deadline_lateness_s" in record:
            stats["deadline_lateness_max_s"] = max(
                stats.get("deadline_lateness_max_s", 0.0), record["deadline_lateness_s"])
        return record

    def _event(self, record):
        if len(self.events) < self.max_events:
            self.events.append(record)
        else:
            self.dropped_events += 1

    @contextmanager
    def span(self, name, **metadata):
        start = self._point()
        depth = self._depth
        self._depth += 1
        error = None
        try:
            yield
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            self._depth -= 1
            record = self._finish(name, start, depth=depth, error_type=error,
                                  tick=None if self._active is None else self._active["tick"], **metadata)
            if self._active is not None:
                if len(self._active["spans"]) < 64:
                    self._active["spans"].append(record)
                else:
                    self._active["dropped_spans"] += 1
            if record["wall_s"] >= self.slow_s or record.get("deadline_lateness_s", 0) >= self.slow_s:
                self._event({"kind": "slow_stage", **record})

    @contextmanager
    def tick(self, index, deadline_monotonic_s):
        start = self._point()
        if self._previous is not None:
            interval = self._finish("tick_interval", self._previous["start"], end=start, tick=index)
            if interval["wall_s"] >= self.slow_s:
                self._event({"kind": "slow_tick_interval", **interval,
                             "between_ticks_s": max(0.0, start["monotonic_s"] - self._previous["end_mono"]),
                             "previous_tick": self._previous["summary"]})
        self._active = {"tick": index, "spans": [], "dropped_spans": 0}
        try:
            yield
        finally:
            end = self._point()
            summary = self._finish("tick", start, end=end, **self._active,
                                   deadline_monotonic_s=deadline_monotonic_s)
            # Subtract only top-level spans; on_tick includes the sim spans.
            summary["unattributed_wall_s"] = max(0.0, summary["wall_s"] - sum(
                s["wall_s"] for s in summary["spans"] if s["depth"] == 0))
            if summary["wall_s"] >= self.slow_s:
                self._event({"kind": "slow_tick", **summary})
            self._previous = {"start": start, "end_mono": end["monotonic_s"], "summary": summary}
            self._active = None

    def to_dict(self, **metadata):
        """Snapshot a profiler after its owning thread has finished."""
        def local_times(value):
            if isinstance(value, dict):
                result = {k: local_times(v) for k, v in value.items()}
                if "wall_timestamp_s" in result:
                    result["local_time"] = datetime.fromtimestamp(result["wall_timestamp_s"]).astimezone().isoformat()
                return result
            if isinstance(value, list):
                return [local_times(v) for v in value]
            return value

        data = {"version": 1, "slow_threshold_s": self.slow_s,
                "max_events": self.max_events, "dropped_events": self.dropped_events,
                "metadata": metadata, "stats": self.stats, "events": self.events,
                "notes": ["Tick indices are zero based; ticks may exit before a CSV row is written.",
                          "Nested spans overlap: do not sum on_tick with its sim.* children.",
                          "tick_interval is start-to-start and differs from the monitor's image-read checkpoint.",
                          "Thread CPU time does not include other processes/threads or identify a specific stall cause."]}
        return local_times(data)

    def write_json(self, path, *, recorder=None, **metadata):
        data = self.to_dict(**metadata)
        if recorder is not None:
            data["recorder"] = recorder
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
