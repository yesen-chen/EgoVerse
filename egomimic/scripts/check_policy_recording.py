"""Synthetic SHM recording benchmark. Never opens a camera, CAN, or a model.

Compare disabled/enabled recording with three RGB publishers at 30 Hz, two
feedback publishers at 100 Hz, a 30 Hz fake action loop and request descriptors.
This is a host-load test, not a real-device timing guarantee.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import signal
import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _publisher(cameras, arms, stop):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from egomimic.robot.piper_pico.arm_process import state_examples

    images = {
        name: np.random.default_rng(i).integers(
            0, 256, ring.shared_arrays["color"].get().shape[1:], dtype=np.uint8
        )
        for i, (name, ring) in enumerate(cameras.items())
    }
    frame_id = feedback_id = 0
    camera_tick = arm_tick = time.monotonic()
    while not stop.is_set():
        now = time.monotonic()
        if now >= camera_tick:
            stamp = time.time()
            for name, ring in cameras.items():
                ring.put(
                    {
                        "frame_id": np.int64(frame_id),
                        "color": images[name],
                        "timestamp": stamp - 0.08,
                        "camera_receive_timestamp": stamp,
                    }
                )
            frame_id += 1
            camera_tick = max(camera_tick + 1 / 30, now)
        if now >= arm_tick:
            stamp = time.time()
            for ring in arms.values():
                row = state_examples()
                row.update(feedback_id=np.int64(feedback_id))
                for key in row:
                    if "timestamp" in key:
                        row[key] = np.float64(stamp)
                ring.put(row)
            feedback_id += 1
            arm_tick = max(arm_tick + 0.01, now)
        stop.wait(min(0.002, max(0, min(camera_tick, arm_tick) - time.monotonic())))


def _distribution(values, scale=1.0):
    values = np.asarray(values, dtype=float) * scale
    return (
        dict(
            zip(
                ("p50", "p95", "p99", "max"),
                map(float, (*np.percentile(values, (50, 95, 99)), values.max())),
            )
        )
        if values.size
        else {}
    )


def run_case(directory, *, enabled, seconds, width, height, video_format):
    from egomimic.robot.piper_pico.arm_process import state_examples
    from egomimic.robot.piper_pico.camera_process import (
        SharedMemoryRingBuffer,
        camera_examples,
    )
    from egomimic.robot.piper_pico.live_policy import SessionLog
    from egomimic.robot.piper_pico.policy_audit import RecordedPolicy
    from egomimic.robot.piper_pico.policy_recording import PolicyRecorder

    ctx = mp.get_context("spawn")
    with SharedMemoryManager(ctx=ctx) as manager:
        cameras = {
            name: SharedMemoryRingBuffer.create_from_examples(
                manager,
                camera_examples(height, width),
                get_max_k=32,
                get_time_budget=0.2,
                put_desired_frequency=30,
            )
            for name in ("front_1", "left_wrist", "right_wrist")
        }
        arms = {
            name: SharedMemoryRingBuffer.create_from_examples(
                manager,
                state_examples(),
                get_max_k=128,
                get_time_budget=0.01,
                put_desired_frequency=100,
            )
            for name in ("left", "right")
        }
        stop = ctx.Event()
        publisher = ctx.Process(target=_publisher, args=(cameras, arms, stop))
        recorder = PolicyRecorder(
            directory,
            cameras,
            arms,
            image_mode="all" if enabled else "off",
            video_format=video_format,
            record_feedback=enabled,
        )
        log = SessionLog(
            directory.parent / f"{directory.name}_logs",
            SimpleNamespace(log=enabled, execute=False),
        )

        class FakePolicy:
            def predict(self, frame, *, rtc=None):
                actions = np.repeat(frame.base_state[None], 100, axis=0)
                return {
                    "base_actions": actions,
                    "camera_actions": actions.copy(),
                    "observation_timestamp": frame.timestamp,
                    "inference_elapsed_s": 0,
                    "inference_steps": 10,
                }

        policy = (
            RecordedPolicy(FakePolicy(), log, recorder) if enabled else FakePolicy()
        )
        publisher.start()
        try:
            deadline = time.monotonic() + 15
            while not all(ring.count for ring in (*cameras.values(), *arms.values())):
                if not publisher.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError("synthetic publisher failed to start")
                time.sleep(0.01)
            if enabled:
                recorder.start()
            initial = {name: ring.count for name, ring in cameras.items()}
            sends, enqueue_ms, request_id = [], [], 0
            start = tick = time.monotonic()
            while time.monotonic() - start < seconds:
                now = time.monotonic()
                if now < tick:
                    time.sleep(min(tick - now, 0.002))
                    continue
                sends.append(now)
                tick = now + 1 / 30
                log.command(np.zeros(14), time.time() + 0.1, now, time.monotonic())
                if len(sends) % 9 == 1:
                    # Reads model existing observation-copy cost, not an image
                    # payload submitted to the parent logging queue.
                    refs = {}
                    for name, ring in cameras.items():
                        row = ring.get()
                        refs[name] = {
                            "frame_id": int(row["frame_id"]),
                            "timestamp": float(row["timestamp"]),
                        }
                    before = time.perf_counter()
                    frame = SimpleNamespace(
                        timestamp=time.time(),
                        base_state=np.zeros(14),
                        base_state_history=np.zeros((2, 14)),
                        state_history_offsets_s=(-2 / 30, -1 / 30),
                        source_timestamps={
                            **{name: row["timestamp"] for name, row in refs.items()},
                            "left": time.time(),
                            "right": time.time(),
                        },
                        image_frame_ids={
                            name: row["frame_id"] for name, row in refs.items()
                        },
                        image_receive_timestamps={},
                        image_space="raw",
                    )
                    prediction = policy.predict(
                        frame,
                        rtc={
                            "schema": "synthetic",
                            "request_id": request_id,
                            "base_actions": np.zeros((100, 14)),
                            "weights": np.ones(100),
                        },
                    )
                    log.cycle(
                        {
                            "observation_timestamp": frame.timestamp,
                            "audit_request_id": prediction.get("audit_request_id"),
                            "base_actions": prediction["base_actions"],
                        }
                    )
                    enqueue_ms.append((time.perf_counter() - before) * 1000)
                    request_id += 1
            elapsed = time.monotonic() - start
            counts = {
                name: ring.count - initial[name] for name, ring in cameras.items()
            }
        finally:
            # The producer stops before bounded drain; source buffers remain
            # alive until recorder children have been reaped.
            stop.set()
            publisher.join(timeout=5)
            if publisher.is_alive():
                publisher.terminate()
                publisher.join(timeout=2)
            if enabled:
                recorder.close()
            log.close()
        return {
            "enabled": enabled,
            "elapsed_s": elapsed,
            "synthetic": True,
            "camera_publication_fps": {
                name: count / elapsed for name, count in counts.items()
            },
            "fake_action_interval_ms": _distribution(np.diff(sends), 1000),
            "request_audit_ms": _distribution(enqueue_ms),
            "recording": recorder.stats if enabled else {"enabled": False},
            "control_logging": log.summary["logging"],
            "request_logging": log.summary["request_logging"],
            "publisher_exitcode": publisher.exitcode,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--video-format", choices=("mp4", "lossless"), default="mp4")
    args = parser.parse_args(argv)
    if (
        not np.isfinite(args.seconds)
        or args.seconds < 2
        or min(args.width, args.height) < 2
    ):
        parser.error("seconds >= 2, width/height >= 2 required")
    if args.width % 2 or args.height % 2:
        parser.error("width and height must be even for MP4")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for enabled in (False, True):
        results.append(
            run_case(
                args.output_dir / ("enabled" if enabled else "disabled"),
                enabled=enabled,
                seconds=args.seconds,
                width=args.width,
                height=args.height,
                video_format=args.video_format,
            )
        )
    report = {
        "schema": "piper_recording_synthetic_benchmark_v1",
        "limitations": "Synthetic publishers and fake actions; no camera, CAN, model, or guarantee of real-device timing. Cases run sequentially.",
        "image_shape": [args.height, args.width, 3],
        "results": results,
    }
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
