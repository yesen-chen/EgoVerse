"""Camera-only native UVC recording acceptance; no robot or PICO pose SDK.

Each cycle opens all configured cameras, records them together, closes every
worker, and verifies the saved videos and timestamp sidecars before reopening.
Host receive timing cannot establish physical exposure synchronization.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import signal
import time
from datetime import datetime, timezone
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import numpy as np

from egomimic.robot.piper_pico.camera_config import load_camera_config
from egomimic.robot.piper_pico.camera_process import CameraProcess

LIMITATION = (
    "Receive timestamps measure host frame arrival, not sensor exposure. "
    "Receive-minus-timestamp is configured compensation, not measured latency. "
    "PASS does not prove physical exposure synchronization or geometric calibration."
)


def positive_float(value):
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError("must be a positive finite number")
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def positive_int(value):
    try:
        number = int(value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if str(number) != str(value) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def decode_video(path: Path) -> dict:
    """Decode every saved frame; container/driver frame counts are not evidence."""
    import av

    sizes, count = set(), 0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            sizes.add((frame.width, frame.height))
            count += 1
    return {"decoded_frames": count, "decoded_sizes": [list(x) for x in sorted(sizes)]}


def _timestamps(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"{path.name}: need at least two finite 1-D timestamps")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{path.name}: timestamps must be strictly increasing")
    return values


def inspect_recording(
    path, spec, *, max_gap_seconds=None, min_fps_ratio=0.9, video_probe=decode_video
) -> tuple[dict, np.ndarray]:
    """Validate real decoded counts, native sizes and saved arrival intervals."""
    path = Path(path)
    stamps = _timestamps(path.with_suffix(".npy"))
    received = _timestamps(path.with_suffix(".receive_timestamps.npy"))
    if len(stamps) != len(received):
        raise ValueError(f"{spec['name']}: timestamp sidecar count mismatch")
    result = video_probe(path)
    if result["decoded_frames"] != len(stamps):
        raise ValueError(f"{spec['name']}: decoded video / timestamp count mismatch")
    if result["decoded_sizes"] != [[spec["width"], spec["height"]]]:
        raise ValueError(f"{spec['name']}: decoded video size differs from native mode")
    metadata = json.loads(path.with_suffix(".stream_metadata.json").read_text())
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") != "uvc_stream_metadata_v1"
    ):
        raise ValueError(f"{spec['name']}: invalid stream metadata schema")
    if metadata.get("frame_count") != len(stamps):
        raise ValueError(f"{spec['name']}: stream metadata frame count mismatch")
    if metadata.get("native_frame_shape") != [spec["height"], spec["width"], 3]:
        raise ValueError(f"{spec['name']}: stream metadata native frame size mismatch")
    gaps = np.diff(received)
    observed_fps = float((len(received) - 1) / (received[-1] - received[0]))
    threshold = max_gap_seconds or 3.0 / spec["fps"]
    compensation = received - stamps
    errors = []
    if metadata.get("end_verification_error"):
        errors.append(
            f"stream end verification failed: {metadata['end_verification_error']}"
        )
    gap_count = int(np.count_nonzero(gaps > threshold + 1e-9))
    if gap_count:
        errors.append(f"{gap_count} receive gaps exceed {threshold:g}s")
    if observed_fps < spec["fps"] * min_fps_ratio:
        errors.append(
            f"observed FPS {observed_fps:.3f} below {min_fps_ratio:g} of requested FPS"
        )
    if not np.allclose(compensation, spec["latency"], atol=1e-6, rtol=0):
        errors.append("receive-minus-timestamp differs from configured compensation")
    result.update(
        name=spec["name"],
        video=str(path),
        frame_count=len(received),
        requested_fps=spec["fps"],
        observed_mean_fps=observed_fps,
        stream_metadata=metadata,
        driver_fps_interpretation="stream_metadata.actual_mode.fps is driver readback, not measured FPS",
        receive_span_s=float(received[-1] - received[0]),
        first_receive_timestamp=float(received[0]),
        last_receive_timestamp=float(received[-1]),
        receive_gap_mean_s=float(gaps.mean()),
        receive_gap_max_s=float(gaps.max()),
        receive_gap_p95_s=float(np.percentile(gaps, 95)),
        gap_threshold_s=threshold,
        excessive_gap_count=gap_count,
        configured_latency_s=spec["latency"],
        latency_status=spec["latency_status"],
        receive_minus_timestamp_mean_s=float(compensation.mean()),
        receive_minus_timestamp_min_s=float(compensation.min()),
        receive_minus_timestamp_max_s=float(compensation.max()),
        errors=errors,
        status="FAIL" if errors else "PASS",
    )
    return result, received


def receive_differences(streams: dict[str, np.ndarray]) -> list[dict]:
    """Nearest host arrivals in overlapping intervals, never exposure alignment."""
    result = []
    for (name_a, a), (name_b, b) in itertools.combinations(streams.items(), 2):
        start, end = max(a[0], b[0]), min(a[-1], b[-1])
        a = a[(a >= start) & (a <= end)]
        if not len(a) or start >= end:
            raise ValueError(
                f"{name_a}/{name_b}: recorded streams have no receive-time overlap"
            )
        right = np.clip(np.searchsorted(b, a), 0, len(b) - 1)
        left = np.maximum(0, right - 1)
        closest = np.where(np.abs(b[left] - a) <= np.abs(b[right] - a), left, right)
        differences = b[closest] - a
        result.append(
            {
                "camera_a": name_a,
                "camera_b": name_b,
                "pairs": len(a),
                "method": "nearest host receive timestamp, B minus A; not exposure sync",
                "signed_mean_s": float(differences.mean()),
                "absolute_p95_s": float(np.percentile(np.abs(differences), 95)),
                "absolute_max_s": float(np.max(np.abs(differences))),
            }
        )
    return result


def _healthy(camera, spec, wall_time, stale_seconds):
    if camera.process.exitcode is not None or not camera.process.is_alive():
        raise RuntimeError(
            f"{spec['name']}: camera worker exited (code {camera.process.exitcode})"
        )
    sample = camera.get_last(1)
    stamp = float(np.asarray(sample["camera_receive_timestamp"])[-1])
    age = wall_time() - stamp
    if (
        not math.isfinite(stamp)
        or not math.isfinite(age)
        or age > stale_seconds
        or age < -0.05
    ):
        raise RuntimeError(
            f"{spec['name']}: missing, stale or future camera frame (age {age:g}s)"
        )
    if np.asarray(sample["color"])[-1].shape != (spec["height"], spec["width"], 3):
        raise RuntimeError(
            f"{spec['name']}: received image is not the configured native size"
        )
    return age


def _cycle(
    specs,
    folder,
    options,
    *,
    camera_factory,
    manager_factory,
    monotonic,
    wall_time,
    sleep,
    video_probe,
):
    result = {
        "status": "FAIL",
        "errors": [],
        "cameras": [],
        "cleanup": [],
        "manager_released": False,
        "receive_differences": [],
    }
    cameras, recording, manager = [], set(), None
    folder.mkdir()
    try:
        manager = manager_factory()
        manager.start()
        for spec in specs:
            camera = camera_factory(
                manager,
                **{
                    key: spec[key]
                    for key in (
                        "device",
                        "name",
                        "width",
                        "height",
                        "fps",
                        "latency",
                        "fourcc",
                        "controls",
                        "expected_identity",
                        "calibration",
                        "latency_status",
                    )
                    if key in spec
                },
            )
            cameras.append((camera, spec))
        # start() is nonblocking: all devices run before waiting on any camera.
        for camera, _ in cameras:
            camera.start()
        deadline = monotonic() + options["timeout"]
        while not all(camera.ready_event.is_set() for camera, _ in cameras):
            for camera, spec in cameras:
                if camera.process.exitcode is not None or not camera.process.is_alive():
                    raise RuntimeError(
                        f"{spec['name']}: worker exited before readiness"
                    )
            if monotonic() >= deadline:
                raise TimeoutError("camera readiness timed out")
            sleep(min(options["poll_seconds"], max(0, deadline - monotonic())))
        for camera, spec in cameras:
            _healthy(camera, spec, wall_time, options["stale_seconds"])
            # A lost start acknowledgement may still leave an active writer.
            recording.add(spec["name"])
            camera.start_recording(
                str(folder / f"{spec['name']}.mp4"), timeout=options["timeout"]
            )
        started = monotonic()
        deadline = started + options["duration"]
        max_ages = {spec["name"]: 0.0 for spec in specs}
        while True:
            for camera, spec in cameras:
                age = _healthy(camera, spec, wall_time, options["stale_seconds"])
                max_ages[spec["name"]] = max(max_ages[spec["name"]], age)
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            sleep(min(options["poll_seconds"], remaining))
        result["observed_duration_s"] = monotonic() - started
        result["max_polled_frame_age_s"] = max_ages
    except (Exception, KeyboardInterrupt) as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        for camera, spec in cameras:
            if spec["name"] in recording:
                try:
                    camera.stop_recording(timeout=options["timeout"])
                except (Exception, KeyboardInterrupt) as exc:
                    result["errors"].append(
                        f"{spec['name']}: recording stop failed: {exc}"
                    )
        # Signal every worker before joining, so cleanup cannot hold another
        # stream open for the timeout of each preceding worker.
        for camera, spec in cameras:
            try:
                camera.stop_event.set()
            except (Exception, KeyboardInterrupt) as exc:
                result["errors"].append(f"{spec['name']}: stop signal failed: {exc}")
        for camera, spec in cameras:
            cleanup = {"name": spec["name"]}
            try:
                camera.stop(timeout=options["timeout"])
            except (Exception, KeyboardInterrupt) as exc:
                cleanup["error"] = str(exc)
                result["errors"].append(f"{spec['name']}: worker cleanup failed: {exc}")
            cleanup.update(
                exit_code=camera.process.exitcode, alive=camera.process.is_alive()
            )
            cleanup["clean"] = (
                not cleanup["alive"]
                and cleanup["exit_code"] == 0
                and "error" not in cleanup
            )
            if not cleanup["clean"]:
                result["errors"].append(
                    f"{spec['name']}: clean worker exit was not confirmed"
                )
            result["cleanup"].append(cleanup)
        if manager is not None:
            try:
                manager.shutdown()
                result["manager_released"] = True
            except (Exception, KeyboardInterrupt) as exc:
                result["errors"].append(f"shared memory cleanup failed: {exc}")
    streams = {}
    for spec in specs:
        try:
            metric, received = inspect_recording(
                folder / f"{spec['name']}.mp4",
                spec,
                max_gap_seconds=options["max_gap_seconds"],
                min_fps_ratio=options["min_fps_ratio"],
                video_probe=video_probe,
            )
            minimum_span = max(0.0, options["duration"] - 3.0 / spec["fps"])
            if metric["receive_span_s"] + 1e-9 < minimum_span:
                metric["errors"].append(
                    "recorded receive timestamps do not cover the requested duration"
                )
                metric["status"] = "FAIL"
            result["cameras"].append(metric)
            streams[spec["name"]] = received
            result["errors"].extend(
                f"{spec['name']}: {error}" for error in metric["errors"]
            )
        except Exception as exc:
            result["errors"].append(
                f"{spec['name']}: saved recording validation failed: {exc}"
            )
    if len(streams) == len(specs):
        try:
            result["receive_differences"] = receive_differences(streams)
        except ValueError as exc:
            result["errors"].append(str(exc))
    result["status"] = "FAIL" if result["errors"] else "PASS"
    return result


def run_diagnostic(
    camera_config,
    output,
    *,
    duration=60.0,
    cycles=3,
    timeout=10.0,
    stale_seconds=1.0,
    poll_seconds=0.05,
    max_gap_seconds=None,
    min_fps_ratio=0.9,
    camera_factory=CameraProcess,
    manager_factory=SharedMemoryManager,
    monotonic=time.monotonic,
    wall_time=time.time,
    sleep=time.sleep,
    video_probe=decode_video,
):
    options = {
        key: positive_float(value)
        for key, value in {
            "duration": duration,
            "timeout": timeout,
            "stale_seconds": stale_seconds,
            "poll_seconds": poll_seconds,
            "min_fps_ratio": min_fps_ratio,
        }.items()
    }
    options["cycles"] = positive_int(cycles)
    options["max_gap_seconds"] = (
        None if max_gap_seconds is None else positive_float(max_gap_seconds)
    )
    if options["min_fps_ratio"] > 1:
        raise ValueError("min_fps_ratio must be at most 1")
    specs = load_camera_config(camera_config)
    if any(spec["type"] != "uvc" for spec in specs):
        raise ValueError(
            "camera-only stage 1 accepts UVC cameras only; PICO specs are not supported"
        )
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "camera_config.json").write_text(
        json.dumps(
            {"schema": "camera_rig_v1", "cameras": specs}, indent=2, allow_nan=False
        )
        + "\n"
    )
    report = {
        "schema": "camera_rig_diagnostic_v1",
        "status": "FAIL",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "camera_config_source": str(Path(camera_config).resolve()),
        "options": options,
        "limitation": LIMITATION,
        "cycles": [],
        "errors": [],
    }
    try:
        for index in range(options["cycles"]):
            print(
                f"Camera-only cycle {index + 1}/{options['cycles']}: {options['duration']:g}s",
                flush=True,
            )
            cycle = _cycle(
                specs,
                output / f"cycle_{index + 1:03d}",
                options,
                camera_factory=camera_factory,
                manager_factory=manager_factory,
                monotonic=monotonic,
                wall_time=wall_time,
                sleep=sleep,
                video_probe=video_probe,
            )
            cycle["cycle"] = index + 1
            report["cycles"].append(cycle)
            if cycle["status"] != "PASS":
                report["errors"].extend(cycle["errors"])
                break
        report["completed_cycles"] = len(report["cycles"])
        if report["completed_cycles"] != options["cycles"]:
            report["errors"].append(
                "not all requested reopen/record/close cycles completed"
            )
        report["mean_observed_fps_by_camera"] = {
            spec["name"]: float(np.mean(values))
            for spec in specs
            if (
                values := [
                    camera["observed_mean_fps"]
                    for cycle in report["cycles"]
                    for camera in cycle["cameras"]
                    if camera["name"] == spec["name"]
                ]
            )
        }
        report["status"] = "FAIL" if report["errors"] else "PASS"
    except (Exception, KeyboardInterrupt) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-config", required=True, type=Path)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new directory; existing paths are rejected",
    )
    parser.add_argument(
        "--duration",
        type=positive_float,
        default=60.0,
        help="recording seconds per cycle",
    )
    parser.add_argument("--cycles", type=positive_int, default=3)
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=10.0,
        help="readiness/ack/cleanup timeout seconds",
    )
    parser.add_argument("--stale-seconds", type=positive_float, default=1.0)
    parser.add_argument("--poll-seconds", type=positive_float, default=0.05)
    parser.add_argument(
        "--max-gap-seconds",
        type=positive_float,
        help="default: three requested frame periods",
    )
    parser.add_argument(
        "--min-fps-ratio",
        type=positive_float,
        default=0.9,
        help="minimum observed / requested FPS",
    )
    return parser


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    previous = signal.signal(signal.SIGTERM, _interrupt)
    try:
        report = run_diagnostic(**vars(args))
    except (ValueError, OSError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(f"{report['status']}: {args.output.resolve() / 'report.json'}")
    print(LIMITATION)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
