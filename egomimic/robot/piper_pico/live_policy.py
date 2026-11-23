"""Keyboard-operated policy deployment with optional Pi0.5 RTC, independent of PICO.

Importing this module never connects devices. Camera / CAN construction happens
only after the explicit runtime calibration and training profile have passed
preflight. Without --execute the robot connection is feedback-only.
With --offline-eval, recorded training inputs are scored without opening devices.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import select
import sys
import termios
import threading
import time
import tty
import uuid
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from egomimic.robot.piper_pico.camera_preview import CameraPreview, preview_fps
from egomimic.robot.piper_pico.policy_execution import (
    SafetyLimits,
    resample_action_block,
)
from egomimic.robot.piper_pico.policy_inference import PiperPolicy
from egomimic.robot.piper_pico.policy_logging import AsyncLogWriter
from egomimic.robot.piper_pico.policy_rollout import run_sync_cycle


def _positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _boolean(value):
    if str(value).lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("use true or false")
    return str(value).lower() == "true"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-backend", choices=("hpt", "pi05"), default="hpt")
    parser.add_argument(
        "--checkpoint", type=Path, help="Required for the local HPT backend"
    )
    parser.add_argument(
        "--server-url", help="Pi0.5 WebSocket URL, e.g. ws://127.0.0.1:8000"
    )
    parser.add_argument("--server-timeout", type=_positive_float, default=10.0)
    parser.add_argument(
        "--task-description",
        help="Pi0.5 prompt; defaults to nonempty training metadata",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Raw intrinsics + per-arm T_base_camera JSON used for conversion",
    )
    parser.add_argument(
        "--training-data",
        type=Path,
        required=True,
        help="Converted Piper episode or directory for this rig/profile",
    )
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument(
        "--device", default=None, help="Default cuda:0 live, cpu with --offline-eval"
    )
    parser.add_argument(
        "--offline-eval",
        action="store_true",
        help="Evaluate recorded training images/state; never open hardware",
    )
    parser.add_argument("--eval-samples-per-episode", type=_positive_int, default=8)
    parser.add_argument(
        "--eval-max-episodes", type=_positive_int, help="Default all selected episodes"
    )
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--eval-parity-samples", type=_positive_int, default=4)
    parser.add_argument("--eval-plots", type=_positive_int, default=6)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Enable physical commands; otherwise robot feedback only",
    )
    parser.add_argument(
        "--home-on-start",
        type=_boolean,
        default=False,
        help="true/false; true requires --execute and also opens grippers",
    )
    parser.add_argument("--left-can", default="can0")
    parser.add_argument("--right-can", default="can1")
    parser.add_argument(
        "--move-spd-rate", type=int, choices=range(1, 101), default=50, metavar="1..100"
    )
    parser.add_argument("--control-frequency", type=_positive_float, default=30.0)
    parser.add_argument(
        "--execute-steps",
        type=_positive_int,
        default=30,
        help="Prefix length AFTER time-preserving resampling; with --rtc, request period = steps/control_frequency from the previous request; with --async-inference, sends before requesting the next chunk",
    )
    parser.add_argument(
        "--rtc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pi0.5 RTC: asynchronous inference with model prefix guidance; requires server --rtc and a shorter --execute-steps (e.g. 12)",
    )
    parser.add_argument(
        "--rtc-delay-ms",
        type=_positive_float,
        default=200.0,
        help="RTC observation/inference deadline; takeover uses the next control tick with guarded prefix coverage; late results pause execution (default 200 ms)",
    )
    parser.add_argument(
        "--async-inference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Infer the next chunk while the current one runs, time-aligned by the training observation_to_send_s, then crossfade",
    )
    parser.add_argument(
        "--rtc-transition-ms",
        type=_nonnegative_float,
        default=200.0,
        help="RTC execution pose transition duration (quintic, default 200 ms; 0 disables)",
    )
    parser.add_argument(
        "--rtc-smoothing-ms",
        type=_nonnegative_float,
        default=100.0,
        help="RTC centred within-prediction pose smoothing window (default about 100 ms; 0 disables)",
    )
    parser.add_argument(
        "--rtc-max-position-gap-mm",
        type=_positive_float,
        default=20.0,
        help="Reject RTC takeover when old/new trajectories disagree beyond this distance in the transition window",
    )
    parser.add_argument(
        "--rtc-max-rotation-gap-deg",
        type=_positive_float,
        default=5.0,
        help="Reject RTC takeover when SO(3) disagreement exceeds this in the transition window",
    )
    parser.add_argument(
        "--blend-steps",
        type=int,
        default=5,
        help="With --async-inference: control intervals to crossfade old -> new chunk (0 = hard switch)",
    )
    parser.add_argument(
        "--inference-steps",
        type=_positive_int,
        default=10,
        help="Local HPT generation steps; Pi0.5 uses the server's configured steps",
    )
    parser.add_argument(
        "--max-chunks",
        type=_positive_int,
        help="Optional prediction count limit; default runs until quit",
    )
    parser.add_argument("--max-observation-age", type=_positive_float, default=0.5)
    parser.add_argument("--home-timeout", type=_positive_float, default=20.0)
    parser.add_argument(
        "--preview", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--preview-fps", type=preview_fps, default=15.0)
    parser.add_argument(
        "--log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Live JSONL logs in a background process (default on); --no-log keeps only configuration/summary. Offline evaluation reports are unaffected.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New directory; default diagnostics/piper_live_<UTC>_<id> or piper_offline_eval_<UTC>_<id>",
    )
    return parser


class KeyboardControl:
    """Terminal keys only; the independent image viewer does not control motion."""

    def __init__(self, stream=None):
        self.stream = sys.stdin if stream is None else stream
        self.state = "pause"
        self.fd = None
        self.old = None

    def __enter__(self):
        if not self.stream.isatty():
            raise RuntimeError("请在交互终端运行；空格开始/暂停，Q/Esc 退出。")
        self.fd = self.stream.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self.fd is not None and self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def feed(self, key):
        if self.state == "stop":
            return
        if key.lower() == "q" or key in ("\x1b", "\x03", ""):
            self.state = "stop"
        elif key == " ":
            self.state = "pause" if self.state == "run" else "run"
        elif key.lower() == "p":
            self.state = "pause"

    def poll(self):
        if self.fd is not None:
            # os.read avoids TextIO buffering hiding a second queued key from select.
            while select.select([self.fd], [], [], 0)[0]:
                value = os.read(self.fd, 1)
                self.feed(value.decode("ascii", errors="ignore") if value else "")
        return self.state

    def pause(self):
        if self.state != "stop":
            self.state = "pause"


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot log {type(value).__name__}")


def _json(value, *, indent=None):
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, default=_json_default, indent=indent
    )


class SessionLog:
    """Small counters locally; optional JSONL encoding/writes in a child process."""

    def __init__(self, directory, args):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.enabled = bool(getattr(args, "log", True))
        self._closed = False
        self._warnings = set()
        # The streaming loop's inference thread may publish gripper events.
        self._publish_lock = threading.Lock()
        self.summary = {
            "status": "initializing",
            "execute": bool(args.execute),
            "started_walltime": time.time(),
            "prediction_count": 0,
            "action_count": 0,
            "schedule_pause_count": 0,
            "rtc_pause_count": 0,
            "rtc_pause_reasons": {},
            "inference_elapsed_total_s": 0.0,
            "max_inference_elapsed_s": 0.0,
            "max_action_interval_s": None,
            "action_interval_mean_s": None,
            "action_semantics": "SDK submission acknowledgements"
            if args.execute
            else "dry-run candidates; no robot commands",
        }
        self.first_send = self.last_send = None
        # Metadata is written at startup/shutdown, outside the action loop.
        (self.directory / "arguments.json").write_text(_json(vars(args), indent=2))
        self._writer = AsyncLogWriter(self.directory) if self.enabled else None
        self.summary["logging"] = (
            self._writer.stats
            if self._writer is not None
            else {"enabled": False, "mode": "disabled", "incomplete": False}
        )

    def _publish(self, kind, value):
        if self._writer is not None:
            # Queue transport is asynchronous: never lend mutable predictions,
            # receipts or executor reports to its feeder thread.
            value = copy.deepcopy(value)
            with self._publish_lock:
                self._writer.submit(kind, value)

    def event(self, name, **fields):
        if self.enabled:
            self._publish("events", {"event": name, "walltime": time.time(), **fields})

    def write_manifest(self, value):
        (self.directory / "manifest.json").write_text(_json(value, indent=2))

    def command(
        self,
        row,
        deadline,
        started,
        finished,
        receipt=None,
        *,
        health_check_elapsed_s=None,
    ):
        if self.last_send is not None:
            gap = started - self.last_send
            old = self.summary["max_action_interval_s"]
            self.summary["max_action_interval_s"] = (
                gap if old is None else max(old, gap)
            )
        else:
            self.first_send = started
        self.last_send = started
        self.summary["action_count"] += 1
        count = self.summary["action_count"]
        if count > 1:
            self.summary["action_interval_mean_s"] = (started - self.first_send) / (
                count - 1
            )
        if self.enabled:
            self._publish(
                "commands",
                {
                    "index": count - 1,
                    "action_base": row,
                    "deadline_walltime": deadline,
                    "submit_monotonic": started,
                    "returned_monotonic": finished,
                    "send_elapsed_s": finished - started,
                    "health_check_elapsed_s": health_check_elapsed_s,
                    "execute": self.summary["execute"],
                    "device_receipt": receipt,
                },
            )

    def cycle(self, report):
        if report.get("timing_mode") == "rtc" and report.get("status") == "paused":
            self.summary["rtc_pause_count"] += 1
            reason = report.get("pause_reason", "unknown")
            reasons = self.summary["rtc_pause_reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1
        if report.get("pause_reason") == "schedule_slip":
            self.summary["schedule_pause_count"] += 1
        if "observation_timestamp" in report:
            self.summary["prediction_count"] += 1
        elapsed = report.get("measured_inference_elapsed_s", 0.0)
        self.summary["inference_elapsed_total_s"] += elapsed
        self.summary["max_inference_elapsed_s"] = max(
            elapsed, self.summary["max_inference_elapsed_s"]
        )
        if self.enabled:
            self._publish("cycles", report)

    def check(self):
        """Poll/report logger failures only between chunks or while paused."""
        if self._writer is None:
            return
        self._writer.poll()
        stats = self._writer.stats
        reason = "error" if stats["errors"] else "dropped"
        if stats["incomplete"] and reason not in self._warnings:
            self._warnings.add(reason)
            detail = stats["errors"][-1] if stats["errors"] else "后台日志队列已满"
            print(
                f"[推理] 日志不完整：{detail}；详情见结束摘要。动作循环不等待日志写入。",
                file=sys.stderr,
                flush=True,
            )

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._writer is not None:
            self._writer.close()
            self.check()
        self.summary["ended_walltime"] = time.time()
        count = self.summary["prediction_count"]
        self.summary["mean_inference_elapsed_s"] = (
            self.summary["inference_elapsed_total_s"] / count if count else None
        )
        (self.directory / "summary.json").write_text(_json(self.summary, indent=2))


class LoggedSink:
    """Check live device health before every command, then log the acknowledged send."""

    def __init__(self, sink, provider, log, monotonic=time.monotonic):
        self.sink, self.provider, self.log = sink, provider, log
        self.monotonic = monotonic

    def send_action(self, row, deadline):
        health_started = self.monotonic()
        self.provider.check_health()
        started = self.monotonic()
        receipt = self.sink.send_action(row, deadline)
        finished = self.monotonic()
        self.log.command(
            row,
            deadline,
            started,
            finished,
            receipt,
            health_check_elapsed_s=started - health_started,
        )
        # RTC measures cadence after variable-cost health checks. This is a
        # host submission time, not proof that the robot reached its target.
        return {
            "send_started_monotonic": started,
            "send_finished_monotonic": finished,
        }

    def hold(self, reason):
        try:
            self.log.event("hold_requested", reason=reason)
        finally:
            # A full disk or logging failure must never suppress physical hold.
            self.sink.hold(reason)
        self.log.event("hold_completed", reason=reason)


class ActuatorPolicy:
    """Bound gripper actuator targets; retain raw model predictions in the audit."""

    def __init__(self, policy, open_m, log):
        self.policy, self.timing, self.log = policy, policy.timing, log
        self.open_m = float(open_m)
        if not math.isfinite(self.open_m) or self.open_m <= 0:
            raise ValueError("gripper opening must be finite and positive")

    def predict(self, frame, *, rtc=None):
        prediction = (
            self.policy.predict(frame)
            if rtc is None
            else self.policy.predict(frame, rtc=rtc)
        )
        actions = np.asarray(prediction["base_actions"], dtype=float).copy()
        if (
            actions.shape != (self.timing.output_chunk_length, 14)
            or not np.isfinite(actions).all()
        ):
            raise ValueError(
                "Model actions must be finite with checkpoint action shape"
            )
        raw = actions[:, [6, 13]].copy()
        clipped = np.clip(raw, 0.0, self.open_m)
        actions[:, [6, 13]] = clipped
        if not np.array_equal(raw, clipped):
            self.log.event(
                "gripper_clamped",
                observation_timestamp=frame.timestamp,
                raw_widths_m=raw,
                raw_min_m=raw.min(axis=0),
                raw_max_m=raw.max(axis=0),
                clipped_count_per_arm=np.count_nonzero(raw != clipped, axis=0),
                command_range_m=[0.0, self.open_m],
            )
        # camera_actions remains the unmodified prediction; base actions and
        # the execution schedule carry the actual bounded gripper commands.
        return {**prediction, "base_actions": actions}


def run_live_loop(
    policy,
    provider,
    sink,
    keys,
    limits,
    log,
    args,
    *,
    monotonic=time.monotonic,
    wall_time=time.time,
    sleep=time.sleep,
):
    """Synchronous cycles with no latency compensation and no chunk-boundary hold."""
    last_stamp, next_send = None, None
    used = 0
    previous_state = None
    last_cycle_held = False
    next_health = 0.0
    while True:
        log.check()
        state = keys.poll()
        if state != previous_state:
            if state == "pause" and previous_state == "run" and not last_cycle_held:
                # A pause can arrive AFTER a completed block's last callback.
                sink.hold("paused")
            log.event("control", state=state)
            print(
                {
                    "pause": "[推理] 已暂停；终端空格开始，Q 退出。",
                    "run": "[推理] 运行中；终端空格/P 暂停，Q 退出。",
                    "stop": "[推理] 正在退出。",
                }[state],
                flush=True,
            )
            previous_state = state
        if state == "stop":
            sink.hold("stopped")
            return "stopped"
        if state == "pause":
            # No repeated physical hold commands while the user is idle.
            if monotonic() >= next_health:
                provider.check_health()
                next_health = monotonic() + 0.1
            sleep(0.01)
            continue
        report = run_sync_cycle(
            policy,
            provider,
            sink,
            limits,
            control_frequency=args.control_frequency,
            execute_steps=args.execute_steps,
            hold_on_completion=False,
            # Deliberately omit observation_to_send_s. First version follows
            # upstream synchronous timing, from chunk offset zero after inference.
            control_callback=keys.poll,
            newer_than=last_stamp,
            send_not_before_monotonic=next_send,
            monotonic=monotonic,
            wall_time=wall_time,
            sleep=sleep,
        )
        execution = report.get("execution", {})
        # Recover only an explicit scheduling slip AFTER a successful hold.
        # Clock faults, bad feedback, sink errors and failed holds stay fatal.
        error = report.get("error") or {}
        schedule_pause = (
            report["status"] == "failed"
            and error.get("type") == "ScheduleOverrun"
            and execution.get("schedule_overrun", {}).get("kind") == "schedule_slip"
            and execution.get("hold_succeeded") is True
            and "hold_error" not in execution
            and "hold_error" not in report
        )
        if schedule_pause:
            keys.pause()
            report.update(
                status="paused",
                pause_reason="schedule_slip",
                discarded_remaining_actions=True,
            )
            log.event(
                "schedule_paused",
                details=execution["schedule_overrun"],
                last_attempt=execution.get("last_attempt"),
                resume_requires="terminal_space_new_observation",
            )
            print(
                "[推理] 动作调度超时，双臂已保持，剩余动作已丢弃。终端空格重新观察并推理，Q 退出。",
                flush=True,
            )
        log.cycle(report)
        last_cycle_held = bool(report["hold_attempted"])
        samples = execution.get("samples", [])
        if samples:
            next_send = (
                samples[-1]["actual_monotonic"]
                + 1 / execution["actual_control_frequency"]
            )
        if schedule_pause:
            next_send = None
        if "observation_timestamp" in report:
            last_stamp = report["observation_timestamp"]
            used += 1
        status = report["status"]
        elapsed = report.get("measured_inference_elapsed_s", 0.0)
        print(
            f"[推理] chunk {used}: {status}，预测 {elapsed * 1000:.1f} ms，"
            f"动作 {execution.get('sent_count', 0)} 点",
            flush=True,
        )
        if status == "failed":
            error = report.get("error") or execution.get("error") or {}
            raise RuntimeError(
                error.get("message", "Policy cycle failed; see cycles.jsonl")
            )
        if status in ("stopped", "exhausted"):
            return status
        if args.max_chunks is not None and used >= args.max_chunks:
            sink.hold("completed")
            return "completed"


def _fingerprint(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy_contract(args, log):
    from egomimic.robot.piper_pico.live_policy_contract import prepare_live_contract

    if args.home_on_start and not args.execute:
        raise ValueError("--home-on-start true 需要 --execute；只观察模式不会回 home。")
    backend = getattr(args, "policy_backend", "hpt")
    if getattr(args, "rtc", False):
        if backend != "pi05":
            raise ValueError("--rtc 当前只支持 Pi0.5 JAX 后端。")
        if getattr(args, "async_inference", False):
            raise ValueError(
                "--rtc 与 --async-inference 不能同时使用；RTC 已包含异步执行。"
            )
    if backend == "hpt" and getattr(args, "checkpoint", None) is None:
        raise ValueError("HPT 后端需要 --checkpoint。")
    if backend == "pi05" and not getattr(args, "server_url", None):
        raise ValueError("Pi0.5 后端需要 --server-url ws://服务器地址:端口。")
    if backend == "pi05" and getattr(args, "offline_eval", False):
        raise ValueError("--offline-eval 是本地 HPT 评估入口，不适用于 Pi0.5 RPC。")
    if not args.left_can or not args.right_can or args.left_can == args.right_can:
        raise ValueError("左右机械臂必须使用两个不同的 CAN 接口。")
    contract = prepare_live_contract(
        args.training_data, args.calibration, args.camera_config
    )
    if any(spec.get("type", "uvc") != "uvc" for spec in contract["camera_specs"]):
        raise ValueError(
            "该入口使用三个 UVC 相机，不启动 PICO；请检查 --camera-config。"
        )
    print("[推理] 标定及训练图像配置核对通过，正在连接策略。", flush=True)
    if backend == "pi05":
        from egomimic.robot.piper_pico.pi05_remote import Pi05RemotePolicy

        policy = Pi05RemotePolicy.connect(
            args.server_url,
            contract["metadata"],
            timeout_s=args.server_timeout,
            task_description=args.task_description,
        )
        source = {
            "policy_backend": "pi05",
            "checkpoint": None,
            "checkpoint_sha256": None,
            "server_url": args.server_url,
            "server_metadata": policy.server_metadata,
            "task_description": policy.prompt,
        }
    else:
        checkpoint = args.checkpoint.expanduser().resolve(strict=True)
        signature = _fingerprint(checkpoint)
        sha = _sha256(checkpoint)
        policy = PiperPolicy.from_checkpoint(
            checkpoint,
            contract["metadata"],
            device=args.device,
            inference_steps=args.inference_steps,
        )
        if signature != _fingerprint(checkpoint):
            raise RuntimeError(
                "读取期间 checkpoint 改变了；请指定已保存的 epoch 文件。"
            )
        source = {
            "policy_backend": "hpt",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha,
        }
    try:
        return _finish_policy_contract(args, log, contract, policy, source)
    except BaseException:
        if hasattr(policy, "close"):
            policy.close()
        raise


def _finish_policy_contract(args, log, contract, policy, source):
    if not math.isclose(
        policy.timing.source_fps,
        contract["metadata"]["fps"],
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("checkpoint source_fps 与所选训练数据 fps 不一致。")
    from egomimic.robot.piper_pico.units import GRIPPER_MAX_M

    open_m = float(policy.adapter.gripper_open_m)
    if open_m > GRIPPER_MAX_M + 1e-9:
        raise ValueError("训练夹爪开度超过当前 PiPER 支持的物理范围。")
    limits = SafetyLimits(
        workspace_min=None,
        workspace_max=None,
        max_translation_step_m=None,
        max_rotation_step_rad=None,
        gripper_max_m=open_m,
        max_observation_age_s=args.max_observation_age,
    )
    schedule = resample_action_block(
        np.zeros((policy.timing.output_chunk_length, 14)),
        policy.timing,
        args.control_frequency,
        execute_steps=args.execute_steps,
    ).to_report()
    streaming = None
    rtc = None
    if getattr(args, "rtc", False):
        from egomimic.robot.piper_pico.policy_rtc import check_rtc_schedule

        capability = policy.require_rtc()
        rtc = {
            **check_rtc_schedule(
                policy.timing,
                args.control_frequency,
                args.execute_steps,
                args.rtc_delay_ms / 1000,
                transition_s=args.rtc_transition_ms / 1000,
            ),
            "server": capability,
            "observation_to_send_s": policy.observation_to_send_s,
            "smoothing_s": args.rtc_smoothing_ms / 1000,
            "max_position_gap_m": args.rtc_max_position_gap_mm / 1000,
            "max_rotation_gap_rad": np.deg2rad(args.rtc_max_rotation_gap_deg),
        }
        if args.rtc_smoothing_ms / 1000 >= policy.timing.effective_span_s:
            raise ValueError(
                "RTC smoothing window must be shorter than the prediction horizon"
            )
    if getattr(args, "async_inference", False):
        from egomimic.robot.piper_pico.policy_streaming import (
            check_streaming_schedule,
        )

        if args.blend_steps < 0:
            raise ValueError("--blend-steps 不能为负数。")
        streaming = {
            **check_streaming_schedule(
                policy.timing,
                args.control_frequency,
                args.execute_steps,
                args.blend_steps,
            ),
            "observation_to_send_s": contract["metadata"]["time_alignment"][
                "observation_to_send_s"
            ],
        }
    manifest = {
        "schema": "piper_live_policy_v1",
        **source,
        "contract": contract["audit"],
        "runtime_metadata": contract["metadata"],
        "camera_specs": contract["camera_specs"],
        "action_timing": policy.timing.to_dict(),
        "schedule": schedule,
        "coordinate_transform": "JSON T_base_camera; inverse for observations, forward for actions",
        "timing_mode": "rtc; model prefix guidance on timestamp-aligned 100-point grid; fixed inference deadline and guarded control-tick takeover; request-start cadence; configured execution pose conditioning"
        if rtc
        else "streaming; chunk offset s at observation + observation_to_send_s + s; crossfade between chunks"
        if streaming
        else "synchronous; offset zero after inference; no latency compensation",
        "streaming": streaming,
        "rtc": rtc,
        "workspace_limits": None,
        "step_limits": None,
        "gripper_open_m": open_m,
        "gripper_commands": "clip to [0, open_m]; raw predictions retained when logging enabled; pose unchanged",
        "checkpoint_geometry_bound": source["policy_backend"] == "pi05",
        "execute": args.execute,
        "home_on_start": args.home_on_start,
    }
    log.write_manifest(manifest)
    if not rtc:
        print(
            f"[推理] {policy.timing.output_chunk_length} → {schedule['full_point_count']} 点，"
            f"首尾 {policy.timing.effective_span_s:.3f} s；每轮执行前 "
            f"{schedule['selected_point_count']} 点，目标 {schedule['actual_control_frequency']:.2f} Hz。",
            flush=True,
        )
    if rtc:
        print(
            f"[推理] RTC：从上次请求起每 {rtc['replan_interval_s'] * 1000:g} ms 重新观察，"
            f"推理期限 {args.rtc_delay_ms:g} ms，接管对齐控制节拍；"
            f"前缀保护额外 {rtc['takeover_guard_s'] * 1000:.1f} ms 并覆盖右侧插值点。"
            f"预测保留 {policy.timing.output_chunk_length} 点 / {policy.timing.effective_span_s:.3f} s，"
            f"目标 {args.control_frequency:g} Hz；超时或错过前缀保护范围会保持并暂停。",
            flush=True,
        )
        print(
            f"[推理] RTC 执行连续性：位姿过渡 {args.rtc_transition_ms:g} ms，"
            f"段内平滑约 {args.rtc_smoothing_ms:g} ms；夹爪保持原时机。"
            f"过渡窗口预测分歧超过 {args.rtc_max_position_gap_mm:g} mm / "
            f"{args.rtc_max_rotation_gap_deg:g}° 会保持并暂停。",
            flush=True,
        )
    if streaming:
        print(
            f"[推理] 流水线推理：每 {args.execute_steps} 点取新观测，"
            f"交叉过渡 {args.blend_steps} 点，按训练 observation_to_send "
            f"{streaming['observation_to_send_s'] * 1000:.1f} ms 对齐；"
            f"推理延迟上限 {streaming['max_inference_latency_s'] * 1000:.0f} ms。",
            flush=True,
        )
    extrinsics = contract["metadata"]["extrinsics"]
    identity = all(
        np.allclose(extrinsics[side], np.eye(4), atol=1e-10, rtol=0)
        for side in ("left", "right")
    )
    print(
        "[推理] 使用 JSON 中的单位外参，保持当前训练的各臂 base 数值约定。"
        if identity
        else "[推理] 使用 JSON 中的相机与双臂 base 外参。",
        flush=True,
    )
    return contract, policy, limits


def run_session(args, log):
    # Reject an unattended terminal before initializing any device.
    if not sys.stdin.isatty():
        raise RuntimeError("请在交互终端运行，使用空格开始/暂停、Q/Esc 退出。")
    contract, policy, limits = load_policy_contract(args, log)
    try:
        return _run_device_session(args, log, contract, policy, limits)
    finally:
        if hasattr(policy, "close"):
            policy.close()


def _run_device_session(args, log, contract, policy, limits):
    history_offsets = tuple(getattr(policy, "state_history_offsets_s", ()))
    policy = ActuatorPolicy(policy, policy.adapter.gripper_open_m, log)
    import multiprocessing as mp
    from multiprocessing.managers import SharedMemoryManager

    from egomimic.robot.piper_pico.env import PiperPicoEnv
    from egomimic.robot.piper_pico.live_policy_devices import (
        LiveActionSink,
        LiveObservationProvider,
    )

    with ExitStack() as stack:
        keys = stack.enter_context(KeyboardControl())
        # Model loading may initialize CUDA and thread pools. No device or
        # manager process should inherit that state through fork.
        shm = stack.enter_context(SharedMemoryManager(ctx=mp.get_context("spawn")))
        env = PiperPicoEnv(
            shm,
            log.directory / "devices",
            cameras=contract["camera_specs"],
            left_can=args.left_can,
            right_can=args.right_can,
            frequency=args.control_frequency,
            horizon=2,
            move_spd_rate=args.move_spd_rate,
            enable_pico=False,
            enable_motion=args.execute,
            preserve_gripper_on_enable=True,
            **({"state_history_offsets_s": history_offsets} if history_offsets else {}),
        )
        stack.callback(env.stop)
        env.start()
        provider = LiveObservationProvider(
            env, max_source_age_s=args.max_observation_age
        )
        hardware_sink = LiveActionSink(
            env, execute=args.execute, max_source_age_s=args.max_observation_age
        )
        sink = LoggedSink(hardware_sink, provider, log)
        stack.enter_context(
            CameraPreview(env.cameras, args.preview_fps)
            if args.preview
            else nullcontext()
        )
        log.event("devices_ready", execute=args.execute)
        try:
            if history_offsets:
                from egomimic.robot.piper_pico.state_history import HistoryCoverageError

                print("[推理] 等待反馈共享内存覆盖模型所需历史。", flush=True)
                deadline = time.monotonic() - history_offsets[0] + 3.0
                while True:
                    if keys.poll() == "stop":
                        return "stopped"
                    try:
                        provider.latest()
                        break
                    except HistoryCoverageError:
                        if time.monotonic() >= deadline:
                            raise RuntimeError(
                                "反馈历史覆盖不足，无法开始历史模型推理"
                            ) from None
                        time.sleep(0.01)
            print("[推理] 模型预热中，本次预测不执行。", flush=True)
            prediction = policy.predict(provider.latest())
            log.event("warmup", inference_elapsed_s=prediction["inference_elapsed_s"])
            if keys.poll() == "stop":
                return "stopped"
            if args.home_on_start:
                print("[推理] 双臂回 home 并打开夹爪，正在等待反馈到位。", flush=True)
                log.event("home_started")
                hardware_sink.home(
                    timeout_s=args.home_timeout,
                    should_stop=lambda: keys.poll() == "stop",
                )
                log.event("home_completed")
            # Keys entered during warmup/home cannot start a policy accidentally.
            keys.poll()
            keys.pause()
            provider.check_health()
            print(
                "[推理] 实际执行模式。"
                if args.execute
                else "[推理] 只观察模式：不使能、不回 home、不发送机械臂目标。",
                flush=True,
            )
            print("[推理] 按键请在终端操作；预览窗口 Q 只关闭窗口。", flush=True)
            if getattr(args, "rtc", False):
                from egomimic.robot.piper_pico.policy_rtc import run_rtc_loop

                return run_rtc_loop(
                    policy,
                    provider,
                    sink,
                    keys,
                    limits,
                    log,
                    args,
                    observation_to_send_s=contract["metadata"]["time_alignment"][
                        "observation_to_send_s"
                    ],
                )
            if getattr(args, "async_inference", False):
                from egomimic.robot.piper_pico.policy_streaming import (
                    run_streaming_loop,
                )

                return run_streaming_loop(
                    policy,
                    provider,
                    sink,
                    keys,
                    limits,
                    log,
                    args,
                    observation_to_send_s=contract["metadata"]["time_alignment"][
                        "observation_to_send_s"
                    ],
                )
            return run_live_loop(policy, provider, sink, keys, limits, log, args)
        except InterruptedError:
            if keys.poll() == "stop":
                log.event("home_cancelled")
                return "stopped"
            raise
        finally:
            # Cleanup also covers warmup/home errors and KeyboardInterrupt.
            active_error = sys.exc_info()[0] is not None
            try:
                sink.hold("session_exit")
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
                log.summary.setdefault("cleanup_errors", []).append(error)
                log.event("cleanup_failed", **error)
                if not active_error:
                    raise


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.policy_backend == "pi05" and args.offline_eval:
        raise SystemExit("--offline-eval 只支持本地 HPT；Pi0.5 请使用独立评估流程。")
    if args.device is None:
        args.device = "cpu" if args.offline_eval else "cuda:0"
    if args.offline_eval:
        from egomimic.robot.piper_pico.offline_policy_eval import run_evaluation

        return run_evaluation(args)
    output = args.output_dir or (
        Path(__file__).resolve().parents[3]
        / "diagnostics"
        / f"piper_live_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    )
    log = SessionLog(output, args)
    print(
        f"[推理] {'后台日志已开启' if log.enabled else '运行日志已关闭，仅保留配置和结束摘要'}："
        f"{log.directory.resolve()}",
        flush=True,
    )
    code = 0
    try:
        log.summary["status"] = run_session(args, log)
    except KeyboardInterrupt:
        log.summary["status"] = "interrupted"
    except Exception as exc:
        code = 1
        log.summary.update(
            status="failed", error={"type": type(exc).__name__, "message": str(exc)}
        )
        print(f"[推理] 已停止：{exc}", file=sys.stderr, flush=True)
    finally:
        log.close()
    return code
