"""Separate camera publication, control-loop sampling, and pose-source health.

Two asynchronous 30 Hz clocks can repeat and skip the latest image while the
producer still publishes 30 fps. Producer rates use ring counters and monotonic
time. The two eye buffers share a decode timestamp, not exposure timestamps.
"""
from __future__ import annotations

import time
import numpy as np
from egomimic.robot.piper_pico.clutch import looks_like_tracking_origin, xyzw_to_wxyz

CONTROL_GAP_THRESHOLD_S = 0.1


def observation_k(horizon: int, frequency: float) -> int:
    """Match ``PiperPicoEnv.get_obs``: how many frames one tick copies."""
    return int(np.ceil(max(horizon * (60.0 / frequency) + 2.0, horizon)))


def _counter_hz(samples: list[tuple[float, int]], window_s: float | None = None) -> float:
    if len(samples) < 2:
        return float("nan")
    first = 0
    if window_s is not None:
        cutoff = samples[-1][0] - window_s
        # Keep the sample straddling the window boundary.
        while first + 1 < len(samples) - 1 and samples[first + 1][0] <= cutoff:
            first += 1
    elapsed = samples[-1][0] - samples[first][0]
    return (samples[-1][1] - samples[first][1]) / elapsed if elapsed > 0 else float("nan")


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else float("nan")


def _number(value: float) -> str:
    return f"{value:.1f}" if np.isfinite(value) else "未知"


class RateMonitor:
    def __init__(self, control_hz: float = 30.0, camera_fps: float = 30.0,
                 camera_enabled: bool = True, window_s: float = 5.0,
                 camera_mode: str = "stereo"):
        if camera_mode not in ("stereo", "left", "right"):
            raise ValueError("camera_mode must be 'stereo', 'left', or 'right'")
        self.control_hz = float(control_hz)
        self.camera_fps = float(camera_fps)
        self.camera_enabled = bool(camera_enabled)
        self.camera_mode = camera_mode
        self.enabled_eyes = ("left", "right") if camera_mode == "stereo" else (camera_mode,)
        self._camera_label = {"stereo": "左/右", "left": "左眼", "right": "右眼"}[camera_mode]
        self.window_s = float(window_s)
        self.periods: list[float] = []
        self.ticks: list[tuple[float, int]] = []
        self.copy_ms: list[float] = []
        self.producer = {"left": [], "right": []}
        self.consumer = {"left": [], "right": []}
        self._last_camera_stamp = {"left": None, "right": None}
        self._new_frames = {"left": 0, "right": 0}
        self.read_phase_skew_ms: list[float] = []
        self.camera_max_age_s = {"left": 0.0, "right": 0.0}
        self.camera_misses = 0
        self.camera_diagnostics_missing = 0
        self.origin_ticks = {"left": 0, "right": 0, "both": 0}
        self.pose_diagnostics_missing = 0
        self.source_diagnostics_missing = 0
        self.source_stall_count = 0
        self.observed_source_stalls = 0
        self.source_max_gap_s = 0.0
        self.source_regressions = 0
        self._source_stamp: int | None = None
        self._source_changed_at: float | None = None
        self._source_stalled = False
        self._prev_mono: float | None = None
        self._last_print = time.monotonic()

    def observe(self, sample: dict, diagnostics: dict | None = None, *,
                now_mono: float | None = None) -> None:
        now = time.monotonic() if now_mono is None else float(now_mono)
        if self._prev_mono is not None:
            self.periods.append(now - self._prev_mono)
        self._prev_mono = now
        self.ticks.append((now, len(self.ticks)))
        self._observe_source(sample, now)
        lost = {}
        for side in ("left", "right"):
            if f"{side}_pos" not in sample or f"{side}_quat_xyzw" not in sample:
                self.pose_diagnostics_missing += 1
                lost[side] = False
            else:
                lost[side] = looks_like_tracking_origin(
                    sample[f"{side}_pos"], xyzw_to_wxyz(sample[f"{side}_quat_xyzw"])
                )
            self.origin_ticks[side] += int(lost[side])
        self.origin_ticks["both"] += int(lost["left"] and lost["right"])
        if self.camera_enabled:
            self._observe_camera(diagnostics or {}, now)
        if now - self._last_print >= 2.0:
            self._last_print = now
            print(self.line(), flush=True)

    def _observe_source(self, sample: dict, now: float) -> None:
        stamp = int(sample.get("source_timestamp_ns", 0))
        age = float(sample.get("source_age_s", float("nan")))
        maximum = float(sample.get("source_max_gap_s", float("nan")))
        stalls = sample.get("source_stall_count")
        if stamp <= 0 or not np.isfinite(age) or not np.isfinite(maximum) or stalls is None:
            self.source_diagnostics_missing += 1
            return
        self.source_stall_count = max(self.source_stall_count, int(stalls))
        if stamp != self._source_stamp:
            if self._source_stamp is not None and stamp < self._source_stamp:
                self.source_regressions += 1
            self._source_stamp = stamp
            self._source_changed_at = now
            self._source_stalled = False
        # A dead producer can leave its last shared sample readable forever.
        # receive_timestamp is the SDK poll clock, not proof of fresh poses.
        observed_age = max(age, now - self._source_changed_at)
        self.source_max_gap_s = max(self.source_max_gap_s, maximum, observed_age)
        if observed_age >= 0.1 and not self._source_stalled:
            self.observed_source_stalls += 1
            self._source_stalled = True

    def _observe_camera(self, data: dict, now: float) -> None:
        copy_s = float(data.get("camera_copy_s", float("nan")))
        if np.isfinite(copy_s):
            self.copy_ms.append(copy_s * 1000.0)
        else:
            self.camera_diagnostics_missing += 1
        stamps = []
        for side in self.enabled_eyes:
            count = data.get(f"camera_{side}_count")
            sampled = float(data.get(f"camera_{side}_count_mono", now))
            if count is None or not np.isfinite(sampled):
                self.camera_diagnostics_missing += 1
            else:
                self.producer[side].append((sampled, int(count)))
            stamp = float(data.get(f"camera_{side}_receive_timestamp", float("nan")))
            age = float(data.get(f"camera_{side}_age_s", float("nan")))
            if not np.isfinite(stamp) or not np.isfinite(age):
                self.camera_misses += 1
            else:
                stamps.append(stamp)
                self.camera_max_age_s[side] = max(self.camera_max_age_s[side], age)
                if stamp != self._last_camera_stamp[side]:
                    self._new_frames[side] += 1
                    self._last_camera_stamp[side] = stamp
            self.consumer[side].append((now, self._new_frames[side]))
        if len(stamps) == 2:
            self.read_phase_skew_ms.append(abs(stamps[0] - stamps[1]) * 1000.0)

    def camera_hz(self, side: str, *, recent: bool = False, consumer: bool = False) -> float:
        samples = self.consumer[side] if consumer else self.producer[side]
        return _counter_hz(samples, self.window_s if recent else None)

    def line(self) -> str:
        loop = _number(_counter_hz(self.ticks, self.window_s))
        parts = [f"近 {self.window_s:g}s 动作 {loop} Hz"]
        if self.camera_enabled:
            recent = "/".join(_number(self.camera_hz(side, recent=True)) for side in self.enabled_eyes)
            total = "/".join(_number(self.camera_hz(side)) for side in self.enabled_eyes)
            consumer = "/".join(_number(self.camera_hz(side, recent=True, consumer=True)) for side in self.enabled_eyes)
            consumer_label = "" if self.camera_mode == "stereo" else self._camera_label + " "
            parts.extend([f"解码发布 {self._camera_label} {recent} fps（累计 {total}）",
                          f"控制拍新帧 {consumer_label}{consumer} Hz", f"读图累计95% {_number(_percentile(self.copy_ms, 95))} ms"])
        parts.append(f"双手origin累计 {self.origin_ticks['both']} 拍，源停更≥100ms {max(self.source_stall_count, self.observed_source_stalls)} 次")
        if self.source_diagnostics_missing:
            parts.append(f"源诊断缺失 {self.source_diagnostics_missing} 拍")
        return "，".join(parts)

    def failures(self, cmd_jumps: int, *, held_rows: int = 0,
                 reanchor_rows: int = 0) -> list[str]:
        problems = []
        if len(self.periods) < 30:
            problems.append(f"控制拍只有 {len(self.periods)}，至少需要 30 拍")
        long_periods = [p for p in self.periods if p >= CONTROL_GAP_THRESHOLD_S]
        if long_periods:
            problems.append(
                f"控制循环间隔≥100ms {len(long_periods)} 次，最大 {max(long_periods) * 1000:.1f} ms"
                "（诊断阈值；平均频率正常不能排除短暂停顿）"
            )
        for recent, label in ((False, "累计"), (True, f"近{self.window_s:g}s")):
            hz = _counter_hz(self.ticks, self.window_s if recent else None)
            if np.isfinite(hz) and hz < self.control_hz - 2.0:
                problems.append(f"{label}动作循环 {hz:.1f} Hz，低于 {self.control_hz - 2.0:.0f} Hz（包括仿真绘制）")
        if self.camera_enabled:
            copy = _percentile(self.copy_ms, 95)
            if np.isfinite(copy) and copy >= 1000.0 / self.control_hz:
                eyes = "左右眼" if self.camera_mode == "stereo" else self._camera_label
                problems.append(f"读{eyes}图像95% {copy:.1f} ms，超过控制周期")
            for side in self.enabled_eyes:
                label = "左眼" if side == "left" else "右眼"
                for recent, window in ((False, "累计"), (True, f"近{self.window_s:g}s")):
                    hz = self.camera_hz(side, recent=recent)
                    if not np.isfinite(hz):
                        problems.append(f"{label}{window}解码发布帧率未知")
                    elif hz < self.camera_fps - 3.0:
                        problems.append(f"{label}{window}解码发布 {hz:.1f} fps，低于 {self.camera_fps - 3.0:.0f} fps")
                if self.camera_max_age_s[side] > 0.5:
                    problems.append(f"{label}最新图像最大接收年龄 {self.camera_max_age_s[side] * 1000:.0f} ms")
            if self.camera_misses or self.camera_diagnostics_missing:
                problems.append(f"相机诊断不完整：缺图 {self.camera_misses}，缺字段 {self.camera_diagnostics_missing}")
        if self.origin_ticks["left"] or self.origin_ticks["right"]:
            problems.append(f"origin pose：左 {self.origin_ticks['left']} 拍，右 {self.origin_ticks['right']} 拍，同时 {self.origin_ticks['both']} 拍")
        if self.source_stall_count or self.observed_source_stalls:
            problems.append(f"姿态源停更≥100ms：采集进程记录 {self.source_stall_count} 次，控制循环观察 {self.observed_source_stalls} 次")
        elif self.source_max_gap_s >= 0.1:
            problems.append(f"姿态源更新/采样间隔达到 {self.source_max_gap_s * 1000:.1f} ms；需区分源停更与采集进程停顿")
        if self.source_regressions:
            problems.append(f"姿态源时间戳倒退 {self.source_regressions} 次")
        if self.source_diagnostics_missing or self.pose_diagnostics_missing:
            problems.append(f"姿态诊断不完整：源时间信息缺失 {self.source_diagnostics_missing} 拍，手柄pose缺失 {self.pose_diagnostics_missing} 项")
        if cmd_jumps:
            problems.append(f"末端指令单拍超过 0.1 m 的次数是 {cmd_jumps}")
        if held_rows or reanchor_rows:
            problems.append(
                f"手柄跳变保护：保持 {held_rows}，重新锚定 {reanchor_rows}"
                "（按手计的采样数，不等于独立异常次数）；需检查原始位姿和采样间隔"
            )
        return problems

    def report(self, cmd_jumps: int, *, held_rows: int = 0,
               reanchor_rows: int = 0) -> str:
        source_gap = _number(self.source_max_gap_s * 1000) if self._source_stamp is not None else "未知"
        lines = ["运行诊断", "  " + self.line(),
                 f"  动作累计 {_number(_counter_hz(self.ticks))} Hz，周期95% {_number(_percentile([p * 1000 for p in self.periods], 95))} ms",
                 f"  控制周期最大 {_number(max(self.periods) * 1000 if self.periods else float('nan'))} ms，≥100ms {sum(p >= CONTROL_GAP_THRESHOLD_S for p in self.periods)} 次（诊断阈值）",
                 f"  姿态源最大更新间隔/年龄 {source_gap} ms（源戳可用时）",
                 f"  origin pose：左 {self.origin_ticks['left']} 拍，右 {self.origin_ticks['right']} 拍，同时 {self.origin_ticks['both']} 拍",
                 f"  指令单拍超过 0.1 m：{cmd_jumps}",
                 f"  跳变保持/重新锚定：{held_rows}/{reanchor_rows}（按手计的采样数，不等于独立异常次数）"]
        if self.camera_enabled:
            consumer_total = "/".join(_number(self.camera_hz(side, consumer=True)) for side in self.enabled_eyes)
            lines.append(f"  控制拍新帧累计 {self._camera_label} {consumer_total} Hz")
            if self.camera_mode == "stereo":
                lines.append(f"  左右最新帧读取相位差95% {_number(_percentile(self.read_phase_skew_ms, 95))} ms；共享解码帧时间不能证明相机曝光同步")
            else:
                lines.append("  左右最新帧读取相位差：不适用（单目模式）")
            lines.append(f"  相机目标 {self.camera_fps:g} fps，帧率容差 3 fps；控制拍新帧率不用于判定相机产帧")
        problems = self.failures(cmd_jumps, held_rows=held_rows, reanchor_rows=reanchor_rows)
        if problems:
            lines.append("  未通过/诊断不足：")
            lines.extend(f"  - {item}" for item in problems)
        else:
            lines.append("  本次观测在检查阈值内，未发现origin pose或≥100ms姿态源停更")
        return "\n".join(lines)
