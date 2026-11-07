"""Pico teleop of two simulated Piper arms.

Pass ``--headset-ip`` to also decode the headset cameras while the control
loop runs. ``--camera-mode`` selects stereo, left, or right output on the
headset. ``--frequency`` selects 30, 60, or 90 Hz independently of camera
fps. The check copies the same frame history ``get_obs`` would copy.
"""

import argparse
import sys
import tempfile
import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

from egomimic.robot.piper_pico.collect import run_teleop
from egomimic.robot.piper_pico.env import PiperPicoEnv
from egomimic.robot.piper_pico.loop_profile import LoopProfiler, profile_span
from egomimic.robot.piper_pico.pico_camera import PicoStereoProcess
from egomimic.robot.piper_pico.rate_monitor import RateMonitor, observation_k
from egomimic.robot.piper_pico.sim_view import SimBimanualView
from egomimic.robot.piper_pico.teleop_log import TeleopRecorder, default_log_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate Piper teleop, optionally with the Pico cameras")
    parser.add_argument("--headset-ip", default="")
    parser.add_argument("--pc-ip", default=None)
    parser.add_argument(
        "--frequency", type=int, choices=(30, 60, 90), default=30,
        help="simulation control Hz; independent of camera fps (default: 30)",
    )
    parser.add_argument(
        "--no-view", action="store_true",
        help="skip the 3D window when measuring control-loop performance",
    )
    parser.add_argument(
        "--profile-loop", action="store_true",
        help="measure loop stages in bounded memory; save <CSV stem>_timing.json at exit",
    )
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--camera-mode", choices=("stereo", "left", "right"), default="stereo",
        help="headset camera output: stereo or a single eye (default: stereo)",
    )
    parser.add_argument("--startup-timeout", type=float, default=25.0)
    return parser


def _wait_until_ready(stereo: PicoStereoProcess) -> None:
    deadline = time.time() + stereo.startup_timeout
    while not stereo.ready_event.is_set():
        if stereo.process.exitcode not in (None, 0):
            raise RuntimeError(f"头显相机进程退出，code {stereo.process.exitcode}")
        if time.time() >= deadline:
            raise TimeoutError("等待头显相机图像超时")
        time.sleep(0.05)


def _read_eyes(stereo: PicoStereoProcess, k: int):
    """Copy the observation history and sample producer counters independently.

    The counter belongs to the producer, whereas the latest image is the frame
    this particular control tick saw. Reading left then right can straddle a
    publication; their latest timestamps are not an exposure-sync measurement.
    """
    started = time.perf_counter()
    sides = stereo.enabled_eyes
    data = {"camera_mode": stereo.mode}
    for side in sides:
        eye = getattr(stereo, side)
        data[f"camera_{side}_count"] = int(eye.ring.count)
        data[f"camera_{side}_count_mono"] = time.monotonic()
        try:
            frames = eye.get_last(k)
        except RuntimeError:
            data[f"camera_{side}_receive_timestamp"] = float("nan")
        else:
            data[f"camera_{side}_receive_timestamp"] = float(frames["camera_receive_timestamp"][-1])
    now = time.time()
    for side in sides:
        data[f"camera_{side}_age_s"] = now - data[f"camera_{side}_receive_timestamp"]
    if len(sides) == 2:
        data["camera_read_phase_skew_s"] = abs(
            data["camera_left_receive_timestamp"] - data["camera_right_receive_timestamp"]
        )
    data["camera_copy_s"] = time.perf_counter() - started
    return data


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(
        "仿真双臂，不打开 CAN。\n"
        "握住 Grip 后手柄按 1:1 带动末端。Y 回 home，A 退出。\n"
        "终端里的 position jump / tracking lost 是离合在挡瞬移，不是末端已经追出去。\n"
        "红叉和「末端指令跳变」才是发给真机的目标单帧超过了 0.1 m。"
    )
    print(
        f"动作循环目标 {args.frequency} Hz；"
        + ("不打开 3D 视图。" if args.no_view else "3D 视图最多刷新 30 Hz。")
    )
    log_path = Path(default_log_path("piper_pico_sim"))
    profiler = LoopProfiler() if args.profile_loop else None
    recorder = TeleopRecorder(log_path, profiler=profiler)
    timing_path = log_path.with_name(log_path.stem + "_timing.json")
    view = None
    monitor = RateMonitor(
        control_hz=args.frequency, camera_fps=args.camera_fps,
        camera_enabled=bool(args.headset_ip),
        camera_mode=args.camera_mode,
    )
    error_type = None
    try:
        view = None if args.no_view else SimBimanualView()
        with tempfile.TemporaryDirectory(prefix="piper_pico_sim_") as tmp:
            with SharedMemoryManager() as shm:
                stereo = None
                try:
                    if args.headset_ip:
                        mode_label = {"stereo": "双目", "left": "左目", "right": "右目"}[args.camera_mode]
                        print(
                            f"同时打开头显相机。动作循环目标 {args.frequency} Hz；"
                            f"头显输出{mode_label}，每个启用眼目标 {args.camera_fps} fps。\n"
                            "保持 XRoboToolkit 运行，停止头显本地 Record，不要同时开预览。",
                            flush=True,
                        )
                        stereo = PicoStereoProcess(
                            shm,
                            headset_ip=args.headset_ip,
                            pc_ip=args.pc_ip,
                            fps=args.camera_fps,
                            mode=args.camera_mode,
                            startup_timeout=args.startup_timeout,
                        )
                        stereo.start()
                        _wait_until_ready(stereo)
                        print("头显相机已出图", flush=True)
                    env = PiperPicoEnv(
                        shm,
                        output_dir=Path(tmp),
                        cameras=[],
                        frequency=args.frequency,
                        simulate=True,
                    )
                    k = observation_k(env.horizon, env.frequency)

                    def on_tick(env, sample):
                        with profile_span(profiler, "sim.read_eyes"):
                            diagnostics = _read_eyes(stereo, k) if stereo is not None else None
                        with profile_span(profiler, "sim.monitor"):
                            monitor.observe(sample, diagnostics)
                        with profile_span(profiler, "sim.recorder_write"):
                            recorder.write(env, sample, diagnostics=diagnostics)
                        with profile_span(profiler, "sim.view"):
                            return view.update(env) if view is not None else True

                    with env:
                        # Launch the CSV thread after the device processes,
                        # and finish opening the file before control starts.
                        recorder.start()
                        profile_kwargs = {} if profiler is None else {"loop_profiler": profiler}
                        run_teleop(env, env.frequency, 3.0, on_tick=on_tick, **profile_kwargs)
                finally:
                    if stereo is not None:
                        stereo.stop()
    except BaseException as exc:
        error_type = type(exc).__name__
        raise
    finally:
        had_error = error_type is not None
        recorder_error = None
        try:
            try:
                recorder.close()
            except Exception as exc:
                recorder_error = exc
                if not had_error:
                    error_type = type(exc).__name__
                else:
                    print(f"CSV 关闭失败（保留原始运行异常）：{exc}", file=sys.stderr, flush=True)
            finally:
                if view is not None:
                    view.close()
                print(
                    monitor.report(
                        recorder.cmd_jumps,
                        held_rows=recorder.held_rows,
                        reanchor_rows=recorder.reanchor_rows,
                    ),
                    flush=True,
                )
        finally:
            if profiler is not None:
                try:
                    profiler.write_json(timing_path, csv_path=str(log_path.resolve()),
                                        recorder=recorder.timing_report(),
                                        frequency_hz=args.frequency, camera_fps=args.camera_fps,
                                        camera_mode=args.camera_mode,
                                        no_view=args.no_view, view_enabled=not args.no_view,
                                        error_type=error_type)
                    print(f"控制循环分段计时已保存：{timing_path.resolve()}", flush=True)
                except OSError as exc:
                    print(f"分段计时保存失败：{exc}", file=sys.stderr, flush=True)
        if recorder_error is not None and not had_error:
            raise recorder_error


if __name__ == "__main__":
    main()
