"""30 Hz Piper + Pico collection loop.

B starts and stops an episode, X drops the episode that is recording, Y sends
both arms home with MOVE J, and A leaves. Grip closes the clutch. Each tick
takes one observation, decides one absolute cartesian target, sends it at
once, and records the pair. A tick that overruns is not caught up: the next
deadline restarts one period after the late tick.
"""

from __future__ import annotations

import argparse
import select
import signal
import sys
import termios
import time
import tty
from contextlib import nullcontext
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import numpy as np

from egomimic.robot.piper_pico.camera_config import (
    DEFAULT_UVC_BRIGHTNESS,
    load_camera_config,
    validate_camera_specs,
)
from egomimic.robot.piper_pico.camera_preview import CameraPreview, preview_fps
from egomimic.robot.piper_pico.env import PiperPicoEnv
from egomimic.robot.piper_pico.loop_profile import profile_span


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic) -> None:
    remaining = t_end - time_func()
    if remaining <= 0.0:
        return
    sleep_for = remaining - slack_time
    if sleep_for > 0.0:
        time.sleep(sleep_for)
    while time_func() < t_end:
        pass


class KeyPoller:
    def __init__(self):
        self._fd = None
        self._old = None

    def __enter__(self) -> "KeyPoller":
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is not None and self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll(self) -> set[str]:
        if self._fd is None:
            return set()
        pressed = set()
        while True:
            ready, _, _ = select.select([self._fd], [], [], 0.0)
            if not ready:
                return pressed
            pressed.add(sys.stdin.read(1).lower())


def parse_camera(value: str) -> dict:
    if "=" not in value:
        raise argparse.ArgumentTypeError("camera must be name=/dev/videoN")
    name, device = value.split("=", 1)
    if not name or not device:
        raise argparse.ArgumentTypeError("camera must be name=/dev/videoN")
    return {"name": name, "device": device}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect a Piper bimanual episode from Pico 4 Ultra")
    parser.add_argument("--output", type=Path, default=Path("piper_pico_data"))
    parser.add_argument("--left-can", default="can0")
    parser.add_argument("--right-can", default="can1")
    parser.add_argument("--camera", action="append", type=parse_camera, default=[],
                        help="UVC name=/dev/videoN; repeat for multiple UVC cameras")
    parser.add_argument(
        "--camera-config", type=Path,
        help=("camera_rig_v1 JSON with a complete native mode for each camera; "
              "cannot combine with --camera/--headset-ip or nondefault legacy "
              "camera options. Config values take precedence over legacy defaults."),
    )
    parser.add_argument("--headset-ip", default="",
                        help="enable recording the PICO left eye as a camera")
    parser.add_argument("--pc-ip", default=None)
    parser.add_argument("--pico-camera-name", default="front_1")
    parser.add_argument("--pico-camera-fps", type=int, default=60)
    parser.add_argument("--pico-camera-width", type=int, default=1080)
    parser.add_argument("--pico-camera-height", type=int, default=810)
    parser.add_argument("--pico-camera-bitrate", type=int, default=5_000_000)
    parser.add_argument("--pico-camera-latency", type=float, default=0.1,
                        help="PICO left-eye receive latency in seconds (0.1 = 100 ms, current rig)")
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--move-spd-rate", type=int, default=50)
    parser.add_argument("--robot-obs-latency", type=float, default=0.0)
    parser.add_argument("--camera-latency", type=float, default=0.08,
                        help="UVC receive latency in seconds for every --camera (0.08 = 80 ms, current rig)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--camera-brightness", type=int, default=DEFAULT_UVC_BRIGHTNESS,
                        help="UVC brightness in driver units for every --camera (default: 48); "
                             "with --camera-config set each camera's controls.brightness instead")
    parser.add_argument("--home-seconds", type=float, default=3.0)
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True,
                        help="show cameras in a separate SHM-only preview process (default: on)")
    parser.add_argument("--preview-fps", type=preview_fps, default=15.0,
                        help="preview refresh rate, 1-30 (default: 15); does not set camera/control FPS")
    return parser


def run_teleop(env, frequency: float, home_seconds: float, on_tick=None, *, loop_profiler=None) -> None:
    """30 Hz loop shared by the real collector and the cartesian simulation."""
    stop = False

    def _request_stop(signum, frame):
        del signum, frame
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    dt = 1.0 / frequency
    t_end = time.monotonic() + dt
    iteration = 0
    home_until = 0.0
    held = {"button_a": False, "button_b": False, "button_x": False, "button_y": False}
    with KeyPoller() as keys:
        while not stop:
            tick_scope = nullcontext() if loop_profiler is None else loop_profiler.tick(iteration, t_end)
            with tick_scope:
                obs = None
                if env.cameras:
                    with profile_span(loop_profiler, "get_obs"):
                        obs = env.get_obs()
                with profile_span(loop_profiler, "get_pico"):
                    sample = env.get_pico()
                with profile_span(loop_profiler, "keyboard_buttons"):
                    pressed = keys.poll()
                    for name, key in (
                        ("button_b", "b"),
                        ("button_x", "x"),
                        ("button_y", "y"),
                        ("button_a", "a"),
                    ):
                        current = float(sample[name]) > 0.5
                        if current and not held[name]:
                            pressed.add(key)
                        held[name] = current
                if "a" in pressed:
                    break
                if "y" in pressed:
                    with profile_span(loop_profiler, "home"):
                        env.home()
                        home_until = time.time() + home_seconds
                        print("双臂回 home")
                elif time.time() >= home_until:
                    if home_until > 0.0:
                        with profile_span(loop_profiler, "seed_clutches"):
                            env.seed_clutches_from_measured()
                            home_until = 0.0
                    with profile_span(loop_profiler, "teleop_action"):
                        action = env.teleop_action(sample, time.time())
                    with profile_span(loop_profiler, "exec_actions"):
                        decided = time.time()
                        deadline = decided + dt
                        command_ids = env.exec_actions(action, np.array([deadline]))
                    if obs is not None:
                        env.record_tick(obs, action, decided, deadline, int(command_ids[0]))
                with profile_span(loop_profiler, "episode_io"):
                    if env.cameras and "b" in pressed:
                        if env._recording:
                            env.end_episode()
                        else:
                            env.start_episode()
                    if env.cameras and "x" in pressed and env._recording:
                        env.drop_episode()
                if on_tick is not None:
                    with profile_span(loop_profiler, "on_tick"):
                        keep_going = on_tick(env, sample)
                    if keep_going is False:
                        break
                with profile_span(loop_profiler, "precise_wait", deadline_monotonic_s=t_end):
                    precise_wait(t_end)
                iteration += 1
                t_end += dt
                now = time.monotonic()
                if t_end <= now:
                    t_end = now + dt
        if env._recording:
            with profile_span(loop_profiler, "episode_close"):
                env.end_episode()


def camera_specs_from_args(args) -> list[dict]:
    if getattr(args, "camera_config", None) is not None:
        if args.camera or args.headset_ip:
            raise ValueError("--camera-config cannot combine with --camera or --headset-ip")
        defaults = build_parser().parse_args([])
        legacy_options = (
            "width", "height", "camera_fps", "fourcc", "camera_latency", "camera_brightness",
            "pc_ip", "pico_camera_name", "pico_camera_fps", "pico_camera_width",
            "pico_camera_height", "pico_camera_bitrate", "pico_camera_latency",
        )
        conflicts = [
            "--" + key.replace("_", "-") for key in legacy_options
            if getattr(args, key) != getattr(defaults, key)
        ]
        if conflicts:
            raise ValueError(
                "--camera-config cannot combine with nondefault legacy camera options: "
                + ", ".join(conflicts)
            )
        return load_camera_config(args.camera_config)
    cameras = []
    if args.headset_ip:
        cameras.append({
            "type": "pico", "name": args.pico_camera_name, "eye": "left",
            "headset_ip": args.headset_ip, "pc_ip": args.pc_ip,
            "width": args.pico_camera_width, "height": args.pico_camera_height,
            "fps": args.pico_camera_fps, "bitrate": args.pico_camera_bitrate,
            "latency": args.pico_camera_latency,
        })
    for camera in args.camera:
        cameras.append({
            **camera, "type": "uvc", "latency": args.camera_latency,
            "width": args.width, "height": args.height, "fps": args.camera_fps,
            "fourcc": args.fourcc,
            "controls": {"brightness": args.camera_brightness},
        })
    return validate_camera_specs(cameras)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cameras = camera_specs_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not cameras:
        parser.error("至少指定 --camera-config、--camera 或 --headset-ip")
    print(
        "B 开始/停止并保存，X 丢掉当前段，Y 双臂回 home，A 退出。\n"
        "握住 Grip 后手柄的位移和旋转按 1:1 加到握住时的手臂位姿上。"
    )
    for spec in cameras:
        brightness = (f"，brightness={spec['controls']['brightness']}"
                      if spec["type"] == "uvc" else "")
        print(
            f"图像 {spec['name']}：{spec['type']} "
            f"{spec['width']}×{spec['height']}@{spec['fps']}，"
            f"延迟补偿 {spec['latency']:g} s ({spec['latency_status']}){brightness}"
        )
    if any(spec["type"] == "pico" for spec in cameras):
        print("PICO 左目与 UVC 同段保存；保持 XRoboToolkit 运行，关闭头显本地 Record/预览。")
    with SharedMemoryManager() as shm:
        env = PiperPicoEnv(
            shm,
            output_dir=args.output,
            cameras=cameras,
            left_can=args.left_can,
            right_can=args.right_can,
            frequency=args.frequency,
            horizon=args.horizon,
            move_spd_rate=args.move_spd_rate,
            robot_obs_latency=args.robot_obs_latency,
            camera_width=args.width,
            camera_height=args.height,
            camera_fps=args.camera_fps,
        )
        with env:
            preview = CameraPreview(env.cameras, args.preview_fps) if args.preview else nullcontext()
            with preview:
                run_teleop(env, args.frequency, args.home_seconds)


if __name__ == "__main__":
    main()
