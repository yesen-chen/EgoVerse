"""Replay converted Piper robot commands; default dry run never opens CAN."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from egomimic.robot.piper_pico.dataset_replay import (
    PiperReplayDevice,
    RealClock,
    ReplayOptions,
    SimulatedDevice,
    VirtualClock,
    discover_episodes,
    load_episode,
    replay_episode,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Converted episode .zarr or batch directories, in replay order",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute", action="store_true", help="Enable and move real arms on CAN"
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Default: virtual-clock validation, no CAN",
    )
    parser.add_argument("--left-can", default="can0")
    parser.add_argument("--right-can", default="can1")
    parser.add_argument(
        "--move-spd-rate",
        type=int,
        help="SDK rate [1,100]; default each recording's value",
    )
    parser.add_argument(
        "--report", type=Path, help="New JSON report path; never overwrite"
    )
    help_text = {
        "speed": "Replay time scale (0,1]; 0.5 doubles duration, independent of SDK speed rate",
        "approach_duration": "Minimum seconds to reach each episode's first command",
        "approach_frequency": "Approach command frequency; replay uses recorded decision intervals",
        "approach_linear_speed": "Approach Cartesian speed bound, m/s",
        "approach_angular_speed": "Approach angular speed bound, rad/s",
        "approach_gripper_speed": "Approach gripper width change bound, m/s",
        "position_tolerance": "Both arms must reach the first/last target within this distance (m)",
        "rotation_tolerance": "Arrival orientation tolerance, rad",
        "gripper_tolerance": "Arrival gripper width tolerance, m",
        "settle_duration": "Both arms must remain within arrival tolerances for this many seconds",
        "settle_timeout": "Abort if first/last target is not reached in this many seconds",
        "max_translation_step": "Reject any recorded command step above this distance, m",
        "max_rotation_step": "Reject any recorded command rotation step above this angle, rad",
        "max_tick_gap": "Reject source decision-time gaps above this many seconds",
        "max_schedule_slip": "Abort rather than catch up if playback slips this many seconds",
    }
    for name, value in asdict(ReplayOptions()).items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=float,
            default=value,
            help=help_text[name],
        )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    options = ReplayOptions(
        **{name: getattr(args, name) for name in asdict(ReplayOptions())}
    )
    if args.move_spd_rate is not None and not 1 <= args.move_spd_rate <= 100:
        parser.error("--move-spd-rate must be in [1,100]")
    if not args.left_can or not args.right_can or args.left_can == args.right_can:
        parser.error("CAN interfaces must be nonempty and different")
    # Validate EVERY selected episode before any device is constructed/opened.
    paths = discover_episodes(args.input)
    episodes = [load_episode(path, options) for path in paths]
    output = (
        (
            args.report
            or Path("diagnostics")
            / (
                "piper_dataset_replay_"
                + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                + ".json"
            )
        )
        .expanduser()
        .resolve()
    )
    if output.suffix != ".json" or any(
        path == output or path in output.parents for path in paths
    ):
        parser.error("--report must be a JSON file outside source episodes")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Open exclusively before motion; fail early on bad/unwritable destinations.
    with output.open("x", encoding="utf-8") as handle:
        report = {
            "mode": "real_CAN" if args.execute else "dry_run_no_CAN",
            "options": asdict(options),
            "trajectory": "cmd_ee_pose and cmd_gripper; separate left/right Piper bases",
            "timing": "recorded decision intervals / speed, shared by both arms; starts after measured arrival",
            "dry_run_scope": "ideal tracking on virtual time; not physical motion, IK or collision verification",
            "status": "starting",
            "episodes": [],
        }
        device = None
        code = 1
        try:
            print(
                "真机回放：将使能左右臂。Ctrl+C 停止并尝试保持当前位置。"
                if args.execute
                else "无 CAN 验收：虚拟时间、理想跟随，不连接机械臂。"
            )
            print(f"已校验 {len(episodes)} 条轨迹；每条都会先移到首点并等待双臂到位。")
            for episode in episodes:
                rate = (
                    episode.move_spd_rate
                    if args.move_spd_rate is None
                    else args.move_spd_rate
                )
                print(
                    f"  {episode.path}: {len(episode.actions)} 点，回放 {episode.offsets[-1] / options.speed:.2f}s，SDK speed={rate}"
                )
            if args.execute:
                device = PiperReplayDevice(args.left_can, args.right_can)
                device.open()
                clock = RealClock()
            else:
                device = SimulatedDevice(episodes[0].initial_state)
                clock = VirtualClock()
            for episode in episodes:
                result = {}
                report["episodes"].append(result)
                replay_episode(
                    episode,
                    device,
                    options,
                    clock,
                    result,
                    move_spd_rate=args.move_spd_rate,
                )
            report["status"], code = "completed", 0
        except KeyboardInterrupt:
            report["status"], code = "interrupted", 130
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(report["error"])
        finally:
            if device is not None:
                try:
                    device.close()
                except Exception as exc:
                    report.update(status="failed", close_error=str(exc))
                    code = 1
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"{report['status']}: {output}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
