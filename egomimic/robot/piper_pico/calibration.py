"""PiPER eye-to-hand capture, image checking, solving and raw rig export."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import cv2
import numpy as np

from egomimic.robot.piper_pico.calibration_capture import (
    CalibrationControls,
    CaptureLimits,
    CaptureWorker,
    SessionStore,
    json_value,
    write_json,
)
from egomimic.robot.piper_pico.calibration_vision import (
    CharucoEstimator,
    _check_transform,
    default_board_config,
    default_camera_config,
    solve_session,
)
from egomimic.robot.piper_pico.camera_config import (
    load_camera_config,
    validate_camera_specs,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="固定 UVC 或 PICO 左目到 PiPER base 的眼在手外标定"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="真机遥操并采集单个机械臂的静止标定样本")
    collect.add_argument("--arm", choices=("left", "right"), required=True)
    collect.add_argument(
        "--headset-ip", help="旧 PICO 图像源必填；UVC --camera-rig 无需此项"
    )
    collect.add_argument("--pc-ip")
    collect.add_argument("--left-can", default="can0")
    collect.add_argument("--right-can", default="can1")
    collect.add_argument("--output", type=Path, help="全新会话目录；省略则自动生成")
    collect.add_argument("--no-preview", action="store_true")
    collect.add_argument("--frequency", type=float, default=30.0)
    collect.add_argument("--home-seconds", type=float, default=3.0)
    collect.add_argument("--move-spd-rate", type=int, default=50)
    collect.add_argument("--robot-obs-latency", type=float, default=0.0)
    collect.add_argument("--pico-camera-fps", type=int, default=60)
    collect.add_argument("--pico-camera-bitrate", type=int, default=5_000_000)
    collect.add_argument("--pico-camera-latency", type=float, default=0.1)
    for name, default in asdict(CaptureLimits()).items():
        collect.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    check = sub.add_parser(
        "check-image", help="用同一个检测器离线检查单张原始相机图，无硬件连接"
    )
    check.add_argument("--image", type=Path, required=True)
    check.add_argument("--output", type=Path, help="全新检查输出目录")
    for command in (collect, check):
        source = command.add_mutually_exclusive_group()
        source.add_argument("--camera-config", type=Path, help="旧 PICO 单目内参 JSON")
        source.add_argument(
            "--camera-rig",
            type=Path,
            help="camera_rig_v1 JSON；选择带原始内参的 UVC 相机",
        )
        command.add_argument(
            "--camera-name",
            default="front_1",
            help="--camera-rig 中的相机名，默认 front_1",
        )
        command.add_argument("--board-config", type=Path)
        command.add_argument("--min-corners", type=int, default=16)
        command.add_argument("--max-rms-px", type=float, default=0.8)
    solve = sub.add_parser("solve", help="离线求解 T_base_camera，并用留出样本验证")
    solve.add_argument("--session", type=Path, required=True)
    solve.add_argument("--output", type=Path)
    solve.add_argument(
        "--visualization-dir",
        type=Path,
        help="验收图片的新目录；默认会话内 verification/时间戳目录",
    )
    solve.add_argument(
        "--axis-length-mm",
        type=float,
        default=40.0,
        help="验收图 SDK 末端坐标轴长度，默认 40 mm",
    )
    export = sub.add_parser(
        "export", help="验证双臂结果并导出原始 UVC 标定，供离线转换使用"
    )
    export.add_argument("--left-result", type=Path, required=True)
    export.add_argument("--right-result", type=Path, required=True)
    export.add_argument("--camera-rig", type=Path, required=True)
    export.add_argument("--camera-name", default="front_1")
    export.add_argument("--calibration-id", required=True)
    export.add_argument(
        "--output", type=Path, required=True, help="新的 calibration.json 文件"
    )
    return parser


def _rig_camera(path, name):
    specs = load_camera_config(path)
    selected = [spec for spec in specs if spec["name"] == name]
    if not selected:
        raise ValueError(f"camera rig 中没有相机 {name!r}")
    spec = selected[0]
    if spec["type"] != "uvc" or "calibration" not in spec:
        raise ValueError("camera-rig 必须选择附带原始 calibration_file 的 UVC 相机")
    return spec


def _configs(args):
    spec = None
    if args.camera_rig:
        if getattr(args, "headset_ip", None):
            raise ValueError("--camera-rig cannot combine with --headset-ip")
        for key, default in (
            ("pc_ip", None),
            ("pico_camera_fps", 60),
            ("pico_camera_bitrate", 5_000_000),
            ("pico_camera_latency", 0.1),
        ):
            if getattr(args, key, default) != default:
                raise ValueError(
                    f"--camera-rig cannot combine with --{key.replace('_', '-')}"
                )
        spec = _rig_camera(args.camera_rig, args.camera_name)
        camera = {
            **copy.deepcopy(spec["calibration"]),
            "name": spec["name"],
            "type": "uvc",
        }
    else:
        if args.camera_name != "front_1":
            raise ValueError("--camera-name requires --camera-rig")
        camera = (
            json.loads(args.camera_config.read_text())
            if args.camera_config
            else default_camera_config()
        )
        if camera.get("type", "pico") != "pico":
            raise ValueError(
                "UVC intrinsics require --camera-rig instead of --camera-config"
            )
        camera = {"type": "pico", "name": "pico_left", **camera}
    board = (
        json.loads(args.board_config.read_text())
        if args.board_config
        else default_board_config()
    )
    estimator = CharucoEstimator(
        camera, board, min_corners=args.min_corners, max_rms_px=args.max_rms_px
    )
    return camera, board, estimator, spec


def _output_name(prefix):
    return Path(f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}")


def check_image(args):
    camera, board, estimator, spec = _configs(args)
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise ValueError(f"无法读取图像：{args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = estimator.detect(rgb)
    output = args.output or _output_name("piper_calibration_image_check")
    output.mkdir(parents=True, exist_ok=False)
    annotated = estimator.annotate(rgb, result)
    if not cv2.imwrite(
        str(output / "overlay.jpg"), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    ):
        raise OSError("检查图像写入失败")
    write_json(
        output / "detection.json",
        {
            "image": str(args.image.resolve()),
            "camera": camera,
            "camera_stream": spec,
            "board": board,
            "detection": result,
        },
    )
    print(json.dumps(json_value(result), ensure_ascii=False, indent=2))
    print(f"离线检查输出：{output.resolve()}")


def run_calibration(env, worker, arm, frequency, home_seconds, preview=True):
    # Importing these functions reuses the terminal and pacing implementation;
    # the original collector and run_teleop are intentionally left untouched.
    from egomimic.robot.piper_pico.collect import KeyPoller, precise_wait

    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    old_handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    controls = CalibrationControls(arm, home_seconds)
    dt = 1.0 / frequency
    deadline = time.monotonic() + dt
    last_preview = None
    last_report = 0.0
    window = "Camera | PiPER eye-to-hand"
    try:
        with KeyPoller() as keys:
            while not stop:
                if worker.fatal_error is not None:
                    raise RuntimeError(
                        "后台采样失败，停止本次标定"
                    ) from worker.fatal_error
                for device in (env.pico, *env.arms, *env.cameras):
                    if device.process.exitcode is not None:
                        raise RuntimeError(f"设备进程 {device.name} 已退出")
                sample = env.get_pico()
                keyboard = keys.poll()
                display, status, counts = worker.display()
                if preview:
                    if display is not None and display is not last_preview:
                        rgb, detection, received, stable = display
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        cv2.rectangle(
                            bgr, (0, 0), (bgr.shape[1], 108), (24, 24, 24), -1
                        )
                        rms = detection["reprojection_rms_px"]
                        rms_label = "--" if rms is None else f"{rms:.3f} px"
                        lines = [
                            f"{arm.upper()} BASE | next {controls.split.upper()} | train {counts['train']} val {counts['validation']}",
                            f"corners {detection['corner_count']} | RMS {rms_label} | {detection['reason']}",
                            f"{worker.feedback_label} | stable {stable['ok']} | received {(time.time() - received) * 1000:.0f} ms ago",
                            "X left home | Y left capture | A right home | B right capture | V split | U undo | Q exit",
                        ]
                        for y, line in zip((22, 47, 71, 96), lines):
                            cv2.putText(
                                bgr,
                                line,
                                (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.48,
                                (230, 230, 230),
                                1,
                                cv2.LINE_AA,
                            )
                        cv2.imshow(window, bgr)
                        last_preview = display
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:
                        keyboard.add(chr(key).lower())
                    if (
                        last_preview is not None
                        and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
                    ):
                        break
                elif display is not None and time.monotonic() - last_report >= 1.0:
                    _, detection, _, stable = display
                    print(
                        f"[预览] {arm} | {controls.split} | train={counts['train']} val={counts['validation']} | "
                        f"角点={detection['corner_count']} RMS={detection['reprojection_rms_px']} | "
                        f"{stable['reason']} | {status}",
                        flush=True,
                    )
                    last_report = time.monotonic()
                pressed = controls.edges(sample, keyboard)
                if not controls.tick(env, worker, sample, pressed, time.time(), dt):
                    break
                precise_wait(deadline)
                deadline += dt
                now = time.monotonic()
                if deadline <= now:
                    deadline = now + dt
    finally:
        worker.cancel("退出，取消未完成的采样")
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if preview:
            cv2.destroyAllWindows()


def collect_session(args):
    from egomimic.robot.piper_pico.env import PiperPicoEnv

    camera, board, estimator, spec = _configs(args)
    if spec is None and not args.headset_ip:
        raise ValueError("collect requires --camera-rig or --headset-ip")
    limits = CaptureLimits(
        **{key: getattr(args, key) for key in asdict(CaptureLimits())}
    )
    limits.validate()
    for key in ("frequency", "home_seconds", "pico_camera_fps", "pico_camera_bitrate"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} 必须为有限正数")
    if not 1 <= args.move_spd_rate <= 100:
        raise ValueError("move-spd-rate 应为 1–100")
    for key in ("robot_obs_latency", "pico_camera_latency"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            raise ValueError(f"{key} 必须非负且有限")
    selected_latency = spec["latency"] if spec is not None else args.pico_camera_latency
    if selected_latency + limits.max_pose_age >= limits.stable_seconds:
        raise ValueError("稳定窗口须大于相机延迟与反馈最大年龄之和")
    if not args.left_can or not args.right_can or args.left_can == args.right_can:
        raise ValueError("左右机械臂必须指定不同 CAN 端口")
    width, height = estimator.image_size
    if spec is None and (width % 2 or height % 2):
        raise ValueError("PICO H.264 的单目宽高须为偶数")
    preview = not args.no_preview
    if preview and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("当前没有桌面显示，使用终端反馈（可显式传 --no-preview）", flush=True)
        preview = False
    if preview:
        # Fail on GUI setup before any robot is enabled.
        cv2.namedWindow("Camera | PiPER eye-to-hand", cv2.WINDOW_NORMAL)
    output = args.output or _output_name(f"piper_calibration_{args.arm}")
    spec = spec or {
        "type": "pico",
        "name": "pico_left",
        "eye": "left",
        "headset_ip": args.headset_ip,
        "pc_ip": args.pc_ip,
        "width": width,
        "height": height,
        "fps": args.pico_camera_fps,
        "bitrate": args.pico_camera_bitrate,
        "latency": args.pico_camera_latency,
    }
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "arm": args.arm,
        "camera": camera,
        "board": board,
        "camera_stream": spec,
        "limits": asdict(limits),
        "detection_limits": {
            "min_corners": args.min_corners,
            "max_rms_px": args.max_rms_px,
        },
        "robot": {
            "left_can": args.left_can,
            "right_can": args.right_can,
            "frequency": args.frequency,
            "move_spd_rate": args.move_spd_rate,
            "observation_latency_s": args.robot_obs_latency,
            "home_seconds": args.home_seconds,
        },
        "frames": {
            "base": f"{args.arm}_base",
            "end": "SDK measured end frame (not assumed fingertip)",
            "camera": f"{spec['name']} ({spec['type']}) optical: x right, y down, z forward",
        },
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "pairing": "stationary direct snapshots; no get_obs and no pose interpolation",
        "timestamp_limit": "Camera time is host decode minus configured latency, not hardware exposure. SDK pose stamp is latest CAN partition, not an atomic 6-DOF timestamp.",
    }
    store = SessionStore(output, metadata)
    print(
        f"会话：{output.resolve()}\n当前标定：{args.arm} base ← {spec['name']} ({spec['type']})\n"
        "X 左臂 home；Y 左臂采样；A 右臂 home；B 右臂采样。\n"
        "V 切换训练/验证；U 作废上一条；Q/Esc/Ctrl+C 退出。Grip、Trigger 和遥操映射保持原样。\n"
        "夹持柄在整次会话中保持不动；松开 Trigger 仍会按原遥操逻辑打开夹爪。",
        flush=True,
    )
    worker = None
    try:
        with SharedMemoryManager() as shm:
            env = PiperPicoEnv(
                shm,
                output_dir=output / "runtime",
                cameras=[spec],
                left_can=args.left_can,
                right_can=args.right_can,
                frequency=args.frequency,
                move_spd_rate=args.move_spd_rate,
                robot_obs_latency=args.robot_obs_latency,
            )
            try:
                with env:
                    worker = CaptureWorker(
                        env.cameras[0],
                        env.arms[0 if args.arm == "left" else 1],
                        estimator,
                        store,
                        limits,
                    )
                    worker.start()
                    try:
                        run_calibration(
                            env,
                            worker,
                            args.arm,
                            args.frequency,
                            args.home_seconds,
                            preview,
                        )
                    finally:
                        worker.stop_event.set()
                        worker.cancel("退出，取消未完成的采样")
            finally:
                # Stop devices first; then wait for bounded background cleanup
                # while shared memory remains mapped.
                if worker is not None:
                    worker.close()
    finally:
        if preview:
            cv2.destroyAllWindows()
    print(f"采集结束：{store.path.resolve()}，有效样本 {len(store.saved)}", flush=True)


def _raw_geometry(camera):
    """Compare physical geometry and source identity without depending on paths."""
    return {
        **{
            key: camera.get(key)
            for key in (
                "K",
                "D",
                "image_size",
                "camera_model",
                "preprocessing",
                "source_sha256",
            )
        },
        "identity": {
            key: value
            for key, value in camera.get("camera", {}).items()
            if key.startswith("ID_")
        },
    }


def _capture_geometry(spec):
    lens = {
        ("focus_automatic_continuous" if key == "focus_auto" else key): value
        for key, value in spec.get("controls", {}).items()
        if "focus" in key or "zoom" in key
    }
    return {
        **{
            key: spec.get(key)
            for key in (
                "name",
                "type",
                "device",
                "width",
                "height",
                "fps",
                "fourcc",
            )
        },
        "lens_controls": lens,
        "expected_identity": spec.get("expected_identity", {}),
        "intrinsics": _raw_geometry(spec["calibration"]),
    }


def export_calibration(args):
    """Combine verified solves without changing camera geometry or inventing poses."""
    if args.camera_name != "front_1":
        raise ValueError("converter export requires reference camera front_1")
    if not args.calibration_id.strip():
        raise ValueError("calibration-id must be nonempty")
    spec = _rig_camera(args.camera_rig, args.camera_name)
    raw = spec["calibration"]
    expected = _capture_geometry(spec)
    extrinsics, sources, quality = {}, {}, {}
    for arm in ("left", "right"):
        path = getattr(args, f"{arm}_result").resolve()
        content = path.read_bytes()
        result = json.loads(content)
        if not isinstance(result, dict) or result.get("arm") != arm:
            raise ValueError(f"{arm} result must describe the {arm} arm")
        if (
            result.get("solve_status") != "solved"
            or result.get("optimization", {}).get("success") is not True
            or result.get("validation_available") is not True
            or len(result.get("training_sample_ids", [])) < 8
            or len(result.get("validation_sample_ids", [])) < 3
        ):
            raise ValueError(
                f"{arm} result needs a successful solve and held-out validation"
            )
        if (
            result.get("extrinsics_convention") != "T_base_camera"
            or result.get("translation_unit") != "metre"
        ):
            raise ValueError(f"{arm} result must use T_base_camera in metres")
        camera = result.get("camera", {})
        if (
            result.get("camera_type") != "uvc"
            or result.get("camera_name") != args.camera_name
            or result.get("camera_eye") is not None
            or camera.get("type") != "uvc"
            or _raw_geometry(camera) != _raw_geometry(raw)
        ):
            raise ValueError(f"{arm} result camera geometry/identity differs from rig")
        stream = result.get("camera_stream")
        if not isinstance(stream, dict):
            raise ValueError(f"{arm} result lacks camera_stream provenance")
        validate_camera_specs([stream])
        if _capture_geometry(stream) != expected:
            raise ValueError(f"{arm} result camera mode/identity differs from rig")
        extrinsics[arm] = _check_transform(
            result.get("T_base_camera"), f"{arm}.T_base_camera"
        ).tolist()
        sources[arm] = {
            "path": str(path),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        quality[arm] = {
            key: result.get(key)
            for key in (
                "solve_status",
                "optimization",
                "training",
                "validation",
                "validation_available",
                "warnings",
                "production_approved",
                "visualization",
            )
        }
    exported = {
        "schema": "piper_uvc_raw_calibration_v1",
        "calibration_id": args.calibration_id,
        "reference_camera": "front_1",
        "extrinsics_convention": "T_base_camera",
        "translation_unit": "metre",
        "image_geometry": "raw_distorted",
        "extrinsics": extrinsics,
        "intrinsics": {"front_1": raw["K"]},
        "image_sizes": {"front_1": raw["image_size"]},
        "distortion_coefficients": {"front_1": raw["D"]},
        "camera_models": {"front_1": raw["camera_model"]},
        "raw_camera_calibration": raw,
        "camera_stream": spec,
        "source_results": sources,
        "quality": {
            "intrinsics": {
                "quality_warnings": raw.get("quality_warnings", []),
                "production_approved": raw.get("production_approved", False),
            },
            "extrinsics": quality,
        },
        "production_approved": False,
        "usage": "Raw native camera K/D. Rectify images and update K before pinhole-only training. Physical extrinsic accuracy remains subject to validation.",
    }
    text = json.dumps(exported, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(text)
    print(f"原始 UVC 双臂标定已导出：{args.output.resolve()}")
    return exported


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            collect_session(args)
        elif args.command == "check-image":
            check_image(args)
        elif args.command == "export":
            export_calibration(args)
        else:
            result = solve_session(
                args.session,
                args.output,
                visualization_dir=args.visualization_dir,
                axis_length_m=args.axis_length_mm / 1000.0,
            )
            print(json.dumps(json_value(result), ensure_ascii=False, indent=2))
            pictures = result["visualization"]
            print(
                f"外参结果：{(args.output or args.session / 'calibration_result.json').resolve()}"
            )
            if pictures.get("directory"):
                print(
                    f"验收图片：{pictures['directory']}\n"
                    f"已生成 {pictures['sample_count']} 个样本、{pictures['image_count']} 张图；优先查看 validation/。"
                )
            if pictures["status"] != "complete":
                print(
                    "外参数值已保存，但验收图片不完整；请查看结果中的 visualization 和图片目录的 summary.json。"
                )
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"标定失败：{exc}\n")


if __name__ == "__main__":
    main()
