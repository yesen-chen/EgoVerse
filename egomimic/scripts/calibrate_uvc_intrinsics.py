"""Standalone UVC ChArUco intrinsics experiment; no PICO service or robot required."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from egomimic.robot.piper_pico.calibration_vision import default_board_config
from egomimic.robot.piper_pico.uvc_intrinsics import CharucoCorners, solve_session


def write_session(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def read_controls(device: str) -> dict:
    """Read named V4L2 controls; missing tools/support remain explicit unknowns."""
    if shutil.which("v4l2-ctl") is None:
        return {"available": False, "reason": "v4l2-ctl not installed", "values": {}}
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, "--list-ctrls"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": str(exc), "values": {}}
    values = {}
    for line in result.stdout.splitlines():
        match = re.match(r"\s*(\w+)\s+0x[0-9a-fA-F]+\s+\([^)]*\)\s*:\s*(.*)", line)
        if match:
            value = re.search(r"\bvalue=(-?\d+)", match[2])
            if value:
                values[match[1]] = int(value[1])
    return {
        "available": result.returncode == 0,
        "values": values,
        "raw": result.stdout,
        "error": result.stderr.strip(),
    }


def lock_focus(device: str, controls: dict, requested: int | None) -> dict:
    """Only called on explicit --lock-focus/--focus. Settings persist after exit."""
    if not controls["available"]:
        raise ValueError(
            "无法读取对焦控制项；请手动固定镜头后不使用 --lock-focus/--focus"
        )
    values = controls["values"]
    changes = {}
    for key in ("focus_automatic_continuous", "focus_auto"):
        if key in values:
            changes[key] = 0
    if requested is not None:
        if "focus_absolute" not in values:
            raise ValueError("此相机没有 focus_absolute，不能使用 --focus")
        changes["focus_absolute"] = requested
    if not changes:
        print("未发现可设置的自动对焦项；请确认相机为定焦或已手动固定镜头。")
        return controls
    for key, value in changes.items():
        subprocess.run(
            ["v4l2-ctl", "-d", device, f"--set-ctrl={key}={value}"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
    verified = read_controls(device)
    if not verified["available"] or any(
        verified["values"].get(k) != v for k, v in changes.items()
    ):
        raise ValueError("对焦设置读回不一致，请先在 V4L2 中检查相机支持的控制项")
    print(f"已设置并核对对焦：{changes}；退出后保留这些设置。")
    return verified


def focus_signature(controls: dict) -> dict:
    return {
        k: v
        for k, v in controls.get("values", {}).items()
        if "focus" in k or "zoom" in k
    }


def validate_focus(controls: dict, baseline: dict | None = None) -> None:
    values = controls.get("values", {})
    if any(values.get(k) == 1 for k in ("focus_automatic_continuous", "focus_auto")):
        raise ValueError(
            "自动对焦仍开启；先对准工作距离对焦，再加 --lock-focus 固定，或先手动关闭"
        )
    if baseline and (
        not controls["available"] or focus_signature(controls) != baseline
    ):
        raise ValueError(
            "录制过程中焦点/变焦设置变化或无法读回；请结束本会话、固定设置后重新采集"
        )


def camera_identity(device: str) -> dict:
    identity = {"device": device, "resolved_device": str(Path(device).resolve())}
    if shutil.which("udevadm"):
        try:
            result = subprocess.run(
                ["udevadm", "info", "--query=property", f"--name={device}"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            for line in result.stdout.splitlines():
                key, sep, value = line.partition("=")
                if sep and key in {
                    "ID_SERIAL",
                    "ID_MODEL",
                    "ID_VENDOR",
                    "ID_PATH",
                    "ID_V4L_CAPABILITIES",
                }:
                    identity[key] = value
        except (OSError, subprocess.TimeoutExpired):
            pass
    return identity


def open_camera(device: str, width: int, height: int, fps: float, fourcc: str):
    """Require the native requested mode; never resize a calibration frame."""
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"打不开 UVC 相机 {device}")
        # Same format-setting order as the existing UVC collector.
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        actual = {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "fourcc_code": int(capture.get(cv2.CAP_PROP_FOURCC)),
        }
        actual["fourcc"] = "".join(
            chr((actual["fourcc_code"] >> (8 * i)) & 255) for i in range(4)
        )
        if (
            (actual["width"], actual["height"]) != (width, height)
            or not np.isfinite(actual["fps"])
            or abs(actual["fps"] - fps) > 0.5
            or actual["fourcc"] != fourcc
        ):
            raise RuntimeError(
                f"相机模式未生效，实际 {actual}；请先检查 --list-formats-ext"
            )
        return capture, actual
    except BaseException:
        capture.release()
        raise


def save_sample(
    session_dir: Path,
    session: dict,
    bgr: np.ndarray,
    split: str,
    detection: dict,
    controls: dict,
) -> bool:
    width, height = session["camera"]["image_size"]
    if bgr.dtype != np.uint8 or bgr.shape != (height, width, 3):
        raise ValueError("原始图像尺寸/格式变化，拒绝保存；不会自动缩放")
    if not detection["accepted"]:
        print(f"本张未保存：{detection['reason']}")
        return False
    digest = hashlib.sha256(bgr.tobytes()).hexdigest()
    if any(row["pixel_sha256"] == digest for row in session["samples"]):
        print("本张未保存：与已保存图像完全相同。")
        return False
    number = len(session["samples"]) + 1
    relative = f"{split}/{number:06d}.png"
    target = session_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    if not cv2.imwrite(str(target), bgr):
        raise OSError(f"保存失败：{target}")
    session["samples"].append(
        {
            "file": relative,
            "split": split,
            "pixel_sha256": digest,
            "save_time_unix_s": time.time(),
            "corner_count": detection["corner_count"],
            "coverage_fraction": detection["coverage_fraction"],
            "controls": controls,
        }
    )
    write_session(session_dir / "session.json", session)
    print(
        f"已保存 {relative}，角点 {detection['corner_count']}，覆盖 {detection['coverage_fraction']:.1%}"
    )
    return True


def capture_session(args) -> None:
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError(
            "采集需要桌面窗口；请在本机图形终端运行。solve 可无窗口运行。"
        )
    if args.output.exists():
        raise FileExistsError(f"采集目录已存在，请更换 --output：{args.output}")
    board = (
        json.loads(args.board_config.read_text())
        if args.board_config
        else default_board_config()
    )
    detector = CharucoCorners(board, min_corners=args.min_corners)
    device = f"/dev/video{args.device}" if args.device.isdigit() else args.device
    capture, actual = open_camera(
        device, args.width, args.height, args.fps, args.fourcc
    )
    session = None
    owns_output = False
    window = "UVC intrinsics | SPACE/C save | V train/validation | Q quit"
    try:
        controls_before = read_controls(device)
        controls = controls_before
        if args.lock_focus or args.focus is not None:
            controls = lock_focus(device, controls, args.focus)
        validate_focus(controls)
        baseline = focus_signature(controls)
        session = {
            "schema": "uvc_intrinsics_capture_v1",
            "created_unix_s": time.time(),
            "opencv_version": cv2.__version__,
            "board": board,
            "min_corners": args.min_corners,
            "camera": {
                "name": args.name,
                **camera_identity(device),
                "image_size": [args.width, args.height],
                "requested_mode": {
                    "width": args.width,
                    "height": args.height,
                    "fps": args.fps,
                    "fourcc": args.fourcc,
                },
                "actual_mode": actual,
                "focus_note": args.focus_note,
                "controls": controls,
                "controls_before": controls_before,
            },
            "preprocessing": {"crop": None, "resize": None, "mirror": False},
            "samples": [],
            "closed_cleanly": False,
        }
        args.output.mkdir(parents=True, exist_ok=False)
        owns_output = True
        write_session(args.output / "session.json", session)
        print(
            f"使用 {device}：{actual['width']}×{actual['height']}@{actual['fps']:g} {actual['fourcc']}"
        )
        print(
            f"板规格：{board['squares_x']}×{board['squares_y']} 格，方格 {board['square_length_m'] * 1000:g} mm，标记 {board['marker_length_m'] * 1000:g} mm"
        )
        print(
            "无需 PICO 或机械臂。固定镜头设置，手持板覆盖画面中心、四边和四角，改变距离和多轴倾角。"
        )
        print(
            "空格/C 保存；V 切换训练/验证；Q/Esc 结束。建议 25–35 张训练、8–10 张独立验证。"
        )
        if not baseline:
            print(
                "未读到可核对的对焦/变焦控制值，请确认镜头固定，并用 --focus-note 记录设置。"
            )
        split, failures, message, message_until = "train", 0, "", 0.0
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, min(args.width, 1280), min(args.height, 800))
        while True:
            ok, bgr = capture.read()
            if not ok:
                failures += 1
                if failures >= 10:
                    raise RuntimeError("连续读取相机失败，请检查 USB 连接")
                time.sleep(0.05)
                continue
            failures = 0
            if bgr.shape != (args.height, args.width, 3):
                raise RuntimeError(f"相机输出尺寸变化：{bgr.shape}；拒绝缩放标定图片")
            detection = detector.detect(bgr)
            preview = bgr.copy()
            if detection["corner_count"]:
                cv2.aruco.drawDetectedCornersCharuco(
                    preview,
                    np.asarray(detection["corners_px"], dtype=np.float32).reshape(
                        -1, 1, 2
                    ),
                    np.asarray(detection["ids"], dtype=np.int32).reshape(-1, 1),
                )
            train_count = sum(row["split"] == "train" for row in session["samples"])
            val_count = len(session["samples"]) - train_count
            rows = [
                f"{split.upper()} | TRAIN {train_count} / VALIDATION {val_count}",
                f"corners={detection['corner_count']} | {detection['reason']}",
                "SPACE/C save | V switch split | Q/Esc finish",
            ]
            if time.monotonic() < message_until:
                rows.append(message)
            for i, line in enumerate(rows):
                cv2.putText(
                    preview,
                    line,
                    (12, 26 + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 220, 0) if detection["accepted"] else (0, 0, 255),
                    2,
                )
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 255
            if (
                key in (ord("q"), ord("Q"), 27)
                or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
            ):
                break
            if key in (ord("v"), ord("V")):
                split = "validation" if split == "train" else "train"
            if key in (32, ord("c"), ord("C")):
                current = read_controls(device)
                validate_focus(current, baseline)
                saved = save_sample(
                    args.output, session, bgr, split, detection, current
                )
                message = (
                    "SAVED: move board before next sample"
                    if saved
                    else "NOT SAVED: need valid, distinct board image"
                )
                message_until = time.monotonic() + 1.5
        session["closed_cleanly"] = True
    except KeyboardInterrupt:
        if session is not None:
            session["closed_cleanly"] = True
        print("停止采集；已保存的图片保留。")
    finally:
        capture.release()
        try:
            cv2.destroyAllWindows()
        finally:
            if session is not None and owns_output:
                session["ended_unix_s"] = time.time()
                write_session(args.output / "session.json", session)
    print(f"采集目录：{args.output.resolve()}；下一步运行 solve。")


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value):
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(
        description="独立 UVC 内参/畸变标定（ChArUco，无机器人、无 PICO）"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser(
        "capture", help="保存原始分辨率 PNG；手动区分 TRAIN/VALIDATION"
    )
    capture.add_argument(
        "--device", required=True, help="例如 /dev/video0 或 /dev/v4l/by-id/..."
    )
    capture.add_argument("--output", type=Path, required=True, help="新的采集目录")
    capture.add_argument("--name", default="front_1")
    capture.add_argument("--width", type=positive_int, default=1280)
    capture.add_argument("--height", type=positive_int, default=720)
    capture.add_argument("--fps", type=positive_float, default=30.0)
    capture.add_argument("--fourcc", default="MJPG")
    capture.add_argument(
        "--board-config", type=Path, help="可选板参数 JSON；默认现有 6×6、19/13.3 mm 板"
    )
    capture.add_argument("--min-corners", type=positive_int, default=8)
    capture.add_argument(
        "--lock-focus",
        action="store_true",
        help="通过 V4L2 关闭自动对焦；退出后保留设置",
    )
    capture.add_argument(
        "--focus",
        type=int,
        help="设置 focus_absolute 并关闭自动对焦；单位为驱动原始单位，不是 mm",
    )
    capture.add_argument(
        "--focus-note",
        default="not specified; verify lens remained fixed",
        help="记录镜头/工作距离/手动对焦位置",
    )
    solve = sub.add_parser(
        "solve", help="只用 TRAIN 求解，在 VALIDATION 上验证并输出去畸变图片"
    )
    solve.add_argument("--session", type=Path, required=True)
    solve.add_argument(
        "--output", type=Path, required=True, help="新的结果目录；不会覆盖之前的求解"
    )
    solve.add_argument("--min-train", type=positive_int, default=15)
    solve.add_argument("--min-validation", type=positive_int, default=5)
    solve.add_argument("--max-rms-px", type=positive_float, default=1.0)
    solve.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="去畸变视场保留系数 0..1；不做 ROI 裁剪",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "capture":
        if len(args.fourcc) != 4:
            parser.error("--fourcc 必须为四个字符，例如 MJPG")
        capture_session(args)
    else:
        if not np.isfinite(args.alpha) or not 0 <= args.alpha <= 1:
            parser.error("--alpha 必须在 0..1 之间")
        result = solve_session(
            args.session,
            args.output,
            min_train=args.min_train,
            min_validation=args.min_validation,
            max_rms_px=args.max_rms_px,
            alpha=args.alpha,
        )
        print(f"已输出 {args.output.resolve()}")
        print("原图内参 K：\n", np.asarray(result["K"]))
        print("原图畸变 D：", result["D"])
        print(
            f"训练 RMS：{result['train']['rms_px']:.3f} px；留出验证 RMS：{result['validation']['rms_px']:.3f} px"
        )
        for warning in result["quality_warnings"]:
            print(f"注意：{warning}")
        print(
            "请检查 calibration.json 中的训练/留出误差、覆盖范围与 quality_warnings，以及 comparison/。"
        )
        print("这是内参实验结果，不包含相机到机械臂外参，也未自动接入采集/训练。")


if __name__ == "__main__":
    main()
