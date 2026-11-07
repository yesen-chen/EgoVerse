"""Offline image reports from the calibrated transform and frozen SDK feedback."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from egomimic.robot.piper_pico.calibration_overlay import render_verification


def _write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def _write_rgb(path, rgb):
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Cannot write verification image: {path}")


def _contact_sheets(directory, records):
    """Six evenly sized thumbnails per page; native-resolution PNGs stay separate."""
    paths = []
    for split in ("validation", "train"):
        rows = [record for record in records if record["split"] == split]
        for start in range(0, len(rows), 6):
            group = rows[start : start + 6]
            tiles = []
            for row in group:
                image = cv2.imread(str(directory / row["board_check_image"]))
                if image is None:
                    raise OSError(
                        f"Cannot read generated image: {row['board_check_image']}"
                    )
                h = round(image.shape[0] * 540 / image.shape[1])
                tiles.append(cv2.resize(image, (540, h), interpolation=cv2.INTER_AREA))
            tile_h = max(tile.shape[0] for tile in tiles)
            sheet = np.full(
                (44 + tile_h * ((len(tiles) + 1) // 2), 1080, 3), 24, np.uint8
            )
            label = (
                "HELD-OUT VALIDATION"
                if split == "validation"
                else "TRAINING / USED IN FIT"
            )
            cv2.putText(
                sheet,
                f"{label} | page {start // 6 + 1} | open PNGs for full resolution",
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            for index, tile in enumerate(tiles):
                x, y = (index % 2) * 540, 44 + (index // 2) * tile_h
                sheet[y : y + tile.shape[0], x : x + 540] = tile
            name = f"{split}/contact_{start // 6 + 1:03d}.jpg"
            if not cv2.imwrite(str(directory / name), sheet):
                raise OSError(f"Cannot write contact sheet: {name}")
            paths.append(name)
    return paths


def export_verification_images(
    session_dir,
    result,
    samples,
    object_points,
    K,
    D,
    image_size,
    output_dir=None,
    axis_length_m=0.04,
):
    """Export only committed, accepted samples supplied by the solver.

    Neither detection nor optimization runs here. Every SDK_END projection is
    inv(T_base_camera) @ the paired measured T_base_end. The PnP board transform
    is deliberately not an input to the renderer. Missing images are explicit
    omissions; they never invalidate an otherwise successful numeric fit.
    """
    if not np.isfinite(axis_length_m) or axis_length_m <= 0:
        raise ValueError("axis_length_m must be finite and positive")
    session_dir = Path(session_dir).resolve()
    directory = (
        Path(output_dir).resolve()
        if output_dir is not None
        else session_dir
        / "verification"
        / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    for split in ("train", "validation"):
        (directory / split).mkdir()
    # This copy binds each report to its exact fit even after a later solve
    # updates calibration_result.json. No timestamps or result are recomputed.
    calibration_text = (
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    (directory / "calibration_used.json").write_text(calibration_text)
    calibration_sha256 = hashlib.sha256(calibration_text.encode()).hexdigest()
    C_base = np.linalg.inv(np.asarray(result["T_base_camera"], dtype=float))
    E_board = np.asarray(result["T_end_board"], dtype=float)
    width, height = map(int, image_size)
    records, skipped = [], []
    # Put held-out samples first in the manifest and printed browsing guidance.
    ordered = sorted(samples, key=lambda row: (row["split"] != "validation", row["id"]))
    for sample in ordered:
        sample_id, split, data = sample["id"], sample["split"], sample["data"]
        source_dir = (session_dir / "samples" / sample_id).resolve()
        image = None
        try:
            if not sample_id.isdecimal() or split not in ("train", "validation"):
                raise ValueError("Invalid sample ID or split")
            image = (source_dir / data.get("image", "raw.png")).resolve()
            if not image.is_relative_to(source_dir):
                raise ValueError("Image path must stay inside its sample directory")
            if not image.is_file():
                raise ValueError("raw_image_missing")
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise ValueError("raw_image_unreadable")
            if bgr.shape != (height, width, 3):
                raise ValueError(
                    f"image_size_mismatch: expected {width}x{height}, got {bgr.shape}"
                )
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            detection = data["detection"]
            axes, check, metrics = render_verification(
                rgb,
                T_camera_end=C_base @ np.asarray(sample["G"], dtype=float),
                T_end_board=E_board,
                object_points=object_points,
                corner_ids=detection.get("corner_ids", []),
                corners_px=detection.get("corners_px", []),
                K=K,
                D=D,
                sample_id=sample_id,
                arm=result["arm"],
                split=split,
                axis_length_m=axis_length_m,
            )
            axes_name = f"{split}/{sample_id}_sdk_end.png"
            check_name = f"{split}/{sample_id}_board_check.png"
            # Hide incomplete writes. Source raw.png/overlay.jpg are never modified.
            axes_temporary = directory / split / f".{sample_id}_sdk_end.pending.png"
            check_temporary = (
                directory / split / f".{sample_id}_board_check.pending.png"
            )
            _write_rgb(axes_temporary, axes)
            _write_rgb(check_temporary, check)
            axes_temporary.replace(directory / axes_name)
            check_temporary.replace(directory / check_name)
            records.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "source_image": str(image),
                    "source_image_sha256": hashlib.sha256(
                        image.read_bytes()
                    ).hexdigest(),
                    "sdk_end_image": axes_name,
                    "board_check_image": check_name,
                    "metrics": metrics,
                }
            )
        except (OSError, ValueError, TypeError, KeyError, cv2.error) as exc:
            skipped.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "source_image": str(image) if image is not None else None,
                    "reason": str(exc),
                }
            )
    warnings = []
    try:
        contacts = _contact_sheets(directory, records)
    except (OSError, ValueError, cv2.error) as exc:
        contacts = []
        warnings.append(f"Contact sheets incomplete: {exc}")
    status = (
        "complete"
        if records and not skipped and not warnings
        else "partial"
        if records
        else "failed"
    )
    manifest = {
        "schema_version": 1,
        "status": status,
        "arm": result["arm"],
        "directory": str(directory),
        "manifest": str(directory / "summary.json"),
        "calibration_file": "calibration_used.json",
        "calibration_sha256": calibration_sha256,
        "display_frame": "SDK_END",
        "tcp_offset_applied": False,
        "axis_length_m": axis_length_m,
        "native_image_size": [width, height],
        "transform": "T_camera_end = inverse(T_base_camera) @ T_base_end",
        "board_prediction": "T_camera_end @ T_end_board; never image PnP for predictions",
        "sample_count": len(records),
        "attempted_sample_count": len(samples),
        "image_count": 2 * len(records),
        "split_counts": {
            split: sum(row["split"] == split for row in records)
            for split in ("train", "validation")
        },
        "contact_sheets": contacts,
        "samples": records,
        "skipped_samples": skipped,
        "warnings": warnings,
    }
    _write_json(directory / "summary.json", manifest)
    (directory / "README.txt").write_text(
        "相机外参验收图片\n\n"
        "优先查看 validation/：这些姿态未参与本次拟合；train/ 是拟合样本。\n"
        "若 validation/ 为空，本次没有可显示的留出验证图，请查看 summary.json。\n"
        "*_sdk_end.png：SDK 实际末端三轴，X 红、Y 绿、Z 蓝；未添加 TCP 偏移。\n"
        f"坐标轴长度：{axis_length_m * 1000:g} mm。原图保持原始分辨率，信息栏在图外。\n"
        "*_board_check.png：叠加检测角点与由机器人位姿预测的角点，以及误差连线。\n"
        "contact_*.jpg：缩略总览；精细检查请打开对应 PNG。\n"
        "summary.json：每张图的投影、误差和缺失图片原因。\n"
        "calibration_used.json：生成这批图片所用的完整标定结果。\n\n"
        "训练拟合效果不等于独立精度验证；轴在画面外或相机后方时不会伪造其位置。\n"
    )
    # Keep the main calibration JSON compact; full per-corner metrics live here.
    return {key: value for key, value in manifest.items() if key != "samples"}
