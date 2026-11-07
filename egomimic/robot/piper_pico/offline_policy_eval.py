"""Recorded-input evaluation through the live inference implementation, without devices.

Ground truth is the training loader's command chunk, never future measured
feedback. Stored images are already processed. This test therefore verifies
the model-side pipeline, not raw-camera acquisition or hardware timing.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from egomimic.rldb.embodiment.piper import Piper
from egomimic.rldb.zarr.piper_dataset import PiperDataset
from egomimic.robot.piper_pico.live_policy import (
    ActuatorPolicy,
    _json,
    load_policy_contract,
)
from egomimic.robot.piper_pico.policy_execution import resample_action_block
from egomimic.robot.piper_pico.policy_inference import (
    ACTION,
    CAMERA_KEYS,
    DOMAIN,
    EMBODIMENT,
    STATE,
)
from egomimic.robot.piper_pico.policy_rollout import OfflineReplayClock
from egomimic.scripts.replay_piper_policy import RecordedObservationProvider

SIDES = ("left", "right")
UNITS = ("position_mm", "rotation_deg", "gripper_mm")


def sample_indices(length, horizon, count):
    """Evenly cover the episode; do not pad, repeat indices or score fake future."""
    valid = int(length) - int(horizon) + 1
    if valid < 1:
        return []
    return np.unique(
        np.rint(np.linspace(0, valid - 1, min(count, valid))).astype(int)
    ).tolist()


def physical_errors(prediction, target):
    """Euclidean translation, SO(3) geodesic rotation and physical gripper width."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    if (
        prediction.shape != target.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 14
    ):
        raise ValueError("error inputs must have matching (N,14) shapes")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("error inputs must be finite")
    result = {}
    for i, side in enumerate(SIDES):
        start = 7 * i
        a = Rotation.from_rotvec(prediction[:, start + 3 : start + 6])
        b = Rotation.from_rotvec(target[:, start + 3 : start + 6])
        result[side] = {
            "position_mm": np.linalg.norm(
                prediction[:, start : start + 3] - target[:, start : start + 3], axis=1
            )
            * 1000,
            "rotation_deg": np.rad2deg((a.inv() * b).magnitude()),
            "gripper_mm": np.abs(prediction[:, start + 6] - target[:, start + 6])
            * 1000,
        }
    return result


def describe(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values):
        return None
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "max": float(values.max()),
    }


def summarize_errors(errors):
    return {
        side: {unit: describe(values) for unit, values in fields.items()}
        for side, fields in errors.items()
    }


def direct_command_targets(store, index, timing, open_m, offsets):
    """Independent base-frame oracle from original command rows, no extrinsics.

    Use the same temporal support declared by training, but compute position /
    quaternion interpolation directly with NumPy and SciPy. This checks the
    training camera transform and runtime inverse without sharing those helpers.
    """
    indices = np.asarray(timing.support_indices)
    times = indices / timing.source_fps
    offsets = np.asarray(offsets, dtype=float)
    if offsets.min() < 0 or offsets.max() > times[-1] + 1e-9:
        raise ValueError("target times are outside the recorded command support")
    result = np.empty((len(offsets), 14))
    for i, side in enumerate(SIDES):
        pose = np.asarray(
            store[f"{side}.cmd_ee_pose"][index : index + timing.source_horizon]
        )[indices]
        width = (
            np.asarray(
                store[f"{side}.cmd_gripper"][index : index + timing.source_horizon]
            ).reshape(-1)[indices]
            * open_m
        )
        start = i * 7
        for dimension in range(3):
            result[:, start + dimension] = np.interp(offsets, times, pose[:, dimension])
        rotations = Rotation.from_quat(pose[:, [4, 5, 6, 3]])
        result[:, start + 3 : start + 6] = Slerp(times, rotations)(offsets).as_rotvec()
        result[:, start + 6] = np.interp(offsets, times, width)
    return result


def check_input_parity(policy, frame, sample):
    prepared = policy.adapter.prepare(frame)[EMBODIMENT]
    state = prepared[STATE][0].numpy()
    expected = sample[STATE].numpy()
    state_error = float(np.max(np.abs(state - expected)))
    if not np.allclose(state, expected, atol=2e-6, rtol=2e-6):
        raise ValueError(
            f"Pipeline mismatch: inference vs training state, max abs={state_error}"
        )
    images = {}
    for camera, key in CAMERA_KEYS.items():
        actual = prepared[key][0].float().numpy() / 255.0
        expected = sample[key].numpy()
        if actual.shape != expected.shape:
            raise ValueError(f"Pipeline mismatch: {camera} image shape")
        difference = float(np.max(np.abs(actual - expected)))
        if difference > 1e-7:
            raise ValueError(f"Pipeline mismatch: {camera} image max abs={difference}")
        images[camera] = difference
    return {"state_max_abs": state_error, "image_max_abs_0_to_1": images}


class EvaluationLog:
    def __init__(self, directory, args):
        self.directory = Path(directory).expanduser().resolve()
        if any(
            parent.suffix == ".zarr"
            for parent in (self.directory, *self.directory.parents)
        ):
            raise ValueError("Evaluation output cannot be inside a source Zarr store")
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "samples").mkdir()
        (self.directory / "plots").mkdir()
        (self.directory / "arguments.json").write_text(_json(vars(args), indent=2))
        self.events = (self.directory / "events.jsonl").open("x", buffering=1)
        self.metrics = (self.directory / "samples.jsonl").open("x", buffering=1)
        self.manifest = None

    def write_manifest(self, manifest):
        self.manifest = manifest
        manifest["mode"] = "offline_recorded_input_evaluation_no_hardware"
        (self.directory / "manifest.json").write_text(_json(manifest, indent=2))

    def event(self, name, **fields):
        self.events.write(
            _json({"event": name, "walltime": time.time(), **fields}) + "\n"
        )

    def close(self):
        self.events.close()
        self.metrics.close()


def plot_sample(output, name, frame, target, predicted, offsets, label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 11), layout="constrained")
    grid = fig.add_gridspec(3, 6, height_ratios=[1, 1.5, 1])
    for i, camera in enumerate(("left_wrist", "front_1", "right_wrist")):
        ax = fig.add_subplot(grid[0, 2 * i : 2 * i + 2])
        ax.imshow(frame.images[camera])
        ax.set_title(camera + " (training pixels)")
        ax.axis("off")
    for i, side in enumerate(SIDES):
        ax = fig.add_subplot(grid[1, 3 * i : 3 * i + 3], projection="3d")
        start = 7 * i
        for rows, color, title in (
            (target, "#222222", "Recorded command"),
            (predicted, "#d34e24", "Policy command"),
        ):
            xyz = rows[:, start : start + 3] * 1000
            ax.plot(*xyz.T, color=color, label=title, linewidth=2)
            ax.scatter(*xyz[0], color=color, s=20)
        ax.scatter(
            *(frame.base_state[start : start + 3] * 1000),
            color="#1a8d7d",
            label="Measured state",
            s=35,
        )
        ax.set(
            xlabel="X (mm)",
            ylabel="Y (mm)",
            zlabel="Z (mm)",
            title=f"{side}: its own base frame",
        )
        ax.legend(fontsize=8)
    errors = physical_errors(predicted, target)
    for i, unit in enumerate(UNITS):
        ax = fig.add_subplot(grid[2, 2 * i : 2 * i + 2])
        for side, color in zip(SIDES, ("#218380", "#d34e24")):
            ax.plot(offsets, errors[side][unit], color=color, label=side)
        ax.set(xlabel="Time from chunk start (s)", ylabel=unit.replace("_", " "))
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(
        label
        + "\nOne recorded observation -> one predicted chunk; not a closed-loop rollout",
        fontsize=12,
    )
    fig.savefig(output / "plots" / f"{name}.png", dpi=130)
    plt.close(fig)


def plot_horizon(output, accumulated, offsets):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 6), layout="constrained")
    for i, side in enumerate(SIDES):
        for j, unit in enumerate(UNITS):
            ax = axes[i, j]
            for key, color, label in (
                ("raw_model", "#d34e24", "Policy"),
                ("hold_current_model", "#218380", "Hold current state"),
            ):
                values = np.stack(accumulated[key][side][unit])
                ax.plot(offsets, values.mean(axis=0), label=label, color=color)
            ax.set(title=f"{side} / {unit}", xlabel="Prediction offset (s)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle(
        "Mean recorded-command error by prediction horizon (sampled training inputs)"
    )
    fig.savefig(output / "error_by_horizon.png", dpi=150)
    plt.close(fig)


def evaluate(args, log):
    contract, policy, _ = load_policy_contract(args, log)
    actuator = ActuatorPolicy(policy, policy.adapter.gripper_open_m, log)
    episodes = contract["audit"]["episodes"]
    if args.eval_max_episodes and len(episodes) > args.eval_max_episodes:
        selected = np.rint(
            np.linspace(0, len(episodes) - 1, args.eval_max_episodes)
        ).astype(int)
        episodes = [episodes[i] for i in selected]
    plan, skipped = [], []
    for episode in episodes:
        path = Path(episode["episode"])
        metadata_file = Path(episode["metadata"]["path"])
        metadata = json.loads(metadata_file.read_text())
        metadata = metadata.get("attributes", metadata)
        indices = sample_indices(
            metadata["total_frames"],
            policy.timing.source_horizon,
            args.eval_samples_per_episode,
        )
        if not indices:
            skipped.append(str(path))
        plan.extend((path, index) for index in indices)
    if not plan:
        raise ValueError(
            "No full action windows: episodes shorter than checkpoint source_horizon"
        )
    plot_indices = set(
        np.rint(np.linspace(0, len(plan) - 1, min(args.eval_plots, len(plan)))).astype(
            int
        )
    )
    aggregate, stratified = {}, {}
    max_input = {"state": 0.0, "images": 0.0}
    oracle_max = {unit: 0.0 for unit in UNITS}
    parity_diffs, times = [], []
    previous, loader, provider = None, None, None
    for number, (path, index) in enumerate(plan):
        if path != previous:
            provider = RecordedObservationProvider(path, OfflineReplayClock())
            loader = PiperDataset(
                path,
                Piper.get_keymap(timing=policy.timing),
                Piper.get_transform_list(timing=policy.timing),
            )
            previous = path
        frame = provider.read_frame(index)
        sample = loader[index]
        parity = check_input_parity(policy, frame, sample)
        max_input["state"] = max(max_input["state"], parity["state_max_abs"])
        max_input["images"] = max(
            max_input["images"], *parity["image_max_abs_0_to_1"].values()
        )
        target_camera = sample[ACTION].numpy()
        target_base = policy.adapter.camera_to_base_actions(target_camera)
        direct = direct_command_targets(
            provider.store,
            index,
            policy.timing,
            policy.adapter.gripper_open_m,
            policy.timing.output_offsets_s,
        )
        oracle = physical_errors(target_base, direct)
        for unit, tolerance in (
            ("position_mm", 0.002),
            ("rotation_deg", 0.002),
            ("gripper_mm", 0.002),
        ):
            maximum = max(float(oracle[side][unit].max()) for side in SIDES)
            oracle_max[unit] = max(oracle_max[unit], maximum)
            if maximum > tolerance:
                raise ValueError(
                    f"Pipeline mismatch: label camera->base vs direct command oracle: {unit}={maximum}"
                )
        seed_text = f"{args.eval_seed}:{path}:{index}".encode()
        seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:4], "little")
        torch.manual_seed(seed)
        prediction = actuator.predict(frame)
        raw_base = policy.adapter.camera_to_base_actions(prediction["camera_actions"])
        times.append(float(prediction["inference_elapsed_s"]))
        prediction_parity = None
        if number < args.eval_parity_samples:
            # A second path starts from the real training loader's observations.
            # Labels are kept OUT of the network input on both paths.
            batch = {key: sample[key][None] for key in (STATE, *CAMERA_KEYS.values())}
            torch.manual_seed(seed)
            expected = (
                policy.model.predict_from_observations(
                    {EMBODIMENT: batch}, num_inference_steps=args.inference_steps
                )[f"{DOMAIN}_{ACTION}"]
                .detach()
                .cpu()
                .numpy()[0]
            )
            prediction_parity = float(
                np.max(np.abs(expected - prediction["camera_actions"]))
            )
            parity_diffs.append(prediction_parity)
            if not np.allclose(
                expected, prediction["camera_actions"], atol=1e-4, rtol=1e-5
            ):
                raise ValueError(
                    f"Pipeline mismatch: training-input vs inference output: {prediction_parity}"
                )
        block = resample_action_block(
            prediction["base_actions"],
            policy.timing,
            args.control_frequency,
            execute_steps=args.execute_steps,
        )
        target_commands = direct_command_targets(
            provider.store,
            index,
            policy.timing,
            policy.adapter.gripper_open_m,
            block.offsets_s,
        )
        target_roundtrip = resample_action_block(
            target_base,
            policy.timing,
            args.control_frequency,
            execute_steps=args.execute_steps,
        )
        hold = np.repeat(frame.base_state[None], len(target_base), axis=0)
        metrics = {
            "raw_model": physical_errors(raw_base, target_base),
            "hold_current_model": physical_errors(hold, target_base),
            "model_first_action": physical_errors(raw_base[:1], target_base[:1]),
            "executed_prefix": physical_errors(block.actions, target_commands),
            "hold_current_prefix": physical_errors(
                np.repeat(frame.base_state[None], len(block.actions), axis=0),
                target_commands,
            ),
            "resampling_only": physical_errors(
                target_roundtrip.actions, target_commands
            ),
        }
        for group, errors in metrics.items():
            bucket = aggregate.setdefault(
                group, {side: {unit: [] for unit in UNITS} for side in SIDES}
            )
            for side in SIDES:
                for unit in UNITS:
                    bucket[side][unit].append(errors[side][unit])
        movement = metrics["hold_current_model"]
        motion_flags = {}
        for side in SIDES:
            moving = (
                movement[side]["position_mm"].max() > 5
                or movement[side]["rotation_deg"].max() > 3
                or movement[side]["gripper_mm"].max() > 5
            )
            motion_flags[side] = bool(moving)
            bucket = stratified.setdefault(side, {}).setdefault(
                "moving" if moving else "near_static", {}
            )
            for group in ("raw_model", "hold_current_model"):
                dest = bucket.setdefault(group, {unit: [] for unit in UNITS})
                for unit in UNITS:
                    dest[unit].append(metrics[group][side][unit])
        name = f"{number:05d}_{path.stem}_frame_{index:06d}"
        np.savez_compressed(
            log.directory / "samples" / f"{name}.npz",
            measured_base_state=frame.base_state,
            predicted_camera=prediction["camera_actions"],
            target_camera=target_camera,
            predicted_base_raw=raw_base,
            target_base=target_base,
            command_base=block.actions,
            target_command_base=target_commands,
            output_offsets_s=np.asarray(policy.timing.output_offsets_s),
            command_offsets_s=block.offsets_s,
        )
        row = {
            "sample": name,
            "episode": str(path),
            "frame_index": index,
            "recorded_timestamp": frame.timestamp,
            "source_timestamps": dict(frame.source_timestamps),
            "seed": seed,
            "inference_elapsed_s": times[-1],
            "input_parity": parity,
            "prediction_parity_max_abs": prediction_parity,
            "moving": motion_flags,
            "metrics": {key: summarize_errors(value) for key, value in metrics.items()},
        }
        log.metrics.write(_json(row) + "\n")
        if number in plot_indices:
            plot_sample(
                log.directory,
                name,
                frame,
                target_commands,
                block.actions,
                block.offsets_s,
                f"{path.parent.name}/{path.stem}, frame {index}",
            )
        if number == 0 or (number + 1) % 8 == 0 or number + 1 == len(plan):
            print(
                f"[离线验证] {number+1}/{len(plan)}，{path.stem} frame {index}",
                flush=True,
            )
    summaries = {
        group: {
            side: {
                unit: describe(np.concatenate(values))
                for unit, values in fields.items()
            }
            for side, fields in bucket.items()
        }
        for group, bucket in aggregate.items()
    }
    motion_summary = {
        side: {
            kind: {
                group: {
                    unit: describe(np.concatenate(values))
                    for unit, values in fields.items()
                }
                for group, fields in groups.items()
            }
            for kind, groups in kinds.items()
        }
        for side, kinds in stratified.items()
    }
    summary = {
        "status": "completed",
        "pipeline_checks": "passed",
        "hardware_opened": False,
        "sample_count": len(plan),
        "episode_count": len({str(path) for path, _ in plan}),
        "skipped_short_episodes": skipped,
        "model_output_length": policy.timing.output_chunk_length,
        "command_prefix_length": len(block.actions),
        "timing": policy.timing.to_dict(),
        "input_max_abs": max_input,
        "label_inverse_oracle_max": oracle_max,
        "prediction_parity_samples": len(parity_diffs),
        "prediction_parity_max_abs": max(parity_diffs, default=None),
        "metrics": summaries,
        "motion_stratified": motion_summary,
        "motion_thresholds": {
            "max_position_from_current_mm": 5,
            "max_rotation_from_current_deg": 3,
            "max_gripper_from_current_mm": 5,
            "rule": "any threshold; descriptive, not accuracy criteria",
        },
        "inference_elapsed_s": describe(times),
        "scope": {
            "ground_truth": "training command labels, not future observed pose",
            "images": "stored processed training RGB; no second undistortion",
            "sampling": "uniform episode/frame selection, complete unpadded windows only",
            "seeds": "fixed per episode/frame; one stochastic FM draw, no best-of-N selection",
            "metrics": "per-point pooled; windows can overlap, not independent trials",
            "not_tested": [
                "raw camera acquisition",
                "CAN and robot execution",
                "physical latency",
                "closed-loop success",
                "held-out generalization",
            ],
            "training_membership": "user-selected training-data; not proven by checkpoint",
        },
    }
    plot_horizon(log.directory, aggregate, policy.timing.output_offsets_s)
    return summary


def write_report(directory, summary):
    lines = [
        "# PiPER 离线推理验证",
        "",
        f"Pipeline 检查：{summary['pipeline_checks']}。",
        f"采样 {summary['episode_count']} 集、{summary['sample_count']} 个完整动作窗口；未连接硬件。",
        "",
        "输入是训练集已处理图像与实际 obs state；标签是同索引的未来 command，未输入模型。",
        f"模型输出 {summary['model_output_length']} 点，首尾跨度 {summary['timing']['effective_span_s']:.6f} s；执行前缀 {summary['command_prefix_length']} 点（只计算，不下发）。",
        f"图像输入最大差 {summary['input_max_abs']['images']:.3g}；state 最大差 {summary['input_max_abs']['state']:.3g}。",
        f"固定种子输出对照 {summary['prediction_parity_samples']} 个窗口，最大差 {summary['prediction_parity_max_abs']:.3g}。",
        "",
        "| 范围 / 手臂 | 位置均值 / P95 (mm) | 旋转均值 / P95 (°) | 夹爪均值 / P95 (mm) |",
        "| --- | --- | --- | --- |",
    ]
    labels = {
        "raw_model": "原始模型输出",
        "hold_current_model": "保持当前状态基线",
        "model_first_action": "首个模型动作",
        "executed_prefix": "重采样及夹爪截断后",
        "resampling_only": "仅重采样误差",
    }
    for group, title in labels.items():
        for side in SIDES:
            cells = [
                f"{summary['metrics'][group][side][unit]['mean']:.3f} / {summary['metrics'][group][side][unit]['p95']:.3f}"
                for unit in UNITS
            ]
            lines.append(f"| {title} / {side} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "按每个手臂分别分组，以下是原始模型输出的均值。一个窗口内标签相对当前反馈的最大位移超过 5 mm、旋转超过 3° 或夹爪变化超过 5 mm，任一满足即计为运动；这是描述性分组，不是精度验收标准。",
        "",
        "| 手臂 / 分组 | 窗口数 | 位置：模型 / 保持状态 (mm) | 旋转：模型 / 保持状态 (°) | 夹爪：模型 / 保持状态 (mm) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for side in SIDES:
        for kind, title in (("moving", "运动"), ("near_static", "近静止")):
            groups = summary["motion_stratified"][side].get(kind)
            if not groups:
                continue
            count = (
                groups["raw_model"]["position_mm"]["count"]
                // summary["model_output_length"]
            )
            cells = [
                f"{groups['raw_model'][unit]['mean']:.3f} / {groups['hold_current_model'][unit]['mean']:.3f}"
                for unit in UNITS
            ]
            lines.append(f"| {side} / {title} | {count} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Pipeline passed 只表示输入一致性、坐标逆变换和固定随机种子的模型输出对照通过，不代表任务成功率或模型误差已达标。",
        "基线可能在基本不动的机械臂上很强，不能只看总体均值。训练集拟合不代表未见数据泛化；这些抽样窗口可以重叠，不是独立成功率试验。FM 每个窗口固定种子单次预测，未挑选多次预测中的最佳结果。",
        "本报告验证已处理图像到模型动作的流程，不覆盖原始相机采集、CAN 下发和闭环效果。所选数据是否用于训练由调用者保证，旧 checkpoint 没有绑定训练文件清单。",
        "CPU/GPU 推理耗时不含机械臂执行，离线结果不证明真机能连续 30 Hz。",
        "",
        "![按预测时间的误差](error_by_horizon.png)",
        "",
        "plots/ 是抽样输入图像和双臂 base 下的轨迹对比；samples/ 保存可复算的预测、标签和时间轴。",
    ]
    (directory / "report.md").write_text("\n".join(lines) + "\n")


def run_evaluation(args):
    output = args.output_dir or (
        Path(__file__).resolve().parents[3]
        / "diagnostics"
        / f"piper_offline_eval_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    )
    log = EvaluationLog(output, args)
    summary = {
        "status": "failed",
        "pipeline_checks": "incomplete",
        "hardware_opened": False,
    }
    code = 1
    print(f"[离线验证] 输出：{log.directory}", flush=True)
    try:
        if args.execute or args.home_on_start:
            raise ValueError(
                "--offline-eval cannot be combined with --execute or --home-on-start true"
            )
        if args.max_chunks is not None:
            raise ValueError(
                "Offline evaluation uses --eval-max-episodes and --eval-samples-per-episode, not --max-chunks"
            )
        if str(args.device) == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        summary = evaluate(args, log)
        write_report(log.directory, summary)
        code = 0
        print(
            f"[离线验证] Pipeline passed；完成 {summary['sample_count']} 个窗口。",
            flush=True,
        )
    except KeyboardInterrupt:
        summary.update(status="interrupted", pipeline_checks="incomplete")
        code = 130
    except Exception as exc:
        summary.update(status="failed", pipeline_checks="incomplete")
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"[离线验证] 未通过：{exc}", file=sys.stderr, flush=True)
    finally:
        (log.directory / "summary.json").write_text(_json(summary, indent=2))
        log.close()
    return code
