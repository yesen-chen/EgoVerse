"""Render a completed offline evaluation; needs only the CPU plotting environment."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"hpt_30": "#d97706", "pi05_30": "#2563eb", "target_30": "#16936a"}
LABELS = {
    "hpt_30": "HPT epoch 49",
    "pi05_30": "Pi0.5 step 12000",
    "target_30": "Recorded target",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    root = args.evaluation
    summary = json.loads((root / "summary.json").read_text())
    rows = [
        json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()
    ]
    LABELS["pi05_30"] = f"Pi0.5 step {summary['checkpoint_directory_label']}"
    metrics = summary["metrics"]
    sides = ("left", "right")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    settings = [
        ("all", "position_error_mm", "All windows: mean position error", "mm"),
        (
            "all",
            "second_difference_mm",
            "Within chunk: mean second difference",
            "mm / sample step",
        ),
        (
            "static",
            "max_from_current_mm",
            "Static: mean of each window's max drift",
            "mm from observed pose",
        ),
    ]
    for ax, (scope, key, title, unit) in zip(axes, settings):
        x = np.arange(2)
        for j, model in enumerate(COLORS):
            values = [metrics[scope][model][side][key]["mean"] for side in sides]
            bars = ax.bar(
                x + (j - 1) * 0.24,
                values,
                width=0.24,
                color=COLORS[model],
                label=LABELS[model],
            )
            ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
        ax.set(xticks=x, xticklabels=["Left", "Right"], ylabel=unit, title=title)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"Matched {len(rows)} recorded inputs; 30 control points / 0.967 s; no smoothing",
        fontsize=13,
    )
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)

    chosen = {}
    for category, worst in (
        ("static_median", False),
        ("static_worst", True),
        ("moving_median", False),
    ):
        fig, axes = plt.subplots(3, 2, figsize=(13, 8), layout="constrained")
        chosen[category] = {}
        for side_index, side in enumerate(sides):
            if category.startswith("static"):
                eligible = [r for r in rows if r["static"][side]["static"]]
                key = "max_from_current_mm"
            else:
                eligible = [
                    r
                    for r in rows
                    if r["static"][side]["cmd"]["max_position_from_first_mm"] > 10
                ]
                key = "position_error_mm"
            if not eligible:
                continue
            ordered = sorted(
                eligible, key=lambda r: r["metrics"]["pi05_30"][side][key]["mean"]
            )
            selected = ordered[-1] if worst else ordered[len(ordered) // 2]
            chosen[category][side] = selected["sample"]
            with np.load(root / "samples" / (selected["sample"] + ".npz")) as a:
                state = a["measured_base_state"]
                time = np.linspace(
                    0, summary["timing"]["effective_span_s"], len(a["pi05_30"])
                )
                for dimension, coord in enumerate("XYZ"):
                    ax = axes[dimension, side_index]
                    for model in COLORS:
                        ax.plot(
                            time,
                            (
                                a[model][:, 7 * side_index + dimension]
                                - state[7 * side_index + dimension]
                            )
                            * 1000,
                            label=LABELS[model],
                            color=COLORS[model],
                            lw=1.6,
                        )
                    ax.axhline(0, color="gray", ls=":", label="Current measured pose")
                    ax.set(
                        xlabel="Time in chunk (s)",
                        ylabel=f"{coord} displacement from current (mm)",
                    )
                    ax.grid(alpha=0.2)
                axes[0, side_index].set_title(
                    f"{side}: {selected['sample']}\nPi0.5 {key}: {selected['metrics']['pi05_30'][side][key]['mean']:.2f}",
                    fontsize=9,
                )
                axes[0, side_index].legend(fontsize=8)
        fig.suptitle(
            f"{category.replace('_', ' ')}: independently selected per arm; recorded inputs"
        )
        fig.savefig(root / (category + ".png"), dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    hold_rates = {}
    for ax, side in zip(axes, sides):
        eligible = [r for r in rows if r["static"][side]["static"]]
        hold_rates[side] = {}
        for model in COLORS:
            values = np.sort(
                [
                    r["metrics"][model][side]["max_from_current_mm"]["mean"]
                    for r in eligible
                ]
            )
            ax.step(
                values,
                np.arange(1, len(values) + 1) / len(values),
                where="post",
                label=LABELS[model],
                color=COLORS[model],
            )
            hold_rates[side][model] = {
                f"within_{threshold}_mm_fraction": float((values <= threshold).mean())
                for threshold in (5, 10, 20)
            }
        ax.set(
            title=f"{side}: {len(eligible)} static windows",
            xlabel="Maximum predicted drift from observed pose (mm)",
            ylabel="Fraction of windows",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(root / "static_drift_distribution.png", dpi=160)
    plt.close(fig)

    audit = (
        json.loads((root / "hold_audit.json").read_text())
        if (root / "hold_audit.json").exists()
        else None
    )
    repeat_rows = (
        [r for r in audit["results"] if r["representative"]]
        if audit
        else summary["repeated_input_tests"]
    )
    repeats = {}
    for side in sides:
        valid = [r for r in repeat_rows if r["static"][side]["static"]]
        repeats[side] = {"windows": len(valid)}
        for key in ("position_std_rms_mm", "endpoint_std_rms_mm"):
            values = np.asarray([r["metrics"][side][key] for r in valid])
            repeats[side][key] = (
                {"mean": float(values.mean()), "max": float(values.max())}
                if len(values)
                else None
            )
    (root / "report_details.json").write_text(
        json.dumps(
            dict(hold_rates=hold_rates, repeated_static=repeats, examples=chosen),
            indent=2,
        )
        + "\n"
    )

    lines = [
        "# Pi0.5 离线动作与静止保持评估",
        "",
        f"权重：`{summary['checkpoint']}`。微调模式 {summary['finetune_mode']}；checkpoint 目录标识 `{summary['checkpoint_directory_label']}`，没有证据确认用户所述的 120000 步。",
        f"通过真实 server/client 推理入口运行，{summary['sample_count']} 个输入、{summary['episode_count']} 条轨迹，与历史 HPT epoch 49 使用相同录制帧和 state。Pi0.5 / HPT 分别为 {summary['num_inference_steps']} / {summary['reference_arguments']['inference_steps']} 步 flow 推理；各自随机采样，不能等同为相同噪声。",
        f"训练清单分组：{summary['split_counts']}。这些评估输入来自训练清单，不能据此衡量新场景成功率。",
        "没有连接机械臂或相机，没有平滑模型输出。保留原始 100 点；通过现有生产转换与时间重采样恢复到 30 点、约 0.967 秒。30 点夹爪按现有控制逻辑限幅，位置和姿态不修改。",
        "",
        "## 同输入比较",
        "",
        "位置二阶差分是相邻位移向量的变化，不等同于纯噪声；正常曲线也会贡献。这里用相同时间步长、同输入、同标签对照。",
        "",
        "| 范围 / 指标 | HPT 左 / 右 | Pi0.5 左 / 右 | 标签左 / 右 |",
        "| --- | --- | --- | --- |",
    ]
    for scope, label, key in (
        ("all", "全部位置误差 mm", "position_error_mm"),
        ("all", "全部旋转误差 °", "rotation_error_deg"),
        ("all", "全部夹爪误差 mm", "gripper_error_mm"),
        ("all", "全部二阶差分 mm", "second_difference_mm"),
        ("moving", "移动窗口位置误差 mm", "position_error_mm"),
        ("moving", "移动窗口二阶差分 mm", "second_difference_mm"),
        ("static", "静止窗口二阶差分 mm", "second_difference_mm"),
        ("static", "静止最大偏离当前位置 mm，逐窗均值", "max_from_current_mm"),
        ("static", "静止最大偏离当前位置 mm，逐窗 P95", "max_from_current_mm"),
        ("static", "静止最大偏离第一预测点 mm，逐窗均值", "max_from_first_mm"),
        ("static", "静止最大姿态偏离 °，逐窗均值", "max_rotation_from_current_deg"),
        ("static", "静止最大夹爪偏离 mm，逐窗均值", "max_gripper_from_current_mm"),
    ):
        statistic = "p95" if "P95" in label else "mean"
        cells = [
            " / ".join(
                f"{metrics[scope][model][side][key][statistic]:.3f}" for side in sides
            )
            for model in ("hpt_30", "pi05_30", "target_30")
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend(["", "原始 100 点二阶差分（左 / 右，mm）："])
    for model in ("hpt_100", "pi05_100", "target_100"):
        lines.append(
            f"- {model}: "
            + " / ".join(
                f"{metrics['all'][model][s]['second_difference_mm']['mean']:.3f}"
                for s in sides
            )
        )
    lines.extend(
        [
            "",
            "## 静止与重复预测",
            "",
            "静止筛选：完整原始窗口内，命令和实际反馈都满足相对首帧位置变化 ≤2 mm、旋转 ≤1°、夹爪开度 ≤1 mm；起点目标与反馈误差 ≤5 mm / 2° / 2 mm。排除结尾补帧。每只臂分别筛选，另一只臂可能在动。",
            "移动筛选：目标相对窗口首点的位置变化超过 10 mm。静止与移动集合不是互补集。",
            "",
        ]
    )
    for side in sides:
        lines.append(
            f"- {side}: 静止 {metrics['static']['pi05_30'][side]['windows']} 窗；移动 {metrics['moving']['pi05_30'][side]['windows']} 窗。"
        )
        for model in ("hpt_30", "pi05_30"):
            rate = hold_rates[side][model]
            lines.append(
                f"  - {LABELS[model]}: 全段位于当前姿态 5 / 10 / 20 mm 内的比例 "
                + " / ".join(
                    f"{rate[f'within_{t}_mm_fraction']:.1%}" for t in (5, 10, 20)
                )
                + "。"
            )
        r = repeats[side]
        if r["windows"]:
            lines.append(
                f"  - 同一输入重复 5 次，其中 {r['windows']} 个静止输入：逐点预测分散 RMS 的窗口均值 {r['position_std_rms_mm']['mean']:.3f} mm，最差 {r['position_std_rms_mm']['max']:.3f} mm。"
            )
    lines.append(
        f"\n其中双臂同时满足静止条件的输入仅 {sum(all(r['static'][s]['static'] for s in sides) for r in rows)} 个；多数结果是一只臂保持、另一只臂操作。"
    )
    if audit:
        lines.extend(
            [
                "",
                "## 最差静止案例复查",
                "",
                "左右臂各选 24 个跨 episode 的代表静止输入重复 5 次，另对主评估各臂最差案例重复 20 次。代表样本统计与最差案例分开报告；没有删掉失败结果。",
            ]
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for ax, side_index, side in zip(axes, range(2), sides):
            case = next(r for r in audit["results"] if side in r["worst_for"])
            m = case["metrics"][side]
            lines.append(
                f"- {side} `{case['sample']}`：重复 {case['repetitions']} 次，最大偏离均值 {m['max_drift_mm']['mean']:.2f} mm、最大 {m['max_drift_mm']['max']:.2f} mm；超过 10 mm 为 {m['drift_over_10mm_count']} 次，超过 20 mm 为 {m['drift_over_20mm_count']} 次。"
            )
            with np.load(root / "hold_repeats" / (case["sample"] + ".npz")) as data:
                predicted = data["commands"][:, :, 7 * side_index : 7 * side_index + 3]
                state = data["state"][7 * side_index : 7 * side_index + 3]
                time = np.linspace(
                    0, summary["timing"]["effective_span_s"], predicted.shape[1]
                )
                drift = np.linalg.norm(predicted - state, axis=-1) * 1000
                for curve in drift:
                    ax.plot(time, curve, color="#2563eb", alpha=0.3, lw=1)
                ax.plot(
                    time,
                    np.median(drift, axis=0),
                    color="#111827",
                    lw=2,
                    label="Median of repeats",
                )
                ax.plot(
                    time,
                    np.linalg.norm(
                        data["target"][:, 7 * side_index : 7 * side_index + 3] - state,
                        axis=1,
                    )
                    * 1000,
                    color="#16936a",
                    lw=2,
                    label="Recorded target",
                )
                ax.set(
                    title=f"{side}: {case['repetitions']} repeats of worst input",
                    xlabel="Time in chunk (s)",
                    ylabel="Distance from observed pose (mm)",
                )
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
        fig.savefig(root / "hold_repeats_worst.png", dpi=160)
        plt.close(fig)
        lines.extend(
            [
                "",
                "右臂失败轨迹主要是主动离开静止姿态，不能简单当作高频抖动。可能涉及动作启动时机/多种合理续接的随机选择，单凭这些离线结果不能断言原因。简单平滑也不能保证消除这种整体位移。",
                "",
                "![最差静止输入重复预测](hold_repeats_worst.png)",
            ]
        )
    lines.extend(
        [
            "",
            "固定输入重复预测只测生成随机性；未把预测动作回灌 state、未模拟物体变化，也未实机执行，因此不证明长期闭环稳定。偏移可能是平滑的目标偏置，不能全部叫抖动。",
            "",
            "## 核对与运行信息",
            "",
            f"- state / 标签最大绝对数值差：{summary['input_label_parity']}；均小于 2e-6 的输入一致性检查阈值。",
            f"- GPU 服务预热后客户端总请求耗时：中位数 {summary['warm_inference_seconds']['median']*1000:.1f} ms，P95 {summary['warm_inference_seconds']['p95']*1000:.1f} ms；包含三图传输和预处理，不是纯模型计时。",
            "- HPT epoch 49 与 Pi0.5 全量 checkpoint 的模型、训练步数和预训练不同；改进不能单独归因于 JAX 或 server/client 架构。",
            "- 新增指标的 CPU 单元测试覆盖固定偏置、直线与锯齿、旋转跨 ±180°、静止筛选、重复随机性及非法值。",
            "",
            "## 图像",
            "",
            "![汇总](comparison.png)",
            "",
            "![静止中位样本](static_median.png)",
            "",
            "![静止最差样本](static_worst.png)",
            "",
            "![移动中位样本](moving_median.png)",
            "",
            "![静止分布](static_drift_distribution.png)",
            "",
            "样例各臂独立选取：按 Pi0.5 静止最大偏移或移动误差排序，展示中位和最差；不是人工挑选好看的案例。原始逐窗结果见 `samples.jsonl`、`samples/*.npz`，汇总见 `summary.json`。",
        ]
    )
    (root / "report_zh.md").write_text("\n".join(lines) + "\n")
    print(root / "report_zh.md")


if __name__ == "__main__":
    main()
