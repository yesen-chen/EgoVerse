"""CPU-only paired report for the three recorded-observation RTC experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MODELS = ("baseline", "history_33ms", "history_66_33ms")
LABELS = ("Current only", "History -33 ms", "History -67/-33 ms")
MODES = ("unguided_self", "rtc_self", "rtc_shared_recorded")
SIDES = ("left", "right")
RUNTIME_FIELDS = (
    "pairs_per_episode",
    "execute_steps",
    "delay_ms",
    "transition_ms",
    "smoothing_ms",
    "control_frequency",
    "num_inference_steps",
    "rtc_max_guidance",
    "seed",
)
CORE_METRICS = (
    "raw_switch_mm",
    "raw_switch_deg",
    "raw_prefix_max_mm",
    "raw_target_position_rmse_mm",
    "transition_d2_p95_mm",
    "raw_internal_d2_p95_mm",
    "raw_max_from_current_mm",
)


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def load_paired_runs(root):
    """Reject reordered, incomplete, differently timed, or unpaired experiments."""
    root = Path(root)
    runs = {}
    for name in MODELS:
        folder = root / name
        selection = read_json(folder / "selection.json")
        summary = read_json(folder / "summary.json")
        arguments = read_json(folder / "arguments.json")
        samples = [
            json.loads(line)
            for line in (folder / "samples.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if not selection or len(samples) != len(selection):
            raise ValueError(f"{name}: incomplete or empty sample selection")
        if summary.get("selection_sha256") != canonical_hash(selection):
            raise ValueError(f"{name}: selection_sha256 mismatch")
        ids = [(p["episode"], p["previous_row"], p["current_row"]) for p in selection]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name}: duplicate pair IDs")
        episodes = len({p["episode"] for p in selection})
        if summary.get("pairs") != len(samples) or summary.get("episodes") != episodes:
            raise ValueError(f"{name}: summary sample/episode count mismatch")
        for index, (selected, row) in enumerate(zip(selection, samples, strict=True)):
            if row.get("pair_index") != index or any(
                row.get(key) != value for key, value in selected.items()
            ):
                raise ValueError(
                    f"{name}: sample ID/timestamp mismatch at pair {index}"
                )
            if set(row.get("static", {})) != set(SIDES) or any(
                not isinstance(row["static"][side], bool) for side in SIDES
            ):
                raise ValueError(f"{name}: invalid static flags at pair {index}")
            if set(row.get("metrics", {})) != set(MODES):
                raise ValueError(f"{name}: missing evaluation modes at pair {index}")
            for mode in MODES:
                for side in SIDES:
                    metrics = row["metrics"][mode][side]
                    if not set(CORE_METRICS).issubset(metrics):
                        raise ValueError(f"{name}: missing core metrics")
                    if any(
                        not np.isfinite(value) or value < 0
                        for value in metrics.values()
                    ):
                        raise ValueError(f"{name}: nonfinite or negative metrics")
        runtime = {key: arguments[key] for key in RUNTIME_FIELDS}
        run = {
            "folder": str(folder.resolve()),
            "selection": selection,
            "summary": summary,
            "samples": samples,
            "runtime": runtime,
            "checkpoint": arguments["checkpoint_path"],
        }
        if runs:
            baseline = runs[MODELS[0]]
            if selection != baseline["selection"]:
                raise ValueError(f"{name}: paired selection differs from baseline")
            if (
                runtime != baseline["runtime"]
                or summary["timing"] != baseline["summary"]["timing"]
            ):
                raise ValueError(f"{name}: runtime timing/noise parameters differ")
            if [r["static"] for r in samples] != [
                r["static"] for r in baseline["samples"]
            ]:
                raise ValueError(f"{name}: static sample definitions differ")
            for row, reference in zip(samples, baseline["samples"], strict=True):
                for mode in MODES:
                    for side in SIDES:
                        if set(row["metrics"][mode][side]) != set(
                            reference["metrics"][mode][side]
                        ):
                            raise ValueError(f"{name}: metric definitions differ")
        runs[name] = run
    return runs


def paired_episode_bootstrap(reference, candidate, episodes, *, repeats=2000, seed=42):
    """Paired candidate-minus-baseline means, resampling entire episodes.

    Each episode has equal weight; sample windows inside an episode never become
    independent bootstrap draws. Static subsets may contain fewer than 44 groups.
    """
    reference, candidate = np.asarray(reference, float), np.asarray(candidate, float)
    episodes = np.asarray(episodes)
    if (
        reference.ndim != 1
        or reference.shape != candidate.shape
        or reference.shape != episodes.shape
    ):
        raise ValueError("paired bootstrap requires matching one-dimensional arrays")
    if (
        repeats < 1
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(candidate))
    ):
        raise ValueError("invalid bootstrap samples or repeats")
    groups = np.unique(episodes)
    if len(groups) == 0:
        return {"episodes": 0, "pairs": 0, "mean_difference": None, "ci95": None}
    group_differences = np.asarray(
        [
            (candidate[episodes == group] - reference[episodes == group]).mean()
            for group in groups
        ]
    )
    draws = np.random.default_rng(seed).integers(0, len(groups), (repeats, len(groups)))
    means = group_differences[draws].mean(axis=1)
    return {
        "episodes": len(groups),
        "pairs": len(reference),
        "mean_difference": float(group_differences.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "episode_differences": group_differences.tolist(),
    }


def describe(values):
    values = np.asarray(values, float)
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def build_comparison(runs, *, repeats=2000, seed=42):
    baseline = runs[MODELS[0]]
    samples = baseline["samples"]
    episodes = np.array([row["episode"] for row in samples])
    result = {
        "schema": "pi05_rtc_history_comparison_v1",
        "selection_sha256": baseline["summary"]["selection_sha256"],
        "pairs": len(samples),
        "episodes": len(np.unique(episodes)),
        "runtime": baseline["runtime"],
        "timing": baseline["summary"]["timing"],
        "checkpoints": {key: value["checkpoint"] for key, value in runs.items()},
        "bootstrap": {
            "seed": seed,
            "repeats": repeats,
            "ci": "percentile 95%",
            "resampling_unit": "episode",
            "estimand": "equal-episode mean of paired window differences",
            "direction": "candidate minus baseline; negative means smaller metric",
        },
        "definitions": {
            "raw_switch_mm": "old/new targets at the same activation time, before client smoothing/blending",
            "transition_d2_p95_mm": "per-window P95 of position second differences; table mean averages these window P95 values, not a global P95",
            "raw_max_from_current_mm": "maximum predicted displacement from current measured pose; static subset only for a holding interpretation",
            "scope": "recorded training observations; paired two-chunk offline experiment, not a robot closed-loop rollout or held-out generalization estimate",
        },
        "static_counts": {},
        "metrics": {},
        "paired_differences": {},
    }
    for side in SIDES:
        static = np.array([row["static"][side] for row in samples])
        result["static_counts"][side] = {
            "pairs": int(static.sum()),
            "episodes": len(np.unique(episodes[static])),
        }
    for scope in ("all", "static", "moving"):
        result["metrics"][scope] = {}
        result["paired_differences"][scope] = {}
        for mode in MODES:
            aggregates, comparisons = {}, {}
            for side in SIDES:
                mask = np.array(
                    [
                        scope == "all" or row["static"][side] == (scope == "static")
                        for row in samples
                    ]
                )
                aggregates[side], comparisons[side] = {}, {}
                metric_names = samples[0]["metrics"][mode][side]
                for metric in metric_names:
                    arrays = {
                        name: np.array(
                            [
                                row["metrics"][mode][side][metric]
                                for row in run["samples"]
                            ]
                        )[mask]
                        for name, run in runs.items()
                    }
                    aggregates[side][metric] = {
                        name: describe(array) for name, array in arrays.items()
                    }
                    if metric in CORE_METRICS:
                        comparisons[side][metric] = {
                            name: paired_episode_bootstrap(
                                arrays[MODELS[0]],
                                arrays[name],
                                episodes[mask],
                                repeats=repeats,
                                seed=seed,
                            )
                            for name in MODELS[1:]
                        }
            result["metrics"][scope][mode] = aggregates
            result["paired_differences"][scope][mode] = comparisons
    return result


def format_number(value):
    return "n/a" if value is None else f"{value:.3f}"


def write_tables(result, path):
    lines = [
        "# Pi0.5 paired RTC history comparison",
        "",
        f"Identical selection: {result['pairs']} pairs from {result['episodes']} episodes.",
        "",
        "All table entries are means over windows unless explicitly labelled. Raw switch gaps compare targets at the same time. Transition d2 is the **mean of per-window P95 values**, not the P95 of all trajectory points.",
        "",
        f"Bootstrap: {result['bootstrap']['repeats']:,} draws of whole episodes, paired candidate minus baseline; negative means a smaller metric. Episodes receive equal weight. These intervals describe this recorded-training-data comparison, not unseen task success.",
        "",
        "Runtime parameters: `" + json.dumps(result["runtime"], sort_keys=True) + "`.",
        "",
    ]
    columns = (
        ("raw_switch_mm", "Raw switch mm"),
        ("raw_switch_deg", "Raw switch deg"),
        ("raw_prefix_max_mm", "Prefix max mm"),
        ("raw_target_position_rmse_mm", "Target RMSE mm"),
        ("transition_d2_p95_mm", "Transition d2 window-P95 mm"),
        ("raw_internal_d2_p95_mm", "Internal d2 window-P95 mm"),
    )
    for mode in MODES:
        lines += [
            f"## {mode}",
            "",
            "| Model | Arm | " + " | ".join(title for _, title in columns) + " |",
            "|---|---|" + "---:|" * len(columns),
        ]
        for name, label in zip(MODELS, LABELS, strict=True):
            for side in SIDES:
                cells = [
                    format_number(
                        result["metrics"]["all"][mode][side][key][name]["mean"]
                    )
                    for key, _ in columns
                ]
                lines.append(f"| {label} | {side} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "Paired episode-bootstrap differences from current-only baseline:",
            "",
            "| Model | Arm | Metric | Mean difference | 95% CI | Episodes |",
            "|---|---|---|---:|---|---:|",
        ]
        for name, label in zip(MODELS[1:], LABELS[1:], strict=True):
            for side in SIDES:
                for metric, title in columns:
                    stats = result["paired_differences"]["all"][mode][side][metric][
                        name
                    ]
                    lo, hi = stats["ci95"]
                    lines.append(
                        f"| {label} | {side} | {title} | {stats['mean_difference']:.3f} | [{lo:.3f}, {hi:.3f}] | {stats['episodes']} |"
                    )
        lines += [
            "",
            "Static subset (flags are identical across models):",
            "",
            "| Model | Arm | Windows | Episodes | Raw max from current mm | Raw target RMSE mm | Raw switch mm |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for name, label in zip(MODELS, LABELS, strict=True):
            for side in SIDES:
                count = result["static_counts"][side]
                cells = [
                    format_number(
                        result["metrics"]["static"][mode][side][key][name]["mean"]
                    )
                    for key in (
                        "raw_max_from_current_mm",
                        "raw_target_position_rmse_mm",
                        "raw_switch_mm",
                    )
                ]
                lines.append(
                    f"| {label} | {side} | {count['pairs']} | {count['episodes']} | "
                    + " | ".join(cells)
                    + " |"
                )
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n")


def plot_comparison(runs, result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#444444", "#2477b5", "#cc7331")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for axis, side in zip(axes[0], SIDES, strict=True):
        for name, label, color in zip(MODELS, LABELS, colors, strict=True):
            x = np.sort(
                [
                    row["metrics"]["rtc_self"][side]["raw_switch_mm"]
                    for row in runs[name]["samples"]
                ]
            )
            axis.step(
                x,
                np.arange(1, len(x) + 1) / len(x),
                where="post",
                color=color,
                label=label,
            )
        axis.set(
            title=f"{side.title()} arm: raw same-time handoff gap",
            xlabel="Position gap (mm)",
            ylabel="Fraction of paired windows",
        )
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9, loc="lower right")
    axis = axes[1, 0]
    for model_index, name in enumerate(MODELS[1:]):
        for side_index, side in enumerate(SIDES):
            stats = result["paired_differences"]["all"]["rtc_self"][side][
                "raw_switch_mm"
            ][name]
            mean, (lo, hi) = stats["mean_difference"], stats["ci95"]
            y = model_index * 2 + side_index
            axis.plot([lo, hi], [y, y], color=colors[model_index + 1], linewidth=2)
            axis.scatter([mean], [y], color=colors[model_index + 1], s=45, zorder=3)
    axis.set_yticks(
        range(4), [f"{label}, {side}" for label in LABELS[1:] for side in SIDES]
    )
    axis.axvline(0, color="black", linewidth=1, linestyle="--")
    axis.set(
        title="Raw gap: paired mean change and 95% CI",
        xlabel="Change vs current only (mm); negative = smaller",
    )
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.2)
    axis = axes[1, 1]
    for model_index, (name, label, color) in enumerate(
        zip(MODELS, LABELS, colors, strict=True)
    ):
        values = [
            result["metrics"]["all"]["rtc_self"][side]["transition_d2_p95_mm"][name][
                "mean"
            ]
            for side in SIDES
        ]
        axis.bar(
            np.arange(2) + (model_index - 1) * 0.24,
            values,
            width=0.23,
            color=color,
            label=label,
        )
    axis.set_xticks([0, 1], ["Left", "Right"])
    axis.set(
        title="After identical smoothing and transition",
        ylabel="Mean of window-P95 position second differences (mm)",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(fontsize=9)
    fig.suptitle(
        f"RTC self-prefix | {result['pairs']} paired windows / {result['episodes']} episodes\nOffline training-observation comparison; episode-grouped bootstrap",
        fontsize=13,
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    runs = load_paired_runs(args.root)
    result = build_comparison(runs, repeats=args.bootstrap_repeats, seed=args.seed)
    (args.root / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    write_tables(result, args.root / "comparison_tables.md")
    plot_comparison(runs, result, args.root / "comparison_plot.png")
    print(
        f"Verified {result['pairs']} paired windows / {result['episodes']} episodes; report written to {args.root}"
    )


if __name__ == "__main__":
    main()
