"""Controlled offline interventions on RTC prefixes and recorded observations.

These inputs are intentionally inconsistent for diagnosis, not a robot rollout.
All variants retain the same noise, checkpoint and activation clock. No devices.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from _bootstrap import setup


def select_moving_pairs(rows):
    """One middle eligible moving-left pair per episode, without ranking errors."""
    groups = {}
    for row in rows:
        if not row["static"]["left"] and row["current_row"] >= 3:
            groups.setdefault(row["episode"], []).append(row)
    return [group[len(group) // 2] for group in groups.values()]


def observation_variant(current, older, kind):
    """Return copied observations, perturbing only the explicitly named input."""
    result = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in current.items()
    }
    if kind in ("state_100ms_older", "inputs_100ms_older"):
        result["observation.state"] = older["observation.state"].copy()
    if kind in ("images_100ms_older", "inputs_100ms_older"):
        for key in result:
            if key.startswith("observation.images."):
                result[key] = older[key].copy()
    return result


def translated_actions(actions, distance_m):
    result = np.array(actions, dtype=float, copy=True)
    if result.ndim != 2 or result.shape[1] != 14 or not np.isfinite(result).all():
        raise ValueError("Expected finite N x 14 base actions")
    if not np.isfinite(distance_m):
        raise ValueError("Translation must be finite")
    result[:, 0] += distance_m
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    setup()
    from egoverse_pi05.assets import load_inference_assets
    from egoverse_pi05.evaluation import describe
    from egoverse_pi05.inference import create_policy
    from egoverse_pi05.rtc import RtcConfig
    from egoverse_pi05.rtc_evaluation import BaseCameraAdapter, evaluate_handoff

    from egomimic.robot.piper_pico.pi05_data import Pi05Dataset
    from egomimic.robot.piper_pico.policy_continuity import conditioned_plan
    from egomimic.robot.piper_pico.policy_rtc import build_rtc_prefix

    original = json.loads((args.baseline_evaluation / "arguments.json").read_text())
    rows = [
        json.loads(line)
        for line in (args.baseline_evaluation / "samples.jsonl")
        .read_text()
        .splitlines()
    ]
    selected = select_moving_pairs(rows)
    assets = load_inference_assets(
        Path(original["checkpoint_path"]), original["num_inference_steps"]
    )
    dataset = Pi05Dataset(Path(original["manifest"]), split="all", include_images=True)
    if dataset.state_history_offsets_s:
        raise ValueError("This intervention is defined for the no-history baseline")
    if dataset.manifest != assets.metadata["manifest"]:
        raise ValueError("Training manifest mismatch")
    policy = create_policy(
        assets,
        require_single_gpu=True,
        rtc_config=RtcConfig(original["rtc_max_guidance"]),
    )
    variants = {
        "unchanged": (0, 0, False),
        "prefix_left_x_plus10mm": (0.010, 0, False),
        "prefix_left_x_plus30mm": (0.030, 0, False),
        "prefix_left_x_minus30mm": (-0.030, 0, False),
        "prefix_left_x_plus30mm_full_overlap_weight": (0.030, 0, True),
        "state_left_x_plus30mm": (0, 0.030, False),
        "state_left_x_minus30mm": (0, -0.030, False),
        "state_100ms_older": (0, 0, False),
        "images_100ms_older": (0, 0, False),
        "inputs_100ms_older": (0, 0, False),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "selection.json").write_text(json.dumps(selected, indent=2))
    (args.output_dir / "arguments.json").write_text(
        json.dumps(
            {"baseline": str(args.baseline_evaluation), "source": original}, indent=2
        )
    )
    pools, baseline_error = {}, []
    started = time.monotonic()
    with (args.output_dir / "samples.jsonl").open("w") as output:
        for index, row in enumerate(selected):
            ep, current = row["episode"], row["current_row"]
            contract = dataset.episodes[ep]["contract"]
            adapter = BaseCameraAdapter(
                contract["extrinsics"], contract["gripper_encoding"]["open_m"]
            )
            sample = dataset[dataset.index_for_source(ep, current)]
            sample.pop("actions")
            older = dataset[dataset.index_for_source(ep, current - 3)]
            older.pop("actions")
            path = (
                args.baseline_evaluation
                / "samples"
                / f"pair_{row['pair_index']:05d}.npz"
            )
            with np.load(path) as stored:
                saved = {key: stored[key] for key in stored.files}
            detail = {"pair_index": row["pair_index"], "episode": ep, "variants": {}}
            arrays = {}
            for name, (prefix_shift, state_shift, full_weight) in variants.items():
                old_actions = translated_actions(
                    saved["previous_prediction"], prefix_shift
                )
                old = conditioned_plan(
                    old_actions,
                    dataset.timing,
                    row["previous_anchor"],
                    original["smoothing_ms"] / 1000,
                )
                prefix = build_rtc_prefix(
                    old,
                    dataset.timing,
                    row["current_time"],
                    row["observation_to_send_s"],
                    row["hard_until"],
                    row["pair_index"] + 1,
                )
                weights = prefix["weights"].astype(np.float32)
                if full_weight:
                    weights = (weights > 0).astype(np.float32)
                obs = observation_variant(sample, older, name)
                if state_shift:
                    state = saved["current_base_state"].copy()
                    state[0] += state_shift
                    obs["observation.state"] = adapter.base_to_camera_state(
                        state
                    ).astype(np.float32)
                obs["rtc"] = {
                    "schema": prefix["schema"],
                    "request_id": prefix["request_id"],
                    "previous_actions": np.stack(
                        [
                            adapter.base_to_camera_state(p)
                            for p in prefix["base_actions"]
                        ]
                    ).astype(np.float32),
                    "weights": weights,
                }
                actions = adapter.camera_to_base_actions(
                    policy.infer(obs, noise=saved["noise_current"])["actions"]
                )
                result = evaluate_handoff(
                    old,
                    actions,
                    dataset.timing,
                    row["new_anchor"],
                    row["activation"],
                    smoothing_s=original["smoothing_ms"] / 1000,
                    transition_s=original["transition_ms"] / 1000,
                    control_frequency=original["control_frequency"],
                    reference_base_actions=saved["recorded_target"],
                    current_base_state=saved["current_base_state"],
                )
                detail["variants"][name] = result["metrics"]
                arrays[name] = actions
                if name == "unchanged":
                    baseline_error.append(
                        float(np.max(np.abs(actions - saved["rtc_self_actions"])))
                    )
                for side, metrics in result["metrics"].items():
                    for key, value in metrics.items():
                        pools.setdefault((name, side, key), []).append(value)
            output.write(json.dumps(detail) + "\n")
            output.flush()
            np.savez_compressed(
                args.output_dir / f"pair_{row['pair_index']:05d}.npz", **arrays
            )
            print(
                f"{index + 1}/{len(selected)} episodes; {time.monotonic() - started:.1f}s",
                flush=True,
            )
    summary = {
        "scope": "Controlled inconsistent-prefix/input interventions on recorded training observations; not real feedback rollout or proof of real-scene distribution shift",
        "selected_pairs": len(selected),
        "variants": {},
        "baseline_max_abs_action_reproduction_error": max(baseline_error),
        "older_inputs": "3 source rows earlier, nominally 100ms; action clock and noise retained",
    }
    for (variant, side, key), values in pools.items():
        summary["variants"].setdefault(variant, {}).setdefault(side, {})[key] = (
            describe(values)
        )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Completed", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
