"""Repeat static inputs for both arms and audit the worst recorded hold failures."""

import argparse
import atexit
import json
from pathlib import Path

from _bootstrap import setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--per-side", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--worst-repeats", type=int, default=20)
    args = parser.parse_args()
    if args.per_side < 1 or args.repeats < 2 or args.worst_repeats < 2:
        parser.error("Positive count and at least two repeats required")
    setup(cpu=True, openpi_root=args.openpi_root)
    import numpy as np
    import zarr
    from egoverse_pi05.assets import load_inference_assets
    from egoverse_pi05.evaluation import (
        describe,
        repeat_metrics,
        select_static_repeats,
        trajectory_metrics,
    )
    from egoverse_pi05.inference import BimanualInferencePolicy

    from egomimic.robot.piper_pico.pi05_data import Pi05Dataset
    from egomimic.robot.piper_pico.pi05_remote import WebSocketRPC
    from egomimic.robot.piper_pico.policy_execution import resample_action_block
    from egomimic.robot.piper_pico.policy_inference import PiperObservationAdapter

    root = args.evaluation
    output = root / "hold_audit.json"
    if output.exists():
        raise FileExistsError(output)
    summary = json.loads((root / "summary.json").read_text())
    arguments = json.loads((root / "arguments.json").read_text())
    rows = [
        json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()
    ]
    assets = load_inference_assets(
        summary["checkpoint"], summary["num_inference_steps"]
    )
    dataset = Pi05Dataset(arguments["manifest"], split="all")
    if dataset.manifest != assets.metadata["manifest"]:
        raise ValueError("Manifest differs")
    selected = select_static_repeats(rows, args.per_side)
    worst = {
        side: max(
            (i for i, r in enumerate(rows) if r["static"][side]["static"]),
            key=lambda i: rows[i]["metrics"]["pi05_30"][side]["max_from_current_mm"][
                "mean"
            ],
            default=None,
        )
        for side in ("left", "right")
    }
    rpc = WebSocketRPC(args.server_url, timeout_s=120)
    atexit.register(rpc.close)
    if rpc.metadata != assets.metadata:
        raise ValueError("Server checkpoint does not match")
    policy = BimanualInferencePolicy(rpc, assets.metadata)
    results, adapters = [], {}
    (root / "hold_repeats").mkdir(exist_ok=True)
    for index in sorted(set(selected) | {i for i in worst.values() if i is not None}):
        row = rows[index]
        episode = row["episode"]
        if episode not in adapters:
            adapters[episode] = PiperObservationAdapter(
                dict(zarr.open_group(str(dataset._paths[episode]), mode="r").attrs),
                allow_synthetic_calibration=dataset.manifest[
                    "allow_synthetic_calibration"
                ],
            )
        adapter = adapters[episode]
        sample = dataset[row["dataset_index"]]
        sample.pop("actions")
        with np.load(root / "samples" / (row["sample"] + ".npz")) as original:
            state, label = original["measured_base_state"], original["target_30"]
        count = args.worst_repeats if index in worst.values() else args.repeats
        commands = []
        for _ in range(count):
            base = adapter.camera_to_base_actions(policy.infer(sample)["actions"])
            base[:, [6, 13]] = np.clip(base[:, [6, 13]], 0, adapter.gripper_open_m)
            commands.append(
                resample_action_block(
                    base, dataset.timing, dataset.timing.source_fps
                ).actions
            )
        entry = dict(
            sample=row["sample"],
            static=row["static"],
            repetitions=count,
            representative=index in selected,
            worst_for=[side for side, i in worst.items() if i == index],
            metrics={},
        )
        for side_index, side in enumerate(("left", "right")):
            values = [trajectory_metrics(c, label, state, side_index) for c in commands]
            maxima = [v["max_from_current_mm"][0] for v in values]
            entry["metrics"][side] = {
                **repeat_metrics(commands, side_index),
                "max_drift_mm": describe(maxima),
                "drift_over_10mm_count": int(np.count_nonzero(np.asarray(maxima) > 10)),
                "drift_over_20mm_count": int(np.count_nonzero(np.asarray(maxima) > 20)),
            }
        results.append(entry)
        np.savez_compressed(
            root / "hold_repeats" / (row["sample"] + ".npz"),
            commands=np.asarray(commands),
            state=state,
            target=label,
        )
        print(
            f"Repeated {len(results)} inputs: {row['sample']}, {count} samples",
            flush=True,
        )
    rpc.close()
    report = {
        "checkpoint": summary["checkpoint"],
        "per_side_requested": args.per_side,
        "scope": "Fixed recorded input; representative windows across episodes and separately selected worst cases. Fresh stochastic predictions, no closed-loop simulation.",
        "results": results,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
