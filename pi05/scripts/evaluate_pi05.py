"""Compare Pi0.5 with saved HPT predictions on identical recorded input windows.

Uses the production Pi0.5 inference factory and robot coordinate/time adapters.
Never opens a camera, CAN device or robot; no smoothing is applied.
"""

import argparse
import atexit
import json
import time
from pathlib import Path

from _bootstrap import setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--server-url", default="ws://127.0.0.1:8000")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--repeat-windows", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 2 or args.repeat_windows < 0:
        parser.error("repeats >= 2 and repeat-windows >= 0 required")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    setup(cpu=True, openpi_root=args.openpi_root)
    import numpy as np
    import zarr
    from egoverse_pi05.assets import load_inference_assets
    from egoverse_pi05.evaluation import (
        aggregate_metrics,
        describe,
        repeat_metrics,
        select_static_repeats,
        static_window,
        trajectory_metrics,
    )
    from egoverse_pi05.inference import BimanualInferencePolicy

    from egomimic.robot.piper_pico.pi05_data import Pi05Dataset
    from egomimic.robot.piper_pico.pi05_remote import WebSocketRPC
    from egomimic.robot.piper_pico.policy_execution import resample_action_block
    from egomimic.robot.piper_pico.policy_inference import PiperObservationAdapter

    assets = load_inference_assets(args.checkpoint_path, args.num_inference_steps)
    dataset = Pi05Dataset(args.manifest, split="all", include_images=True)
    if dataset.manifest != assets.metadata["manifest"]:
        raise ValueError("Dataset must exactly match checkpoint training manifest")
    reference_summary = json.loads((args.reference_eval / "summary.json").read_text())
    if reference_summary["timing"] != dataset.timing.to_dict():
        raise ValueError("HPT reference timing differs")
    rows = [
        json.loads(line)
        for line in (args.reference_eval / "samples.jsonl").read_text().splitlines()
    ]
    paths = {path.resolve(): i for i, path in enumerate(dataset._paths)}
    starts = np.r_[0, dataset._ends[:-1]]
    adapters = {}
    for i, path in enumerate(dataset._paths):
        metadata = dict(zarr.open_group(str(path), mode="r").attrs)
        adapters[i] = PiperObservationAdapter(
            metadata,
            allow_synthetic_calibration=dataset.manifest["allow_synthetic_calibration"],
        )
    windows = []
    for row in rows:
        episode = paths[Path(row["episode"]).resolve()]
        index = row["frame_index"]
        if (
            index + dataset.timing.source_horizon
            > dataset.episodes[episode]["total_frames"]
        ):
            raise ValueError("Reference includes padded episode-end windows")
        static = {
            side: static_window(
                dataset._numeric[episode],
                side,
                index,
                dataset.timing.source_horizon,
                adapters[episode].gripper_open_m,
            )
            for side in ("left", "right")
        }
        windows.append(
            dict(
                sample=row["sample"],
                episode=episode,
                frame_index=index,
                dataset_index=int(starts[episode] + index),
                static=static,
            )
        )
    selected = set(select_static_repeats(windows, args.repeat_windows // 2))
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "samples").mkdir()
    (args.output_dir / "arguments.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            indent=2,
        )
    )
    (args.output_dir / "selection.json").write_text(json.dumps(windows, indent=2))
    print(
        f"Verified {len(windows)} matching windows; {len(selected)} repeated inputs. Connecting to model service.",
        flush=True,
    )
    rpc = WebSocketRPC(args.server_url, timeout_s=120)
    atexit.register(rpc.close)
    if rpc.metadata != assets.metadata:
        rpc.close()
        raise ValueError(
            "Server checkpoint/steps/assets do not match evaluation request"
        )
    policy = BimanualInferencePolicy(rpc, assets.metadata)
    pools = {
        scope: {
            model: {side: [] for side in ("left", "right")}
            for model in (
                "pi05_100",
                "hpt_100",
                "target_100",
                "pi05_30",
                "hpt_30",
                "target_30",
            )
        }
        for scope in ("all", "static", "moving")
    }
    repeats_report, durations = [], []
    parity = {"state_max_abs": 0.0, "target_max_abs": 0.0}
    clip_counts = np.zeros(2, dtype=int)
    detail_file = (args.output_dir / "samples.jsonl").open("w")
    for n, window in enumerate(windows):
        sample = dataset[window["dataset_index"]]
        target = sample.pop("actions")
        adapter = adapters[window["episode"]]
        target_base = adapter.camera_to_base_actions(target)
        state_base = adapter.camera_to_base_actions(sample["observation.state"][None])[
            0
        ]
        with np.load(
            args.reference_eval / "samples" / (window["sample"] + ".npz")
        ) as old:
            old = {k: old[k] for k in old.files}
        for name, a, b in (
            ("state_max_abs", state_base, old["measured_base_state"]),
            ("target_max_abs", target_base, old["target_base"]),
        ):
            error = float(np.max(np.abs(a - b)))
            parity[name] = max(parity[name], error)
            if error > 2e-6:
                raise ValueError(
                    f"Same-input/label parity failed: {window['sample']}, {name}={error}"
                )
        begin = time.monotonic()
        predicted = policy.infer(sample)["actions"]
        elapsed = time.monotonic() - begin
        durations.append(elapsed)
        base = adapter.camera_to_base_actions(predicted)
        bounded = base.copy()
        bounded[:, [6, 13]] = np.clip(base[:, [6, 13]], 0, adapter.gripper_open_m)
        clip_counts += np.count_nonzero(base[:, [6, 13]] != bounded[:, [6, 13]], axis=0)
        command = resample_action_block(
            bounded, dataset.timing, dataset.timing.source_fps
        ).actions.copy()
        # Reuse the precisely matched historical labels for a paired comparison.
        trajectories = {
            "pi05_100": base,
            "hpt_100": old["predicted_base_raw"],
            "target_100": old["target_base"],
            "pi05_30": command,
            "hpt_30": old["command_base"],
            "target_30": old["target_command_base"],
        }
        detail = {**window, "elapsed_s": elapsed, "metrics": {}}
        for model, trajectory in trajectories.items():
            label = (
                old["target_base"]
                if model.endswith("100")
                else old["target_command_base"]
            )
            detail["metrics"][model] = {}
            for side_index, side in enumerate(("left", "right")):
                m = trajectory_metrics(trajectory, label, state_base, side_index)
                pools["all"][model][side].append(m)
                if window["static"][side]["static"]:
                    pools["static"][model][side].append(m)
                if window["static"][side]["cmd"]["max_position_from_first_mm"] > 10:
                    pools["moving"][model][side].append(m)
                detail["metrics"][model][side] = aggregate_metrics([m])
        repeat_raw, repeat_cmd = [base], [command]
        if n in selected:
            for _ in range(args.repeats - 1):
                more = adapter.camera_to_base_actions(policy.infer(sample)["actions"])
                bounded_more = more.copy()
                bounded_more[:, [6, 13]] = np.clip(
                    more[:, [6, 13]], 0, adapter.gripper_open_m
                )
                more_command = resample_action_block(
                    bounded_more, dataset.timing, dataset.timing.source_fps
                ).actions.copy()
                repeat_raw.append(more)
                repeat_cmd.append(more_command)
            repeat = {
                "sample": window["sample"],
                "static": window["static"],
                "metrics": {
                    side: repeat_metrics(repeat_cmd, i)
                    for i, side in enumerate(("left", "right"))
                },
            }
            repeats_report.append(repeat)
        np.savez_compressed(
            args.output_dir / "samples" / (window["sample"] + ".npz"),
            predicted_camera=predicted,
            target_camera=target,
            measured_base_state=state_base,
            **trajectories,
            repeated_base_raw=np.asarray(repeat_raw),
            repeated_command=np.asarray(repeat_cmd),
        )
        detail_file.write(json.dumps(detail) + "\n")
        detail_file.flush()
        if n % 25 == 0 or n + 1 == len(windows):
            print(
                f"Evaluated {n+1}/{len(windows)}, latest {elapsed*1000:.1f} ms",
                flush=True,
            )
    detail_file.close()
    result = {
        "checkpoint": str(assets.checkpoint_path),
        "checkpoint_directory_label": assets.checkpoint_path.name,
        "reference_eval": str(args.reference_eval.resolve()),
        "reference_arguments": json.loads(
            (args.reference_eval / "arguments.json").read_text()
        ),
        "manifest_sha256": dataset.manifest["manifest_sha256"],
        "finetune_mode": assets.metadata["finetune_mode"],
        "num_inference_steps": args.num_inference_steps,
        "sample_count": len(windows),
        "episode_count": len(set(w["episode"] for w in windows)),
        "split_counts": {
            s: sum(r["split"] == s for r in dataset.episodes)
            for s in ("train", "validation")
        },
        "timing": dataset.timing.to_dict(),
        "input_label_parity": parity,
        "hardware_opened": False,
        "smoothing": False,
        "first_rpc_seconds_server_already_warmed": durations[0],
        "warm_inference_seconds": describe(durations[1:]),
        "gripper_clipped_raw100_points_left_right": clip_counts.tolist(),
        "static_definition": "Full native window, BOTH command and actual feedback excursions <=2mm/1deg/1mm grip; initial tracking gap <=5mm/2deg/2mm grip. No episode-end padding.",
        "moving_definition": "Command position excursion from first point >10mm over the native window, separately for each arm.",
        "scope": "Offline teacher-forced recorded inputs. Repeated identical-input predictions measure sampling variation, not closed-loop stability or task success. Second difference includes intended curvature; compare same timestamps and labels.",
        "metrics": {
            scope: {
                model: {side: aggregate_metrics(items) for side, items in sides.items()}
                for model, sides in models.items()
            }
            for scope, models in pools.items()
        },
        "repeated_input_tests": repeats_report,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    rpc.close()
    print(f"Saved {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
