"""Paired offline RTC handoffs on recorded observations; never opens devices.

Three conditions: unguided new plan vs its own preceding prediction; RTC with
that preceding prediction; RTC with a shared recorded-command prefix. The last
condition isolates prefix following, while the self-prefix condition includes
model-dependent preceding plans. Recorded observations do not react to outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from _bootstrap import setup


def paired_noise(seed, pair_index, role, *, horizon=100, dimension=32):
    import numpy as np

    return (
        np.random.default_rng(np.random.SeedSequence([seed, pair_index, role]))
        .normal(size=(horizon, dimension))
        .astype(np.float32)
    )


def select_pairs(dataset, count, steps, delay, transition, frequency):
    """Evenly span each episode, preserving original source indices/timestamps."""
    import numpy as np

    from egomimic.robot.piper_pico.policy_rtc import check_rtc_schedule

    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("pairs-per-episode must be a positive integer")
    check_rtc_schedule(dataset.timing, frequency, steps, delay, transition_s=transition)
    if not math.isclose(float(frequency), dataset.timing.source_fps, rel_tol=1e-12):
        raise ValueError(
            "Offline RTC source-row selection requires control frequency to equal "
            "the dataset source_fps"
        )
    result = []
    interval = 1 / frequency
    for ep, record in enumerate(dataset.episodes):
        stamps = dataset._numeric[ep]["timestamp"]
        available = set(record.get("eligible_indices", range(record["total_frames"])))
        send = float(record["contract"]["time_alignment"]["observation_to_send_s"])
        eligible = []
        for before in range(
            record["total_frames"] - steps - dataset.timing.source_horizon + 1
        ):
            after = before + steps
            if before not in available or after not in available:
                continue
            t0, t1 = float(stamps[before]), float(stamps[after])
            anchor = t0 + send
            activation = (
                anchor + math.ceil((t1 + delay - anchor) / interval - 1e-9) * interval
            )
            # Preserve the real source timing, but exclude recording stalls and
            # insufficient overlap explicitly instead of clamping trajectories.
            if abs(t1 - t0 - steps / frequency) > interval:
                continue
            if (
                activation + interval + dataset.timing.output_offsets_s[1] + transition
                >= anchor + dataset.timing.effective_span_s
            ):
                continue
            eligible.append(
                dict(
                    episode=ep,
                    episode_name=Path(record["path"]).name,
                    previous_row=before,
                    current_row=after,
                    previous_time=t0,
                    current_time=t1,
                    previous_anchor=anchor,
                    new_anchor=t1 + send,
                    activation=activation,
                    hard_until=activation + interval,
                    observation_to_send_s=send,
                )
            )
        if not eligible:
            continue
        indices = np.unique(
            np.linspace(0, len(eligible) - 1, min(count, len(eligible)), dtype=int)
        )
        result.extend(eligible[int(i)] for i in indices)
    if not result:
        raise ValueError("No complete, timing-valid pairs")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--pairs-per-episode", type=int, default=6)
    parser.add_argument("--execute-steps", type=int, default=27)
    parser.add_argument("--delay-ms", type=float, default=170)
    parser.add_argument("--transition-ms", type=float, default=300)
    parser.add_argument("--smoothing-ms", type=float, default=150)
    parser.add_argument("--control-frequency", type=float, default=30)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--rtc-max-guidance", type=float, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    setup(cpu=args.prepare_only, openpi_root=args.openpi_root)
    import numpy as np
    from egoverse_pi05.assets import load_inference_assets
    from egoverse_pi05.evaluation import describe, static_window

    from egomimic.robot.piper_pico.pi05_data import Pi05Dataset, canonical_hash
    from egomimic.robot.piper_pico.policy_continuity import conditioned_plan
    from egomimic.robot.piper_pico.policy_rtc import build_rtc_prefix

    assets = load_inference_assets(args.checkpoint_path, args.num_inference_steps)
    dataset = Pi05Dataset(args.manifest, split="all", include_images=True)
    if dataset.manifest != assets.metadata["manifest"]:
        raise ValueError("Expected the exact checkpoint training manifest")
    pairs = select_pairs(
        dataset,
        args.pairs_per_episode,
        args.execute_steps,
        args.delay_ms / 1000,
        args.transition_ms / 1000,
        args.control_frequency,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "selection.json").write_text(json.dumps(pairs, indent=2))
    (args.output_dir / "arguments.json").write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            indent=2,
        )
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(assets.metadata, indent=2)
    )
    print(
        f"Verified {len(pairs)} pairs in {len(set(p['episode'] for p in pairs))} episodes; history={dataset.state_history_offsets_s}",
        flush=True,
    )
    if args.prepare_only:
        return
    from egoverse_pi05.inference import create_policy
    from egoverse_pi05.rtc import RtcConfig
    from egoverse_pi05.rtc_evaluation import BaseCameraAdapter, evaluate_handoff

    policy = create_policy(
        assets, require_single_gpu=True, rtc_config=RtcConfig(args.rtc_max_guidance)
    )
    adapters, pools, elapsed = {}, {}, []
    (args.output_dir / "samples").mkdir()
    started_all = time.monotonic()
    modes = ("unguided_self", "rtc_self", "rtc_shared_recorded")
    with (args.output_dir / "samples.jsonl").open("w") as output:
        for number, pair in enumerate(pairs):
            episode, current = pair["episode"], pair["current_row"]
            if episode not in adapters:
                contract = dataset.episodes[episode]["contract"]
                adapters[episode] = BaseCameraAdapter(
                    contract["extrinsics"], contract["gripper_encoding"]["open_m"]
                )
            adapter = adapters[episode]
            previous_sample = dataset[
                dataset.index_for_source(episode, pair["previous_row"])
            ]
            sample = dataset[dataset.index_for_source(episode, current)]
            previous_target = adapter.camera_to_base_actions(
                previous_sample.pop("actions")
            )
            target = adapter.camera_to_base_actions(sample.pop("actions"))
            current_state = adapter.camera_to_base_actions(
                sample["observation.state"][None]
            )[0]
            noise = paired_noise(args.seed, number, 1)

            def predict(observation, noise_array):
                begin = time.monotonic()
                actions = policy.infer(observation, noise=noise_array)["actions"]
                elapsed.append(time.monotonic() - begin)
                return adapter.camera_to_base_actions(actions)

            previous = predict(previous_sample, paired_noise(args.seed, number, 0))
            own = conditioned_plan(
                previous,
                dataset.timing,
                pair["previous_anchor"],
                args.smoothing_ms / 1000,
            )
            shared = conditioned_plan(
                previous_target,
                dataset.timing,
                pair["previous_anchor"],
                args.smoothing_ms / 1000,
            )
            plain = predict(sample, noise)
            detail = {
                "pair_index": number,
                **pair,
                "metrics": {},
                "static": {
                    side: static_window(
                        dataset._numeric[episode],
                        side,
                        current,
                        dataset.timing.source_horizon,
                        adapter.gripper_open_m,
                    )["static"]
                    for side in ("left", "right")
                },
            }
            arrays = {
                "previous_prediction": previous,
                "previous_recorded_target": previous_target,
                "recorded_target": target,
                "current_base_state": current_state,
                "noise_current": noise,
                "noise_previous": paired_noise(args.seed, number, 0),
            }
            if dataset.state_history_offsets_s:
                arrays["current_camera_history"] = sample["observation.state_history"]
            for mode in modes:
                old = shared if mode == "rtc_shared_recorded" else own
                if mode == "unguided_self":
                    actions = plain
                else:
                    prefix = build_rtc_prefix(
                        old,
                        dataset.timing,
                        pair["current_time"],
                        pair["observation_to_send_s"],
                        pair["hard_until"],
                        number + 1,
                    )
                    request = {
                        "schema": prefix["schema"],
                        "request_id": prefix["request_id"],
                        "previous_actions": np.stack(
                            [
                                adapter.base_to_camera_state(row)
                                for row in prefix["base_actions"]
                            ]
                        ).astype(np.float32),
                        "weights": prefix["weights"].astype(np.float32),
                    }
                    actions = predict(dict(sample, rtc=request), noise)
                    arrays[f"{mode}_prefix"] = prefix["base_actions"]
                    arrays[f"{mode}_weights"] = prefix["weights"]
                result = evaluate_handoff(
                    old,
                    actions,
                    dataset.timing,
                    pair["new_anchor"],
                    pair["activation"],
                    smoothing_s=args.smoothing_ms / 1000,
                    transition_s=args.transition_ms / 1000,
                    control_frequency=args.control_frequency,
                    reference_base_actions=target,
                    current_base_state=current_state,
                )
                detail["metrics"][mode] = result["metrics"]
                arrays[f"{mode}_actions"] = actions
                arrays.update(
                    {f"{mode}_{key}": value for key, value in result["traces"].items()}
                )
                for side in ("left", "right"):
                    scopes = ("all", "static" if detail["static"][side] else "moving")
                    for scope in scopes:
                        for key, value in result["metrics"][side].items():
                            pools.setdefault((scope, mode, side, key), []).append(value)
            np.savez_compressed(
                args.output_dir / "samples" / f"pair_{number:05d}.npz", **arrays
            )
            output.write(json.dumps(detail) + "\n")
            output.flush()
            if number == 0 or (number + 1) % 12 == 0:
                print(
                    f"{number + 1}/{len(pairs)} pairs; elapsed {time.monotonic() - started_all:.1f}s",
                    flush=True,
                )
    summary = {
        "pairs": len(pairs),
        "episodes": len(adapters),
        "selection_sha256": canonical_hash(pairs),
        "history_offsets_s": list(dataset.state_history_offsets_s),
        "metrics": {},
        "inference_seconds_including_compilation": describe(elapsed),
        "scope": "offline paired two-chunk replanning on recorded training observations; not a closed-loop rollout",
        "shared_prefix": "recorded previous command targets, smoothed identically; not new prediction labels",
        "noise": "same explicit Gaussian noise per pair and role across checkpoints and new-plan modes",
        "timing": dataset.timing.to_dict(),
        "total_seconds": time.monotonic() - started_all,
    }
    for (scope, mode, side, key), values in pools.items():
        summary["metrics"].setdefault(scope, {}).setdefault(mode, {}).setdefault(
            side, {}
        )[key] = describe(values)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Completed {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
