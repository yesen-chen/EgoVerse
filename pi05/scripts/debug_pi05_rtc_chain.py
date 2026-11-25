"""Compare independent RTC pairs with recursive RTC on recorded observations.

This is teacher-forced input replay, NOT robot simulation or closed-loop policy
evaluation: the measured state and images always come from the recording. Only
the previous planned commands are propagated recursively. This isolates whether
model-side RTC prefix recursion itself changes the handoff behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from _bootstrap import setup


def activation_time(request_time, first_control_tick, frequency, delay_s):
    """Ideal production control grid, without OS wakeup or send-duration drift."""
    if (
        not all(
            math.isfinite(float(v))
            for v in (request_time, first_control_tick, frequency, delay_s)
        )
        or frequency <= 0
        or delay_s <= 0
    ):
        raise ValueError("Finite times and positive frequency/delay required")
    interval = 1 / frequency
    return (
        first_control_tick
        + math.ceil((request_time + delay_s - first_control_tick) / interval - 1e-9)
        * interval
    )


def select_chains(
    dataset,
    *,
    max_episodes=8,
    max_handoffs=32,
    steps=27,
    delay_s=0.170,
    transition_s=0.300,
    frequency=30,
    start_delay_s=0,
):
    """Select complete, continuous chains; rank episodes by commanded travel."""
    import numpy as np

    from egomimic.robot.piper_pico.policy_rtc import check_rtc_schedule

    if any(
        isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1
        for v in (max_episodes, max_handoffs)
    ):
        raise ValueError("Episode and handoff limits must be positive integers")
    if not math.isfinite(start_delay_s) or start_delay_s < 0:
        raise ValueError("start delay must be finite and nonnegative")
    check_rtc_schedule(
        dataset.timing, frequency, steps, delay_s, transition_s=transition_s
    )
    if not math.isclose(frequency, dataset.timing.source_fps, rel_tol=1e-12):
        raise ValueError(
            "Control frequency must match source fps for row-strided replay"
        )
    selected = []
    dt = 1 / frequency
    for episode, record in enumerate(dataset.episodes):
        arrays = dataset._numeric[episode]
        timestamps = np.asarray(arrays["timestamp"], dtype=float)
        eligible = set(record.get("eligible_indices", range(record["total_frames"])))
        complete = record["total_frames"] - dataset.timing.source_horizon + 1
        first = next((i for i in range(complete) if i in eligible), None)
        if first is None:
            continue
        rows = [first]
        send = float(record["contract"]["time_alignment"]["observation_to_send_s"])
        first_tick = float(timestamps[first]) + send + start_delay_s
        for row in range(first + steps, complete, steps):
            if len(rows) - 1 >= max_handoffs or row not in eligible:
                break
            if (
                abs(float(timestamps[row] - timestamps[rows[-1]]) - steps / frequency)
                > dt
            ):
                break
            activation = activation_time(
                float(timestamps[row]), first_tick, frequency, delay_s
            )
            old_anchor = (
                first_tick if len(rows) == 1 else float(timestamps[rows[-1]]) + send
            )
            new_anchor = float(timestamps[row]) + send
            # evaluate_handoff requires two samples before entry and two after
            # transition. No incomplete/padded prefix or label enters the test.
            trace_end = activation + (math.ceil(transition_s / dt) + 2) * dt
            if activation - 2 * dt < max(old_anchor, new_anchor) - 1e-9:
                break
            if (
                trace_end
                > min(old_anchor, new_anchor) + dataset.timing.effective_span_s + 1e-9
            ):
                break
            rows.append(row)
        if len(rows) < 2:
            continue
        end = rows[-1] + dataset.timing.source_horizon
        motion = sum(
            float(
                np.linalg.norm(
                    np.diff(
                        np.asarray(arrays[f"{s}.cmd_ee_pose"])[first:end, :3], axis=0
                    ),
                    axis=1,
                ).sum()
            )
            for s in ("left", "right")
        )
        selected.append(
            dict(
                episode=episode,
                episode_name=Path(record["path"]).name,
                rows=rows,
                timestamps=timestamps[rows].tolist(),
                observation_to_send_s=send,
                first_control_tick=first_tick,
                commanded_travel_m=motion,
            )
        )
    selected.sort(key=lambda r: (-r["commanded_travel_m"], r["episode"]))
    if not selected:
        raise ValueError("No complete timing-valid continuous chains")
    return selected[:max_episodes]


def replay_chain(
    samples,
    timestamps,
    *,
    predict,
    adapter,
    timing,
    observation_to_send_s,
    delay_s=0.170,
    transition_s=0.300,
    smoothing_s=0.150,
    frequency=30,
    start_delay_s=0,
    seed=42,
    episode=0,
):
    """Yield paired metrics and arrays. ``predict(observation, noise)`` returns base actions.

    The independent condition uses the preceding ordinary prediction as its
    prefix; the recursive condition uses its own previously RTC-guided result.
    Current input and explicit Gaussian noise match both conditions exactly.
    Initial old plans are identical. No output state is used as a later input.
    """
    import numpy as np
    from egoverse_pi05.rtc_evaluation import evaluate_handoff

    from egomimic.robot.piper_pico.policy_continuity import (
        ConstantPosePlan,
        ContinuousPlan,
        conditioned_plan,
    )
    from egomimic.robot.piper_pico.policy_rtc import build_rtc_prefix

    timestamps = np.asarray(timestamps, dtype=float)
    if (
        timestamps.ndim != 1
        or len(timestamps) < 2
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise ValueError("Need at least two strictly increasing finite timestamps")
    if not math.isfinite(start_delay_s) or start_delay_s < 0:
        raise ValueError("start delay must be finite and nonnegative")
    first_tick = float(timestamps[0]) + observation_to_send_s + start_delay_s
    recursive = independent = None
    iterator = iter(samples)
    for step, timestamp in enumerate(timestamps):
        try:
            sample = dict(next(iterator))
        except StopIteration as exc:
            raise ValueError("Missing recorded observation") from exc
        target = adapter.camera_to_base_actions(sample.pop("actions"))
        state = adapter.camera_to_base_actions(sample["observation.state"][None])[0]
        noise = (
            np.random.default_rng(np.random.SeedSequence([seed, episode, step]))
            .normal(size=(timing.output_chunk_length, 32))
            .astype(np.float32)
        )
        plain = np.asarray(predict(dict(sample), noise.copy()), dtype=float)
        anchor = float(timestamp) + observation_to_send_s
        if step == 0:
            initial = conditioned_plan(plain, timing, first_tick, smoothing_s)
            if transition_s:
                initial = ContinuousPlan(
                    initial,
                    ConstantPosePlan(state, initial.end_walltime),
                    first_tick,
                    transition_s,
                )
            recursive = independent = initial
            continue
        activation = activation_time(float(timestamp), first_tick, frequency, delay_s)
        new_recursive = None
        detail = dict(
            chain_step=step,
            observation_timestamp=float(timestamp),
            new_anchor=anchor,
            activation=activation,
            metrics={},
            prefix_counts={},
        )
        arrays = dict(
            noise=noise,
            recorded_target=target,
            current_base_state=state,
            plain_actions=plain,
        )
        if "observation.state_history" in sample:
            arrays["current_camera_history"] = sample["observation.state_history"]
        for mode, old in (
            ("independent_pair_rtc", independent),
            ("recursive_rtc", recursive),
        ):
            prefix = build_rtc_prefix(
                old,
                timing,
                timestamp,
                observation_to_send_s,
                activation + 1 / frequency,
                step,
            )
            request = {
                "schema": prefix["schema"],
                "request_id": prefix["request_id"],
                "previous_actions": np.stack(
                    [adapter.base_to_camera_state(r) for r in prefix["base_actions"]]
                ).astype(np.float32),
                "weights": prefix["weights"].astype(np.float32),
            }
            actions = np.asarray(
                predict(dict(sample, rtc=request), noise.copy()), dtype=float
            )
            result = evaluate_handoff(
                old,
                actions,
                timing,
                anchor,
                activation,
                smoothing_s=smoothing_s,
                transition_s=transition_s,
                control_frequency=frequency,
                reference_base_actions=target,
                current_base_state=state,
            )
            detail["metrics"][mode] = result["metrics"]
            detail["prefix_counts"][mode] = {
                "hard": int(np.sum(prefix["weights"] == 1)),
                "overlap": int(np.sum(prefix["weights"] > 0)),
            }
            arrays[f"{mode}_actions"] = actions
            arrays[f"{mode}_prefix"] = prefix["base_actions"]
            arrays[f"{mode}_weights"] = prefix["weights"]
            arrays.update({f"{mode}_{k}": v for k, v in result["traces"].items()})
            if mode == "recursive_rtc":
                new_recursive = conditioned_plan(actions, timing, anchor, smoothing_s)
                if transition_s:
                    new_recursive = ContinuousPlan(
                        new_recursive, old, activation, transition_s
                    )
        recursive = new_recursive
        independent = conditioned_plan(plain, timing, anchor, smoothing_s)
        yield detail, arrays
    try:
        next(iterator)
    except StopIteration:
        return
    raise ValueError("Extra recorded observations without timestamps")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--max-episodes", type=int, default=8)
    parser.add_argument("--max-handoffs-per-episode", type=int, default=32)
    parser.add_argument("--execute-steps", type=int, default=27)
    parser.add_argument("--delay-ms", type=float, default=170)
    parser.add_argument("--transition-ms", type=float, default=300)
    parser.add_argument("--smoothing-ms", type=float, default=150)
    parser.add_argument("--start-delay-ms", type=float, default=0)
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
    from egoverse_pi05.evaluation import describe
    from egoverse_pi05.rtc_evaluation import BaseCameraAdapter

    from egomimic.robot.piper_pico.pi05_data import Pi05Dataset, canonical_hash

    assets = load_inference_assets(args.checkpoint_path, args.num_inference_steps)
    dataset = Pi05Dataset(
        args.manifest, split="all", include_images=not args.prepare_only
    )
    if dataset.manifest != assets.metadata["manifest"]:
        raise ValueError("Exact checkpoint training manifest required")
    chains = select_chains(
        dataset,
        max_episodes=args.max_episodes,
        max_handoffs=args.max_handoffs_per_episode,
        steps=args.execute_steps,
        delay_s=args.delay_ms / 1000,
        transition_s=args.transition_ms / 1000,
        frequency=args.control_frequency,
        start_delay_s=args.start_delay_ms / 1000,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "selection.json").write_text(json.dumps(chains, indent=2) + "\n")
    (args.output_dir / "arguments.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            indent=2,
        )
        + "\n"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(assets.metadata, indent=2) + "\n"
    )
    print(
        f"Selected {len(chains)} chains, {sum(len(x['rows'])-1 for x in chains)} handoffs; teacher-forced recorded inputs",
        flush=True,
    )
    if args.prepare_only:
        return
    from egoverse_pi05.inference import create_policy
    from egoverse_pi05.rtc import RtcConfig

    policy = create_policy(
        assets, require_single_gpu=True, rtc_config=RtcConfig(args.rtc_max_guidance)
    )
    started = time.monotonic()
    pools = {}
    (args.output_dir / "samples").mkdir()
    with (args.output_dir / "samples.jsonl").open("w") as log:
        for chain in chains:
            episode = chain["episode"]
            contract = dataset.episodes[episode]["contract"]
            adapter = BaseCameraAdapter(
                contract["extrinsics"], contract["gripper_encoding"]["open_m"]
            )

            def predict(observation, noise):
                return adapter.camera_to_base_actions(
                    policy.infer(observation, noise=noise)["actions"]
                )

            observations = (
                dataset[dataset.index_for_source(episode, row)] for row in chain["rows"]
            )
            for detail, arrays in replay_chain(
                observations,
                chain["timestamps"],
                predict=predict,
                adapter=adapter,
                timing=dataset.timing,
                observation_to_send_s=chain["observation_to_send_s"],
                delay_s=args.delay_ms / 1000,
                transition_s=args.transition_ms / 1000,
                smoothing_s=args.smoothing_ms / 1000,
                frequency=args.control_frequency,
                start_delay_s=args.start_delay_ms / 1000,
                seed=args.seed,
                episode=episode,
            ):
                step = detail["chain_step"]
                detail.update(
                    episode=episode,
                    episode_name=chain["episode_name"],
                    source_row=chain["rows"][step],
                )
                np.savez_compressed(
                    args.output_dir
                    / "samples"
                    / f"episode_{episode:03d}_step_{step:03d}.npz",
                    **arrays,
                )
                log.write(json.dumps(detail) + "\n")
                log.flush()
                for mode, arms in detail["metrics"].items():
                    for side, metrics in arms.items():
                        for metric, value in metrics.items():
                            pools.setdefault((mode, side, metric), []).append(value)
            print(
                f"Episode {episode}: {len(chain['rows'])-1} handoffs; elapsed {time.monotonic()-started:.1f}s",
                flush=True,
            )
    summary = dict(
        episodes=len(chains),
        handoffs=sum(len(x["rows"]) - 1 for x in chains),
        selection_sha256=canonical_hash(chains),
        metrics={},
        seconds=time.monotonic() - started,
        scope=__doc__,
        initial_anchor="first observation + observation_to_send_s + start_delay_ms; initial held-pose transition included",
        request_clock="recorded observation timestamps separated by execute_steps source rows",
        control_clock="ideal fixed first-control-tick grid; no OS drift, hardware delay or actuator simulation",
        paired_noise="same explicit noise for current unguided and both RTC conditions; previous plain noise retained across chain",
        recursive_definition="previous recursively RTC-guided output, conditioned by production SG and quintic transition",
        independent_definition="preceding ordinary prediction; no inherited RTC guidance from older requests",
    )
    for (mode, side, key), values in pools.items():
        summary["metrics"].setdefault(mode, {}).setdefault(side, {})[key] = describe(
            values
        )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Completed {args.output_dir/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
