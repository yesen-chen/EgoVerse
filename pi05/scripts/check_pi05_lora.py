"""Real-data Pi0.5 LoRA test: base loading, updates, frozen weights, checkpoint and resume.

Run fresh training and resume in separate processes. Uses the same manifest,
tokenizer, normalization and three-camera 100-point batches as production.
It tests training mechanics and preprocessing, not learned task success.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from _bootstrap import setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Total target updates; use 4 when resuming a 2-step test",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="CPU real-data and shape checks; no pretrained model allocated",
    )
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("steps must be at least 2")
    setup(cpu=args.inspect, openpi_root=args.openpi_root)
    import flax.nnx as nnx
    import jax
    import numpy as np
    from egoverse_pi05.assets import load_inference_assets, read_json
    from egoverse_pi05.config import make_train_config
    from egoverse_pi05.training import dry_run, run_training
    from openpi.shared.nnx_utils import PathRegex

    cfg = make_train_config(
        manifest_path=args.manifest,
        stats_dir=args.stats_dir,
        tokenizer_path=args.tokenizer_path,
        params_path=args.params_path,
        checkpoint_root=args.run_dir / "checkpoints",
        exp_name="lora_probe",
        batch_size=args.batch_size,
        steps=args.steps,
        warmup_steps=1,
        decay_steps=100,
        save_interval=1,
        keep_period=1,
        resume=args.resume,
    )
    # Both paths exercise genuine three-camera/task/state tokenization.
    print(json.dumps(dry_run(cfg), indent=2), flush=True)
    shape_model = nnx.eval_shape(cfg.model.create, jax.random.key(0))
    shape_params = nnx.state(shape_model, nnx.Param)
    trainable = shape_params.filter(cfg.trainable_filter)
    paths = ["/".join(map(str, path)) for path in trainable.flat_state()]
    if not paths or any("llm" in path and "lora" not in path for path in paths):
        raise AssertionError(
            "Base language/expert weights must be frozen during LoRA training"
        )
    if any(path.startswith("PaliGemma/img/") for path in paths):
        raise AssertionError("Vision weights must be frozen in this LoRA configuration")

    def count(tree):
        return sum(math.prod(value.shape) for value in jax.tree.leaves(tree))

    configuration = {
        "action_horizon": 100,
        "physical_action_dim": 14,
        "padded_action_dim": 32,
        "camera_count": 3,
        "batch_size": args.batch_size,
        "total_parameters": count(shape_params),
        "trainable_parameters": count(trainable),
    }
    print(json.dumps(configuration, indent=2), flush=True)
    if args.inspect:
        print("PI05 LORA CPU INSPECTION PASSED; no pretrained weights allocated")
        return
    if not args.resume and not (args.params_path / "_METADATA").is_file():
        raise FileNotFoundError(f"Missing local pretrained params: {args.params_path}")
    result_path = args.run_dir / "lora_result.json"
    previous = read_json(result_path)["final_state"] if args.resume else None
    if not args.resume and result_path.exists():
        raise FileExistsError("Probe already exists; use a new run-dir or --resume")
    initial, final = None, None

    def fingerprint(tree, sampled=False):
        digest = hashlib.sha256()
        leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
        if not leaves:
            raise AssertionError("Cannot validate an empty parameter group")
        for path, leaf in leaves:
            value = leaf
            if sampled:
                flat = value.reshape(-1)
                indices = np.unique(
                    np.linspace(0, flat.size - 1, min(8, flat.size), dtype=np.int32)
                )
                value = flat[indices]
            array = np.asarray(jax.device_get(value))
            if not np.isfinite(array).all():
                raise AssertionError(
                    f"Non-finite parameter/optimizer state at {jax.tree_util.keystr(path)}"
                )
            digest.update(jax.tree_util.keystr(path).encode())
            digest.update(str((leaf.shape, leaf.dtype)).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def signature(state):
        step = int(jax.device_get(state.step))
        counts = [
            int(jax.device_get(value))
            for path, value in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]
            if any(getattr(key, "name", None) == "count" for key in path)
        ]
        if not counts or any(value != step for value in counts):
            raise AssertionError(
                f"Optimizer counters differ from completed updates: {step}, {counts}"
            )
        if state.ema_params is not None:
            raise AssertionError("LoRA probe expects EMA disabled")
        return {
            "step": step,
            "optimizer_counts": counts,
            "trainable": fingerprint(state.params.filter(cfg.trainable_filter)),
            "text_lora": fingerprint(
                state.params.filter(
                    nnx.All(
                        PathRegex(".*llm.*lora.*"), nnx.Not(PathRegex(".*llm.*_1.*"))
                    )
                )
            ),
            "expert_lora": fingerprint(
                state.params.filter(
                    nnx.All(PathRegex(".*llm.*_1.*"), PathRegex(".*lora.*"))
                )
            ),
            "optimizer": fingerprint(state.opt_state),
            "frozen_samples": fingerprint(
                state.params.filter(cfg.freeze_filter), sampled=True
            ),
        }

    def observer(event, state):
        nonlocal initial, final
        current = signature(state)
        if event in ("initial", "restored"):
            initial = current
            if event == "restored" and current != previous:
                raise AssertionError(
                    "Restored LoRA/optimizer/frozen samples differ from the saved state"
                )
        elif event == "saved":
            final = current
            if initial is None or final["frozen_samples"] != initial["frozen_samples"]:
                raise AssertionError("Frozen pretrained parameter samples changed")
            for key in ("trainable", "text_lora", "expert_lora", "optimizer"):
                if final[key] == initial[key]:
                    raise AssertionError(f"{key} did not update")
        print(f"[LoRA probe] {event}: completed updates {current['step']}", flush=True)

    report = run_training(
        cfg,
        manifest_path=args.manifest,
        stats_dir=args.stats_dir,
        tokenizer_path=args.tokenizer_path,
        finetune_mode="lora",
        training_contract={**configuration, "probe": "real_dataset_lora_v1"},
        observer=observer,
    )
    if final is None or final["step"] != args.steps:
        raise AssertionError("LoRA training did not reach the requested updates")
    # Verify every retained intermediate checkpoint has portable/hash-valid assets.
    checkpoints = sorted(
        path for path in Path(cfg.checkpoint_dir).iterdir() if path.name.isdigit()
    )
    for checkpoint in checkpoints:
        load_inference_assets(checkpoint)
    result = {
        **report,
        "configuration": configuration,
        "initial_state": initial,
        "final_state": final,
        "portable_checkpoints": [str(path) for path in checkpoints],
        "frozen_check": "8 evenly spaced samples per frozen leaf; all trainable/optimizer leaves fully hashed",
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    (args.run_dir / f"lora_result_step_{args.steps}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(f"PI05 LORA {'RESUME' if args.resume else 'TRAIN'} PASSED: {result_path}")


if __name__ == "__main__":
    main()
