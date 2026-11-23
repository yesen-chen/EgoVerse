"""Train 100-point, three-camera bimanual Pi0.5 using the pinned OpenPI JAX trainer."""

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT, setup


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--finetune-mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-token-len",
        type=int,
        help="Task/current-state/history token budget (default 200, or 256 with history); overflow fails",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--decay-steps", type=int, default=1000)
    parser.add_argument("--end-learning-rate", type=float, default=1e-6)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--keep-period", type=int)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU real-data input validation; no model weights allocated",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup(cpu=args.dry_run, openpi_root=args.openpi_root)
    from egoverse_pi05.config import make_train_config
    from egoverse_pi05.training import dry_run, run_training

    config_values = dict(
        manifest_path=args.manifest,
        stats_dir=args.stats_dir,
        tokenizer_path=args.tokenizer_path,
        params_path=args.params_path,
        checkpoint_root=args.checkpoint_root,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        steps=args.train_steps,
        workers=args.num_workers,
        finetune_mode=args.finetune_mode,
        fsdp_devices=args.fsdp_devices,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        end_learning_rate=args.end_learning_rate,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        keep_period=args.keep_period,
        resume=args.resume,
        seed=args.seed,
        max_token_len=args.max_token_len,
    )
    cfg = make_train_config(**config_values)
    if args.dry_run:
        print(json.dumps(dry_run(cfg), indent=2))
        print("PI05 CPU INPUT CHECK PASSED; no model weights loaded")
        return
    if not args.resume and not (args.params_path / "_METADATA").is_file():
        raise FileNotFoundError(f"Missing local Pi0.5 base params: {args.params_path}")
    contract = {
        key: config_values[key]
        for key in (
            "batch_size",
            "finetune_mode",
            "fsdp_devices",
            "learning_rate",
            "warmup_steps",
            "decay_steps",
            "end_learning_rate",
            "seed",
        )
    }
    report = run_training(
        cfg,
        manifest_path=args.manifest,
        stats_dir=args.stats_dir,
        tokenizer_path=args.tokenizer_path,
        finetune_mode=args.finetune_mode,
        training_contract=contract,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
