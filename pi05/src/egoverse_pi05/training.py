"""Thin wrapper over the pinned JAX trainer with portable transactional assets."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

from egoverse_pi05.assets import (
    build_contract,
    read_json,
    validate_resume_contract,
    write_portable_assets,
)


def validate_device_topology(batch_size, fsdp_devices, device_count, process_count=1):
    from egoverse_pi05.assets import positive_int

    for name, value in (
        ("batch_size", batch_size),
        ("fsdp_devices", fsdp_devices),
        ("device_count", device_count),
    ):
        positive_int(value, name)
    if process_count != 1 or device_count % fsdp_devices or batch_size % device_count:
        raise ValueError(
            "Use one host/process, with batch_size divisible by GPUs and GPUs divisible by fsdp_devices"
        )


def initialize_with_step_zero(initializer, *args, **kwargs):
    manager, resuming = initializer(*args, **kwargs)
    # Pinned upstream labels completed update 1 as step 0, but mistakenly treats
    # a single valid checkpoint 0 as an empty directory on resume.
    if kwargs.get("resume") and tuple(manager.all_steps()) == (0,):
        resuming = True
    if kwargs.get("resume") and not tuple(manager.all_steps()):
        raise ValueError("Resume requested but no committed checkpoint exists")
    return manager, resuming


def dry_run(cfg):
    from openpi.training import data_loader

    from egoverse_pi05.config import local_dataset_loader

    with local_dataset_loader():
        loader = data_loader.create_data_loader(cfg, num_batches=1, shuffle=False)
        observation, actions = next(iter(loader))
        actions = np.asarray(actions)
        if actions.shape != (cfg.batch_size, 100, 32) or not np.isfinite(actions).all():
            raise ValueError("Invalid Pi0.5 training batch")
        if (
            np.any(actions[..., 14:] != 0)
            or not np.asarray(observation.tokenized_prompt_mask).any(axis=-1).all()
        ):
            raise ValueError("Invalid action padding or empty tokenized task/state")
        if not np.asarray(observation.image_masks["base_0_rgb"]).all():
            raise ValueError("Every sample requires a real front camera input")
        return {
            "actions_shape": list(actions.shape),
            "images": {k: list(v.shape) for k, v in observation.images.items()},
            "image_masks": {
                k: np.asarray(v).tolist() for k, v in observation.image_masks.items()
            },
        }


def run_training(
    cfg,
    *,
    manifest_path,
    stats_dir,
    tokenizer_path,
    finetune_mode,
    training_contract,
    observer=None,
):
    import jax
    import wandb
    from openpi.shared import array_typing as at
    from openpi.training import checkpoints
    from scripts import train

    from egomimic.robot.piper_pico.pi05_training_data import load_training_manifest
    from egoverse_pi05.config import local_dataset_loader

    devices = jax.devices()
    if not all(device.platform == "gpu" for device in devices):
        raise RuntimeError(
            "Training requires visible CUDA GPU(s); use --dry-run for CPU input checks"
        )
    validate_device_topology(
        cfg.batch_size, cfg.fsdp_devices, len(devices), jax.process_count()
    )
    manifest = load_training_manifest(manifest_path)
    stats_path = Path(stats_dir) / "norm_stats.json"
    contract = build_contract(
        manifest=manifest,
        stats_path=stats_path,
        tokenizer_path=tokenizer_path,
        finetune_mode=finetune_mode,
        training=training_contract,
    )
    record_path = Path(cfg.checkpoint_dir).parent / f"{cfg.exp_name}.run.json"
    if cfg.resume:
        validate_resume_contract(read_json(record_path), contract)
    else:
        if record_path.exists() or cfg.checkpoint_dir.exists():
            raise FileExistsError(
                "Run already exists; choose a new exp-name or explicitly resume"
            )
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(contract, indent=2, ensure_ascii=False) + "\n"
        )
    metrics_path = record_path.with_suffix(".metrics.jsonl")
    original_log, original_initialize = wandb.log, checkpoints.initialize_checkpoint_dir
    original_init, original_restore = train.init_train_state, checkpoints.restore_state
    final_step, last_metrics = None, {}

    def initialize(*args, **kwargs):
        return initialize_with_step_zero(original_initialize, *args, **kwargs)

    def init_state(*args, **kwargs):
        state, sharding = original_init(*args, **kwargs)
        if observer is not None and not kwargs.get("resume"):
            observer("initial", state)
        return state, sharding

    def restore_state(*args, **kwargs):
        state = original_restore(*args, **kwargs)
        if int(jax.device_get(state.step)) >= cfg.num_train_steps:
            raise ValueError(
                "Resume target train-steps must exceed the checkpoint's completed updates"
            )
        if observer is not None:
            observer("restored", state)
        return state

    def save_state(manager, state, loader, step):
        nonlocal final_step
        if observer is not None:
            observer("saved", state)

        # The callback is part of the checkpoint transaction, so every committed
        # intermediate checkpoint is portable even if training later crashes.
        def save_assets(directory):
            write_portable_assets(directory, contract, stats_path, tokenizer_path)

        with at.disable_typechecking():
            train_state, params = checkpoints._split_params(state)
        manager.save(
            step,
            {
                "assets": save_assets,
                "train_state": train_state,
                "params": {"params": params},
            },
        )
        final_step = int(jax.device_get(state.step))

    def log_metrics(data, *args, **kwargs):
        if "loss" in data:
            values = {key: float(np.asarray(value)) for key, value in data.items()}
            if not all(np.isfinite(value) for value in values.values()):
                raise FloatingPointError(f"Non-finite training metrics: {values}")
            values.update(loop_index=kwargs.get("step"), wall_time=time.time())
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(values) + "\n")
            last_metrics.update(values)
        return original_log(data, *args, **kwargs)

    started = time.monotonic()
    with (
        local_dataset_loader(),
        patch.object(wandb, "log", log_metrics),
        patch.object(checkpoints, "save_state", save_state),
        patch.object(checkpoints, "initialize_checkpoint_dir", initialize),
        patch.object(train, "init_train_state", init_state),
        patch.object(checkpoints, "restore_state", restore_state),
    ):
        train.main(cfg)
    if final_step != cfg.num_train_steps:
        raise RuntimeError("Target training checkpoint was not saved")
    report = {
        "checkpoint_dir": str(cfg.checkpoint_dir),
        "completed_updates": final_step,
        "last_metrics": last_metrics,
        "elapsed_seconds_including_compilation_and_io": time.monotonic() - started,
        "resumed": cfg.resume,
    }
    record_path.with_suffix(".result.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
