"""JAX Pi0.5 training using EgoVerse's explicit 100-point dataset contract."""

import dataclasses
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import flax.nnx as nnx
import numpy as np
import sentencepiece
from openpi import transforms
from openpi.models import pi0_config, tokenizer
from openpi.shared import normalize
from openpi.shared.nnx_utils import PathRegex
from openpi.training import config, data_loader, optimizer, weight_loaders

from egoverse_pi05.assets import (
    CAMERAS,
    positive_int,
    read_json,
    validate_manifest_contract,
    validate_stats,
)


@dataclasses.dataclass(frozen=True)
class BimanualInputs(transforms.DataTransformFn):
    def __call__(self, data):
        state = np.asarray(data["observation.state"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Expected 14 finite absolute XYZ/YPR/gripper state values")
        images = {}
        for camera, slot in CAMERAS.items():
            image = np.asarray(data[f"observation.images.{camera}"])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Expected uint8 HWC RGB image for {camera}")
            images[slot] = image
        result = {
            "state": state,
            "image": images,
            "image_mask": {slot: np.True_ for slot in images},
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape != (100, 14) or not np.isfinite(actions).all():
                raise ValueError(
                    "Expected 100 x 14 finite physical actions; interpolation belongs to Pi05Dataset"
                )
            result["actions"] = actions
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty task prompt is required")
        result["prompt"] = prompt
        return result


@dataclasses.dataclass(frozen=True)
class BimanualOutputs(transforms.DataTransformFn):
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"], dtype=np.float32)[..., :14]}


class LocalTokenizer(tokenizer.PaligemmaTokenizer):
    def __init__(self, path, max_len):
        self._max_len = max_len
        if not Path(path).is_file():
            raise FileNotFoundError(f"Local tokenizer is required: {path}")
        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(path))


def model_transforms(tokenizer_path, model_config):
    return [
        transforms.ResizeImages(224, 224),
        transforms.TokenizePrompt(
            LocalTokenizer(tokenizer_path, model_config.max_token_len),
            discrete_state_input=True,
        ),
        transforms.PadStatesAndActions(model_config.action_dim),
    ]


@dataclasses.dataclass(frozen=True)
class EgoVerseDataConfig(config.DataConfig):
    manifest_path: str = ""


@dataclasses.dataclass(frozen=True)
class EgoVerseDataFactory(config.DataConfigFactory):
    manifest_path: str = ""
    stats_dir: str = ""
    tokenizer_path: str = ""

    def create(self, assets_dirs, model_config):
        from egomimic.robot.piper_pico.pi05_data import load_manifest

        manifest = load_manifest(self.manifest_path)
        validate_manifest_contract(manifest)
        report = read_json(Path(self.stats_dir) / "stats_report.json")
        if (
            report.get("manifest_sha256") != manifest["manifest_sha256"]
            or report.get("split") != "train"
        ):
            raise ValueError(
                "Normalization statistics do not match this training manifest/split"
            )
        validate_stats(read_json(Path(self.stats_dir) / "norm_stats.json"))
        return EgoVerseDataConfig(
            repo_id="local/egoverse_piper",
            asset_id="egoverse",
            manifest_path=str(self.manifest_path),
            norm_stats=normalize.load(self.stats_dir),
            use_quantile_norm=True,
            action_sequence_keys=("actions",),
            data_transforms=transforms.Group(
                inputs=[BimanualInputs()], outputs=[BimanualOutputs()]
            ),
            model_transforms=transforms.Group(
                inputs=model_transforms(self.tokenizer_path, model_config)
            ),
        )


@contextmanager
def local_dataset_loader():
    """Reuse upstream batching/sharding, without LeRobot timestamp re-windowing."""
    original = data_loader.create_torch_dataset

    def create(data_config, action_horizon, model_config):
        if not isinstance(data_config, EgoVerseDataConfig):
            return original(data_config, action_horizon, model_config)
        if action_horizon != 100:
            raise ValueError("EgoVerse Pi0.5 requires 100 model action points")
        from egomimic.robot.piper_pico.pi05_data import Pi05Dataset

        return Pi05Dataset(
            data_config.manifest_path, split="train", include_images=True
        )

    with patch.object(data_loader, "create_torch_dataset", create):
        yield


def make_model_config(finetune_mode):
    if finetune_mode not in ("lora", "full"):
        raise ValueError("finetune_mode must be lora or full")
    suffix = "_lora" if finetune_mode == "lora" else ""
    return pi0_config.Pi0Config(
        pi05=True,
        action_horizon=100,
        action_dim=32,
        paligemma_variant=f"gemma_2b{suffix}",
        action_expert_variant=f"gemma_300m{suffix}",
    )


def make_train_config(
    *,
    manifest_path,
    stats_dir,
    tokenizer_path,
    params_path,
    checkpoint_root,
    exp_name,
    batch_size=1,
    steps=1000,
    workers=0,
    learning_rate=1e-5,
    warmup_steps=10,
    decay_steps=1000,
    end_learning_rate=1e-6,
    save_interval=1000,
    log_interval=1,
    resume=False,
    seed=42,
    finetune_mode="lora",
    fsdp_devices=1,
    keep_period=None,
):
    for name, value in (
        ("batch_size", batch_size),
        ("steps", steps),
        ("save_interval", save_interval),
        ("log_interval", log_interval),
        ("fsdp_devices", fsdp_devices),
    ):
        positive_int(value, name)
    if workers < 0 or not exp_name.strip() or Path(exp_name).name != exp_name:
        raise ValueError(
            "workers must be nonnegative and exp_name must be one directory name"
        )
    if not 0 <= warmup_steps < decay_steps or not all(
        np.isfinite(v) and v > 0 for v in (learning_rate, end_learning_rate)
    ):
        raise ValueError("Invalid learning-rate schedule")
    if keep_period is not None and (
        positive_int(keep_period, "keep_period") % save_interval
    ):
        raise ValueError("keep_period must be a multiple of save_interval")
    model = make_model_config(finetune_mode)
    freeze = (
        nnx.Any(model.get_freeze_filter(), PathRegex("PaliGemma/img/.*"))
        if finetune_mode == "lora"
        else nnx.Nothing
    )
    return config.TrainConfig(
        name=f"egoverse_pi05_{finetune_mode}",
        exp_name=exp_name,
        model=model,
        freeze_filter=freeze,
        data=EgoVerseDataFactory(
            manifest_path=str(manifest_path),
            stats_dir=str(stats_dir),
            tokenizer_path=str(tokenizer_path),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(params_path)),
        checkpoint_base_dir=str(checkpoint_root),
        batch_size=batch_size,
        num_workers=workers,
        num_train_steps=steps,
        log_interval=log_interval,
        save_interval=save_interval,
        keep_period=keep_period,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=warmup_steps,
            peak_lr=learning_rate,
            decay_steps=decay_steps,
            decay_lr=end_learning_rate,
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        overwrite=False,
        resume=resume,
        wandb_enabled=False,
        seed=seed,
        fsdp_devices=fsdp_devices,
    )
