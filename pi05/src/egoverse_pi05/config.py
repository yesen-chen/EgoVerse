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
    history_offsets,
    positive_int,
    token_config,
    validate_manifest_contract,
    validate_stats,
)


@dataclasses.dataclass(frozen=True)
class BimanualInputs(transforms.DataTransformFn):
    # Serving stays strict: only the explicitly mixed training path accepts Ego.
    allow_ego: bool = False
    state_history: dict | None = None

    def __call__(self, data):
        domain = data.get("domain", "piper")
        if domain not in ("piper", "ego") or (domain == "ego" and not self.allow_ego):
            raise ValueError(
                "Ego samples are only accepted by the mixed training input"
            )
        gripper_mode = data.get("gripper_mode", "width")
        if gripper_mode not in ("width", "zero") or (
            domain == "piper" and gripper_mode != "width"
        ):
            raise ValueError("Invalid gripper mode for the sample domain")
        state = np.asarray(data["observation.state"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Expected 14 finite absolute XYZ/YPR/gripper state values")
        images = {}
        image_mask = {}
        for camera, slot in CAMERAS.items():
            if domain == "ego" and camera != "front_1":
                if f"observation.images.{camera}" in data:
                    raise ValueError("Ego samples must only provide the front_1 image")
                images[slot] = np.zeros_like(images[CAMERAS["front_1"]])
                image_mask[slot] = np.False_
                continue
            image = np.asarray(data[f"observation.images.{camera}"])
            if (
                image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[-1] != 3
                or min(image.shape[:2]) < 1
            ):
                raise ValueError(f"Expected uint8 HWC RGB image for {camera}")
            images[slot] = image
            image_mask[slot] = np.True_
        result = {
            "state": state,
            "image": images,
            "image_mask": image_mask,
        }
        offsets = history_offsets(self.state_history)
        history = data.get("observation.state_history")
        if offsets:
            history = np.asarray(history, dtype=np.float32)
            if history.shape != (len(offsets), 14) or not np.isfinite(history).all():
                raise ValueError(
                    f"Expected {len(offsets)} x 14 finite state_history values"
                )
            result["state_history"] = history
        elif "observation.state_history" in data:
            raise ValueError("state_history is not enabled in this checkpoint")
        if self.allow_ego:
            result.update(domain=domain, gripper_mode=gripper_mode)
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
class DomainNormalize(transforms.DataTransformFn):
    """Normalize physical values by source domain before state tokenization.

    The zero Ego mode follows Human's original shared-head convention: zero
    gripper supervision is inserted after pose normalization, not interpreted
    as a physical closed-gripper value and then normalized to -1.
    """

    domain_stats: dict

    def __post_init__(self):
        if set(self.domain_stats) != {"piper", "ego"}:
            raise ValueError(
                "Mixed training requires separate Piper and Ego statistics"
            )
        normalizers = {}
        for domain, wrapped in self.domain_stats.items():
            stats = validate_stats(wrapped)
            normalizers[domain] = transforms.Normalize(
                with_history_stats(
                    {
                        key: normalize.NormStats(
                            **{
                                field: np.asarray(value)
                                for field, value in fields.items()
                            }
                        )
                        for key, fields in stats.items()
                    }
                ),
                use_quantiles=True,
            )
        object.__setattr__(self, "_normalizers", normalizers)

    def __call__(self, data):
        result = dict(data)
        domain = result.pop("domain", None)
        gripper_mode = result.pop("gripper_mode", None)
        if domain not in self._normalizers or gripper_mode not in ("width", "zero"):
            raise ValueError("Normalization needs the explicit domain and gripper mode")
        if domain == "piper" and gripper_mode != "width":
            raise ValueError("Piper requires physical gripper-width supervision")
        result = self._normalizers[domain](result)
        with_history = "state_history" in result
        for key in ("state", "actions", "state_history"):
            if key in result:
                # History-enabled training must tokenize the same precision as
                # serving. Rounding only this mixed-data path can change a bin
                # at a boundary. Keep legacy no-history preprocessing intact.
                dtype = None if with_history and key != "actions" else np.float32
                result[key] = np.array(result[key], dtype=dtype, copy=True)
                if domain == "ego" and gripper_mode == "zero":
                    result[key][..., [6, 13]] = 0.0
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

    def tokenize(self, prompt, state=None, *, state_history=None):
        # Legacy checkpoints retain upstream tokens, dtypes and overflow
        # behavior. Strict history budgeting applies only to the new input.
        if state_history is None:
            return super().tokenize(prompt, state)
        text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is None:
            tokens = self._tokenizer.encode(
                text, add_bos=True
            ) + self._tokenizer.encode("\n")
        else:

            def discretize(value):
                bins = np.digitize(value, bins=np.linspace(-1, 1, 257)[:-1]) - 1
                return " ".join(map(str, bins))

            text = f"Task: {text}, State: {discretize(state)};"
            if state_history is not None:
                text += (
                    " History (oldest first): "
                    + "; ".join(discretize(row) for row in state_history)
                    + ";"
                )
            tokens = self._tokenizer.encode(text + "\nAction: ", add_bos=True)
        if len(tokens) > self._max_len:
            raise ValueError(
                f"Pi0.5 prompt/state/history needs {len(tokens)} tokens, exceeding "
                f"max_token_len={self._max_len}; increase --max-token-len and retrain. "
                "Refusing to truncate state history or the action prompt."
            )
        count = len(tokens)
        return (
            np.asarray(tokens + [0] * (self._max_len - count), dtype=np.int32),
            np.arange(self._max_len) < count,
        )


def with_history_stats(stats, enabled=True):
    result = dict(stats)
    if enabled:
        result["state_history"] = result["state"]
    return result


@dataclasses.dataclass(frozen=True)
class TokenizeStateHistory(transforms.DataTransformFn):
    tokenizer: LocalTokenizer
    count: int

    def __call__(self, data):
        result = dict(data)
        # Preserve the normalizer's dtype just like the current state. Casting
        # only history could move a value across a discrete tokenizer bin edge.
        history = np.asarray(result.pop("state_history"))
        if history.shape != (self.count, 14) or not np.isfinite(history).all():
            raise ValueError("Invalid normalized state_history")
        prompt = result.pop("prompt")
        if not isinstance(prompt, str):
            prompt = prompt.item()
        tokens, masks = self.tokenizer.tokenize(
            prompt, result["state"], state_history=history
        )
        # History is consumed here: OpenPI Observation has no history field.
        return {**result, "tokenized_prompt": tokens, "tokenized_prompt_mask": masks}


def model_transforms(tokenizer_path, model_config, state_history=None):
    offsets = history_offsets(state_history)
    local_tokenizer = LocalTokenizer(tokenizer_path, model_config.max_token_len)
    return [
        transforms.ResizeImages(224, 224),
        TokenizeStateHistory(local_tokenizer, len(offsets))
        if offsets
        else transforms.TokenizePrompt(
            local_tokenizer,
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
        from egomimic.robot.piper_pico.pi05_training_data import (
            load_training_manifest,
            load_training_stats,
        )

        manifest = load_training_manifest(self.manifest_path)
        domain_stats = load_training_stats(manifest, self.stats_dir)
        mixed = manifest["schema"] != "egoverse_pi05_manifest_v1"
        history = manifest.get("state_history")
        if not mixed:
            validate_manifest_contract(manifest)
        inputs = (
            [
                BimanualInputs(allow_ego=True, state_history=history),
                DomainNormalize(domain_stats),
            ]
            if mixed
            else [BimanualInputs(state_history=history)]
        )
        return EgoVerseDataConfig(
            repo_id="local/egoverse_piper_ego" if mixed else "local/egoverse_piper",
            asset_id="egoverse",
            manifest_path=str(self.manifest_path),
            # OpenPI adds Normalize after data_transforms. Mixed inputs have
            # already used their own domain stats, so that stage must be a no-op.
            norm_stats={}
            if mixed
            else with_history_stats(
                normalize.load(self.stats_dir), bool(history_offsets(history))
            ),
            use_quantile_norm=True,
            action_sequence_keys=("actions",),
            data_transforms=transforms.Group(
                inputs=inputs, outputs=[BimanualOutputs()]
            ),
            model_transforms=transforms.Group(
                inputs=model_transforms(self.tokenizer_path, model_config, history)
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
        from egomimic.robot.piper_pico.pi05_training_data import TrainingDataset

        return TrainingDataset(
            data_config.manifest_path, split="train", include_images=True
        )

    with patch.object(data_loader, "create_torch_dataset", create):
        yield


def make_model_config(finetune_mode, *, state_history=None, max_token_len=None):
    if finetune_mode not in ("lora", "full"):
        raise ValueError("finetune_mode must be lora or full")
    suffix = "_lora" if finetune_mode == "lora" else ""
    return pi0_config.Pi0Config(
        pi05=True,
        action_horizon=100,
        action_dim=32,
        max_token_len=token_config(state_history, max_token_len)["max_token_len"],
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
    max_token_len=None,
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
    from egomimic.robot.piper_pico.pi05_training_data import load_training_manifest

    manifest = load_training_manifest(manifest_path)
    model = make_model_config(
        finetune_mode,
        state_history=manifest.get("state_history"),
        max_token_len=max_token_len,
    )
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
