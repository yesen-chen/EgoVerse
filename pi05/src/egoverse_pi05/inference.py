"""Strict Pi0.5 restore and the same physical input/output transforms as training."""

import copy

import numpy as np

from egoverse_pi05.assets import CAMERAS


class BimanualInferencePolicy:
    def __init__(self, policy, metadata):
        self._policy = policy
        self._metadata = copy.deepcopy(metadata)

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    def infer(self, observation):
        expected = {
            "observation.state",
            "prompt",
            *(f"observation.images.{camera}" for camera in CAMERAS),
        }
        if set(observation) != expected:
            raise ValueError(f"Expected exactly {sorted(expected)}")
        for camera, shape in self._metadata["image_shapes"].items():
            image = np.asarray(observation[f"observation.images.{camera}"])
            if image.dtype != np.uint8 or image.shape != tuple(shape):
                raise ValueError(
                    f"Expected uint8 RGB {camera} with training shape {shape}"
                )
        state = np.asarray(observation["observation.state"])
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Expected 14 finite physical XYZ/YPR/gripper state values")
        if (
            not isinstance(observation["prompt"], str)
            or not observation["prompt"].strip()
        ):
            raise ValueError("A nonempty task prompt is required")
        result = dict(self._policy.infer(observation))
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (100, 14) or not np.isfinite(actions).all():
            raise ValueError("Pi0.5 returned invalid physical action chunk")
        result["actions"] = actions
        return result

    def reset(self):
        self._policy.reset()


def create_policy(assets, *, require_single_gpu=False):
    import jax
    import jax.numpy as jnp
    from openpi import transforms
    from openpi.models import model
    from openpi.policies.policy import Policy
    from openpi.shared import normalize

    from egoverse_pi05.config import (
        BimanualInputs,
        BimanualOutputs,
        make_model_config,
        model_transforms,
    )

    # Match training/offline inference initialization order. The pinned OpenPI
    # runtime imports its wheel CUDA dependencies; discovering GPUs before that
    # can pick an older system nvJitLink from LD_LIBRARY_PATH and fail cuSPARSE.
    # Validate devices after those imports, still before allocating model weights.
    if require_single_gpu:
        devices = jax.devices()
        if (
            jax.process_count() != 1
            or len(devices) != 1
            or devices[0].platform != "gpu"
        ):
            raise RuntimeError("Expose exactly one CUDA GPU for serving")
    config = make_model_config(assets.metadata["finetune_mode"])
    stats = normalize.load(assets.norm_stats_path.parent)
    preprocessing = model_transforms(assets.tokenizer_path, config)
    params = model.restore_params(assets.checkpoint_path / "params", dtype=jnp.bfloat16)
    # Upstream defaults to intersecting trees. Strict loading prevents silently
    # dropping LoRA adapters by restoring them with a base-only model config.
    network = config.load(params, remove_extra_params=False)
    policy = Policy(
        network,
        transforms=[
            BimanualInputs(),
            transforms.Normalize(stats, use_quantiles=True),
            *preprocessing,
        ],
        output_transforms=[
            transforms.Unnormalize(stats, use_quantiles=True),
            BimanualOutputs(),
        ],
        sample_kwargs={"num_steps": assets.metadata["num_inference_steps"]},
        metadata=assets.metadata,
    )
    return BimanualInferencePolicy(policy, assets.metadata)


def warmup_policy(policy, assets):
    state = assets.norm_stats["state"]
    observation = {
        f"observation.images.{name}": np.zeros(shape, dtype=np.uint8)
        for name, shape in assets.metadata["image_shapes"].items()
    }
    observation.update(
        {
            "observation.state": (
                np.asarray(state["q01"]) + np.asarray(state["q99"])
            ).astype(np.float32)
            / 2,
            "prompt": assets.metadata["manifest"]["episodes"][0]["prompt"],
        }
    )
    result = policy.infer(observation)
    policy.reset()
    return result
