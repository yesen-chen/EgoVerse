"""Strict Pi0.5 restore and the same physical input/output transforms as training."""

import copy

import numpy as np

from egoverse_pi05.assets import CAMERAS, history_offsets


class BimanualInferencePolicy:
    def __init__(self, policy, metadata, *, rtc_runner=None, rtc_config=None):
        self._policy = policy
        self._metadata = copy.deepcopy(metadata)
        self._rtc_runner = rtc_runner
        if rtc_config is not None:
            if rtc_runner is None:
                raise ValueError("RTC capability requires a guided sampler")
            self._metadata["rtc"] = rtc_config.metadata()

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    def infer(self, observation, *, noise=None):
        observation = dict(observation)
        has_rtc_request = "rtc" in observation
        rtc_request = observation.pop("rtc", None)
        if has_rtc_request:
            from egoverse_pi05.rtc import validate_request

            if self._rtc_runner is None:
                raise ValueError(
                    "RTC is not enabled on this server; restart with --rtc"
                )
            rtc_request = validate_request(rtc_request)
        expected = {
            "observation.state",
            "prompt",
            *(f"observation.images.{camera}" for camera in CAMERAS),
        }
        offsets = history_offsets(self._metadata.get("state_history"))
        if offsets:
            expected.add("observation.state_history")
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
        if offsets:
            history = np.asarray(observation["observation.state_history"])
            if history.shape != (len(offsets), 14) or not np.isfinite(history).all():
                raise ValueError(
                    f"Expected {len(offsets)} x 14 finite state_history values"
                )
        if (
            not isinstance(observation["prompt"], str)
            or not observation["prompt"].strip()
        ):
            raise ValueError("A nonempty task prompt is required")
        if rtc_request is not None:
            result = dict(self._rtc_runner.infer(observation, rtc_request, noise=noise))
            result["rtc"] = {
                "schema": rtc_request["schema"],
                "request_id": rtc_request["request_id"],
            }
        elif noise is not None:
            result = dict(self._policy.infer(observation, noise=noise))
        else:
            result = dict(self._policy.infer(observation))
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (100, 14) or not np.isfinite(actions).all():
            raise ValueError("Pi0.5 returned invalid physical action chunk")
        result["actions"] = actions
        return result

    def reset(self):
        self._policy.reset()


def create_policy(assets, *, require_single_gpu=False, rtc_config=None):
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
        with_history_stats,
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
    history = assets.metadata.get("state_history")
    config = make_model_config(
        assets.metadata["finetune_mode"],
        state_history=history,
        max_token_len=assets.metadata.get("model_input", {}).get("max_token_len"),
    )
    stats = normalize.load(assets.norm_stats_path.parent)
    preprocessing = model_transforms(assets.tokenizer_path, config, history)
    params = model.restore_params(assets.checkpoint_path / "params", dtype=jnp.bfloat16)
    # Upstream defaults to intersecting trees. Strict loading prevents silently
    # dropping LoRA adapters by restoring them with a base-only model config.
    network = config.load(params, remove_extra_params=False)
    policy = Policy(
        network,
        transforms=[
            BimanualInputs(state_history=history),
            transforms.Normalize(
                with_history_stats(stats, bool(history_offsets(history))),
                use_quantiles=True,
            ),
            *preprocessing,
        ],
        output_transforms=[
            transforms.Unnormalize(stats, use_quantiles=True),
            BimanualOutputs(),
        ],
        sample_kwargs={"num_steps": assets.metadata["num_inference_steps"]},
        metadata=assets.metadata,
    )
    rtc_runner = None
    if rtc_config is not None:
        from egoverse_pi05.rtc import RtcInference, make_sampler

        rtc_runner = RtcInference(policy, make_sampler(network, rtc_config))
    return BimanualInferencePolicy(
        policy, assets.metadata, rtc_runner=rtc_runner, rtc_config=rtc_config
    )


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
    offsets = history_offsets(assets.metadata.get("state_history"))
    if offsets:
        observation["observation.state_history"] = np.tile(
            observation["observation.state"], (len(offsets), 1)
        )
    result = policy.infer(observation)
    if "rtc" in policy.metadata:
        from egoverse_pi05.rtc import SCHEMA

        # Compile the guided graph, then exercise the disabled-guidance path.
        # Neither touches a robot. Fully normalize the physical baseline output
        # through the same prefix path used for subsequent remote requests.
        for request_id, weights in enumerate(
            (np.ones(100, dtype=np.float32), np.zeros(100, dtype=np.float32))
        ):
            guided_observation = dict(
                observation,
                rtc={
                    "schema": SCHEMA,
                    "request_id": request_id,
                    "previous_actions": result["actions"].copy(),
                    "weights": weights,
                },
            )
            policy.infer(guided_observation)
    policy.reset()
    return result
