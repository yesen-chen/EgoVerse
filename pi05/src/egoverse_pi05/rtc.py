"""Training-free RTC guidance for the pinned OpenPI Pi0.5 JAX flow.

Implements Black et al., Real-Time Execution of Action Chunking Flow Policies,
Eq. 2--4 / Algorithm 1: https://www.pi.website/download/real_time_chunking.pdf.
OpenPI runs noise time t=1 to data time t=0, opposite to the paper's tau.
The denoised estimate is x - t*v and guidance is SUBTRACTED from v, because
the Euler integration step is negative. This is VJP guidance through the full
action expert, not endpoint replacement or output blending.
"""

import time
from dataclasses import dataclass

import numpy as np

SCHEMA = "egoverse_rtc_v1"


@dataclass(frozen=True)
class RtcConfig:
    max_guidance: float = 5.0

    def __post_init__(self):
        if (
            isinstance(self.max_guidance, bool)
            or not np.isfinite(self.max_guidance)
            or self.max_guidance <= 0
        ):
            raise ValueError("RTC max_guidance must be finite and positive")

    def metadata(self):
        return {
            "schema": SCHEMA,
            "max_guidance": float(self.max_guidance),
            "method": "denoised_prefix_vjp",
            "prefix_space": "physical_camera_actions",
            "action_horizon": 100,
            "action_dim": 14,
            "padded_action_dim": 32,
        }


def validate_request(value):
    """Validate the wire request before any transformation or GPU execution."""
    required = {"schema", "previous_actions", "weights", "request_id"}
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema") != SCHEMA
    ):
        raise ValueError("Invalid RTC request schema/fields")
    request_id = value["request_id"]
    if (
        isinstance(request_id, bool)
        or not isinstance(request_id, (int, np.integer))
        or request_id < 0
    ):
        raise ValueError("RTC request_id must be a nonnegative integer")
    previous = np.asarray(value["previous_actions"])
    weights = np.asarray(value["weights"])
    if (
        previous.shape != (100, 14)
        or previous.dtype.kind not in "fiu"
        or not np.isfinite(previous).all()
    ):
        raise ValueError(
            "RTC previous_actions must be 100 x 14 finite physical actions"
        )
    if (
        weights.shape != (100,)
        or weights.dtype.kind not in "fiu"
        or not np.isfinite(weights).all()
        or np.any((weights < 0) | (weights > 1))
    ):
        raise ValueError("RTC weights must be 100 finite values in [0, 1]")
    previous = previous.astype(np.float32)
    if not np.isfinite(previous).all():
        raise ValueError("RTC previous_actions must remain finite in float32")
    return {
        "schema": SCHEMA,
        "previous_actions": previous,
        "weights": weights.astype(np.float32),
        "request_id": int(request_id),
    }


def padded_mask(weights):
    """Only the 14 learned robot dimensions contribute to the prefix error."""
    mask = np.zeros((100, 32), dtype=np.float32)
    mask[:, :14] = np.asarray(weights, dtype=np.float32)[:, None]
    return mask


def guidance_scale(t, maximum):
    import jax.numpy as jnp

    # Eq. 2,4 with paper tau=1-t. Both endpoints approach +infinity before
    # clipping. Guard before division, so no NaNs/infinities enter the graph.
    denominator = jnp.maximum(t * (1 - t), jnp.finfo(jnp.float32).eps)
    return jnp.minimum(maximum, (t * t + (1 - t) ** 2) / denominator)


def guided_velocity(velocity, x, t, previous, mask, maximum=5.0):
    """One RTC vector field evaluation; mask multiplies the error exactly once."""
    import jax

    def denoised(current):
        v = velocity(current, t)
        return current - t * v, v

    clean, pullback, v = jax.vjp(denoised, x, has_aux=True)
    error = (previous - clean) * mask
    correction = pullback(error)[0]
    return v - guidance_scale(t, maximum) * correction


def make_sampler(network, config):
    """Wrap an immutable trained network without changing vendored OpenPI."""
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_module
    from openpi.models.pi0 import make_attn_mask
    from openpi.shared.nnx_utils import module_jit

    class RtcModel(nnx.Module):
        def __init__(self, network):
            self.network = network

        def sample_actions(
            self, rng, observation, previous, mask, *, num_steps=10, noise=None
        ):
            net = self.network
            observation = model_module.preprocess_observation(
                None, observation, train=False
            )
            dt = -1.0 / num_steps
            batch_size = observation.state.shape[0]
            if noise is None:
                noise = jax.random.normal(
                    rng, (batch_size, net.action_horizon, net.action_dim)
                )
            prefix_tokens, prefix_mask, prefix_ar_mask = net.embed_prefix(observation)
            _, kv_cache = net.PaliGemma.llm(
                [prefix_tokens, None],
                mask=make_attn_mask(prefix_mask, prefix_ar_mask),
                positions=jnp.cumsum(prefix_mask, axis=1) - 1,
            )

            def velocity(x, t):
                suffix, suffix_mask, suffix_ar, adarms = net.embed_suffix(
                    observation, x, jnp.broadcast_to(t, batch_size)
                )
                mask_prefix = jnp.broadcast_to(
                    prefix_mask[:, None, :],
                    (batch_size, suffix.shape[1], prefix_mask.shape[1]),
                )
                full_mask = jnp.concatenate(
                    [mask_prefix, make_attn_mask(suffix_mask, suffix_ar)], axis=-1
                )
                positions = (
                    jnp.sum(prefix_mask, axis=-1)[:, None]
                    + jnp.cumsum(suffix_mask, axis=-1)
                    - 1
                )
                (_, output), _ = net.PaliGemma.llm(
                    [None, suffix],
                    mask=full_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms],
                )
                return net.action_out_proj(output[:, -net.action_horizon :])

            def step(carry):
                x, t = carry
                # Branch after the Euler update: the expert's bf16 arithmetic
                # must stay exactly as upstream in the unconditioned branch,
                # while guidance intentionally promotes the correction to fp32.
                next_x = jax.lax.cond(
                    jnp.any(mask > 0),
                    lambda: x
                    + dt
                    * guided_velocity(
                        velocity, x, t, previous, mask, config.max_guidance
                    ),
                    lambda: x + dt * velocity(x, t),
                )
                return next_x, t + dt

            output, _ = jax.lax.while_loop(
                lambda carry: carry[1] >= -dt / 2, step, (noise, 1.0)
            )
            return output

    return module_jit(RtcModel(network).sample_actions)


class RtcInference:
    """Reuse the original Policy's exact transforms, RNG, and output conversion."""

    def __init__(self, policy, sampler):
        self.policy = policy
        self.sampler = sampler

    def infer(self, observation, request, *, noise=None):
        import jax
        import jax.numpy as jnp
        from openpi.models.model import Observation

        if not np.any(request["weights"]):
            # Explicit compatibility guarantee for disabled guidance: same RNG
            # consumption, transforms and sampler as ordinary policy inference.
            return self.policy.infer(observation, noise=noise)
        # BimanualInputs -> Normalize(actions quantiles) -> model preprocessing
        # -> PadStatesAndActions is EXACTLY the training action path. The state
        # tokenizer sees only state; the prefix is never used as a state token.
        physical = dict(observation, actions=request["previous_actions"].copy())
        inputs = self.policy._input_transform(physical)
        previous = np.asarray(inputs.pop("actions"), dtype=np.float32)
        if previous.shape != (100, 32) or not np.isfinite(previous).all():
            raise ValueError("RTC normalized prefix must be finite 100 x 32")
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        self.policy._rng, key = jax.random.split(self.policy._rng)
        kwargs = dict(self.policy._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)
            kwargs["noise"] = noise[None] if noise.ndim == 2 else noise
        start = time.monotonic()
        actions = self.sampler(
            key,
            Observation.from_dict(inputs),
            jnp.asarray(previous)[None],
            jnp.asarray(padded_mask(request["weights"]))[None],
            **kwargs,
        )
        outputs = jax.tree.map(
            lambda x: np.asarray(x[0]), {"state": inputs["state"], "actions": actions}
        )
        elapsed = time.monotonic() - start  # includes device synchronization
        outputs = self.policy._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": elapsed * 1000}
        return outputs
