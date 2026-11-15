"""Offline RTC math/normalization checks and optional real-checkpoint GPU smoke."""

import argparse
import json
import sys
import time
from pathlib import Path

from _bootstrap import setup


def analytic_checks():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from egoverse_pi05.rtc import guidance_scale, guided_velocity, padded_mask

    x = jnp.array([0.4, -0.2])
    a = jnp.array([[0.2, 0.4], [-0.3, 0.1]])
    b = jnp.array([0.1, -0.4])
    target = jnp.array([0.9, 0.7])
    mask = jnp.array([0.5, 0.0])
    t = 0.6

    def velocity(current, _):
        return a @ current + b

    jacobian = jnp.eye(2) - t * a
    clean = x - t * velocity(x, t)
    expected = velocity(x, t) - guidance_scale(t, 5.0) * (
        jacobian.T @ (mask * (target - clean))
    )
    got = jax.jit(guided_velocity, static_argnums=(0,))(
        velocity, x, t, target, mask, 5.0
    )
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    # Non-diagonal Jacobian affects even an output coordinate with error-mask0.
    assert abs(float(got[1] - velocity(x, t)[1])) > 0.01
    zero = guided_velocity(velocity, x, t, target, jnp.zeros(2), 5.0)
    np.testing.assert_array_equal(zero, velocity(x, t))
    np.testing.assert_array_equal(
        np.asarray([guidance_scale(0.0, 5.0), guidance_scale(1.0, 5.0)]), [5.0, 5.0]
    )

    def no_flow(current, _):
        return jnp.zeros_like(current)

    moved = x - 0.1 * guided_velocity(no_flow, x, 0.7, target, jnp.ones(2), 5.0)
    assert float(jnp.linalg.norm(moved - target)) < float(jnp.linalg.norm(x - target))
    assert not np.any(padded_mask(np.ones(100))[:, 14:])
    print(
        "RTC ANALYTIC CHECKS PASSED: reverse-time sign, affine VJP, single soft weight, zero mask, padded dimensions",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cpu_only:
        import os

        os.environ["JAX_PLATFORMS"] = "cpu"
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        analytic_checks()
        return
    if args.checkpoint_path is None or args.output is None:
        parser.error("GPU smoke requires --checkpoint-path and --output")
    if args.output.exists():
        raise FileExistsError(args.output)
    setup(openpi_root=args.openpi_root)
    import numpy as np
    from egoverse_pi05.assets import load_inference_assets
    from egoverse_pi05.inference import create_policy
    from egoverse_pi05.rtc import SCHEMA, RtcConfig

    assets = load_inference_assets(args.checkpoint_path, args.num_inference_steps)
    policy = create_policy(assets, require_single_gpu=True, rtc_config=RtcConfig())
    if args.manifest:
        from egomimic.robot.piper_pico.pi05_data import Pi05Dataset

        dataset = Pi05Dataset(args.manifest, split="train", include_images=True)
        if (
            dataset.manifest["manifest_sha256"]
            != assets.metadata["manifest"]["manifest_sha256"]
        ):
            raise ValueError("Expected exact checkpoint manifest")
        observation = dataset[args.sample_index]
        observation.pop("actions")
    else:
        stats = assets.norm_stats["state"]
        observation = {
            f"observation.images.{c}": np.zeros(shape, np.uint8)
            for c, shape in assets.metadata["image_shapes"].items()
        }
        observation.update(
            {
                "observation.state": (
                    np.asarray(stats["q01"]) + np.asarray(stats["q99"])
                )
                / 2,
                "prompt": assets.metadata["manifest"]["episodes"][0]["prompt"],
            }
        )
    noise = np.random.default_rng(42).normal(size=(100, 32)).astype(np.float32)
    started = time.monotonic()
    baseline = policy.infer(observation, noise=noise)["actions"]
    baseline_compile_s = time.monotonic() - started
    started = time.monotonic()
    baseline_repeat = policy.infer(observation, noise=noise)["actions"]
    baseline_warm_s = time.monotonic() - started
    np.testing.assert_array_equal(baseline, baseline_repeat)
    prefix = baseline.copy()
    prefix[:, 0] += 0.005
    request = {
        "schema": SCHEMA,
        "request_id": 1,
        "previous_actions": prefix,
        "weights": np.zeros(100, np.float32),
    }
    disabled = policy.infer(dict(observation, rtc=request), noise=noise)["actions"]
    np.testing.assert_array_equal(disabled, baseline)
    request["weights"][:30] = 1.0
    request["weights"][30:60] = np.linspace(1, 0, 30, dtype=np.float32)
    started = time.monotonic()
    guided = policy.infer(dict(observation, rtc=request), noise=noise)
    compile_s = time.monotonic() - started
    started = time.monotonic()
    repeated = policy.infer(dict(observation, rtc=request), noise=noise)
    warm_s = time.monotonic() - started
    assert guided["rtc"] == {"schema": SCHEMA, "request_id": 1}
    assert np.isfinite(guided["actions"]).all()
    np.testing.assert_array_equal(guided["actions"], repeated["actions"])
    # Verify the exact deployed training transform, independent of sampling.
    from openpi.shared import normalize
    from openpi.transforms import Normalize

    normalized = policy._policy._input_transform(
        dict(observation, actions=prefix.copy())
    )["actions"]
    expected = Normalize(
        normalize.load(assets.norm_stats_path.parent), use_quantiles=True
    )({"actions": prefix})["actions"]
    np.testing.assert_allclose(normalized[:, :14], expected)
    np.testing.assert_array_equal(normalized[:, 14:], 0)
    # Exercise the copied KV-cache/attention/negative-Euler loop directly.
    # Going only through an all-zero RPC would test the baseline bypass alone.
    import jax
    import jax.numpy as jnp
    from openpi.models.model import Observation

    inputs = policy._policy._input_transform(dict(observation, actions=prefix.copy()))
    direct_prefix = inputs.pop("actions")
    inputs = jax.tree.map(lambda value: jnp.asarray(value)[None], inputs)
    direct = policy._rtc_runner.sampler(
        jax.random.key(42),
        Observation.from_dict(inputs),
        jnp.asarray(direct_prefix)[None],
        jnp.zeros((1, 100, 32), jnp.float32),
        num_steps=args.num_inference_steps,
        noise=jnp.asarray(noise)[None],
    )
    direct = policy._policy._output_transform(
        {
            "actions": np.asarray(direct[0]),
            "state": np.asarray(inputs["state"][0]),
        }
    )["actions"]
    np.testing.assert_allclose(direct, baseline, rtol=1e-5, atol=1e-5)
    normalizer = Normalize(
        normalize.load(assets.norm_stats_path.parent), use_quantiles=True
    )
    normalized_baseline = normalizer({"actions": baseline})["actions"]
    normalized_guided = normalizer({"actions": guided["actions"]})["actions"]
    weights = request["weights"][:, None]

    def weighted_rmse(actions):
        return float(
            np.sqrt(np.sum(weights * (actions - expected) ** 2) / (weights.sum() * 14))
        )

    report = {
        "passed": True,
        "checkpoint": str(assets.checkpoint_path),
        "rtc": policy.metadata["rtc"],
        "zero_mask_exact_baseline": True,
        "direct_zero_mask_sampler_matches_baseline": True,
        "direct_zero_mask_sampler_exact": bool(np.array_equal(direct, baseline)),
        "direct_zero_mask_sampler_max_abs_difference": float(
            np.max(np.abs(direct - baseline))
        ),
        "same_noise_repeat_exact": True,
        "exact_training_prefix_normalization": True,
        "baseline_first_seconds_including_compile": baseline_compile_s,
        "baseline_warm_seconds": baseline_warm_s,
        "guided_first_seconds_including_compile": compile_s,
        "guided_warm_seconds": warm_s,
        "finite_shape": list(guided["actions"].shape),
        "used_recorded_observation": bool(args.manifest),
        "prefix_x_error_before_mm": float(
            np.abs(baseline[:30, 0] - prefix[:30, 0]).mean() * 1000
        ),
        "prefix_x_error_after_mm": float(
            np.abs(guided["actions"][:30, 0] - prefix[:30, 0]).mean() * 1000
        ),
        "weighted_normalized_prefix_rmse_before": weighted_rmse(normalized_baseline),
        "weighted_normalized_prefix_rmse_after": weighted_rmse(normalized_guided),
        "guidance_reduces_prefix_rmse": weighted_rmse(normalized_guided)
        < weighted_rmse(normalized_baseline),
        "timing_caveat": "Existing GPU processes may contend; one warm call is diagnostic, not a throughput benchmark",
        "scope": "Offline implementation smoke, no robot and no closed-loop continuity/success guarantee",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez(
        args.output.with_suffix(".npz"),
        baseline=baseline,
        prefix=prefix,
        guided=guided["actions"],
        weights=request["weights"],
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
