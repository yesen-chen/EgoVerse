"""Time contract for uniformly sampled action chunks, opt-in for new recipes.

Matches EgoVerse's current-index window -> [::stride] -> endpoint-preserving
interpolation. It describes target times, not controller frequency, measured
arrival times, or an execution/latency compensation policy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class ActionTiming:
    source_fps: float
    source_horizon: int
    action_stride: int
    output_chunk_length: int
    includes_current: bool = True

    def __post_init__(self):
        if (
            isinstance(self.source_fps, bool)
            or not isinstance(self.source_fps, Real)
            or not math.isfinite(self.source_fps)
            or self.source_fps <= 0
        ):
            raise ValueError("source_fps must be a finite positive number")
        object.__setattr__(self, "source_fps", float(self.source_fps))
        for key in ("source_horizon", "action_stride", "output_chunk_length"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
            object.__setattr__(self, key, int(value))
        if self.includes_current is not True:
            raise ValueError(
                "Stage 1 requires includes_current=True, matching EgoVerse"
            )
        if len(self.support_indices) < 2 or self.output_chunk_length < 2:
            raise ValueError(
                "Interpolation requires at least two support and output points"
            )

    @property
    def support_indices(self) -> tuple[int, ...]:
        return tuple(range(0, self.source_horizon, self.action_stride))

    @property
    def source_span_s(self) -> float:
        """First-to-last source sample span (not the H/fps bin duration)."""
        return (self.source_horizon - 1) / self.source_fps

    @property
    def effective_span_s(self) -> float:
        """First-to-last support span after stride; interpolation preserves it."""
        return self.support_indices[-1] / self.source_fps

    @property
    def output_offsets_s(self) -> tuple[float, ...]:
        return tuple(
            np.linspace(0.0, self.effective_span_s, self.output_chunk_length).tolist()
        )

    def to_dict(self) -> dict:
        return {
            "schema": "action_timing_v1",
            **asdict(self),
            "anchor": "observation_sample_time",
            "source_sampling": "uniform",
            "source_span_s": self.source_span_s,
            "effective_span_s": self.effective_span_s,
            "support_indices": list(self.support_indices),
            "output_offsets_s": list(self.output_offsets_s),
            "episode_end_padding": "repeat_last; offsets are nominal, not measured future observations",
        }


def parse_action_timing(config: Mapping | None) -> dict[str, ActionTiming]:
    """Parse explicit per-embodiment settings; no chosen experimental defaults."""
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise ValueError("action_timing must be a mapping keyed by embodiment")
    result = {}
    for name, values in config.items():
        if not isinstance(name, str) or not name or not isinstance(values, Mapping):
            raise ValueError(
                "action_timing entries must map embodiment names to settings"
            )
        try:
            result[name] = ActionTiming(**values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"action_timing.{name}: {exc}") from exc
    return result


def validate_dataset_timing(specs, train_datasets, valid_datasets) -> None:
    """Reject declarations that disagree with the selected runtime data pipeline.

    Legacy recipes without a declaration are untouched. This checks stored FPS,
    window and resampling settings; it does not measure sensor capture timing.
    """
    if not specs:
        return
    from egomimic.rldb.zarr.action_chunk_transforms import (
        InterpolateLinear,
        InterpolatePose,
    )

    if set(specs) != set(train_datasets):
        raise ValueError(
            "action_timing names must exactly match selected train datasets"
        )
    if set(valid_datasets) - set(specs):
        raise ValueError(
            "Validation datasets have no matching action_timing declaration"
        )

    def leaves(dataset):
        if hasattr(dataset, "datasets"):
            for child in dataset.datasets.values():
                yield from leaves(child)
        else:
            yield dataset

    for split, datasets in (("train", train_datasets), ("valid", valid_datasets)):
        for name, dataset in datasets.items():
            spec = specs[name]
            found = False
            for leaf in leaves(dataset):
                found = True
                label = f"{split}/{name}/{getattr(leaf, 'episode_path', '<episode>')}"
                fps = getattr(leaf, "metadata", {}).get("fps")
                if (
                    isinstance(fps, bool)
                    or not isinstance(fps, Real)
                    or not math.isfinite(fps)
                    or not math.isclose(
                        fps, spec.source_fps, rel_tol=1e-9, abs_tol=1e-9
                    )
                ):
                    raise ValueError(
                        f"{label}: stored fps {fps!r} != source_fps {spec.source_fps}"
                    )
                horizons = [
                    item["horizon"]
                    for item in leaf.key_map.values()
                    if item.get("key_type") == "action_keys" and "horizon" in item
                ]
                if not horizons or any(h != spec.source_horizon for h in horizons):
                    raise ValueError(
                        f"{label}: action horizon {horizons} != {spec.source_horizon}"
                    )
                interpolators = [
                    t
                    for t in leaf.transform or []
                    if isinstance(t, (InterpolatePose, InterpolateLinear))
                ]
                if not interpolators:
                    raise ValueError(
                        f"{label}: no supported action interpolator to verify"
                    )
                for transform in interpolators:
                    if transform.stride != spec.action_stride:
                        raise ValueError(
                            f"{label}: stride {transform.stride} != {spec.action_stride}"
                        )
                    if transform.new_chunk_length != spec.output_chunk_length:
                        raise ValueError(
                            f"{label}: output_chunk_length {transform.new_chunk_length} "
                            f"!= {spec.output_chunk_length}"
                        )
            if not found:
                raise ValueError(
                    f"{split}/{name}: no episodes to validate action timing"
                )


def validate_model_timing(specs, model_config) -> None:
    """Check actual action output lengths in the configured HPT or PI model.

    ``model_config`` is the wrapper's ``cfg.model`` subtree. HPT trunk action
    tokens are conditioning tokens, not trajectory samples, and are deliberately
    ignored. The check never instantiates a policy or downloads model weights.
    """
    if not specs:
        return
    if not isinstance(model_config, Mapping):
        raise ValueError(
            "Action timing validation requires a model configuration mapping"
        )
    algo = model_config.get("robomimic_model")
    if not isinstance(algo, Mapping):
        raise ValueError(
            "Unsupported action timing model: missing robomimic_model mapping"
        )

    def check_length(value, expected, location):
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value != expected
        ):
            raise ValueError(
                f"{location}: action output length {value!r} != output_chunk_length {expected}"
            )

    target = algo.get("_target_")
    if target == "egomimic.algo.hpt.HPT":
        heads = algo.get("head_specs")
        if not isinstance(heads, Mapping):
            raise ValueError(
                "Unsupported HPT action timing configuration: missing head_specs"
            )
        shared_enabled = algo.get("shared_ac_key") is not None
        for name, spec in specs.items():
            active_heads = []
            if heads.get(name) is not None:
                active_heads.append((name, heads[name]))
            if shared_enabled:
                # HPT can train/execute a domain head and a shared head together.
                active_heads.append(("shared", heads.get("shared")))
            if not active_heads:
                raise ValueError(
                    f"Unsupported HPT action timing configuration: no active head for {name}"
                )
            for head_name, head in active_heads:
                location = f"model.robomimic_model.head_specs.{head_name}"
                if not isinstance(head, Mapping) or "action_horizon" not in head:
                    raise ValueError(
                        f"Unsupported action timing head at {location}: missing action_horizon"
                    )
                check_length(
                    head["action_horizon"],
                    spec.output_chunk_length,
                    f"{location}.action_horizon",
                )
                network = head.get("model")
                if isinstance(network, Mapping) and "act_seq" in network:
                    check_length(
                        network["act_seq"],
                        spec.output_chunk_length,
                        f"{location}.model.act_seq",
                    )
    elif target == "egomimic.algo.pi.PI":
        config = algo.get("config")
        network = config.get("model") if isinstance(config, Mapping) else None
        if not isinstance(network, Mapping) or "action_horizon" not in network:
            raise ValueError(
                "Unsupported PI action timing configuration: missing config.model.action_horizon"
            )
        for name, spec in specs.items():
            check_length(
                network["action_horizon"],
                spec.output_chunk_length,
                f"model.robomimic_model.config.model.action_horizon ({name})",
            )
    else:
        raise ValueError(f"Unsupported model for action timing validation: {target!r}")
