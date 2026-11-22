"""Select eligible Ego training starts without removing future support rows.

An episode's end determines whether padding would invent future supervision.
A loss at its *start* does not determine the meaning of its final observation.
Legacy episodes without preprocessing metadata keep their original behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral


@dataclass(frozen=True)
class EgoBoundaryPolicy:
    require_full_window_after_trim: bool = True
    require_full_window_after_inactivity: bool = True

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")

    def to_dict(self):
        return {
            "schema": "ego_boundary_policy_v1",
            **asdict(self),
            "decision": "episode_end_reason",
            "full_window": "all source_horizon rows before stride or interpolation",
            "legacy_without_preprocessing": "repeat_last",
            "source_end": "repeat_last",
        }

    def resolve(self, metadata, *, episode_path="<episode>"):
        """Return the final boundary reason and its full-window requirement."""
        if "preprocessing" not in metadata:
            return "legacy", False
        preprocessing = metadata["preprocessing"]
        if (
            not isinstance(preprocessing, Mapping)
            or preprocessing.get("schema") != "ego_preprocessing_v1"
        ):
            raise ValueError(f"{episode_path}: invalid ego preprocessing metadata")
        reason = preprocessing.get("end_reason")
        if reason == "ui_trim":
            required = self.require_full_window_after_trim
        elif reason in ("inactivity", "tracking_gap", "video_gap"):
            required = self.require_full_window_after_inactivity
        elif reason == "source_end":
            required = False
        else:
            raise ValueError(
                f"{episode_path}: unsupported preprocessing.end_reason {reason!r}"
            )
        return reason, required


def action_source_horizon(key_map, *, episode_path="<episode>"):
    """Use the longest requested source window, not the stride's last support."""
    horizons = [
        spec.get("horizon")
        for spec in key_map.values()
        if spec.get("key_type") == "action_keys"
    ]
    if not horizons or any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        for value in horizons
    ):
        raise ValueError(f"{episode_path}: action keys require positive horizons")
    return max(horizons)
