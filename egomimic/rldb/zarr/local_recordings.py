"""Strict local recording discovery and source-group train/validation splits.

The legacy SQL/Eva resolver is deliberately untouched. Conversion metadata
identifies a recording independently of its output directory or segment name.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch
from omegaconf import OmegaConf

from egomimic.rldb.embodiment.pico_ego import PicoEgoDataset
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.piper_dataset import PiperDataset
from egomimic.rldb.zarr.zarr_dataset_multi import (
    EpisodeResolver,
    MultiDataset,
    ZarrEpisode,
    split_dataset_names,
)


def recording_group(metadata: dict) -> str:
    """Keep alternate conversions and all segments of a recording together."""
    pico = metadata.get("pico_ego", {})
    tracking = pico.get("source_tracking", {})
    if tracking.get("sha256"):
        return "pico:" + str(tracking["sha256"])
    source = metadata.get("conversion", {}).get("source_dataset")
    if not source:
        raise ValueError(
            "Missing original recording provenance: conversion.source_dataset"
        )
    # A robot capture directory is one group even if it contains many episodes.
    return "recording:" + hashlib.sha256(str(source).encode()).hexdigest()


class RecordingEpisode(torch.utils.data.Dataset):
    """Add unique sample provenance without changing the underlying transforms."""

    def __init__(self, leaf, episode_id):
        self.leaf = leaf
        self.episode_id = episode_id
        self.recording_id = recording_group(leaf.metadata)
        for name in ("metadata", "episode_path", "embodiment", "key_map", "transform"):
            setattr(self, name, getattr(leaf, name))

    def __len__(self):
        return len(self.leaf)

    def normalization_fingerprint(self):
        """Include calibration contents even if a copy preserves file times."""
        from egomimic.rldb.zarr.norm_cache import episode_fingerprint

        digest = hashlib.sha256()
        digest.update(str(episode_fingerprint(self.episode_path)).encode())
        for name in ("zarr.json", ".zattrs", ".zgroup"):
            path = Path(self.episode_path) / name
            if path.is_file():
                digest.update(name.encode())
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def __getitem__(self, index):
        sample = self.leaf[index]
        sample["episode_hash"] = self.episode_id
        sample["recording_id"] = self.recording_id
        sample["sample_index"] = index
        # Epoch seconds must not pass through the generic float32 conversion.
        store = self.leaf.episode_reader._store
        if "timestamp" in store:
            sample["sample_timestamp"] = torch.as_tensor(
                store["timestamp"][index], dtype=torch.float64
            )
        if "timestamp_ns" in store:
            sample["sample_timestamp_ns"] = torch.as_tensor(
                store["timestamp_ns"][index], dtype=torch.int64
            )
        return sample


class LocalRecordingResolver(EpisodeResolver):
    """Load every selected converted episode, or report all failures together."""

    def __init__(
        self,
        folder_path,
        kind,
        key_map,
        transform_list=None,
        allow_synthetic_calibration=False,
        gripper_mode="zero",
        expected_axis_convention="piper_x",
    ):
        super().__init__(folder_path, key_map, transform_list)
        if kind not in ("piper", "pico_ego"):
            raise ValueError(f"Unknown recording kind {kind!r}")
        self.kind = kind
        self.allow_synthetic_calibration = allow_synthetic_calibration
        self.gripper_mode = gripper_mode
        self.expected_axis_convention = expected_axis_convention

    def _memo_fields(self):
        return (str(self.folder_path.resolve()), self.kind)

    def _compute_paths(self, filters, expected_embodiment):
        root = self.folder_path
        if not root.is_dir():
            raise ValueError(f"Recording directory does not exist: {root}")
        paths = [root] if root.suffix == ".zarr" else sorted(root.rglob("*.zarr"))
        paths = [p for p in paths if p.is_dir()]
        selected, errors = [], []
        for path in paths:
            relative = path.name if path == root else path.relative_to(root).as_posix()
            episode_id = relative.removesuffix(".zarr")
            try:
                metadata = dict(ZarrEpisode(path).metadata)
                row = {**metadata, "episode_hash": episode_id}
                if not filters.matches(row):
                    continue
                if (
                    expected_embodiment
                    and metadata.get("embodiment") != expected_embodiment
                ):
                    raise ValueError(
                        f"embodiment {metadata.get('embodiment')!r} != {expected_embodiment}"
                    )
                recording_group(metadata)
                selected.append((str(path), episode_id))
            except Exception as exc:
                errors.append(f"{path}: {exc}")
        missing = set(filters.episode_hashes) - {name for _, name in selected}
        errors.extend(
            f"Pinned episode missing or excluded: {name}" for name in sorted(missing)
        )
        if errors:
            raise ValueError("Recording discovery failed:\n  " + "\n  ".join(errors))
        if not selected:
            raise ValueError(f"No selected converted episodes under {root}")
        return selected

    def load(self, paths):
        if self.kind == "piper":
            cls = PiperDataset
            kwargs = {"allow_synthetic_calibration": self.allow_synthetic_calibration}
        else:
            cls = PicoEgoDataset
            kwargs = {
                "gripper_mode": self.gripper_mode,
                "expected_axis_convention": self.expected_axis_convention,
            }
        datasets, errors = {}, []
        for path, name in paths:
            try:
                leaf = cls(
                    Path(path),
                    key_map=copy.deepcopy(self.key_map),
                    transform_list=copy.deepcopy(self.transform_list),
                    **kwargs,
                )
                datasets[name] = RecordingEpisode(leaf, name)
            except Exception as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
        if errors:
            raise ValueError(
                "Recording preflight failed (no episodes skipped):\n  "
                + "\n  ".join(errors)
            )
        return datasets

    def resolve(self, filters=None, expected_embodiment=None):
        return self.load(self.resolve_paths(filters, expected_embodiment))


class RecordingMultiDataset(MultiDataset):
    """Select whole original recordings before using the existing normalizer."""

    strict_read = True

    def __init__(self, *args, sample_policy="finite", **kwargs):
        if sample_policy not in ("finite", "legacy_quantile"):
            raise ValueError("sample_policy must be finite or legacy_quantile")
        self.sample_policy = sample_policy
        super().__init__(*args, **kwargs)

    def _check_bounds(self, data, dataset, idx, dataset_name):
        if self.sample_policy == "legacy_quantile":
            return super()._check_bounds(data, dataset, idx, dataset_name)
        # Dataset-wide schema/SE3 checks already ran. Finite values outside the
        # training distribution are valid evaluation samples, not corrupt data.
        # Keep their identity; never replace them with an easier random sample.
        emb = data.get("embodiment")
        for key, kind in self.key_types.get(emb, {}).items():
            if kind not in self.NORMALIZE_KEY_TYPES:
                continue
            value = data.get(self.zarr_keys.get(emb, {}).get(key))
            if value is not None and not torch.isfinite(torch.as_tensor(value)).all():
                raise ValueError(f"Nonfinite {key} in {dataset_name}[{idx}]")
        return None

    @classmethod
    def _from_resolver(
        cls,
        resolver,
        dataset_name=None,
        mode="train",
        valid_ratio=0.2,
        split_seed=42,
        filters=None,
        **kwargs,
    ):
        if filters is not None and not isinstance(filters, DatasetFilter):
            raise TypeError("filters must be a DatasetFilter")
        resolved = resolver.resolve(filters, expected_embodiment=dataset_name)
        groups = {leaf.recording_id for leaf in resolved.values()}
        train_groups, valid_groups = split_dataset_names(
            groups, valid_ratio, split_seed
        )
        if mode == "train":
            chosen = train_groups
        elif mode == "valid":
            chosen = valid_groups
        elif mode == "total":
            chosen = groups
        else:
            raise ValueError("Recording split mode must be train, valid or total")
        selected = {k: ds for k, ds in resolved.items() if ds.recording_id in chosen}
        if not selected:
            raise ValueError(
                f"Empty {mode} split from {len(groups)} original recordings. "
                "Provide more recordings or an explicit separate validation directory; "
                "do not split frames from one recording across train/validation."
            )
        result = cls(datasets=selected, mode="total", **kwargs)
        result.train_recordings = train_groups
        result.valid_recordings = valid_groups
        result.selected_recordings = chosen
        return result


def validate_recording_split(train_datasets, valid_datasets):
    """Also catch overlaps when train/valid use different roots or copied exports."""
    train_groups, valid_groups = set(), set()
    for datasets, output in (
        (train_datasets, train_groups),
        (valid_datasets, valid_groups),
    ):
        for dataset in datasets.values():
            if isinstance(dataset, RecordingMultiDataset):
                output.update(dataset.selected_recordings)
    overlap = train_groups & valid_groups
    if overlap:
        raise ValueError(
            f"Original recordings overlap between train/validation: {sorted(overlap)}"
        )


def local_data_contracts(cfg):
    """Save value semantics with the checkpoint, independent of data locations.

    Legacy datasets opt out. Train/validation may use different directories or
    different per-episode calibration, but cannot change the gripper semantics.
    """
    contracts = {}
    for split in ("train_datasets", "valid_datasets"):
        datasets = OmegaConf.select(cfg, f"data.{split}", default={})
        for name, dataset in datasets.items():
            resolver = dataset.get("resolver", {})
            if resolver.get("_target_") != (
                "egomimic.rldb.zarr.local_recordings.LocalRecordingResolver"
            ):
                continue
            kind = resolver["kind"]
            contract = {
                "version": 1,
                "kind": kind,
                "reference_frame": "current_front_1_optical_camera",
                "pose_output": "xyz_m+yaw_pitch_roll_rad_ZYX",
                "hand_order": ["left", "right"],
                "observation_source": "measured_pose",
                "action_source": "command"
                if kind == "piper"
                else "future_observed_pose",
                "gripper_mode": "width"
                if kind == "piper"
                else resolver.get("gripper_mode", "zero"),
                "gripper_encoding": "fixed_width_ratio_no_episode_rescaling",
                "sample_policy": dataset.get("sample_policy", "finite"),
            }
            if kind == "pico_ego":
                contract["axis_convention"] = resolver.get(
                    "expected_axis_convention", "piper_x"
                )
            else:
                contract["calibration"] = "per_episode_T_base_camera"
            if name in contracts and contracts[name] != contract:
                raise ValueError(f"Train/validation data contracts differ for {name}")
            contracts[name] = contract
    return contracts
