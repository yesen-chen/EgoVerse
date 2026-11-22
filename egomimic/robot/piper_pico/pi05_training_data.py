"""Training-only composition of Piper and PICO Ego Pi0.5 manifests.

The robot handshake deliberately continues to accept only Piper manifests.
Each domain keeps its own physical time grid and training normalization. A
mixture aligns normalized trajectory phase, not sensor timestamps or speed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from egomimic.robot.piper_pico import pi05_data as piper

SCHEMA = "egoverse_pi05_mixed_manifest_v1"
DOMAINS = ("piper", "ego")
STATS_FILES = {"piper": "norm_stats.json", "ego": "ego_norm_stats.json"}
POLICY = {
    "normalization": "separate_train_only_per_domain",
    "action_alignment": "normalized_phase_100_points; source_timestamps_unchanged",
    "deployment_domain": "piper",
    "ego_missing_wrists": "zero_images_with_false_masks",
    "ego_zero_gripper": "zero_after_normalization_before_state_tokenization",
}


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _child_validator(domain):
    if domain == "piper":
        return piper.validate_manifest
    from egomimic.robot.piper_pico.pi05_ego_data import validate_ego_manifest

    return validate_ego_manifest


def validate_training_manifest(value):
    """Validate portable metadata without opening the original dataset."""
    if value.get("schema") == piper.SCHEMA:
        return piper.validate_manifest(value)
    if value.get("schema") != SCHEMA:
        raise ValueError("Expected a Piper or mixed Piper/Ego training manifest")
    if value.get("manifest_sha256") != piper._manifest_hash(value):
        raise ValueError("Mixed manifest checksum mismatch")
    if value.get("policy") != POLICY or set(value.get("sources", {})) != set(DOMAINS):
        raise ValueError("Unsupported mixed training contract")
    for domain, source in value["sources"].items():
        path = source.get("manifest_path")
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            raise ValueError("Child manifest path must be relative")
        _child_validator(domain)(source["manifest"])
    sampling = value.get("sampling", {})
    if sampling.get("method") != "weighted_complete_domain_cycles":
        raise ValueError("Unsupported mixed sampling method")
    if set(sampling.get("weights", {})) != set(DOMAINS):
        raise ValueError("Both domain sampling weights are required")
    for domain in DOMAINS:
        _positive_int(sampling["weights"][domain], f"{domain} weight")
    seed = sampling.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Sampling seed must be a nonnegative integer")
    identity = {
        domain: source["manifest"]["data_identity_sha256"]
        for domain, source in value["sources"].items()
    }
    if value.get("data_identity_sha256") != piper.canonical_hash(identity):
        raise ValueError("Mixed data identity mismatch")
    expected_timing = {
        domain: source["manifest"]["action_timing"]
        for domain, source in value["sources"].items()
    }
    if value.get("action_timings") != expected_timing:
        raise ValueError("Mixed timing disagrees with source windows")
    return value


def load_training_manifest(path):
    return validate_training_manifest(json.loads(Path(path).read_text()))


def deployment_manifest(manifest):
    """Expose only the Piper contract to robot execution/inverse transforms."""
    validate_training_manifest(manifest)
    if manifest["schema"] == piper.SCHEMA:
        return copy.deepcopy(manifest)
    return copy.deepcopy(manifest["sources"]["piper"]["manifest"])


def export_mixed_manifest(
    piper_manifest, ego_manifest, output_path, *, piper_weight=1, ego_weight=1, seed=42
):
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    sources = {}
    for domain, path in zip(DOMAINS, (piper_manifest, ego_manifest), strict=True):
        path = Path(path).expanduser().resolve()
        manifest = _child_validator(domain)(json.loads(path.read_text()))
        sources[domain] = {
            "manifest_path": os.path.relpath(path, destination.parent),
            "manifest": manifest,
        }
    value = {
        "schema": SCHEMA,
        "sources": sources,
        "policy": copy.deepcopy(POLICY),
        "sampling": {
            "method": "weighted_complete_domain_cycles",
            "weights": {"piper": piper_weight, "ego": ego_weight},
            "seed": seed,
        },
        "action_timings": {
            domain: source["manifest"]["action_timing"]
            for domain, source in sources.items()
        },
        "data_identity_sha256": piper.canonical_hash(
            {
                domain: source["manifest"]["data_identity_sha256"]
                for domain, source in sources.items()
            }
        ),
    }
    value["manifest_sha256"] = piper._manifest_hash(value)
    validate_training_manifest(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return destination


def domain_datasets(manifest_path, *, split="train", include_images=False):
    """Unweighted unique eligible starts, shared by training and statistics."""
    path = Path(manifest_path).resolve()
    manifest = load_training_manifest(path)
    if manifest["schema"] == piper.SCHEMA:
        return {"piper": piper.Pi05Dataset(path, split, include_images)}
    from egomimic.robot.piper_pico.pi05_ego_data import EgoPi05Dataset

    result = {}
    for domain, cls in (("piper", piper.Pi05Dataset), ("ego", EgoPi05Dataset)):
        source = manifest["sources"][domain]
        child = (path.parent / source["manifest_path"]).resolve()
        if json.loads(child.read_text()) != source["manifest"]:
            raise ValueError(f"{domain} child manifest changed since mixture export")
        result[domain] = cls(child, split=split, include_images=include_images)
    return result


class TrainingDataset:
    """Deterministic weighted mixture with every training start covered per cycle.

    Shorter domains are cycled to achieve the requested integer ratio. Upstream
    DataLoader shuffles this finite schedule each epoch. Validation is a simple
    concatenation, with no oversampling. Statistics never use this schedule.
    """

    def __init__(self, manifest_path, split="train", include_images=True):
        self.manifest = load_training_manifest(manifest_path)
        self.datasets = domain_datasets(
            manifest_path, split=split, include_images=include_images
        )
        self.schedule = []
        if self.manifest["schema"] == SCHEMA and split == "train":
            sampling = self.manifest["sampling"]
            weights = sampling["weights"]
            if any(len(dataset) < 1 for dataset in self.datasets.values()):
                raise ValueError("Both domains need eligible training windows")
            blocks = max(
                math.ceil(len(ds) / weights[d]) for d, ds in self.datasets.items()
            )
            rng = np.random.default_rng(sampling["seed"])
            orders = {d: rng.permutation(len(ds)) for d, ds in self.datasets.items()}
            for block in range(blocks):
                for domain in DOMAINS:
                    order = orders[domain]
                    for offset in range(weights[domain]):
                        index = int(
                            order[(block * weights[domain] + offset) % len(order)]
                        )
                        self.schedule.append((domain, index))
        else:
            self.schedule = [
                (d, i) for d, ds in self.datasets.items() for i in range(len(ds))
            ]

    def __len__(self):
        return len(self.schedule)

    def __getitem__(self, index):
        if (
            isinstance(index, bool)
            or not isinstance(index, (int, np.integer))
            or not 0 <= index < len(self)
        ):
            raise IndexError(index)
        domain, source_index = self.schedule[int(index)]
        sample = dict(self.datasets[domain][source_index])
        sample["domain"] = domain
        if domain == "piper":
            sample["gripper_mode"] = "width"
        return sample


def _dataset_stats(dataset, max_windows):
    if max_windows is not None:
        _positive_int(max_windows, "max_windows")
    if not len(dataset):
        raise ValueError("Cannot compute statistics without eligible training windows")
    count = len(dataset) if max_windows is None else min(len(dataset), max_windows)
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    samples = [dataset[int(i)] for i in indices]
    values = {
        "state": np.stack([s["observation.state"] for s in samples]).astype(np.float64),
        "actions": np.concatenate([s["actions"] for s in samples]).astype(np.float64),
    }
    stats = {
        key: {
            "mean": a.mean(axis=0).tolist(),
            "std": a.std(axis=0).tolist(),
            "q01": np.quantile(a, 0.01, axis=0).tolist(),
            "q99": np.quantile(a, 0.99, axis=0).tolist(),
        }
        for key, a in values.items()
    }
    return {"norm_stats": stats}, count


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_training_stats(manifest_path, output_dir, *, max_windows=None):
    """Fresh separate statistics over each domain's unique training starts."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(
            "Statistics output already exists; use a fresh output-dir"
        )
    manifest = load_training_manifest(manifest_path)
    datasets = domain_datasets(manifest_path, split="train", include_images=False)
    computed = {
        domain: _dataset_stats(ds, max_windows) for domain, ds in datasets.items()
    }
    output.mkdir(parents=True)
    domain_reports = {}
    for domain, (stats, count) in computed.items():
        file = output / STATS_FILES[domain]
        file.write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")
        child = (
            manifest
            if manifest["schema"] == piper.SCHEMA
            else manifest["sources"][domain]["manifest"]
        )
        domain_reports[domain] = {
            "filename": STATS_FILES[domain],
            "sha256": _file_hash(file),
            "manifest_sha256": child["manifest_sha256"],
            "eligible_training_windows": len(datasets[domain]),
            "sampled_windows": count,
        }
    report = {
        "manifest_sha256": manifest["manifest_sha256"],
        "data_identity_sha256": manifest["data_identity_sha256"],
        "split": "train",
        "action_horizon": 100,
        "action_dim": 14,
        "max_windows": max_windows,
        "smoke_test_statistics": max_windows is not None,
        "normalization": "OpenPI q01/q99; separate domains; unique eligible training starts; no clipping",
        "domains": domain_reports,
    }
    (output / "stats_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_training_stats(manifest, stats_dir):
    """Validate identities before normalization; legacy Piper reports remain valid."""
    from egoverse_pi05.assets import validate_stats

    validate_training_manifest(manifest)
    root = Path(stats_dir)
    report = json.loads((root / "stats_report.json").read_text())
    if (
        report.get("manifest_sha256") != manifest["manifest_sha256"]
        or report.get("split") != "train"
    ):
        raise ValueError(
            "Normalization statistics do not match this training manifest/split"
        )
    domains = DOMAINS if manifest["schema"] == SCHEMA else ("piper",)
    if manifest["schema"] == SCHEMA and set(report.get("domains", {})) != set(DOMAINS):
        raise ValueError("Mixed training requires separate Piper and Ego statistics")
    result = {}
    for domain in domains:
        path = root / STATS_FILES[domain]
        entry = report.get("domains", {}).get(domain)
        child = (
            manifest
            if manifest["schema"] == piper.SCHEMA
            else manifest["sources"][domain]["manifest"]
        )
        if entry is not None and (
            entry.get("filename") != STATS_FILES[domain]
            or entry.get("sha256") != _file_hash(path)
            or entry.get("manifest_sha256") != child["manifest_sha256"]
        ):
            raise ValueError(f"{domain} normalization identity mismatch")
        raw = json.loads(path.read_text())
        validate_stats(raw)
        result[domain] = raw
    return result
