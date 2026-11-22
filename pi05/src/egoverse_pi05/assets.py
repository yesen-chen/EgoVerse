"""Portable, hashed checkpoint contract; deliberately independent of JAX."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from egoverse_pi05 import OPENPI_REVISION

CAMERAS = {
    "front_1": "base_0_rgb",
    "left_wrist": "left_wrist_0_rgb",
    "right_wrist": "right_wrist_0_rgb",
}


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_stats(raw):
    stats = raw.get("norm_stats")
    if not isinstance(stats, dict) or set(stats) != {"state", "actions"}:
        raise ValueError("Expected normalization statistics for state and actions")
    for name, fields in stats.items():
        for field in ("mean", "std", "q01", "q99"):
            value = np.asarray(fields.get(field), dtype=np.float64)
            if value.shape != (14,) or not np.isfinite(value).all():
                raise ValueError(f"Expected 14 finite {name}.{field} values")
        if (np.asarray(fields["std"]) < 0).any() or (
            np.asarray(fields["q99"]) < fields["q01"]
        ).any():
            raise ValueError(f"Invalid normalization bounds: {name}")
    return stats


def validate_manifest_contract(manifest):
    from egomimic.robot.piper_pico.pi05_data import validate_manifest

    # This checks the self-hash, per-episode identities and all deployment
    # profiles without opening any original dataset paths.
    validate_manifest(manifest)
    if manifest.get("schema") != "egoverse_pi05_manifest_v1":
        raise ValueError("Unsupported Pi0.5 dataset manifest")
    timing = manifest.get("action_timing", {})
    # Use the same timing validator as conversion/execution; no implied 100 Hz.
    from egomimic.rldb.action_timing import ActionTiming

    parsed = ActionTiming(
        **{
            key: timing[key]
            for key in (
                "source_fps",
                "source_horizon",
                "action_stride",
                "output_chunk_length",
                "includes_current",
            )
        }
    )
    if timing != parsed.to_dict():
        raise ValueError(
            "Action timing derived fields do not match the declared sampling"
        )
    if parsed.output_chunk_length != 100:
        raise ValueError("This Pi0.5 policy requires exactly 100 model action points")
    contract = manifest.get("action_contract", {})
    if (
        contract.get("dimension") != 14
        or contract.get("representation") != "absolute_xyz_ypr_gripper"
    ):
        raise ValueError("Expected absolute 14-D XYZ/YPR/gripper actions")
    if contract.get("rotation_convention") != "intrinsic_ZYX":
        raise ValueError("Expected intrinsic ZYX rotations")
    for key, expected in (
        ("position_unit", "m"),
        ("rotation_unit", "rad"),
        ("gripper_unit", "normalized_width"),
    ):
        if contract.get(key) != expected:
            raise ValueError(f"Unsupported action {key}; expected {expected}")
    if manifest.get("camera_slots", contract.get("camera_slots")) != CAMERAS:
        raise ValueError("Expected front and both wrist camera slots")
    shapes = manifest.get("image_shapes", {})
    if set(shapes) != set(CAMERAS):
        raise ValueError("Manifest must include all three image_shapes")
    for shape in shapes.values():
        if not isinstance(shape, list) or len(shape) != 3 or shape[2] != 3:
            raise ValueError("Expected HWC RGB image_shapes")
        for dim in shape:
            positive_int(dim, "image dimension")
    if not manifest.get("episodes") or not any(
        e["split"] == "train" for e in manifest["episodes"]
    ):
        raise ValueError("Manifest has no training episodes")
    profiles = manifest.get("training_contracts")
    if (
        not isinstance(profiles, list)
        or not profiles
        or not all(isinstance(profile, dict) for profile in profiles)
    ):
        raise ValueError("Expected nonempty training_contracts for deployment geometry")


def build_contract(*, manifest, stats_path, tokenizer_path, finetune_mode, training):
    from egomimic.robot.piper_pico.pi05_training_data import (
        SCHEMA as MIXED_SCHEMA,
    )
    from egomimic.robot.piper_pico.pi05_training_data import (
        deployment_manifest,
        load_training_stats,
    )

    robot_manifest = deployment_manifest(manifest)
    validate_manifest_contract(robot_manifest)
    validate_stats(read_json(stats_path))
    if finetune_mode not in ("lora", "full"):
        raise ValueError("finetune_mode must be lora or full")
    result = {
        "schema": "egoverse_pi05_checkpoint_v1",
        "openpi_revision": OPENPI_REVISION,
        "protocol_version": 1,
        "policy_type": "pi05",
        "action_horizon": 100,
        "action_dim": 14,
        "model_action_dim": 32,
        "finetune_mode": finetune_mode,
        "stats_sha256": sha256(stats_path),
        "tokenizer_sha256": sha256(tokenizer_path),
        "manifest": robot_manifest,
        "training": copy.deepcopy(training),
    }
    if manifest["schema"] == MIXED_SCHEMA:
        load_training_stats(manifest, Path(stats_path).parent)
        result["training_manifest"] = copy.deepcopy(manifest)
        result["domain_stats"] = {
            "piper": {"filename": "norm_stats.json", "sha256": sha256(stats_path)},
            "ego": {
                "filename": "ego_norm_stats.json",
                "sha256": sha256(Path(stats_path).parent / "ego_norm_stats.json"),
            },
        }
    return result


def write_portable_assets(directory, contract, stats_path, tokenizer_path):
    """Called inside Orbax's asset callback, before that checkpoint commits."""
    target = Path(directory) / "egoverse"
    target.mkdir(parents=True, exist_ok=True)
    for source, name, digest in (
        (stats_path, "norm_stats.json", contract["stats_sha256"]),
        (tokenizer_path, "tokenizer.model", contract["tokenizer_sha256"]),
    ):
        if sha256(source) != digest:
            raise ValueError(f"Training asset changed while running: {source}")
        shutil.copyfile(source, target / name)
    for domain, entry in contract.get("domain_stats", {}).items():
        if domain == "piper":
            continue
        source = Path(stats_path).parent / entry["filename"]
        if sha256(source) != entry["sha256"]:
            raise ValueError(f"Training domain statistics changed: {domain}")
        shutil.copyfile(source, target / entry["filename"])
    (target / "contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n"
    )


def validate_resume_contract(saved, expected):
    if saved != expected:
        raise ValueError(
            "Resume contract differs: data, normalization, model, timing, or optimizer configuration changed"
        )


def validate_checkpoint_mode(path, mode):
    metadata = read_json(path).get("tree_metadata", {})
    if not metadata:
        raise ValueError("Missing checkpoint parameter metadata")
    paths = []
    for entry in metadata.values():
        keys = entry.get("key_metadata", [])
        if not keys or keys[0].get("key") != "params":
            raise ValueError("Checkpoint must contain the official OpenPI params tree")
        paths.append("/".join(str(key["key"]) for key in keys[1:]))
    if any("lora" in path for path in paths) != (mode == "lora"):
        raise ValueError(
            "Checkpoint finetune mode differs; refusing to discard LoRA weights"
        )


@dataclass(frozen=True)
class InferenceAssets:
    checkpoint_path: Path
    tokenizer_path: Path
    norm_stats_path: Path
    metadata: dict
    norm_stats: dict


def inference_summary(metadata):
    """Bounded startup output; the complete contract stays in the RPC handshake."""
    timing = metadata["action_timing"]
    return {
        key: copy.deepcopy(metadata[key])
        for key in (
            "policy_type",
            "checkpoint",
            "finetune_mode",
            "action_horizon",
            "action_dim",
            "num_inference_steps",
            "stats_sha256",
            "tokenizer_sha256",
            "image_shapes",
        )
    } | {
        "training_episodes": len(metadata["manifest"]["episodes"]),
        "training_geometry_profiles": len(metadata["training_contracts"]),
        "action_timing": {
            key: timing[key]
            for key in (
                "source_fps",
                "source_horizon",
                "action_stride",
                "output_chunk_length",
                "effective_span_s",
            )
        },
    }


def load_inference_assets(checkpoint_path, num_inference_steps=10):
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_dir() or not checkpoint.name.isdigit():
        raise ValueError("checkpoint-path must name a specific numeric checkpoint step")
    marker = read_json(checkpoint / "_CHECKPOINT_METADATA")
    if (
        not isinstance(marker.get("commit_timestamp_nsecs"), int)
        or marker["commit_timestamp_nsecs"] <= 0
    ):
        raise ValueError("Checkpoint has not finished committing")
    root = checkpoint / "assets/egoverse"
    contract = read_json(root / "contract.json")
    if (
        contract.get("schema") != "egoverse_pi05_checkpoint_v1"
        or contract.get("openpi_revision") != OPENPI_REVISION
    ):
        raise ValueError("Unsupported Pi0.5 checkpoint contract or upstream revision")
    if (
        contract.get("action_dim"),
        contract.get("action_horizon"),
        contract.get("model_action_dim"),
    ) != (14, 100, 32):
        raise ValueError("Expected 100-point, 14-D physical / 32-D padded Pi0.5 policy")
    mode = contract.get("finetune_mode")
    if mode not in ("lora", "full"):
        raise ValueError("Unsupported finetune mode")
    validate_checkpoint_mode(checkpoint / "params/_METADATA", mode)
    manifest = contract["manifest"]
    validate_manifest_contract(manifest)
    if "training_manifest" in contract:
        from egomimic.robot.piper_pico.pi05_training_data import (
            SCHEMA as MIXED_SCHEMA,
        )
        from egomimic.robot.piper_pico.pi05_training_data import (
            deployment_manifest,
            validate_training_manifest,
        )

        mixed = validate_training_manifest(contract["training_manifest"])
        if mixed["schema"] != MIXED_SCHEMA or deployment_manifest(mixed) != manifest:
            raise ValueError(
                "Mixed checkpoint deployment manifest differs from Piper source"
            )
        domain_stats = contract.get("domain_stats", {})
        if set(domain_stats) != {"piper", "ego"}:
            raise ValueError("Mixed checkpoint requires both domain statistics")
        for domain, filename in (
            ("piper", "norm_stats.json"),
            ("ego", "ego_norm_stats.json"),
        ):
            entry = domain_stats[domain]
            if entry.get("filename") != filename or sha256(
                root / filename
            ) != entry.get("sha256"):
                raise ValueError(f"Checkpoint {domain} statistics SHA256 mismatch")
            validate_stats(read_json(root / filename))
        if domain_stats["piper"]["sha256"] != contract.get("stats_sha256"):
            raise ValueError(
                "Mixed checkpoint Piper statistics disagree with deployment"
            )
    elif "domain_stats" in contract:
        raise ValueError("Domain statistics require a mixed training manifest")
    for name, key in (
        ("norm_stats.json", "stats_sha256"),
        ("tokenizer.model", "tokenizer_sha256"),
    ):
        if sha256(root / name) != contract.get(key):
            raise ValueError(f"Checkpoint {name} SHA256 mismatch")
    stats = validate_stats(read_json(root / "norm_stats.json"))
    metadata = {
        key: copy.deepcopy(contract[key])
        for key in (
            "protocol_version",
            "policy_type",
            "action_horizon",
            "action_dim",
            "model_action_dim",
            "finetune_mode",
            "stats_sha256",
            "tokenizer_sha256",
            "manifest",
        )
    }
    metadata.update(
        num_inference_steps=positive_int(num_inference_steps, "num_inference_steps"),
        action_timing=copy.deepcopy(manifest["action_timing"]),
        image_shapes=copy.deepcopy(manifest["image_shapes"]),
        action_contract=copy.deepcopy(manifest["action_contract"]),
        training_contracts=copy.deepcopy(manifest["training_contracts"]),
        checkpoint=str(checkpoint),
    )
    return InferenceAssets(
        checkpoint, root / "tokenizer.model", root / "norm_stats.json", metadata, stats
    )
