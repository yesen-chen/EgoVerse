"""Mixed training checkpoints keep both domains but deploy only Piper semantics."""

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pi05/src"))

from egoverse_pi05.assets import (  # noqa: E402
    build_contract,
    load_inference_assets,
    sha256,
    validate_resume_contract,
    write_portable_assets,
)

from egomimic.rldb.action_timing import ActionTiming  # noqa: E402
from egomimic.rldb.ego_boundary_policy import EgoBoundaryPolicy  # noqa: E402
from egomimic.robot.piper_pico import pi05_ego_data as ego  # noqa: E402
from egomimic.robot.piper_pico.pi05_data import (  # noqa: E402
    canonical_hash,
    validate_manifest,
)
from egomimic.robot.piper_pico.pi05_training_data import (  # noqa: E402
    export_mixed_manifest,
    load_training_manifest,
)
from tests.unit.test_pi05_jax_config import fixture_manifest  # noqa: E402


def _resign(value):
    value["manifest_sha256"] = canonical_hash(
        {k: v for k, v in value.items() if k != "manifest_sha256"}
    )


def mixed_fixture(tmp_path):
    piper = fixture_manifest()
    piper["action_timing"] = ActionTiming(30, 50, 1, 100).to_dict()
    _resign(piper)
    record = {
        "path": "ego_source.zarr",
        "total_frames": 70,
        "eligible_starts": 41,
        "end_reason": "ui_trim",
        "requires_full_window": True,
        "source_recording_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
        "numeric_sha256": "c" * 64,
        "split": "train",
        "prompt": "Pick up the object",
        "contract": {},
        "image_shapes": {"front_1": [12, 16, 3]},
    }
    human = {
        "schema": ego.SCHEMA,
        "action_timing": ActionTiming(30, 30, 1, 100).to_dict(
            episode_end_padding=ego.PADDING
        ),
        "action_contract": copy.deepcopy(ego.ACTION_CONTRACT),
        "gripper_mode": "zero",
        "boundary_policy": EgoBoundaryPolicy().to_dict(),
        "image_shapes": record["image_shapes"],
        "split": {"unit": "source_recording_sha256"},
        "training_contracts": [{}],
        "training_contract": {},
        "episodes": [record],
        "data_identity_sha256": ego._identity([record]),
    }
    _resign(human)
    sources = tmp_path / "source_manifests"
    sources.mkdir()
    p = sources / "piper.json"
    e = sources / "ego.json"
    p.write_text(json.dumps(piper))
    e.write_text(json.dumps(human))
    path = export_mixed_manifest(p, e, tmp_path / "mixed.json")
    mixed = load_training_manifest(path)
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    domains = {}
    for domain, filename, center in (
        ("piper", "norm_stats.json", 1),
        ("ego", "ego_norm_stats.json", 7),
    ):
        stats = {
            "norm_stats": {
                key: {
                    "mean": [center] * 14,
                    "std": [1] * 14,
                    "q01": [center - 1] * 14,
                    "q99": [center + 1] * 14,
                }
                for key in ("state", "actions")
            }
        }
        file = stats_dir / filename
        file.write_text(json.dumps(stats))
        domains[domain] = {
            "filename": filename,
            "sha256": sha256(file),
            "manifest_sha256": mixed["sources"][domain]["manifest"]["manifest_sha256"],
        }
    (stats_dir / "stats_report.json").write_text(
        json.dumps(
            {
                "manifest_sha256": mixed["manifest_sha256"],
                "split": "train",
                "domains": domains,
            }
        )
    )
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"fixture tokenizer")
    return mixed, stats_dir, tokenizer


def checkpoint_fixture(tmp_path):
    mixed, stats, tokenizer = mixed_fixture(tmp_path)
    contract = build_contract(
        manifest=mixed,
        stats_path=stats / "norm_stats.json",
        tokenizer_path=tokenizer,
        finetune_mode="lora",
        training={"batch_size": 2},
    )
    checkpoint = tmp_path / "checkpoints/run/1"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text(
        json.dumps({"commit_timestamp_nsecs": 42})
    )
    (checkpoint / "params/_METADATA").write_text(
        json.dumps(
            {
                "tree_metadata": {
                    "weight": {"key_metadata": [{"key": "params"}, {"key": "lora_a"}]}
                }
            }
        )
    )
    write_portable_assets(
        checkpoint / "assets", contract, stats / "norm_stats.json", tokenizer
    )
    return checkpoint, contract, stats


def test_mixed_portable_assets_use_robot_timing_and_stats_without_sources(tmp_path):
    checkpoint, contract, stats = checkpoint_fixture(tmp_path)
    shutil.rmtree(tmp_path / "source_manifests")
    shutil.rmtree(stats)
    (tmp_path / "mixed.json").unlink()
    (tmp_path / "tokenizer.model").unlink()
    assets = load_inference_assets(checkpoint)
    assert assets.metadata["action_timing"]["effective_span_s"] == pytest.approx(
        49 / 30
    )
    assert contract["training_manifest"]["action_timings"]["ego"][
        "effective_span_s"
    ] == pytest.approx(29 / 30)
    assert assets.norm_stats["actions"]["mean"] == [1] * 14
    assert assets.metadata["manifest"]["schema"] == "egoverse_pi05_manifest_v1"
    validate_manifest(assets.metadata["manifest"])
    with pytest.raises(ValueError, match="schema"):
        validate_manifest(contract["training_manifest"])


@pytest.mark.parametrize("filename", ["norm_stats.json", "ego_norm_stats.json"])
def test_corruption_of_either_domain_statistics_rejected(tmp_path, filename):
    checkpoint, _, _ = checkpoint_fixture(tmp_path)
    file = checkpoint / "assets/egoverse" / filename
    file.write_text(file.read_text() + " ")
    with pytest.raises(ValueError, match="SHA256"):
        load_inference_assets(checkpoint)


def test_ego_change_rejected_on_resume_and_piper_profile_mismatch_rejected(tmp_path):
    checkpoint, original, _ = checkpoint_fixture(tmp_path)
    altered = copy.deepcopy(original)
    altered["training_manifest"]["sources"]["ego"]["manifest"]["gripper_mode"] = "width"
    with pytest.raises(ValueError, match="Resume contract differs"):
        validate_resume_contract(original, altered)
    altered = copy.deepcopy(original)
    altered["manifest"]["action_timing"] = ActionTiming(30, 30, 1, 100).to_dict()
    _resign(altered["manifest"])
    (checkpoint / "assets/egoverse/contract.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="deployment manifest"):
        load_inference_assets(checkpoint)


def test_statistics_changed_after_contract_creation_not_packaged(tmp_path):
    checkpoint, contract, stats = checkpoint_fixture(tmp_path)
    file = stats / "ego_norm_stats.json"
    file.write_text(file.read_text() + " ")
    with pytest.raises(ValueError, match="statistics changed"):
        write_portable_assets(
            checkpoint / "other_assets",
            contract,
            stats / "norm_stats.json",
            tmp_path / "tokenizer.model",
        )
