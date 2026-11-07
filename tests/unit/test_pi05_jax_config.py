"""Portable Pi0.5 contract and serving checks need neither OpenPI nor a GPU."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PI05_SRC = Path(__file__).resolve().parents[2] / "pi05/src"
sys.path.insert(0, str(PI05_SRC))

from egoverse_pi05.assets import (  # noqa: E402
    CAMERAS,
    build_contract,
    inference_summary,
    load_inference_assets,
    validate_resume_contract,
    validate_stats,
    write_portable_assets,
)
from egoverse_pi05.inference import BimanualInferencePolicy  # noqa: E402
from egoverse_pi05.training import (  # noqa: E402
    initialize_with_step_zero,
    validate_device_topology,
)

from egomimic.rldb.action_timing import ActionTiming  # noqa: E402
from egomimic.robot.piper_pico.pi05_data import (  # noqa: E402
    ACTION_CONTRACT,
    canonical_hash,
)


def fixture_manifest():
    profile = {
        key: None
        for key in (
            "calibration_status",
            "extrinsics",
            "intrinsics",
            "gripper_encoding",
            "image_processing",
            "image_space",
            "tool_frame",
        )
    }
    profile["calibration"] = {"data": {"test": True}}
    shapes = {key: [12, 16, 3] for key in CAMERAS}
    record = {
        "path": "source.zarr",
        "total_frames": 10,
        "metadata_sha256": "b" * 64,
        "numeric_sha256": "c" * 64,
        "split": "train",
        "prompt": "Pick up the object",
        "image_shapes": shapes,
        "contract": {**profile, "time_alignment": None, "source_capture_config": None},
    }
    manifest = {
        "schema": "egoverse_pi05_manifest_v1",
        "action_timing": ActionTiming(30, 30, 1, 100).to_dict(),
        "action_contract": copy.deepcopy(ACTION_CONTRACT),
        "image_shapes": shapes,
        "training_contracts": [profile],
        "training_contract": profile,
        "episodes": [record],
        "data_identity_sha256": canonical_hash(
            [
                {
                    key: record[key]
                    for key in (
                        "metadata_sha256",
                        "numeric_sha256",
                        "total_frames",
                    )
                }
            ]
        ),
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    return manifest


@pytest.fixture
def portable_checkpoint(tmp_path):
    stats_path = tmp_path / "norm_stats.json"
    stats = {
        key: {
            "mean": [0.0] * 14,
            "std": [1.0] * 14,
            "q01": [-1.0] * 14,
            "q99": [1.0] * 14,
        }
        for key in ("state", "actions")
    }
    stats_path.write_text(json.dumps({"norm_stats": stats}))
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"fixture tokenizer")
    contract = build_contract(
        manifest=fixture_manifest(),
        stats_path=stats_path,
        tokenizer_path=tokenizer,
        finetune_mode="lora",
        training={"batch_size": 1},
    )
    checkpoint = tmp_path / "checkpoints/run/1"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text(
        json.dumps({"commit_timestamp_nsecs": 123})
    )
    (checkpoint / "params/_METADATA").write_text(
        json.dumps(
            {
                "tree_metadata": {
                    "x": {
                        "key_metadata": [
                            {"key": "params"},
                            {"key": "PaliGemma"},
                            {"key": "lora_a"},
                        ]
                    }
                }
            }
        )
    )
    write_portable_assets(checkpoint / "assets", contract, stats_path, tokenizer)
    return checkpoint, contract


def test_portable_checkpoint_needs_no_original_data_or_tokenizer(portable_checkpoint):
    checkpoint, _ = portable_checkpoint
    original = checkpoint.parents[2]
    (original / "norm_stats.json").unlink()
    (original / "tokenizer.model").unlink()
    assets = load_inference_assets(checkpoint, 20)
    assert assets.metadata["action_horizon"] == 100
    assert assets.metadata["action_dim"] == 14
    assert assets.metadata["num_inference_steps"] == 20
    assert assets.metadata["action_timing"]["effective_span_s"] == pytest.approx(
        29 / 30
    )
    assert "training_contracts" in assets.metadata


def test_startup_summary_is_bounded_and_preserves_full_handshake(portable_checkpoint):
    checkpoint, _ = portable_checkpoint
    metadata = load_inference_assets(checkpoint).metadata
    metadata["manifest"]["episodes"] *= 44
    metadata["manifest"]["diagnostic_blob"] = "x" * 1_000_000
    before = copy.deepcopy(metadata)
    summary = inference_summary(metadata)
    assert summary["training_episodes"] == 44
    assert summary["action_timing"]["output_chunk_length"] == 100
    assert len(json.dumps(summary)) < 4096
    assert "manifest" not in summary
    assert "training_contracts" not in summary
    assert metadata == before


@pytest.mark.parametrize("name", ["norm_stats.json", "tokenizer.model"])
def test_corrupt_checkpoint_assets_rejected(portable_checkpoint, name):
    checkpoint, _ = portable_checkpoint
    with (checkpoint / "assets/egoverse" / name).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_inference_assets(checkpoint)


def test_base_only_tree_cannot_silently_drop_lora(portable_checkpoint):
    checkpoint, _ = portable_checkpoint
    path = checkpoint / "params/_METADATA"
    path.write_text(path.read_text().replace("lora_a", "kernel"))
    with pytest.raises(ValueError, match="discard LoRA"):
        load_inference_assets(checkpoint)


def test_uncommitted_checkpoint_rejected(portable_checkpoint):
    checkpoint, _ = portable_checkpoint
    (checkpoint / "_CHECKPOINT_METADATA").write_text("{}")
    with pytest.raises(ValueError, match="committing"):
        load_inference_assets(checkpoint)


def test_tampered_portable_manifest_rejected_without_source_access(portable_checkpoint):
    checkpoint, _ = portable_checkpoint
    path = checkpoint / "assets/egoverse/contract.json"
    value = json.loads(path.read_text())
    value["manifest"]["episodes"][0]["prompt"] = "Changed task"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="manifest checksum"):
        load_inference_assets(checkpoint)


def test_resume_rejects_changed_schedule_or_data(portable_checkpoint):
    _, contract = portable_checkpoint
    validate_resume_contract(contract, copy.deepcopy(contract))
    changed = copy.deepcopy(contract)
    changed["manifest"]["action_timing"]["source_fps"] = 25
    with pytest.raises(ValueError, match="Resume contract differs"):
        validate_resume_contract(contract, changed)


def test_stats_reject_padded_dimensions_and_nonfinite():
    stats = {
        key: {field: [1.0] * 14 for field in ("mean", "std", "q01", "q99")}
        for key in ("state", "actions")
    }
    validate_stats({"norm_stats": stats})
    stats["actions"]["mean"] = [0.0] * 32
    with pytest.raises(ValueError, match="14 finite"):
        validate_stats({"norm_stats": stats})
    stats["actions"]["mean"] = [float("nan")] * 14
    with pytest.raises(ValueError, match="14 finite"):
        validate_stats({"norm_stats": stats})


def test_topology_and_step_zero_resume():
    validate_device_topology(4, 2, 2)
    for values in ((1, 1, 2), (4, 3, 2), (True, 1, 1), (1, 1, 1, 2)):
        with pytest.raises(ValueError):
            validate_device_topology(*values)

    class Manager:
        def all_steps(self):
            return (0,)

    assert initialize_with_step_zero(lambda **kw: (Manager(), False), resume=True)[1]


def test_rpc_checks_input_shape_and_finite_output(portable_checkpoint):
    checkpoint, _ = portable_checkpoint
    metadata = load_inference_assets(checkpoint).metadata

    class Dummy:
        actions = np.zeros((100, 14), dtype=np.float32)

        def infer(self, obs):
            return {"actions": self.actions}

    dummy = Dummy()
    policy = BimanualInferencePolicy(dummy, metadata)
    observation = {
        f"observation.images.{camera}": np.zeros(shape, dtype=np.uint8)
        for camera, shape in metadata["image_shapes"].items()
    }
    observation.update(
        {"observation.state": np.zeros(14, dtype=np.float32), "prompt": "Pick"}
    )
    assert policy.infer(observation)["actions"].shape == (100, 14)
    wrong = dict(observation, **{"observation.state": np.zeros(7)})
    with pytest.raises(ValueError, match="14 finite"):
        policy.infer(wrong)
    dummy.actions[0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid physical"):
        policy.infer(observation)
