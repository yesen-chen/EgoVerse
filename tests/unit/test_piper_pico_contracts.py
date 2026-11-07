"""Resume semantics, shared-head layout and calibration cache invalidation."""

import copy
import json
import os
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from egomimic.algo.hpt import HPT
from egomimic.eval.eval_hpt import HPTEvalVideo
from egomimic.models.denoising_policy import DenoisingPolicy
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.zarr.local_recordings import (
    RecordingEpisode,
    RecordingMultiDataset,
    local_data_contracts,
)
from egomimic.rldb.zarr.norm_cache import code_modules, episode_fingerprint
from egomimic.trainHydra import _build_model_config_tree


@pytest.mark.parametrize("padding", ["zero", "gaussian"])
def test_shared_pose_only_layout_preserves_both_full_rotations(padding):
    # Each coordinate has a distinct sentinel, especially the final right yaw/pitch/roll.
    head = SimpleNamespace(padding=padding)
    prediction = torch.arange(2 * 5 * 14).reshape(2, 5, 14)
    actual = DenoisingPolicy.restore_action_dimensions(head, prediction, 12)
    expected = prediction[..., [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual, prediction[..., :12])
    torch.testing.assert_close(
        DenoisingPolicy.restore_action_dimensions(head, prediction, 14), prediction
    )


def test_reverse_kl_samples_use_same_unpadding():
    class Head(torch.nn.Module):
        padding = "zero"
        restore_action_dimensions = DenoisingPolicy.restore_action_dimensions

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = torch.nn.ModuleDict({"shared": Head()})

        def forward(self, domain, data):
            return {"shared": torch.arange(14).reshape(1, 1, 14).repeat(2, 3, 1)}

    algo = SimpleNamespace(
        nets=torch.nn.ModuleDict({"policy": Policy()}),
        _clone_batch=copy.deepcopy,
        shared_ac_key="actions_cartesian",
        norm_stats=SimpleNamespace(
            unnormalize=lambda data, emb: {k: v * 2 + 10 for k, v in data.items()}
        ),
    )
    samples = HPTEvalVideo._collect_policy_samples(
        SimpleNamespace(model=algo),
        {"domain": "human_bimanual", "data": {}},
        ref=torch.zeros(2, 3, 12),
        key_name="shared",
        M=2,
    )
    assert samples.shape == (2, 2, 3, 12)
    assert samples[0, 0, 0].tolist() == [10, 12, 14, 16, 18, 20, 24, 26, 28, 30, 32, 34]
    assert algo.nets.training


def test_training_preprocess_preserves_epoch_seconds_precision():
    algo = SimpleNamespace(
        norm_stats=SimpleNamespace(zarr_key_to_keyname=lambda key, emb: key),
        ac_keys={7: "actions_cartesian"},
        annotation_key=None,
        device="cpu",
    )
    times = torch.tensor([1700000000.0, 1700000000.0333333], dtype=torch.float64)
    batch = {
        "piper_bimanual": {
            "actions_cartesian": torch.zeros(2, 3, 14, dtype=torch.float64),
            "sample_timestamp": times,
        }
    }
    processed = HPT.process_batch_for_training(algo, batch)[7]
    assert processed["sample_timestamp"].dtype == torch.float64
    torch.testing.assert_close(processed["sample_timestamp"], times, rtol=0, atol=0)
    assert processed["sample_timestamp"][1] > processed["sample_timestamp"][0]
    assert processed["actions_cartesian"].dtype == torch.float32


def config(mode="zero"):
    resolver = {
        "_target_": "egomimic.rldb.zarr.local_recordings.LocalRecordingResolver",
        "kind": "pico_ego",
        "gripper_mode": mode,
        "expected_axis_convention": "piper_x",
    }
    return OmegaConf.create(
        {
            "data": {
                split: {"human_bimanual": {"resolver": copy.deepcopy(resolver)}}
                for split in ("train_datasets", "valid_datasets")
            },
            "model": {"robomimic_model": {}},
        }
    )


def test_contract_consistency_and_resolved_checkpoint(monkeypatch):
    cfg = config()
    tree = _build_model_config_tree(cfg)
    assert tree.data_contracts.human_bimanual.axis_convention == "piper_x"
    assert tree.data_contracts.human_bimanual.gripper_mode == "zero"
    cfg.data.valid_datasets.human_bimanual.resolver.gripper_mode = "width"
    with pytest.raises(ValueError, match="Train/validation data contracts differ"):
        local_data_contracts(cfg)
    monkeypatch.setattr(
        ModelWrapper,
        "_instantiate_model",
        lambda *_: SimpleNamespace(
            nets=torch.nn.ModuleDict({"policy": torch.nn.Linear(2, 2)})
        ),
    )
    norm_state = {"norm_stats": {3: {"state": {"mean": [0.0]}}}}
    wrapper = ModelWrapper(
        config_tree=_build_model_config_tree(config("width")),
        norm_stats_state=norm_state,
    )
    with pytest.raises(ValueError, match="Checkpoint data_contracts differ"):
        wrapper.on_load_checkpoint({"hyper_parameters": {"config_tree": tree}})
    wrapper.on_load_checkpoint(
        {
            "hyper_parameters": {
                "config_tree": wrapper.hparams.config_tree,
                "norm_stats_state": norm_state,
            }
        }
    )


def test_calibration_cache_uses_contents_even_with_identical_stat(tmp_path):
    path = tmp_path / "episode.zarr"
    path.mkdir()
    metadata_file = path / "zarr.json"
    metadata_file.write_text(json.dumps({"attributes": {"calibration": 1}}))
    stat = metadata_file.stat()
    leaf = SimpleNamespace(
        metadata={"conversion": {"source_dataset": "recording"}},
        episode_path=path,
        embodiment="piper_bimanual",
        key_map={},
        transform=[],
    )
    episode = RecordingEpisode(leaf, "batch/episode")
    generic_before = episode_fingerprint(path)
    strict_before = episode.normalization_fingerprint()
    metadata_file.write_text(json.dumps({"attributes": {"calibration": 2}}))
    os.utime(metadata_file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert episode_fingerprint(path) == generic_before
    assert episode.normalization_fingerprint() != strict_before


def test_norm_cache_hashes_both_specific_reader_implementations():
    recipe = {"_target_": "egomimic.rldb.zarr.local_recordings.LocalRecordingResolver"}
    modules = code_modules(recipe)
    assert "egomimic.rldb.zarr.piper_dataset" in modules
    assert "egomimic.rldb.embodiment.pico_ego" in modules


def test_local_finite_policy_keeps_distribution_shift_and_rejects_nan():
    key = "actions_cartesian"
    state = {
        "embodiments": [7],
        "key_types": {7: {key: "action_keys"}},
        "zarr_keys": {7: {key: key}},
        "norm_stats": {7: {key: {"quantile_1": [0.0], "quantile_99": [1.0]}}},
    }
    sample = {"embodiment": 7, key: torch.tensor([1000.0])}
    local = RecordingMultiDataset(state=state)
    assert local._check_bounds(sample, None, 4, "batch/episode") is None
    old = RecordingMultiDataset(state=state, sample_policy="legacy_quantile")
    assert "Bounds violation" in old._check_bounds(sample, None, 4, "batch/episode")
    sample[key] = torch.tensor([float("nan")])
    with pytest.raises(ValueError, match="Nonfinite.*batch/episode\\[4\\]"):
        local._check_bounds(sample, None, 4, "batch/episode")
