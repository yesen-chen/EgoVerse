"""Ego artificial endings filter starts, never delete action support rows."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr
from fixtures.piper_pico_training import write_recording
from hydra import compose, initialize_config_module
from lightning import Trainer
from omegaconf import OmegaConf

from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.action_timing import ActionTiming
from egomimic.rldb.ego_boundary_policy import EgoBoundaryPolicy
from egomimic.rldb.embodiment.pico_ego import PicoEgo, PicoEgoDataset
from egomimic.rldb.zarr.local_recordings import (
    LocalRecordingResolver,
    RecordingEpisode,
    RecordingMultiDataset,
    local_data_contracts,
    recording_group,
)
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.trainHydra import _build_model_config_tree


def fixture_episode(root, reason, *, name="recording", frames=12, start="ui_trim"):
    path = write_recording(root, "pico_ego", name, frames=frames)
    if reason is not None:
        zarr.open_group(str(path), mode="a").attrs["preprocessing"] = {
            "schema": "ego_preprocessing_v1",
            "start_reason": start,
            "end_reason": reason,
        }
    return path


def load(path, trim=False, inactivity=False, *, spec=None, norm=False, transforms=True):
    spec = spec or ActionTiming(30, 6, 2, 9)
    return PicoEgoDataset(
        path,
        PicoEgo.get_keymap(timing=spec, gripper_mode="width", norm_mode=norm),
        PicoEgo.get_transform_list(timing=spec, gripper_mode="width")
        if transforms
        else None,
        gripper_mode="width",
        require_full_window_after_trim=trim,
        require_full_window_after_inactivity=inactivity,
    )


@pytest.mark.parametrize(
    "trim,inactivity", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize(
    "reason", ["ui_trim", "inactivity", "tracking_gap", "video_gap", "source_end", None]
)
def test_each_end_reason_and_both_switches(tmp_path, reason, trim, inactivity):
    path = fixture_episode(tmp_path, reason)
    strict = (
        trim
        if reason == "ui_trim"
        else inactivity
        if reason in ("inactivity", "tracking_gap", "video_gap")
        else False
    )
    loaded = load(path, trim, inactivity, transforms=False)
    assert len(loaded) == (7 if strict else 12)
    assert loaded.total_frames == 12
    assert list(loaded.eligible_start_indices) == list(range(len(loaded)))
    assert loaded.sample_window_contract["episode_end_padding"] == (
        "none" if strict else "repeat_last"
    )
    last = loaded[len(loaded) - 1]
    raw = zarr.open_group(str(path), mode="r")["left.obs_ee_pose"][:12]
    expected = raw[6:12] if strict else np.repeat(raw[11:12], 6, axis=0)
    np.testing.assert_allclose(last["left.action_ee_pose"], expected, atol=1e-6)
    assert raw.shape[0] == 12  # Tail support remains physically stored.
    with pytest.raises(IndexError):
        loaded[len(loaded)]


def test_start_inactivity_does_not_override_ui_trim_end_policy(tmp_path):
    path = fixture_episode(tmp_path, "ui_trim", start="inactivity")
    assert len(load(path, False, True)) == 12
    assert len(load(path, True, False)) == 7


@pytest.mark.parametrize("reason", ["ui_trim", "inactivity"])
def test_public_loader_defaults_require_full_windows(tmp_path, reason):
    path = fixture_episode(tmp_path, reason)
    spec = ActionTiming(30, 6, 2, 9)
    dataset = PicoEgoDataset(path, PicoEgo.get_keymap(timing=spec))
    assert len(dataset) == 7
    assert dataset.boundary_policy == EgoBoundaryPolicy(True, True)


@pytest.mark.parametrize("recipe", ["pico_ego_local", "piper_pico_local"])
def test_shipped_config_defaults_and_independent_overrides(recipe):
    with initialize_config_module(
        config_module="egomimic.hydra_configs", version_base=None
    ):
        cfg = compose(config_name="train_piper_pico", overrides=[f"data={recipe}"])
        modified = compose(
            config_name="train_piper_pico",
            overrides=[
                f"data={recipe}",
                "recordings.ego_require_full_window_after_trim=false",
            ],
        )
    original = cfg.data.train_datasets.human_bimanual.resolver
    changed = modified.data.train_datasets.human_bimanual.resolver
    assert original.require_full_window_after_trim is True
    assert original.require_full_window_after_inactivity is True
    assert changed.require_full_window_after_trim is False
    assert changed.require_full_window_after_inactivity is True
    assert (
        modified.data.valid_datasets.human_bimanual.resolver.require_full_window_after_trim
        is False
    )


def test_full_source_horizon_even_if_stride_skips_final_row(tmp_path):
    path = fixture_episode(tmp_path, "inactivity", frames=12)
    spec = ActionTiming(30, 6, 2, 9)  # support [0, 2, 4], source still has 6 rows.
    strict = load(path, False, True, spec=spec)
    assert len(strict) == 12 - 6 + 1
    assert strict.source_index(6) == 6
    with pytest.raises(IndexError):
        strict[7]  # support would fit but the complete source window would not.


@pytest.mark.parametrize("frames", [5, 6, 7])
def test_short_fragment_fails_clearly_or_keeps_exact_valid_start_count(
    tmp_path, frames
):
    path = fixture_episode(tmp_path, "ui_trim", frames=frames)
    if frames < 6:
        with pytest.raises(ValueError, match=r"episode_000000.zarr.*5 rows.*6"):
            load(path, True)
    else:
        assert len(load(path, True)) == frames - 6 + 1
    assert len(load(path, False)) == frames


@pytest.mark.parametrize(
    "bad", [None, "mystery", {}, {"schema": "wrong", "end_reason": "inactivity"}]
)
def test_malformed_present_boundary_is_not_silently_treated_as_complete(tmp_path, bad):
    path = fixture_episode(tmp_path, "ui_trim")
    group = zarr.open_group(str(path), mode="a")
    group.attrs["preprocessing"] = bad
    with pytest.raises(ValueError, match="preprocessing"):
        load(path, True, True)


@pytest.mark.parametrize("bad", ["false", 0, 1, None])
def test_policy_flags_are_actual_booleans(bad):
    with pytest.raises(ValueError, match="must be a boolean"):
        EgoBoundaryPolicy(bad, False)


def test_norm_and_training_use_identical_starts_values_and_provenance(tmp_path):
    path = fixture_episode(tmp_path, "ui_trim")
    train_leaf, norm_leaf = load(path, True), load(path, True, norm=True)
    train_wrapped = RecordingEpisode(train_leaf, "recording/segment")
    norm_wrapped = RecordingEpisode(norm_leaf, "recording/segment")
    for idx in range(len(train_leaf)):
        sample, norm_sample = train_wrapped[idx], norm_wrapped[idx]
        assert sample["sample_index"] == norm_sample["sample_index"] == idx
        assert sample["sample_timestamp_ns"].dtype == torch.int64
        assert sample["sample_timestamp_ns"] == norm_sample["sample_timestamp_ns"]
        torch.testing.assert_close(
            sample["actions_cartesian"], norm_sample["actions_cartesian"]
        )
    train = RecordingMultiDataset(datasets={"segment": train_wrapped}, mode="total")
    norm = RecordingMultiDataset(datasets={"segment": norm_wrapped}, mode="total")
    stats = MultiDataset(state={}, norm_mode="zscore")
    stats.populate_from_datasets({"human_bimanual": train})
    stats.infer_norm_from_dataset(norm, "human_bimanual", sample_frac=1, num_workers=0)
    all_actions = np.stack(
        [norm_wrapped[i]["actions_cartesian"].numpy() for i in range(len(norm))]
    )
    expected = MultiDataset._compute_stats_for_array(all_actions)["mean"]
    np.testing.assert_allclose(
        stats.norm_stats[3]["actions_cartesian"]["mean"], expected, atol=1e-6
    )
    assert stats._norm_run_metadata["frames"] == len(train) == len(norm) == 7
    repeated = RecordingEpisode(load(path), "recording/segment")
    assert (
        repeated.normalization_fingerprint()
        != train_wrapped.normalization_fingerprint()
    )


def test_source_groups_and_split_are_unchanged_by_boundary_policy(tmp_path):
    for recording in ("a", "b"):
        for segment, reason in enumerate(("inactivity", "ui_trim")):
            path = fixture_episode(tmp_path, reason, name=f"{recording}/{segment}")
            group = zarr.open_group(str(path), mode="a")
            pico = copy.deepcopy(group.attrs["pico_ego"])
            pico["source_tracking"]["sha256"] = recording
            group.attrs["pico_ego"] = pico
    spec = ActionTiming(30, 6, 2, 9)
    resolver = LocalRecordingResolver(
        tmp_path,
        "pico_ego",
        PicoEgo.get_keymap(timing=spec),
        PicoEgo.get_transform_list(timing=spec),
        require_full_window_after_trim=True,
        require_full_window_after_inactivity=True,
    )
    train = RecordingMultiDataset._from_resolver(
        resolver, mode="train", valid_ratio=0.5
    )
    valid = RecordingMultiDataset._from_resolver(
        resolver, mode="valid", valid_ratio=0.5
    )
    assert len(train) == len(valid) == 14
    assert len(train.datasets) == len(valid.datasets) == 2
    assert train.selected_recordings.isdisjoint(valid.selected_recordings)
    for dataset in (train, valid):
        assert (
            len({recording_group(leaf.metadata) for leaf in dataset.datasets.values()})
            == 1
        )


def policy_config(trim=True, inactivity=True):
    resolver = {
        "_target_": "egomimic.rldb.zarr.local_recordings.LocalRecordingResolver",
        "kind": "pico_ego",
        "require_full_window_after_trim": trim,
        "require_full_window_after_inactivity": inactivity,
    }
    return OmegaConf.create(
        {
            "data": {
                "train_datasets": {"human_bimanual": {"resolver": resolver}},
                "valid_datasets": {
                    "human_bimanual": {"resolver": copy.deepcopy(resolver)}
                },
            },
            "action_timing": {
                "human_bimanual": {
                    "source_fps": 30,
                    "source_horizon": 6,
                    "action_stride": 2,
                    "output_chunk_length": 9,
                }
            },
            "model": {"robomimic_model": {}},
        }
    )


def test_checkpoint_preserves_boundary_flags_and_rejects_changed_supervision(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        ModelWrapper,
        "_instantiate_model",
        lambda *_: SimpleNamespace(
            nets=torch.nn.ModuleDict({"policy": torch.nn.Linear(2, 2)})
        ),
    )
    cfg = policy_config(True, False)
    tree = _build_model_config_tree(cfg)
    assert tree.action_timing.human_bimanual.episode_end_padding.startswith(
        "per_episode_end_reason"
    )
    policy = tree.data_contracts.human_bimanual.future_window_policy
    assert policy.require_full_window_after_trim is True
    assert policy.require_full_window_after_inactivity is False
    # The local checkpoint contract requires saved statistics. Numeric statistics
    # are exercised above; this tiny model only exercises serialization/restoring.
    norm_state = {
        "norm_stats": {3: {"actions_cartesian": {"mean": [0.0], "std": [1.0]}}}
    }
    wrapper = ModelWrapper(
        config_tree=tree, norm_stats_state=norm_state, enable_grad_norm=False
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.strategy.connect(wrapper)
    checkpoint = tmp_path / "boundary.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = ModelWrapper.load_from_checkpoint(
        checkpoint, weights_only=False, map_location="cpu"
    )
    assert OmegaConf.to_container(
        restored.hparams.config_tree
    ) == OmegaConf.to_container(tree)
    changed = _build_model_config_tree(policy_config(False, False))
    with pytest.raises(ValueError, match="data_contracts differ"):
        ModelWrapper.validate_checkpoint_contracts(
            torch.load(checkpoint, weights_only=False), changed
        )


def test_train_valid_cannot_use_different_window_policy():
    cfg = policy_config()
    cfg.data.valid_datasets.human_bimanual.resolver.require_full_window_after_trim = (
        False
    )
    with pytest.raises(ValueError, match="Train/validation data contracts differ"):
        local_data_contracts(cfg)
