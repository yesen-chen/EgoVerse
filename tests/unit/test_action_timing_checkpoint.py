"""Action timing survives the real Lightning checkpoint writer/loader."""

from types import SimpleNamespace

import pytest
import torch
from lightning import Trainer
from omegaconf import OmegaConf

from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.trainHydra import _action_timing_from_config, _build_model_config_tree


def config():
    return OmegaConf.create(
        {
            "sampling": {"fps": 30},
            "action_timing": {
                "piper_bimanual": {
                    "source_fps": "${sampling.fps}",
                    "source_horizon": 45,
                    "action_stride": 1,
                    "output_chunk_length": 100,
                    "includes_current": True,
                }
            },
            "model": {
                "robomimic_model": {
                    "head_specs": {
                        "shared": {
                            "action_horizon": "${action_timing.piper_bimanual.output_chunk_length}"
                        }
                    },
                    "norm_stats": "not saved here",
                }
            },
        }
    )


def test_config_resolves_before_detaching_and_does_not_mutate_source():
    cfg = config()
    tree = _build_model_config_tree(cfg)
    cfg.sampling.fps = 60
    cfg.action_timing.piper_bimanual.output_chunk_length = 45
    saved = OmegaConf.to_container(tree, resolve=True)
    assert (
        saved["model"]["robomimic_model"]["head_specs"]["shared"]["action_horizon"]
        == 100
    )
    assert saved["model"]["robomimic_model"]["norm_stats"] is None
    assert cfg.model.robomimic_model.norm_stats == "not saved here"
    timing = saved["action_timing"]["piper_bimanual"]
    assert timing["source_fps"] == 30
    assert timing["output_chunk_length"] == 100
    assert timing["output_offsets_s"][-1] == pytest.approx(44 / 30)


@pytest.mark.parametrize("with_timing", [False, True])
def test_lightning_checkpoint_roundtrip(tmp_path, monkeypatch, with_timing):
    # No image encoder, downloads, training, GPU or robot. Use the real wrapper,
    # Lightning checkpoint serializer and loader with only its model stubbed.
    monkeypatch.setattr(
        ModelWrapper,
        "_instantiate_model",
        lambda *_: SimpleNamespace(
            nets=torch.nn.ModuleDict({"policy": torch.nn.Linear(2, 2)})
        ),
    )
    cfg = config()
    if not with_timing:
        cfg.action_timing = None
        cfg.model.robomimic_model.head_specs.shared.action_horizon = 100
    tree = _build_model_config_tree(cfg)
    wrapper = ModelWrapper(
        config_tree=tree, norm_stats_state={}, enable_grad_norm=False
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
    path = tmp_path / "timing.ckpt"
    trainer.save_checkpoint(path)
    restored = ModelWrapper.load_from_checkpoint(
        path, weights_only=False, map_location="cpu"
    )
    restored_tree = OmegaConf.to_container(restored.hparams.config_tree, resolve=True)
    assert restored_tree == OmegaConf.to_container(tree, resolve=True)
    assert ("action_timing" in restored_tree) is with_timing
    for key, value in wrapper.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)

    if with_timing:
        changed_cfg = config()
        changed_cfg.action_timing.piper_bimanual.source_horizon = 30
        original_checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        changed_wrapper = ModelWrapper(
            config_tree=_build_model_config_tree(changed_cfg),
            norm_stats_state={},
        )
        with pytest.raises(ValueError, match="Checkpoint action_timing differs"):
            changed_wrapper.on_load_checkpoint(original_checkpoint)
        changed_cfg.action_timing = None
        changed_cfg.model.robomimic_model.head_specs.shared.action_horizon = 100
        changed_wrapper = ModelWrapper(
            config_tree=_build_model_config_tree(changed_cfg),
            norm_stats_state={},
        )
        with pytest.raises(ValueError, match="Checkpoint action_timing differs"):
            changed_wrapper.on_load_checkpoint(original_checkpoint)
    else:
        restored_with_explicit_timing = ModelWrapper(
            config_tree=_build_model_config_tree(config()),
            norm_stats_state={},
        )
        with pytest.warns(UserWarning, match="original time contract is unknown"):
            restored_with_explicit_timing.on_load_checkpoint(
                torch.load(path, weights_only=False, map_location="cpu")
            )
        assert restored_with_explicit_timing.hparams.config_tree.action_timing


def test_legacy_config_opt_out():
    cfg = OmegaConf.create({"model": {"robomimic_model": {}}})
    assert _action_timing_from_config(cfg) == {}
    assert list(_build_model_config_tree(cfg)) == ["model"]
