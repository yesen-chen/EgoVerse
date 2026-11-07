"""Saved normalization is a model contract, including full Trainer resume."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fixtures.piper_pico_training import write_recording, write_training_roots
from fixtures.train_harness import hermetic_env
from hydra import compose, initialize_config_module
from hydra.core.hydra_config import HydraConfig
from lightning import Trainer
from omegaconf import OmegaConf, open_dict

import egomimic.trainHydra as train_hydra
from egomimic.pl_utils.pl_model import ModelWrapper, checkpoint_values_equal
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

STATE = "observations.state.ee_pose"
ACTION = "actions_cartesian"
STAT_NAMES = (
    "mean",
    "std",
    "min",
    "max",
    "median",
    "quantile_1",
    "quantile_99",
    "quantile_0_01",
    "quantile_99_99",
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def schema(*, stats=True):
    inventory = MultiDataset(state={})
    inventory.embodiments = {3, 7}
    inventory.norm_mode = "quantile"
    for emb, dim in ((3, 12), (7, 14)):
        inventory.key_types[emb] = {STATE: "proprio_keys", ACTION: "action_keys"}
        inventory.zarr_keys[emb] = {STATE: STATE, ACTION: ACTION}
        inventory.shapes[emb] = {STATE: (dim,), ACTION: (9, dim)}
        inventory.norm_stats[emb] = {}
        if stats:
            inventory.norm_stats[emb] = {
                key: {
                    stat: np.full(shape, emb + i, dtype=np.float32)
                    for i, stat in enumerate(STAT_NAMES)
                }
                for key, shape in inventory.shapes[emb].items()
            }
    return inventory


def checkpoint(state=None):
    return {
        "hyper_parameters": {
            "norm_stats_state": schema().to_state() if state is None else state,
            "config_tree": {
                "model": {"robomimic_model": {}},
                "data_contracts": {"synthetic": {"version": 1}},
            },
        }
    }


@pytest.mark.parametrize("piper_id", [7, torch.tensor(7), "piper_bimanual"])
def test_shape_inference_is_scoped_to_sample_embodiment(piper_id):
    inventory = schema(stats=False)
    inventory.shapes = {}
    inventory.infer_shapes_from_batch(
        {"embodiment": piper_id, STATE: torch.zeros(14), ACTION: torch.zeros(9, 14)}
    )
    inventory.infer_shapes_from_batch(
        {
            "embodiment": torch.tensor(3),
            STATE: torch.zeros(12),
            ACTION: torch.zeros(9, 12),
        }
    )
    assert inventory.shapes == {
        7: {STATE: (14,), ACTION: (9, 14)},
        3: {STATE: (12,), ACTION: (9, 12)},
    }
    # Legacy callers without sample identity keep their former common-key behavior.
    inventory.infer_shapes_from_batch({STATE: torch.zeros(2)})
    assert inventory.shapes[3][STATE] == inventory.shapes[7][STATE] == (2,)
    with pytest.raises(ValueError, match="one embodiment"):
        inventory.infer_shapes_from_batch({"embodiment": torch.tensor([3, 7])})


def test_saved_values_are_cloned_without_refitting():
    saved = checkpoint()
    current = schema(stats=False)
    restored = train_hydra._restore_local_norm_state(saved, current)
    assert checkpoint_values_equal(
        restored.to_state(), saved["hyper_parameters"]["norm_stats_state"]
    )
    restored.norm_stats[7][STATE]["mean"][0] += 100
    assert (
        saved["hyper_parameters"]["norm_stats_state"]["norm_stats"][7][STATE]["mean"][0]
        == 7
    )
    assert current.norm_stats == {3: {}, 7: {}}


@pytest.mark.parametrize(
    "field", ["norm_mode", "embodiments", "key_types", "zarr_keys", "shapes"]
)
def test_resume_rejects_changed_data_schema(field):
    saved = checkpoint()
    saved["hyper_parameters"]["norm_stats_state"][field] = "wrong"
    with pytest.raises(ValueError, match=field):
        train_hydra._restore_local_norm_state(saved, schema(stats=False))


@pytest.mark.parametrize(
    "problem",
    [
        "missing_state",
        "missing_domain",
        "extra_domain",
        "missing_key",
        "missing_stat",
        "wrong_shape",
        "nan",
        "negative_std",
    ],
)
def test_resume_rejects_corrupt_or_incomplete_saved_statistics(problem):
    saved = checkpoint()
    state = saved["hyper_parameters"]["norm_stats_state"]
    if problem == "missing_state":
        del saved["hyper_parameters"]["norm_stats_state"]
    elif problem == "missing_domain":
        del state["norm_stats"][7]
    elif problem == "extra_domain":
        state["norm_stats"][8] = {}
    elif problem == "missing_key":
        del state["norm_stats"][7][STATE]
    elif problem == "missing_stat":
        del state["norm_stats"][7][STATE]["mean"]
    elif problem == "wrong_shape":
        state["norm_stats"][7][STATE]["mean"] = np.zeros(12)
    elif problem == "nan":
        state["norm_stats"][7][STATE]["mean"][0] = np.nan
    else:
        state["norm_stats"][7][STATE]["std"][0] = -1
    with pytest.raises(ValueError, match="[Cc]heckpoint"):
        train_hydra._restore_local_norm_state(saved, schema(stats=False))


def test_slurm_requeue_checkpoint_resolved_before_model_and_statistics(
    monkeypatch, tmp_path
):
    cfg = OmegaConf.create(
        {"ckpt_path": "explicit.ckpt", "trainer": {"default_root_dir": str(tmp_path)}}
    )
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    assert train_hydra._effective_local_checkpoint_path(cfg) == "explicit.ckpt"
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    assert train_hydra._effective_local_checkpoint_path(cfg) == "explicit.ckpt"
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    expected = str(tmp_path / "checkpoints" / "last.ckpt")
    assert train_hydra._effective_local_checkpoint_path(cfg) == expected
    assert cfg.ckpt_path == expected


def _stub_model(_, config_tree, norm_stats_state):
    return SimpleNamespace(
        nets=torch.nn.ModuleDict({"policy": torch.nn.Linear(2, 2)}),
        norm_stats=MultiDataset.from_state(norm_stats_state),
    )


def test_lightning_loader_and_direct_resume_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(ModelWrapper, "_instantiate_model", _stub_model)
    payload = checkpoint()["hyper_parameters"]
    wrapper = ModelWrapper(**payload, enable_grad_norm=False)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.strategy.connect(wrapper)
    path = tmp_path / "norm.ckpt"
    trainer.save_checkpoint(path)
    restored = ModelWrapper.load_from_checkpoint(
        path, weights_only=False, map_location="cpu"
    )
    saved = torch.load(path, weights_only=False, map_location="cpu")
    assert checkpoint_values_equal(
        restored.model.norm_stats.to_state(), payload["norm_stats_state"]
    )
    changed = copy.deepcopy(payload)
    changed["norm_stats_state"]["norm_stats"][7][STATE]["mean"][0] += 1
    with pytest.raises(ValueError, match="normalization differs"):
        ModelWrapper(**changed).on_load_checkpoint(saved)
    # Hparams can still agree while a caller has replaced actual policy stats.
    restored.model.norm_stats.norm_stats[7][STATE]["mean"][0] += 1
    with pytest.raises(ValueError, match="differs from policy"):
        restored.on_load_checkpoint(saved)
    restored.model.norm_stats = MultiDataset.from_state(payload["norm_stats_state"])
    wrong_data = MultiDataset.from_state(changed["norm_stats_state"])
    restored._trainer = SimpleNamespace(
        datamodule=SimpleNamespace(
            train_datasets={"piper": wrong_data}, valid_datasets={}
        )
    )
    with pytest.raises(ValueError, match="train_datasets/piper"):
        restored.on_load_checkpoint(saved)
    missing_contract = copy.deepcopy(saved)
    del missing_contract["hyper_parameters"]["config_tree"]["data_contracts"]
    with pytest.raises(ValueError, match="no data_contracts"):
        wrapper.on_load_checkpoint(missing_contract)


def _compose_training(roots, out):
    out.mkdir(parents=True, exist_ok=True)
    overrides = [
        "data=piper_pico_local",
        "recordings.gripper_mode=zero",
        f"recordings.piper_root={roots['piper']}",
        f"recordings.pico_ego_root={roots['pico_ego']}",
        "recordings.allow_synthetic_calibration=true",
        "recordings.valid_ratio=0.25",
        "recordings.batch_size=2",
        "recordings.num_workers=0",
        "recordings.output_chunk_length=9",
        "recordings.piper_timing.source_horizon=9",
        "recordings.ego_timing.source_horizon=9",
        f"paths.output_dir={out}",
        f"paths.cache_dir={out}/cache",
        "~callbacks",
        "~evaluator",
        "+mmap_checkpoint=false",
        "norm_stats.use_cache=false",
        "norm_stats.save_cache_dir=null",
        "norm_stats.num_workers=0",
        "norm_stats.sample_frac=1.0",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "trainer.precision=32",
        "trainer.max_epochs=1",
        "trainer.min_epochs=1",
        "trainer.limit_train_batches=1",
        "trainer.limit_val_batches=0",
        "+trainer.num_sanity_val_steps=0",
        "+trainer.enable_checkpointing=false",
        "+trainer.enable_progress_bar=false",
        "model.robomimic_model.trunk.num_blocks=1",
        "model.robomimic_model.head_specs.shared.model.nblocks=1",
        "model.robomimic_model.head_specs.shared.num_inference_steps=1",
        "model.robomimic_model.reverse_kl_samples=0",
        "+model.robomimic_model.device=cpu",
        *[
            f"+model.robomimic_model.encoder_specs.{camera}.weights=null"
            for camera in ("front_img_1", "left_wrist_img", "right_wrist_img")
        ],
    ]
    with initialize_config_module(
        config_module="egomimic.hydra_configs", version_base=None
    ):
        cfg = compose(
            config_name="train_piper_pico",
            overrides=overrides,
            return_hydra_config=True,
        )
    with open_dict(cfg):
        cfg.hydra.runtime.output_dir = str(out)
    HydraConfig.instance().set_config(cfg)
    return cfg


class ResumeProbeEvaluator:
    """Use the custom evaluator branch to inspect the model before execution."""

    override_dict = {}
    expected = None
    calls = 0

    def run(self, trainer, model, datamodule, cfg):
        type(self).calls += 1
        assert checkpoint_values_equal(
            model.model.norm_stats.to_state(),
            self.expected["hyper_parameters"]["norm_stats_state"],
        )
        for split in (datamodule.train_datasets, datamodule.valid_datasets):
            for dataset in split.values():
                assert checkpoint_values_equal(
                    dataset.to_state(), model.model.norm_stats.to_state()
                )
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(
                tensor.cpu(), self.expected["state_dict"][key], rtol=0, atol=0
            )


def test_full_optimizer_resume_and_eval_reuse_saved_stats_without_recomputing(
    tmp_path, monkeypatch
):
    hermetic_env(monkeypatch)
    roots = write_training_roots(tmp_path)
    cfg = _compose_training(roots, tmp_path / "first")
    _, first = train_hydra.train(cfg)
    assert first["trainer"].global_step == 1
    path = tmp_path / "full.ckpt"
    first["trainer"].save_checkpoint(path, weights_only=False)
    saved = torch.load(path, weights_only=False, map_location="cpu")
    assert saved["optimizer_states"] and saved["optimizer_states"][0]["state"]
    expected = saved["hyper_parameters"]["norm_stats_state"]
    assert expected["shapes"][7][STATE] == (14,)
    assert expected["shapes"][3][STATE] == (12,)

    # Today's input distribution really changes, while its semantic schema stays
    # compatible. A new fit would shift the translation statistics by 0.15 m.
    for kind, root in roots.items():
        for index in range(4):
            write_recording(
                root, kind, f"recording_{index}", phase=index * 0.015, offset=0.15
            )

    def forbidden_refit(*args, **kwargs):
        raise AssertionError("resume/eval must not recompute normalization")

    monkeypatch.setattr(MultiDataset, "infer_norm_from_dataset", forbidden_refit)
    resumed_cfg = _compose_training(roots, tmp_path / "resumed")
    resumed_cfg.ckpt_path = str(path)
    resumed_cfg.trainer.max_epochs = 2
    _, resumed = train_hydra.train(resumed_cfg)
    assert resumed["trainer"].global_step == 2
    assert resumed["trainer"].current_epoch == 2
    # Adam's counter continuing to 2 verifies optimizer state, beyond weights.
    states = resumed["trainer"].optimizers[0].state.values()
    assert any(float(state["step"]) == 2 for state in states if "step" in state)
    assert checkpoint_values_equal(
        resumed["model"].model.norm_stats.to_state(), expected
    )
    for split in (
        resumed["datamodule"].train_datasets,
        resumed["datamodule"].valid_datasets,
    ):
        for dataset in split.values():
            assert checkpoint_values_equal(dataset.to_state(), expected)

    # The eval-only custom-run path must also restore the saved stats/weights
    # before handing the policy to its evaluator (formerly it loaded neither).
    ResumeProbeEvaluator.expected, ResumeProbeEvaluator.calls = saved, 0
    eval_cfg = _compose_training(roots, tmp_path / "eval")
    with open_dict(eval_cfg):
        eval_cfg.mode = "eval"
        eval_cfg.ckpt_path = str(path)
        eval_cfg.evaluator = {"_target_": f"{__name__}.ResumeProbeEvaluator"}
    train_hydra.train(eval_cfg)
    assert ResumeProbeEvaluator.calls == 1

    # Exercise the default Trainer.validate branch too; the evaluator's real
    # metrics are covered by the separate local-recipe training acceptance.
    # Here we inspect the exact weights/data statistics at its entry boundary.
    calls = []
    probe = ResumeProbeEvaluator().run

    def validate(trainer, *, model, datamodule):
        probe(trainer, model, datamodule, eval_cfg)
        calls.append(True)
        return []

    monkeypatch.delattr(ResumeProbeEvaluator, "run")
    monkeypatch.setattr(Trainer, "validate", validate)
    train_hydra.train(eval_cfg)
    assert calls == [True]
    assert ResumeProbeEvaluator.calls == 2
