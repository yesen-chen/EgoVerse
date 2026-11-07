"""CPU-only acceptance of local recipes through real shared-FM optimization."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch
from fixtures.piper_pico_training import write_recording, write_training_roots
from fixtures.train_harness import hermetic_env
from hydra import compose, initialize_config_module
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

import egomimic.trainHydra as train_hydra
from egomimic.models.denoising_policy import DenoisingPolicy
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.action_timing import ActionTiming
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.pico_ego import PicoEgo
from egomimic.rldb.embodiment.piper import Piper
from egomimic.rldb.zarr.local_recordings import (
    LocalRecordingResolver,
    RecordingMultiDataset,
)
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

CASES = [
    ("piper_local", "zero"),
    ("pico_ego_local", "zero"),
    ("pico_ego_local", "width"),
    ("piper_pico_local", "zero"),
    ("piper_pico_local", "width"),
]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def compose_local(data, mode, roots, out, evaluator=False):
    out.mkdir(parents=True, exist_ok=True)
    overrides = [
        f"data={data}",
        f"recordings.piper_root={roots['piper']}",
        f"recordings.pico_ego_root={roots['pico_ego']}",
        f"recordings.gripper_mode={mode}",
        "recordings.allow_synthetic_calibration=true",
        "recordings.valid_ratio=0.25",
        "recordings.batch_size=2",
        "recordings.num_workers=0",
        "recordings.output_chunk_length=9",
        "recordings.piper_timing.source_horizon=9",
        "recordings.ego_timing.source_horizon=9",
        "recordings.piper_timing.action_stride=2",
        "recordings.ego_timing.action_stride=2",
        f"paths.output_dir={out}",
        f"paths.cache_dir={out}/cache",
        "~callbacks",
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
        "trainer.limit_train_batches=2",
        "trainer.limit_val_batches=0",
        "+trainer.num_sanity_val_steps=0",
        "+trainer.enable_checkpointing=false",
        "+trainer.enable_progress_bar=false",
        "model.robomimic_model.trunk.num_blocks=1",
        "model.robomimic_model.head_specs.shared.model.nblocks=1",
        "model.robomimic_model.head_specs.shared.num_inference_steps=2",
        "model.robomimic_model.reverse_kl_samples=0",
        "+model.robomimic_model.device=cpu",
        *[
            f"+model.robomimic_model.encoder_specs.{camera}.weights=null"
            for camera in ("front_img_1", "left_wrist_img", "right_wrist_img")
        ],
    ]
    if not evaluator:
        overrides.append("~evaluator")
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


def _first_batch(loader):
    value = next(iter(loader))
    return value[0] if isinstance(value, tuple) else value


def _assert_stats_equal(a, b):
    assert a.keys() == b.keys()
    for emb in a:
        assert a[emb].keys() == b[emb].keys()
        for key in a[emb]:
            assert a[emb][key].keys() == b[emb][key].keys()
            for stat in a[emb][key]:
                np.testing.assert_array_equal(a[emb][key][stat], b[emb][key][stat])


@pytest.mark.parametrize("data,mode", CASES)
def test_local_recipe_optimization_validation_stats_and_checkpoint(
    tmp_path, monkeypatch, data, mode
):
    hermetic_env(monkeypatch)
    roots = write_training_roots(tmp_path)
    cfg = compose_local(
        data, mode, roots, tmp_path / "out", evaluator=data == "piper_pico_local"
    )
    updates, padded_shapes = [], []
    original_step = torch.optim.AdamW.step
    original_preprocess = DenoisingPolicy.preprocess_compute_loss

    def step(optimizer, *args, **kwargs):
        # Capture a representative parameter from each group before the real
        # Lightning closure/AdamW update, not merely a reported global_step.
        params = [group["params"][0] for group in optimizer.param_groups]
        previous = [param.detach().clone() for param in params]
        result = original_step(optimizer, *args, **kwargs)
        updates.append(
            any(
                not torch.equal(old, param.detach())
                for old, param in zip(previous, params)
            )
        )
        return result

    def preprocess(head, cond, payload):
        before = payload["action"]
        result, result_cond = original_preprocess(head, cond, payload)
        padded_shapes.append((before.shape[-1], result.shape[-1]))
        assert result.shape[-1] == 14
        if before.shape[-1] == 12:
            torch.testing.assert_close(result[..., :6], before[..., :6])
            torch.testing.assert_close(result[..., 7:13], before[..., 6:])
            assert torch.count_nonzero(result[..., [6, 13]]).item() == 0
        else:
            torch.testing.assert_close(result, before)
        return result, result_cond

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    monkeypatch.setattr(DenoisingPolicy, "preprocess_compute_loss", preprocess)
    metrics, objects = train_hydra.train(cfg)
    assert math.isfinite(float(metrics["Train/action_loss"]))
    assert objects["trainer"].global_step == 2
    assert updates == [True, True]
    algo = objects["model"].model
    assert next(algo.nets.parameters()).device.type == "cpu"
    dm = objects["datamodule"]
    selected = set(cfg.recordings.domains)
    expected_embs = {get_embodiment_id(name) for name in selected}
    assert set(algo.norm_stats.norm_stats) == expected_embs
    for domain, train in dm.train_datasets.items():
        valid = dm.valid_datasets[domain]
        emb = get_embodiment_id(domain)
        dim = 12 if domain == "human_bimanual" and mode == "zero" else 14
        assert algo.norm_stats.shapes[emb]["observations.state.ee_pose"] == (dim,)
        assert algo.norm_stats.shapes[emb]["actions_cartesian"] == (9, dim)
        assert train.norm_stats is valid.norm_stats
        assert train.selected_recordings.isdisjoint(valid.selected_recordings)
        assert (
            len(train.selected_recordings) == 3 and len(valid.selected_recordings) == 1
        )
        for key in ("actions_cartesian", "observations.state.ee_pose"):
            # Independent enumeration of every raw transformed training sample;
            # validation rows and the other embodiment cannot affect this mean.
            raw_values = np.stack(
                [
                    leaf[i][key].numpy()
                    for leaf in train.datasets.values()
                    for i in range(len(leaf))
                ]
            )
            np.testing.assert_allclose(
                algo.norm_stats.norm_stats[emb][key]["mean"],
                raw_values.mean(axis=0),
                atol=2e-6,
            )
            assert raw_values.shape[-1] == dim
        raw = next(iter(train.datasets.values()))[3]
        normed = algo.norm_stats.normalize(raw, emb)
        restored = algo.norm_stats.unnormalize(normed, emb)
        for key in ("actions_cartesian", "observations.state.ee_pose"):
            torch.testing.assert_close(restored[key], raw[key], atol=2e-6, rtol=2e-6)

    # Invoke the actual HPT inference and validation loss paths. With evaluator
    # disabled, Lightning validation_step alone would do nothing, so it is not
    # counted as validation here.
    batch = _first_batch(dm.val_dataloader())
    algo.nets.eval()
    with torch.no_grad():
        processed = algo.process_batch_for_training(batch)
        predictions = algo.forward_eval(processed)
    for domain in selected:
        dim = 12 if domain == "human_bimanual" and mode == "zero" else 14
        assert predictions[f"{domain}_actions_cartesian"].shape == (2, 9, dim)
        assert torch.isfinite(predictions[f"{domain}_actions_cartesian"]).all()
        assert math.isfinite(float(predictions[f"{domain}_loss"]))
    if data == "piper_pico_local":
        # Exercise the configured evaluator's actual metrics and projection,
        # including stochastic shared-head RKL samples in physical units.
        algo.rkl_samples = 2
        with torch.no_grad():
            eval_metrics, eval_images = objects[
                "model"
            ].evaluator.compute_metrics_and_viz(processed)
        assert set(eval_images) == expected_embs
        for emb, images in eval_images.items():
            assert images.shape == (2, 32, 32, 3)
            assert images.dtype == np.uint8
        for value in eval_metrics.values():
            assert math.isfinite(float(value))
        for domain in selected:
            key = f"Valid/{domain}_actions_cartesian_reverse_kl_M2"
            assert key in eval_metrics
    if "human_bimanual" in selected and mode == "zero":
        assert (12, 14) in padded_shapes
    else:
        assert all(before == after == 14 for before, after in padded_shapes)

    path = tmp_path / "trained.ckpt"
    objects["trainer"].save_checkpoint(path, weights_only=True)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    recovered = ModelWrapper.load_from_checkpoint(
        path, map_location="cpu", weights_only=False
    )
    _assert_stats_equal(
        algo.norm_stats.norm_stats, recovered.model.norm_stats.norm_stats
    )
    tree = OmegaConf.to_container(recovered.hparams.config_tree, resolve=True)
    assert tree == OmegaConf.to_container(
        objects["model"].hparams.config_tree, resolve=True
    )
    for domain in selected:
        assert tree["action_timing"][domain]["output_offsets_s"][-1] == pytest.approx(
            8 / 30
        )
        assert tree["action_timing"][domain]["includes_current"] is True
    if "human_bimanual" in selected:
        assert tree["data_contracts"]["human_bimanual"]["gripper_mode"] == mode
        altered = copy.deepcopy(recovered.hparams.config_tree)
        altered.data_contracts.human_bimanual.gripper_mode = (
            "width" if mode == "zero" else "zero"
        )
        recovered.hparams.config_tree = altered
        with pytest.raises(ValueError, match="data_contracts differ"):
            recovered.on_load_checkpoint(saved)
    # CPU memory/disk budget is bounded even when all five real models run.
    path.unlink()


def build_local(root, kind, mode="width"):
    spec = ActionTiming(30, 9, 2, 9)
    embodiment = Piper if kind == "piper" else PicoEgo
    kwargs = {} if kind == "piper" else {"gripper_mode": mode}
    resolver = LocalRecordingResolver(
        root,
        kind,
        embodiment.get_keymap(timing=spec, **kwargs),
        embodiment.get_transform_list(timing=spec, **kwargs),
        allow_synthetic_calibration=True,
        gripper_mode=mode,
    )
    return RecordingMultiDataset._from_resolver(resolver, mode="total")


@pytest.mark.parametrize("kind", ["piper", "pico_ego"])
def test_two_worker_sample_identity_and_timestamp_remain_with_values(tmp_path, kind):
    for index in range(2):
        write_recording(tmp_path, kind, f"source_{index}", phase=index * 0.4)
    dataset = build_local(tmp_path, kind)
    sequence = []
    # The sampler alternates source episodes in one batch; workers do real Zarr
    # reads and transforms, not synthetic tensors or mocked dataset instances.
    for row in (1, 4, 8, 12):
        sequence.extend([row, 24 + row])
    loader = DataLoader(dataset, batch_size=4, num_workers=2, sampler=sequence)
    seen = []
    for batch in loader:
        for j, episode_id in enumerate(batch["episode_hash"]):
            row = int(batch["sample_index"][j])
            direct = dataset.datasets[episode_id][row]
            assert batch["recording_id"][j] == direct["recording_id"]
            assert batch["sample_timestamp"].dtype == torch.float64
            assert batch["sample_timestamp"][j] == direct["sample_timestamp"]
            torch.testing.assert_close(
                batch["actions_cartesian"][j], direct["actions_cartesian"]
            )
            torch.testing.assert_close(
                batch["observations.state.ee_pose"][j],
                direct["observations.state.ee_pose"],
            )
            seen.append((episode_id, row))
    assert len(seen) == len(sequence)
    assert len({episode for episode, _ in seen}) == 2


@pytest.mark.parametrize("kind", ["piper", "pico_ego"])
def test_extreme_validation_recording_does_not_enter_training_statistics(
    tmp_path, kind
):
    write_recording(tmp_path / "train", kind, "ordinary", offset=0)
    write_recording(tmp_path / "valid", kind, "extreme-independent", offset=1000)
    train, valid = (
        build_local(tmp_path / "train", kind),
        build_local(tmp_path / "valid", kind),
    )
    domain = "piper_bimanual" if kind == "piper" else "human_bimanual"
    emb = get_embodiment_id(domain)
    stats = MultiDataset(state={}, norm_mode="zscore")
    stats.populate_from_datasets({domain: train})
    stats.infer_norm_from_dataset(train, domain, sample_frac=1.0, num_workers=0)
    before = copy.deepcopy(stats.norm_stats)
    train.set_norm_stats_from(stats)
    valid.set_norm_stats_from(stats)
    assert valid.norm_stats is train.norm_stats is stats.norm_stats
    # The new local finite policy keeps out-of-distribution validation samples,
    # with their real identities, instead of replacing them with in-range ones.
    raw = next(iter(valid.datasets.values()))[2]
    assert torch.max(torch.abs(raw["observations.state.ee_pose"])).item() > 500
    normalized = valid[2]
    assert normalized["sample_index"] == 2
    assert normalized["episode_hash"] == raw["episode_hash"]
    assert normalized["sample_timestamp"] == raw["sample_timestamp"]
    restored = stats.unnormalize(normalized, emb)
    torch.testing.assert_close(
        restored["observations.state.ee_pose"],
        raw["observations.state.ee_pose"],
        atol=2e-4,
        rtol=2e-6,
    )
    _assert_stats_equal(before, stats.norm_stats)
    assert (
        np.max(np.abs(stats.norm_stats[emb]["observations.state.ee_pose"]["mean"])) < 10
    )
