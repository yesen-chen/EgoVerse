"""CPU tier: every (vendor, algo) recipe builds and takes 2 optimizer steps on
synthetic data through the real trainHydra.train() path. Pi's openpi network
is a stub; the EgoVerse Pi wrapper (keymaps, converters, prompts, batch
assembly) is real."""

from __future__ import annotations

import math
import os

import pytest
import torch
from fixtures.recipes import (
    RECIPES,
    common_overrides,
    cpu_trainer_overrides,
    hpt_small_overrides,
    pi_cpu_overrides,
)
from fixtures.train_harness import (
    STATE_DIM,
    BatchKeySpy,
    StubPI0,
    compose_recipe,
    hermetic_env,
    pi_unavailable,
    write_fixtures,
)

import egomimic.trainHydra as train_hydra
from egomimic.algo.hpt import HPT
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Human

VENDOR_NAMES = ["eva", "aria", "mecka", "scale"]
STEPS = 2


def _run(cfg) -> float:
    metrics, objects = train_hydra.train(cfg)
    loss = float(metrics["Train/action_loss"])
    assert math.isfinite(loss), f"non-finite loss {loss}"
    assert objects["trainer"].global_step == STEPS
    return loss


@pytest.mark.parametrize("vendor", VENDOR_NAMES)
def test_hpt_train_step(tmp_path, monkeypatch, vendor):
    hermetic_env(monkeypatch)
    recipe = RECIPES[(vendor, "hpt")]
    data, out, hashes = write_fixtures(tmp_path, vendor)
    cfg = compose_recipe(
        recipe,
        common_overrides(
            recipe.embodiment,
            data,
            out,
            batch_size=2,
            num_workers=0,
            episode_hashes=hashes,
        )
        + cpu_trainer_overrides(STEPS)
        + hpt_small_overrides(recipe.embodiment),
        out,
    )
    spy = BatchKeySpy(monkeypatch, HPT)
    _run(cfg)
    seen = spy.keys[recipe.embodiment]
    is_eva = recipe.embodiment == "eva_bimanual"
    front = Eva.VIZ_IMAGE_KEY if is_eva else Human.VIZ_IMAGE_KEY
    expected = {"actions_cartesian", "observations.state.ee_pose", front}
    if is_eva:
        expected |= {
            "observations.images.right_wrist_img",
            "observations.images.left_wrist_img",
        }
    assert (
        expected <= seen
    ), f"{vendor}/hpt missing {expected - seen}; saw {sorted(seen)}"


def test_eva_train_step_with_explicit_action_timing(tmp_path, monkeypatch):
    """Exercise timing checks on real datasets and save after actual CPU steps."""
    hermetic_env(monkeypatch)
    recipe = RECIPES[("eva", "hpt")]
    data, out, hashes = write_fixtures(tmp_path, "eva")
    cfg = compose_recipe(
        recipe,
        common_overrides(
            recipe.embodiment,
            data,
            out,
            batch_size=2,
            num_workers=0,
            episode_hashes=hashes,
        )
        + cpu_trainer_overrides(STEPS)
        + hpt_small_overrides(recipe.embodiment),
        out,
    )
    cfg.action_timing = {
        "eva_bimanual": {
            "source_fps": 30,
            "source_horizon": 45,
            "action_stride": 1,
            "output_chunk_length": 100,
            "includes_current": True,
        }
    }
    metrics, objects = train_hydra.train(cfg)
    assert math.isfinite(float(metrics["Train/action_loss"]))
    assert objects["trainer"].global_step == STEPS
    path = out / "timing_after_training.ckpt"
    objects["trainer"].save_checkpoint(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    timing = saved["hyper_parameters"]["config_tree"]["action_timing"]["eva_bimanual"]
    assert timing["includes_current"] is True
    assert timing["source_horizon"] == 45
    assert len(timing["output_offsets_s"]) == 100
    assert timing["output_offsets_s"][-1] == pytest.approx(44 / 30)
    # Trainer.fit resumes into a current-config model, unlike classmethod load.
    # Ensure its actual restore hook rejects changing the time contract.
    cfg.action_timing.eva_bimanual.source_horizon = 30
    objects["model"].hparams.config_tree = train_hydra._build_model_config_tree(cfg)
    with pytest.raises(ValueError, match="Checkpoint action_timing differs"):
        objects["trainer"].fit(
            model=objects["model"],
            datamodule=objects["datamodule"],
            ckpt_path=path,
            weights_only=False,
        )


@pytest.mark.parametrize("vendor", VENDOR_NAMES)
def test_pi_train_step(tmp_path, monkeypatch, vendor):
    reason = pi_unavailable()
    if reason:
        if os.environ.get("EGOVERSE_REQUIRE_PI"):  # set in CI
            pytest.fail(reason)
        pytest.skip(reason)
    import openpi.models_pytorch.pi0_pytorch as pi0_pytorch

    from egomimic.algo.pi import PI

    hermetic_env(monkeypatch)
    recipe = RECIPES[(vendor, "pi")]
    is_eva = recipe.embodiment == "eva_bimanual"
    monkeypatch.setattr(pi0_pytorch, "PI0Pytorch", StubPI0)
    monkeypatch.setattr(
        StubPI0,
        "expect",
        {"state_dim": STATE_DIM[recipe.embodiment], "wrist_present": is_eva},
    )
    data, out, hashes = write_fixtures(tmp_path, vendor)
    cfg = compose_recipe(
        recipe,
        common_overrides(
            recipe.embodiment,
            data,
            out,
            batch_size=2,
            num_workers=0,
            episode_hashes=hashes,
        )
        + cpu_trainer_overrides(STEPS)
        + pi_cpu_overrides(),
        out,
    )
    spy = BatchKeySpy(monkeypatch, PI)
    _run(cfg)
    # The wrapper sees dataset camera names; StubPI0 asserts it was handed
    # openpi's slot names, which is the remap working end to end.
    seen = spy.keys[recipe.embodiment]
    front = Eva.VIZ_IMAGE_KEY if is_eva else Human.VIZ_IMAGE_KEY
    expected = {"actions_cartesian", "observations.state.ee_pose", front, "annotations"}
    if is_eva:
        expected |= {
            "observations.images.left_wrist_img",
            "observations.images.right_wrist_img",
        }
    assert (
        expected <= seen
    ), f"{vendor}/pi missing {expected - seen}; saw {sorted(seen)}"
