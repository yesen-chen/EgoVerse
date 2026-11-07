"""Observation-only prediction with real HPT/FM and saved physical statistics."""

from __future__ import annotations

import copy

import hydra
import numpy as np
import pytest
import torch
from fixtures.train_harness import hermetic_env
from unit.test_piper_pico_training import compose_local

from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset


@pytest.fixture(params=["zero", "width"])
def model(request, tmp_path, monkeypatch):
    hermetic_env(monkeypatch)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    state = {
        "norm_mode": "quantile",
        "embodiments": [7, 3],
        "key_types": {},
        "zarr_keys": {},
        "shapes": {},
        "norm_stats": {},
    }
    for emb in (7, 3):
        dim = 12 if emb == 3 and request.param == "zero" else 14
        shapes = {
            "observations.state.ee_pose": (dim,),
            "actions_cartesian": (9, dim),
            "observations.images.front_img_1": (3, 32, 32),
        }
        if emb == 7:
            shapes.update(
                {
                    f"observations.images.{side}_wrist_img": (3, 32, 32)
                    for side in ("left", "right")
                }
            )
        state["shapes"][emb] = shapes
        state["key_types"][emb] = {
            key: "action_keys"
            if key == "actions_cartesian"
            else "camera_keys"
            if ".images." in key
            else "proprio_keys"
            for key in shapes
        }
        state["zarr_keys"][emb] = {key: key for key in shapes}
        state["norm_stats"][emb] = {}
        for key, shape in shapes.items():
            if ".images." in key:
                continue
            # Different domains, coordinates and time indices have distinct
            # nontrivial scales; accidental no-normalization cannot pass.
            lo = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * 0.03 + emb
            state["norm_stats"][emb][key] = {
                "quantile_1": lo,
                "quantile_99": lo + 2 + np.arange(dim, dtype=np.float32) * 0.2,
            }
    stats = MultiDataset.from_state(state)
    cfg = compose_local(
        "piper_pico_local",
        request.param,
        {"piper": tmp_path, "pico_ego": tmp_path},
        tmp_path / "config",
    )
    algo = hydra.utils.instantiate(cfg.model.robomimic_model, norm_stats=stats)
    yield algo
    torch.set_num_threads(previous)


def observations(algo, *, uint8=False):
    rng = torch.Generator().manual_seed(38)
    batch = {}
    for emb in (7, 3):
        fields = {}
        for key in algo.camera_keys[emb]:
            image = torch.randint(
                1, 255, (2, 3, 32, 32), dtype=torch.uint8, generator=rng
            )
            fields[key] = image if uint8 else image.float() / 255
        key = "observations.state.ee_pose"
        fields[key] = (
            torch.as_tensor(algo.norm_stats.norm_stats[emb][key]["quantile_1"]).repeat(
                2, 1
            )
            + 0.7
        )
        batch[emb] = fields
    return batch


def reference_eval(algo, raw, monkeypatch, seed):
    batch = {}
    for emb, fields in raw.items():
        dim = algo.norm_stats.key_shape("actions_cartesian", emb)[-1]
        domain = "piper_bimanual" if emb == 7 else "human_bimanual"
        # Labels exist only in the old evaluation reference, never in the new
        # inference API. Their values must have no influence on prediction.
        fields = dict(fields, actions_cartesian=torch.ones(2, 9, dim))
        batch[domain] = algo.norm_stats.normalize(fields, emb)
    original = algo.nets["policy"].compute_loss

    def loss_without_advancing_prediction_rng(*args, **kwargs):
        # forward_eval computes a stochastic validation loss before sampling.
        # Isolate only its RNG use so both prediction paths receive the exact
        # same FM noise; the actual loss calculation still executes.
        with torch.random.fork_rng(devices=[]):
            return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(
            algo.nets["policy"], "compute_loss", loss_without_advancing_prediction_rng
        )
        torch.manual_seed(seed)
        with torch.inference_mode():
            result = algo.forward_eval(algo.process_batch_for_training(batch))
    return {key: value for key, value in result.items() if not key.endswith("_loss")}


def test_physical_predictions_match_existing_eval_without_action_labels(
    model, monkeypatch
):
    raw = observations(model)
    original = copy.deepcopy(raw)
    model.nets.eval()
    expected = reference_eval(model, raw, monkeypatch, seed=71)
    torch.manual_seed(71)
    actual = model.predict_from_observations(raw)
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        assert not actual[key].requires_grad
        assert torch.is_inference(actual[key])
    for emb in raw:
        for key in raw[emb]:
            torch.testing.assert_close(
                raw[emb][key], original[emb][key], rtol=0, atol=0
            )


def test_uint8_aliases_eval_augs_step_override_and_state_restoration(
    model, monkeypatch
):
    raw = observations(model, uint8=True)
    # Both model-side key and saved source aliases are supported.
    key = "observations.state.ee_pose"
    model.norm_stats.zarr_keys[7][key] = "physical_state"
    raw[7]["physical_state"] = raw[7].pop(key)
    model.nets.train()
    policy = model.nets["policy"]
    policy.heads["shared"].model.eval()  # preserve mixed subtree modes exactly
    before_modes = [module.training for module in model.nets.modules()]
    head = policy.heads["shared"]
    head.dt = -0.875
    configured_steps = head.num_inference_steps
    observed_steps, fields_seen = [], []
    original_forward = policy.forward
    original_augs = model.eval_image_augs

    def forward(domain, data):
        assert not model.nets.training and torch.is_inference_mode_enabled()
        assert (
            "action" not in data and "pad_mask" not in data and "embodiment" not in data
        )
        observed_steps.append(head.num_inference_steps)
        fields_seen.append(set(data))
        return original_forward(domain, data)

    aug_calls = []

    def eval_augs(image):
        aug_calls.append(image.clone())
        assert image.dtype == torch.float32 and image.max() <= 1
        return original_augs(image)

    monkeypatch.setattr(policy, "forward", forward)
    monkeypatch.setattr(model, "eval_image_augs", eval_augs)
    monkeypatch.setattr(
        model,
        "train_image_augs",
        lambda _: pytest.fail("training augmentation used in inference"),
    )
    result = model.predict_from_observations(raw, num_inference_steps=3)
    assert observed_steps == [3, 3] and len(aug_calls) == 4
    assert len(fields_seen) == 2 and result
    assert [module.training for module in model.nets.modules()] == before_modes
    assert head.num_inference_steps == configured_steps and head.dt == -0.875
    assert model.training_step == 0


def test_failure_restores_modes_steps_and_does_not_leave_new_dt(model, monkeypatch):
    policy, raw = model.nets["policy"], observations(model)
    head = policy.heads["shared"]
    assert not hasattr(head, "dt")
    before = [module.training for module in model.nets.modules()]
    original = policy.forward

    def fail(domain, data):
        original(domain, data)  # real sampling creates dt
        raise RuntimeError("after sampling")

    monkeypatch.setattr(policy, "forward", fail)
    with pytest.raises(RuntimeError, match="after sampling"):
        model.predict_from_observations(raw, num_inference_steps=3)
    assert head.num_inference_steps == 2 and not hasattr(head, "dt")
    assert [module.training for module in model.nets.modules()] == before


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_reject_invalid_step_override(model, steps):
    with pytest.raises(ValueError, match="positive integer"):
        model.predict_from_observations(observations(model), num_inference_steps=steps)


@pytest.mark.parametrize("steps", [0, -1, True])
def test_reject_invalid_saved_steps_before_sampling(model, steps):
    model.nets["policy"].heads["shared"].num_inference_steps = steps
    with pytest.raises(ValueError, match="Saved Flow Matching"):
        model.predict_from_observations(observations(model))


@pytest.mark.parametrize(
    "fault",
    [
        "labels",
        "missing",
        "nan",
        "image_range",
        "batch",
        "missing_stats",
        "bad_horizon",
    ],
)
def test_reject_invalid_observation_contract(model, fault):
    raw = observations(model)
    key = "observations.state.ee_pose"
    if fault == "labels":
        raw[7]["actions_cartesian"] = torch.zeros(2, 9, 14)
    elif fault == "missing":
        del raw[7][key]
    elif fault == "nan":
        raw[7][key][0, 0] = torch.nan
    elif fault == "image_range":
        raw[7]["observations.images.front_img_1"][0, 0, 0, 0] = 255
    elif fault == "batch":
        raw[7][key] = raw[7][key][:1]
    elif fault == "missing_stats":
        del model.norm_stats.norm_stats[7][key]
    else:
        model.norm_stats.shapes[7]["actions_cartesian"] = (8, 14)
    with pytest.raises(ValueError):
        model.predict_from_observations(raw)
