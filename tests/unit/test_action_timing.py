"""Timing metadata must describe the existing EgoVerse interpolation exactly."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egomimic.rldb.action_timing import (
    ActionTiming,
    parse_action_timing,
    validate_dataset_timing,
    validate_model_timing,
)
from egomimic.rldb.zarr.action_chunk_transforms import (
    InterpolateLinear,
    InterpolatePose,
)


def _fields(**overrides):
    fields = {
        "source_fps": 30.0,
        "source_horizon": 45,
        "action_stride": 1,
        "output_chunk_length": 100,
        "includes_current": True,
    }
    fields.update(overrides)
    return fields


def _leaf(**overrides):
    spec = ActionTiming(**_fields())
    values = {
        "episode_path": Path("/synthetic/recording/episode_000000.zarr"),
        "metadata": {"fps": spec.source_fps},
        "key_map": {
            "left.action_ee_pose": {
                "zarr_key": "left.cmd_ee_pose",
                "key_type": "action_keys",
                "horizon": spec.source_horizon,
            },
            "left.action_gripper": {
                "zarr_key": "left.cmd_gripper",
                "key_type": "action_keys",
                "horizon": spec.source_horizon,
            },
            # Eva's annotation entry can be action_keys without being a window.
            "annotations": {
                "zarr_key": "annotations",
                "key_type": "action_keys",
            },
        },
        "transform": [
            InterpolatePose(
                new_chunk_length=spec.output_chunk_length,
                action_key="left.action_ee_pose",
                output_action_key="left.action_ee_pose",
                stride=spec.action_stride,
            ),
            InterpolateLinear(
                new_chunk_length=spec.output_chunk_length,
                action_key="left.action_gripper",
                output_action_key="left.action_gripper",
                stride=spec.action_stride,
            ),
        ],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _dataset(*leaves):
    return SimpleNamespace(datasets={str(i): leaf for i, leaf in enumerate(leaves)})


@pytest.mark.parametrize("points", [45, 100, 257])
def test_output_density_does_not_change_source_trajectory_duration(points):
    timing = ActionTiming(**_fields(output_chunk_length=points))
    assert timing.support_indices == tuple(range(45))
    assert timing.source_span_s == pytest.approx(44 / 30)
    assert timing.effective_span_s == pytest.approx(44 / 30)
    assert len(timing.output_offsets_s) == points
    assert timing.output_offsets_s[0] == 0
    assert timing.output_offsets_s[-1] == pytest.approx(44 / 30)
    np.testing.assert_allclose(np.diff(timing.output_offsets_s), 44 / 30 / (points - 1))


def test_stride_uses_last_retained_support_instead_of_unsampled_window_end():
    timing = ActionTiming(**_fields(source_horizon=30, action_stride=3))
    assert timing.support_indices == tuple(range(0, 30, 3))
    assert timing.source_span_s == pytest.approx(29 / 30)
    assert timing.effective_span_s == pytest.approx(27 / 30)
    assert timing.output_offsets_s[-1] == pytest.approx(0.9)


@pytest.mark.parametrize(
    "horizon,stride,points", [(45, 1, 100), (30, 3, 100), (11, 4, 9)]
)
def test_time_axis_matches_existing_pose_and_scalar_interpolation(
    horizon, stride, points
):
    timing = ActionTiming(
        **_fields(
            source_horizon=horizon, action_stride=stride, output_chunk_length=points
        )
    )
    source_t = np.arange(horizon, dtype=np.float64) / timing.source_fps
    # Translation is a known physical velocity; yaw crosses pi, and stored
    # quaternions alternate signs without changing their represented rotations.
    positions = np.column_stack((source_t, -2 * source_t, 0.5 * source_t))
    source_rotations = Rotation.from_euler("Z", (2.8 + 0.6 * source_t)[:, None])
    q_xyzw = source_rotations.as_quat()
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]
    q_wxyz[1::2] *= -1
    poses = np.column_stack((positions, q_wxyz))
    grip = (0.2 + 0.1 * source_t)[:, None]
    out = InterpolatePose(points, "pose", "out", stride=stride).transform(
        {"pose": poses}
    )["out"]
    grip_out = InterpolateLinear(points, "grip", "out", stride=stride).transform(
        {"grip": grip}
    )["out"]
    target_t = np.asarray(timing.output_offsets_s)
    np.testing.assert_allclose(
        out[:, :3],
        np.column_stack((target_t, -2 * target_t, 0.5 * target_t)),
        atol=1e-12,
    )
    np.testing.assert_allclose(grip_out[:, 0], 0.2 + 0.1 * target_t, atol=1e-12)
    actual_rotations = Rotation.from_quat(out[:, [4, 5, 6, 3]])
    expected_rotations = Rotation.from_euler("Z", (2.8 + 0.6 * target_t)[:, None])
    np.testing.assert_allclose(
        (actual_rotations.inv() * expected_rotations).magnitude(), 0, atol=1e-12
    )
    np.testing.assert_allclose(out[0, :3], poses[0, :3])
    np.testing.assert_allclose(out[-1, :3], poses[timing.support_indices[-1], :3])


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_fps", True),
        ("source_fps", "30"),
        ("source_fps", 0),
        ("source_fps", -30),
        ("source_fps", float("nan")),
        ("source_fps", float("inf")),
        ("source_horizon", True),
        ("source_horizon", 2.5),
        ("source_horizon", 0),
        ("source_horizon", -1),
        ("action_stride", True),
        ("action_stride", 1.5),
        ("action_stride", 0),
        ("action_stride", -1),
        ("output_chunk_length", True),
        ("output_chunk_length", 2.5),
        ("output_chunk_length", 1),
        ("output_chunk_length", 0),
        ("includes_current", False),
        ("includes_current", 1),
    ],
)
def test_invalid_parameters_fail_before_interpolation(field, value):
    with pytest.raises((TypeError, ValueError)):
        ActionTiming(**_fields(**{field: value}))


@pytest.mark.parametrize("horizon,stride", [(1, 1), (2, 2), (30, 45)])
def test_at_least_two_retained_support_points_are_required(horizon, stride):
    with pytest.raises(ValueError):
        ActionTiming(**_fields(source_horizon=horizon, action_stride=stride))


def test_serialized_timing_is_json_safe_and_immutable():
    timing = ActionTiming(**_fields(source_horizon=30, action_stride=3))
    stored = json.loads(json.dumps(timing.to_dict(), allow_nan=False))
    assert stored["schema"] == "action_timing_v1"
    for key, value in _fields(source_horizon=30, action_stride=3).items():
        assert stored[key] == value
    assert stored["support_indices"] == list(range(0, 30, 3))
    assert stored["source_span_s"] == pytest.approx(29 / 30)
    assert stored["effective_span_s"] == pytest.approx(0.9)
    assert stored["output_offsets_s"] == list(timing.output_offsets_s)
    with pytest.raises(FrozenInstanceError):
        timing.source_horizon = 60


def test_parse_is_opt_in_and_preserves_separate_embodiment_durations():
    assert parse_action_timing(None) == {}
    assert parse_action_timing({}) == {}
    parsed = parse_action_timing(
        {
            "human_bimanual": _fields(source_horizon=30, action_stride=3),
            "piper_bimanual": _fields(),
        }
    )
    assert parsed["human_bimanual"].effective_span_s == pytest.approx(0.9)
    assert parsed["piper_bimanual"].effective_span_s == pytest.approx(44 / 30)


@pytest.mark.parametrize(
    "config",
    [[], {"piper_bimanual": None}, {"piper_bimanual": _fields(unrecognized=1)}],
)
def test_parser_rejects_malformed_or_unknown_fields(config):
    with pytest.raises((TypeError, ValueError)):
        parse_action_timing(config)


def test_runtime_checks_both_splits_and_ignores_unwindowed_annotations():
    specs = parse_action_timing({"piper_bimanual": _fields()})
    validate_dataset_timing(
        specs,
        {"piper_bimanual": _dataset(_leaf(), _leaf())},
        {"piper_bimanual": _dataset(_leaf())},
    )


def test_legacy_training_without_timing_does_not_require_new_metadata():
    validate_dataset_timing(
        {}, {"eva_bimanual": object()}, {"human_bimanual": object()}
    )


@pytest.mark.parametrize("split", ["train", "valid"])
@pytest.mark.parametrize(
    "mismatch",
    [
        "fps",
        "missing_fps",
        "horizon",
        "stride",
        "points",
        "no_horizon",
        "no_interpolator",
    ],
)
def test_runtime_rejects_disagreement_with_loaded_data(split, mismatch):
    leaf = _leaf()
    if mismatch == "fps":
        leaf.metadata["fps"] = 60
    elif mismatch == "missing_fps":
        del leaf.metadata["fps"]
    elif mismatch == "horizon":
        leaf.key_map["left.action_ee_pose"]["horizon"] = 30
    elif mismatch == "stride":
        leaf.transform[0].stride = 3
    elif mismatch == "points":
        leaf.transform[1].new_chunk_length = 50
    elif mismatch == "no_horizon":
        for entry in leaf.key_map.values():
            entry.pop("horizon", None)
    elif mismatch == "no_interpolator":
        leaf.transform = []
    train = {"piper_bimanual": _dataset(_leaf())}
    valid = {"piper_bimanual": _dataset(_leaf())}
    (train if split == "train" else valid)["piper_bimanual"] = _dataset(_leaf(), leaf)
    with pytest.raises(ValueError):
        validate_dataset_timing(
            parse_action_timing({"piper_bimanual": _fields()}), train, valid
        )


def test_runtime_rejects_timing_for_an_unselected_dataset():
    with pytest.raises(ValueError):
        validate_dataset_timing(
            parse_action_timing({"piper_bimanual": _fields()}),
            {"eva_bimanual": _dataset(_leaf())},
            {},
        )


def _hpt_config(domain_head=True, shared_head=False):
    heads = {}
    if domain_head:
        heads["piper_bimanual"] = {
            "action_horizon": 100,
            "model": {"act_seq": 100},
        }
    if shared_head:
        heads["shared"] = {"action_horizon": 100, "model": {"act_seq": 100}}
    return {
        "robomimic_model": {
            "_target_": "egomimic.algo.hpt.HPT",
            "trunk": {"action_horizon": 64},
            "head_specs": heads,
            "shared_ac_key": "actions_cartesian" if shared_head else None,
        }
    }


def _specs():
    return parse_action_timing({"piper_bimanual": _fields()})


@pytest.mark.parametrize("domain,shared", [(True, False), (False, True), (True, True)])
def test_hpt_checks_domain_and_shared_outputs_not_trunk_tokens(domain, shared):
    validate_model_timing(_specs(), _hpt_config(domain, shared))


@pytest.mark.parametrize("field", ["action_horizon", "act_seq"])
@pytest.mark.parametrize("domain,shared", [(True, False), (False, True), (True, True)])
def test_hpt_rejects_wrong_output_length_in_every_active_head(field, domain, shared):
    cfg = _hpt_config(domain, shared)
    # When both heads are present, changing only the shared head must still fail.
    head = cfg["robomimic_model"]["head_specs"][
        "shared" if shared else "piper_bimanual"
    ]
    if field == "act_seq":
        head["model"][field] = 50
    else:
        head[field] = 50
    with pytest.raises(ValueError, match=field):
        validate_model_timing(_specs(), cfg)


def test_hpt_does_not_require_optional_network_act_seq():
    cfg = _hpt_config()
    del cfg["robomimic_model"]["head_specs"]["piper_bimanual"]["model"]
    validate_model_timing(_specs(), cfg)


def test_shared_model_rejects_disagreeing_embodiment_output_lengths():
    specs = parse_action_timing(
        {
            "piper_bimanual": _fields(),
            "human_bimanual": _fields(output_chunk_length=50),
        }
    )
    with pytest.raises(ValueError):
        validate_model_timing(specs, _hpt_config(False, True))


@pytest.mark.parametrize("bad_value", [None, True, 100.0, "100"])
def test_hpt_rejects_noninteger_or_missing_head_length(bad_value):
    cfg = _hpt_config()
    cfg["robomimic_model"]["head_specs"]["piper_bimanual"]["action_horizon"] = bad_value
    with pytest.raises(ValueError):
        validate_model_timing(_specs(), cfg)


@pytest.mark.parametrize("length,valid", [(100, True), (50, False)])
def test_pi_uses_its_nested_network_action_horizon(length, valid):
    cfg = {
        "robomimic_model": {
            "_target_": "egomimic.algo.pi.PI",
            "config": {
                # This sibling is NOT the field read by the local PI wrapper.
                "action_horizon": 999,
                "model": {"action_horizon": length},
            },
        }
    }
    if valid:
        validate_model_timing(_specs(), cfg)
    else:
        with pytest.raises(ValueError, match="config.model.action_horizon"):
            validate_model_timing(_specs(), cfg)


def test_model_validation_supports_omegaconf_references_without_instantiating():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"points": 100, "model": _hpt_config()})
    cfg.model.robomimic_model.head_specs.piper_bimanual.action_horizon = "${points}"
    cfg.model.robomimic_model.head_specs.piper_bimanual.model.act_seq = "${points}"
    validate_model_timing(_specs(), cfg.model)


@pytest.mark.parametrize(
    "cfg",
    [
        None,
        {},
        {"robomimic_model": {"_target_": "unknown.Model", "head_specs": {}}},
        {"robomimic_model": {"_target_": "egomimic.algo.hpt.HPT", "head_specs": {}}},
        {
            "robomimic_model": {
                "_target_": "egomimic.algo.pi.PI",
                "config": {"action_horizon": 100},
            }
        },
    ],
)
def test_unknown_model_structures_cannot_claim_timing_validation(cfg):
    with pytest.raises(ValueError):
        validate_model_timing(_specs(), cfg)


def test_disabled_timing_leaves_legacy_unknown_models_untouched():
    validate_model_timing({}, object())
