"""Offline composition of the three local HPT/shared-FM training recipes."""

import hydra
import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from egomimic.rldb.action_timing import parse_action_timing, validate_model_timing
from egomimic.rldb.zarr.action_chunk_transforms import (
    InterpolateLinear,
    InterpolatePose,
)

RECIPES = {
    "piper_local": {"piper_bimanual"},
    "pico_ego_local": {"human_bimanual"},
    "piper_pico_local": {"piper_bimanual", "human_bimanual"},
}


def configuration(data, mode="zero", extra=()):
    with initialize_config_module(
        config_module="egomimic.hydra_configs", version_base=None
    ):
        return compose(
            config_name="train_piper_pico",
            overrides=[f"data={data}", f"recordings.gripper_mode={mode}", *extra],
        )


@pytest.mark.parametrize("data", RECIPES)
@pytest.mark.parametrize("mode", ["zero", "width"])
def test_selected_domains_shared_head_camera_and_gripper_contracts(data, mode):
    cfg = configuration(data, mode)
    specs = parse_action_timing(OmegaConf.to_container(cfg.action_timing, resolve=True))
    model = OmegaConf.to_container(cfg.model, resolve=True)
    validate_model_timing(specs, model)
    algo = model["robomimic_model"]
    assert (
        set(specs)
        == set(cfg.data.train_datasets)
        == set(cfg.data.valid_datasets)
        == RECIPES[data]
    )
    assert set(algo["domains"]) == RECIPES[data]
    head = algo["head_specs"]["shared"]
    assert head["model"]["act_dim"] == 14
    assert head["padding"] == "zero"
    assert head["infer_ac_dims"] == {"piper_bimanual": 14, "human_bimanual": 14}
    assert algo["stem_specs"]["piper_bimanual"]["state_ee_pose"]["input_dim"] == 14
    assert algo["stem_specs"]["human_bimanual"]["state_ee_pose"]["input_dim"] == (
        12 if mode == "zero" else 14
    )
    for name, node in cfg.data.train_datasets.items():
        keymap = hydra.utils.instantiate(node.resolver.key_map)
        image_keys = {
            item["zarr_key"]
            for item in keymap.values()
            if item["key_type"] == "camera_keys"
        }
        expected = (
            {"images.front_1", "images.left_wrist", "images.right_wrist"}
            if name == "piper_bimanual"
            else {"images.front_1"}
        )
        assert image_keys == expected
        assert (
            node._target_
            == "egomimic.rldb.zarr.local_recordings.RecordingMultiDataset._from_resolver"
        )
        assert node.mode == "train"
        assert cfg.data.valid_datasets[name].mode == "valid"
        assert cfg.data.valid_datasets[name].resolver == node.resolver
        if name == "human_bimanual":
            assert node.resolver.gripper_mode == mode
            assert node.resolver.expected_axis_convention == "piper_x"
            assert ("left.action_gripper" in keymap) is (mode == "width")
        else:
            assert node.resolver.allow_synthetic_calibration is False


@pytest.mark.parametrize("data", RECIPES)
def test_all_model_dataset_timings_follow_overrides_without_changing_spans(data):
    cfg = configuration(
        data,
        extra=[
            "recordings.output_chunk_length=37",
            "recordings.piper_timing.source_horizon=17",
            "recordings.piper_timing.action_stride=3",
            "recordings.ego_timing.source_horizon=17",
            "recordings.ego_timing.action_stride=3",
        ],
    )
    specs = parse_action_timing(OmegaConf.to_container(cfg.action_timing, resolve=True))
    validate_model_timing(specs, OmegaConf.to_container(cfg.model, resolve=True))
    for name, node in cfg.data.train_datasets.items():
        spec = specs[name]
        assert spec.source_horizon == 17
        assert spec.action_stride == 3
        assert spec.output_chunk_length == 37
        assert spec.effective_span_s == 0.5  # Support offsets 0,3,...,15 at 30 Hz.
        keymap = hydra.utils.instantiate(node.resolver.key_map)
        assert {item["horizon"] for item in keymap.values() if "horizon" in item} == {
            17
        }
        transforms = hydra.utils.instantiate(node.resolver.transform_list)
        for transform in transforms:
            if isinstance(transform, (InterpolatePose, InterpolateLinear)):
                assert transform.stride == 3
                assert transform.new_chunk_length == 37
    assert cfg.model.robomimic_model.head_specs.shared.action_horizon == 37
    assert cfg.model.robomimic_model.head_specs.shared.model.act_seq == 37
    # Trunk conditioning tokens do not encode the trajectory's physical duration.
    assert cfg.model.robomimic_model.trunk.action_horizon == 64


def test_roots_and_hardware_axis_and_split_are_explicitly_overridable():
    cfg = configuration(
        "piper_pico_local",
        extra=[
            "recordings.piper_root=/tmp/robot_exports",
            "recordings.pico_ego_root=/tmp/human_exports",
            "recordings.expected_axis_convention=piper",
            "recordings.valid_ratio=0.3",
            "recordings.split_seed=17",
            "recordings.batch_size=4",
            "recordings.num_workers=2",
        ],
    )
    assert (
        cfg.data.train_datasets.piper_bimanual.resolver.folder_path
        == "/tmp/robot_exports"
    )
    assert (
        cfg.data.train_datasets.human_bimanual.resolver.folder_path
        == "/tmp/human_exports"
    )
    assert (
        cfg.data.valid_datasets.human_bimanual.resolver.expected_axis_convention
        == "piper"
    )
    for split in ("train_datasets", "valid_datasets"):
        for node in cfg.data[split].values():
            assert node.valid_ratio == 0.3
            assert node.split_seed == 17
    for split in ("train_dataloader_params", "valid_dataloader_params"):
        for node in cfg.data[split].values():
            assert node.batch_size == 4
            assert node.num_workers == 2


def test_shared_head_rejects_different_domain_output_lengths():
    cfg = configuration(
        "piper_pico_local", extra=["recordings.ego_timing.output_chunk_length=75"]
    )
    specs = parse_action_timing(OmegaConf.to_container(cfg.action_timing, resolve=True))
    with pytest.raises(ValueError, match="output_chunk_length 75"):
        validate_model_timing(specs, OmegaConf.to_container(cfg.model, resolve=True))


def test_new_viz_uses_data_driven_piper_and_pico_classes():
    cfg = configuration("piper_pico_local")
    for name, class_name in (
        ("piper_bimanual", "piper.Piper"),
        ("human_bimanual", "pico_ego.PicoEgo"),
    ):
        node = cfg.evaluator.viz_func[name]
        assert node._target_ == f"egomimic.rldb.embodiment.{class_name}.viz_gt_preds"
        assert node.annotation_key is None
