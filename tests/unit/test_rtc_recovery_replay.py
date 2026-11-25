"""CPU checks for causal support and paired recovery-counterfactual metrics."""

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def runner():
    path = (
        Path(__file__).resolve().parents[2] / "pi05/scripts/debug_pi05_rtc_recovery.py"
    )
    spec = importlib.util.spec_from_file_location("rtc_recovery_test_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mask_example():
    # The first two points have no valid old plan, and the last four are padding.
    grid = np.arange(15, dtype=float) / 10
    weights = np.array([0, 0, 1, 1, 1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.03, 0, 0, 0, 0])
    return weights, grid


@pytest.mark.parametrize(
    "variant",
    [
        "original",
        "unguided",
        "hard_only",
        "overlap300",
        "hold150_release300",
        "hold200_release300",
    ],
)
def test_weight_variants_preserve_input_and_never_constrain_padding(
    runner, mask_example, variant
):
    weights, grid = mask_example
    original = weights.copy()
    result = runner.variant_weights(weights, grid, 0.25, variant)
    np.testing.assert_array_equal(weights, original)
    assert not np.shares_memory(weights, result)
    assert np.all(result[weights == 0] == 0)
    assert np.all((result >= 0) & (result <= 1))


def test_hard_only_keeps_original_tick_guard_beyond_actual_switch(runner, mask_example):
    weights, grid = mask_example
    result = runner.variant_weights(weights, grid, 0.25, "hard_only")
    # The 300 and 400 ms points were committed by the original deadline guard.
    np.testing.assert_array_equal(np.flatnonzero(result), [2, 3, 4])


def test_overlap300_includes_right_interpolation_point_and_keeps_tail(
    runner, mask_example
):
    weights, grid = mask_example
    result = runner.variant_weights(weights, grid, 0.25, "overlap300")
    # entry + 300 ms = 550 ms: both 500 and 600 ms endpoints must be weight 1.
    assert result[5] == result[6] == 1
    assert 0 < result[7] < 1
    assert np.all(np.diff(result[7:11]) < 0)
    assert result[10] > 0 and result[11] == 0


@pytest.mark.parametrize("entry", [0.75, 1.2])
def test_overlap300_rejects_missing_right_support(runner, mask_example, entry):
    weights, grid = mask_example
    with pytest.raises(ValueError, match="support"):
        runner.variant_weights(weights, grid, entry, "overlap300")


@pytest.mark.parametrize(
    "variant,hold_s,hold_index",
    [("hold150_release300", 0.150, 36), ("hold200_release300", 0.200, 41)],
)
def test_quintic_release_uses_physical_time_and_right_hold_point(
    runner, variant, hold_s, hold_index
):
    grid = np.arange(121, dtype=float) / 100
    weights = np.where(grid <= 0.24, 1, np.where(grid < 0.9, 0.4, 0))
    entry = 0.203
    result = runner.variant_weights(weights, grid, entry, variant)
    assert grid[hold_index - 1] < entry + hold_s < grid[hold_index]
    assert result[hold_index - 1] == result[hold_index] == 1
    assert result[hold_index + 1] < 1
    # A 300 ms release has its half-weight point 150 ms after the actual
    # right-grid hold endpoint, regardless of the original overlap length.
    assert result[hold_index + 15] == pytest.approx(0.5, abs=1e-7)
    assert result[hold_index + 29] > 0
    np.testing.assert_allclose(result[hold_index + 30 :], 0, atol=1e-7)
    assert np.all(np.diff(result) <= 0)
    longer_overlap = np.where(grid <= 0.24, 1, np.where(grid < 1.15, 0.8, 0))
    np.testing.assert_allclose(
        runner.variant_weights(longer_overlap, grid, entry, variant), result
    )


@pytest.mark.parametrize("variant", ["hold150_release300", "hold200_release300"])
def test_quintic_hold_does_not_shorten_existing_hard_prefix(runner, variant):
    grid = np.arange(121, dtype=float) / 100
    weights = np.where(grid <= 0.47, 1, np.where(grid < 1.0, 0.5, 0))
    result = runner.variant_weights(weights, grid, 0.203, variant)
    np.testing.assert_array_equal(result[weights == 1], 1)
    assert result[47] == 1 and result[48] < 1
    assert result[62] == pytest.approx(0.5, abs=1e-7)
    np.testing.assert_allclose(result[77:], 0, atol=1e-7)


@pytest.mark.parametrize("variant", ["hold150_release300", "hold200_release300"])
@pytest.mark.parametrize("missing", ["hard", "hold", "release"])
def test_quintic_release_refuses_missing_support(runner, variant, missing):
    grid = np.arange(121, dtype=float) / 100
    weights = np.where(grid <= 0.24, 1, np.where(grid < 1.1, 0.5, 0))
    if missing == "hard":
        weights[weights == 1] = 0.5
        message = "committed prefix"
    elif missing == "hold":
        weights[grid >= 0.35] = 0
        message = "Hold.*support"
    else:
        # Hold is covered, but the right interpolation point at the end of
        # release is unavailable. Do not silently compress release duration.
        end_index = 66 if variant == "hold150_release300" else 71
        weights[end_index:] = 0
        message = "release.*support"
    with pytest.raises(ValueError, match=message):
        runner.variant_weights(weights, grid, 0.203, variant)


@pytest.mark.parametrize(
    "run_name", [".", "..", "../outside", "nested/run", "/tmp/run"]
)
def test_run_name_cannot_escape_output_directory(
    runner, monkeypatch, tmp_path, run_name
):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "index.json").write_text("{}")
    monkeypatch.setattr(runner, "HERE", tmp_path)
    with pytest.raises(ValueError, match="directory name"):
        runner.run("identity", run_name=run_name)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bundle"]


@pytest.mark.parametrize(
    "weights,grid",
    [
        ([1, 0.5, 0], [0, 0.1, np.inf]),
        ([1, 0.5, 0], [0, 0.1, np.nan]),
        ([[1, 0.5], [0.5, 0]], [[0, 0.1], [0.2, 0.3]]),
        ([1], [0]),
    ],
)
def test_invalid_action_grid_fails_before_inference(runner, weights, grid):
    with pytest.raises(ValueError):
        runner.variant_weights(weights, grid, 0.05, "original")


def test_saved_grid_closure_is_strict_threshold_future_only_and_not_fabricated(runner):
    grid = np.arange(6) / 10
    actions = np.zeros((6, 14))
    actions[:, 13] = [0.01, 0.08, 0.02, 0.019, 0.01, 0.01]
    actions[:, 7] = grid
    actions[:, 10] = grid * 2
    event = runner.close_event(actions, grid, 0.15, 0.3, 20)
    # Earlier closed samples are excluded and equality at 20 mm is not closure.
    assert event["offset_s"] == pytest.approx(0.3)
    assert event["after_switch_s"] == pytest.approx(0.15)
    assert event["before_next_recorded_switch"] is False
    np.testing.assert_allclose(event["xyz_mm"], [300, 0, 0])
    np.testing.assert_allclose(event["rotvec_rad"], [0.6, 0, 0])
    assert event["gripper_mm"] == pytest.approx(19)
    assert runner.close_event(actions, grid, 0.15, None, 5) is None
    assert (
        runner.close_event(actions, grid, 0.15, None, 20)["before_next_recorded_switch"]
        is None
    )


def test_measure_matches_analytic_linear_translation_and_rotation(runner):
    grid = np.linspace(0, 1.6, 161)
    old = np.zeros((len(grid), 14))
    raw, sg = old.copy(), old.copy()
    raw[:, 0], sg[:, 0] = 0.01 + 0.2 * grid, 0.005 + 0.1 * grid
    raw[:, 5], sg[:, 5] = 0.1 + 0.2 * grid, 0.05 + 0.1 * grid
    old[:, 13] = raw[:, 13] = sg[:, 13] = 0.08
    raw[:, 9] = sg[:, 9] = 0.02 * grid
    out = runner.measure(raw, sg, old, grid, 0.2, 0.8)
    left = out["arms"]["left"]
    assert left["raw_entry_mm"] == pytest.approx(50)
    assert left["raw_window_max_mm"] == pytest.approx(110)
    assert left["sg_entry_mm"] == pytest.approx(25)
    assert left["sg_window_max_mm"] == pytest.approx(55)
    for delta in (0, 150, 300):
        assert left[f"raw_velocity_gap_{delta}ms_m_s"] == pytest.approx(0.2)
        assert left[f"sg_velocity_gap_{delta}ms_m_s"] == pytest.approx(0.1)
    assert left["raw_entry_deg"] == pytest.approx(np.rad2deg(0.14))
    assert left["raw_window_max_deg"] == pytest.approx(np.rad2deg(0.2))
    assert left["sg_entry_deg"] == pytest.approx(np.rad2deg(0.07))
    assert left["sg_window_max_deg"] == pytest.approx(np.rad2deg(0.1))
    assert out["right_closure"] == {"20": None, "30": None, "40": None}
    assert out["right_future"]["max_upward_excursion_600ms_mm"] == pytest.approx(12)
    np.testing.assert_allclose(
        np.array(out["right_future"]["xyz_at_0_150_300_600ms_mm"])[:, 2],
        [4, 7, 10, 16],
    )


def test_blend_speed_contains_known_quintic_correction_term(runner):
    grid = np.linspace(0, 1.6, 161)
    old = np.zeros((len(grid), 14))
    new = old.copy()
    new[:, 0] = 0.08
    result = runner.measure(new, new, old, grid, 0.2, None)
    # Quintic weight has peak derivative 1.875 / duration; its 1 ms sampled
    # secant is slightly below the exact 0.5 m/s peak for an 80 mm offset.
    peak = result["arms"]["left"]["blended_target_peak_speed_m_s"]
    assert peak == pytest.approx(1.875 * 0.08 / 0.3, abs=2e-5)


@pytest.mark.parametrize("entry", [0, 1.1])
def test_measure_refuses_out_of_horizon_derivatives_or_future(runner, entry):
    grid = np.linspace(0, 1.6, 161)
    actions = np.zeros((len(grid), 14))
    with pytest.raises(ValueError, match="extrapolated"):
        runner.measure(actions, actions, actions, grid, entry, None)


@pytest.mark.parametrize("mode", ["vjp", "identity"])
@pytest.mark.parametrize("custom_run", [False, True])
def test_cpu_fake_run_preserves_pairing_and_isolates_variant_observations(
    runner, monkeypatch, tmp_path, mode, custom_run
):
    """Exercise the actual orchestration without importing JAX or any devices."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    checkpoint = tmp_path / "checkpoint"
    contract = checkpoint / "assets/egoverse/contract.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}")
    grid = np.linspace(0, 1.6333333333333333, 100)
    weights = np.where(grid <= 0.25, 1, np.where(grid < 0.85, 0.5, 0))
    source = {
        "observation.state": np.arange(14, dtype=np.float32),
        "observation.images.front_1": np.zeros((2, 2, 3), dtype=np.uint8),
        "prompt": np.array("pick basket"),
        "rtc.schema": np.array("egoverse_rtc_v1"),
        "rtc.request_id": np.array(3),
        "rtc.previous_actions": np.zeros((100, 14)),
        "rtc.weights": weights,
        "logged.prefix_base": np.zeros((100, 14)),
    }
    record_path = bundle / "request.npz"
    np.savez_compressed(record_path, **source)
    timing_meta = dict(
        source_fps=30,
        source_horizon=50,
        action_stride=1,
        output_chunk_length=100,
        includes_current=True,
        output_offsets_s=grid.tolist(),
    )
    record = {
        "session": "synthetic",
        "request_id": 3,
        "file": record_path.name,
        "input_sha256": runner.digest(record_path),
        "entry_s": 0.2,
        "next_switch_offset_s": 0.9,
    }
    index = {
        "contract_sha256": runner.digest(contract),
        "sessions": {"synthetic": {"extrinsics": {}, "gripper_open_m": 0.08}},
        "records": [record],
        "seeds": [42, 43],
        "limitations": [],
    }
    (bundle / "index.json").write_text(json.dumps(index))
    monkeypatch.setattr(runner, "HERE", tmp_path)
    monkeypatch.setattr(runner, "CHECKPOINT", checkpoint)
    calls = []
    identity_active = False

    @contextmanager
    def identity_context():
        nonlocal identity_active
        identity_active = True
        try:
            yield
        finally:
            identity_active = False

    def infer(observation, *, noise):
        assert identity_active is (mode == "identity")
        calls.append(
            (
                observation["observation.state"].copy(),
                noise.copy(),
                observation["rtc"]["weights"].copy(),
            )
        )
        # An input-consuming model must not change later variants' inputs.
        observation["observation.state"][:] = -99
        observation["rtc"]["previous_actions"][:] = 99
        return {"actions": np.zeros((100, 14))}

    def create_policy(assets, *, require_single_gpu, rtc_config):
        assert identity_active is (mode == "identity")
        assert require_single_gpu is True and rtc_config.maximum == 20
        return SimpleNamespace(infer=infer)

    def install(name, **attributes):
        module = ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)

    install("_bootstrap", setup=lambda **kwargs: None)
    install("egoverse_pi05", __path__=[])
    install(
        "egoverse_pi05.assets",
        load_inference_assets=lambda *args: SimpleNamespace(
            metadata={"action_timing": timing_meta}
        ),
    )
    install("egoverse_pi05.inference", create_policy=create_policy)
    install(
        "egoverse_pi05.rtc", RtcConfig=lambda maximum: SimpleNamespace(maximum=maximum)
    )
    install(
        "egoverse_pi05.rtc_diagnostics",
        identity_guidance_for_diagnostic=identity_context,
    )
    install(
        "egoverse_pi05.rtc_evaluation",
        BaseCameraAdapter=lambda *args: SimpleNamespace(
            camera_to_base_actions=lambda x: np.array(x, copy=True)
        ),
    )
    install(
        "egomimic.rldb.action_timing",
        ActionTiming=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    install(
        "egomimic.robot.piper_pico.policy_continuity",
        smooth_pose_actions=lambda actions, *args: actions.copy(),
    )
    if custom_run:
        variants = ["hold200_release300", "hold150_release300"]
        output_name = "paired_quintic_release"
        runner.run(mode, variants=variants, run_name=output_name)
    else:
        variants = ["original", "hard_only", "overlap300"]
        if mode == "vjp":
            variants.append("unguided")
        output_name = f"replay_{mode}"
        runner.run(mode)
    assert identity_active is False
    assert len(calls) == 2 * len(variants)
    for index, (state, noise, actual_weights) in enumerate(calls):
        np.testing.assert_array_equal(state, source["observation.state"])
        expected_seed = 42 if index < len(variants) else 43
        expected = (
            np.random.default_rng(expected_seed)
            .standard_normal((100, 32))
            .astype(np.float32)
        )
        np.testing.assert_array_equal(noise, expected)
        variant = variants[index % len(variants)]
        np.testing.assert_array_equal(
            actual_weights, runner.variant_weights(weights, grid, 0.2, variant)
        )
    again, prefix = runner.load_record(record_path)
    np.testing.assert_array_equal(
        again["observation.state"], source["observation.state"]
    )
    np.testing.assert_array_equal(again["rtc"]["previous_actions"], 0)
    np.testing.assert_array_equal(prefix, 0)
    output = tmp_path / output_name
    assert json.loads((output / "complete.json").read_text())["samples"] == len(calls)
    arguments = json.loads((output / "arguments.json").read_text())
    assert arguments["variants"] == variants and arguments["mode"] == mode
    assert len(list(output.glob("*.npz"))) == len(calls)
    if custom_run:
        assert not (tmp_path / f"replay_{mode}").exists()
