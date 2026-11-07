"""Offline deployment evaluation: independent labels, metrics, no device path."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fixtures.piper_pico_training import write_recording

from egomimic.rldb.action_timing import ActionTiming
from egomimic.rldb.embodiment.piper import Piper
from egomimic.rldb.zarr.piper_dataset import PiperDataset
from egomimic.robot.piper_pico import live_policy as live
from egomimic.robot.piper_pico import offline_policy_eval as evaluation
from egomimic.robot.piper_pico.policy_inference import (
    ACTION,
    CAMERA_KEYS,
    DOMAIN,
    EMBODIMENT,
    STATE,
    PiperObservationAdapter,
    PiperPolicy,
)
from egomimic.robot.piper_pico.policy_rollout import OfflineReplayClock
from egomimic.scripts.replay_piper_policy import RecordedObservationProvider


def cli_args(tmp_path, *extra):
    return [
        "--checkpoint",
        "model.ckpt",
        "--calibration",
        "calibration.json",
        "--training-data",
        "data",
        "--camera-config",
        "rig.json",
        "--output-dir",
        str(tmp_path / "evaluation"),
        "--offline-eval",
        *extra,
    ]


def test_metric_units_and_rotation_wrap():
    target, predicted = np.zeros((2, 14)), np.zeros((2, 14))
    predicted[:, :3] = [0.003, 0.004, 0]
    target[:, 5] = np.deg2rad(179)
    predicted[:, 5] = np.deg2rad(-179)
    predicted[:, 6] = 0.012
    result = evaluation.physical_errors(predicted, target)
    np.testing.assert_allclose(result["left"]["position_mm"], 5)
    np.testing.assert_allclose(result["left"]["rotation_deg"], 2)
    np.testing.assert_allclose(result["left"]["gripper_mm"], 12)
    for values in result["right"].values():
        np.testing.assert_array_equal(values, 0)
    predicted[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluation.physical_errors(predicted, target)


@pytest.mark.parametrize(
    "length,horizon,count,expected",
    [
        (30, 30, 8, [0]),
        (29, 30, 8, []),
        (32, 30, 8, [0, 1, 2]),
        (100, 30, 3, [0, 35, 70]),
    ],
)
def test_sampling_excludes_padded_targets(length, horizon, count, expected):
    assert evaluation.sample_indices(length, horizon, count) == expected


@pytest.mark.parametrize(
    "timing", [ActionTiming(30, 9, 2, 11), ActionTiming(30, 10, 1, 100)]
)
def test_actual_loader_nonidentity_geometry_and_independent_oracle(tmp_path, timing):
    path = write_recording(tmp_path, "piper", "nonidentity")
    clock = OfflineReplayClock()
    provider = RecordedObservationProvider(
        path, clock, allow_synthetic_calibration=True
    )
    adapter = PiperObservationAdapter(
        provider.metadata, allow_synthetic_calibration=True
    )
    loader = PiperDataset(
        path,
        Piper.get_keymap(timing=timing),
        Piper.get_transform_list(timing=timing),
        allow_synthetic_calibration=True,
    )
    policy = SimpleNamespace(adapter=adapter)
    before = clock.wall_time()
    for index in (0, 24 - timing.source_horizon):
        frame = provider.read_frame(index)
        sample = loader[index]
        assert frame.image_space == "processed"
        assert frame.timestamp == provider.timestamps[index]
        assert provider.next_index == 0 and clock.wall_time() == before
        evaluation.check_input_parity(policy, frame, sample)
        oracle = evaluation.direct_command_targets(
            provider.store, index, timing, 0.08, timing.output_offsets_s
        )
        target = adapter.camera_to_base_actions(sample[ACTION].numpy())
        for fields in evaluation.physical_errors(target, oracle).values():
            for values in fields.values():
                assert values.max() < 0.002
        # Real observation and command differ in this fixture: scoring cannot
        # silently substitute future feedback or feed command labels as state.
        assert not np.allclose(frame.base_state, oracle[0])
        bad = replace(frame, base_state=oracle[0])
        with pytest.raises(ValueError, match="state"):
            evaluation.check_input_parity(policy, bad, sample)
        bad_images = dict(frame.images)
        bad_images["front_1"] = 255 - bad_images["front_1"]
        with pytest.raises(ValueError, match="front_1"):
            evaluation.check_input_parity(
                policy, replace(frame, images=bad_images), sample
            )
    for invalid in (-1, 24, True, 1.5):
        with pytest.raises(IndexError):
            provider.read_frame(invalid)
    assert provider.latest().timestamp == before
    assert provider.next_index == 1


def test_main_offline_dispatch_never_starts_live_session(tmp_path, monkeypatch):
    monkeypatch.setattr(
        live, "run_session", lambda *a: pytest.fail("opened live devices")
    )
    seen = []

    def offline(args):
        seen.append(args)
        return 17

    monkeypatch.setattr(evaluation, "run_evaluation", offline)
    assert live.main(cli_args(tmp_path)) == 17
    assert seen[0].device == "cpu"


@pytest.mark.parametrize(
    "extra", [("--execute",), ("--home-on-start", "true"), ("--max-chunks", "1")]
)
def test_offline_rejects_motion_flags_before_model_or_devices(
    tmp_path, monkeypatch, extra
):
    monkeypatch.setattr(
        evaluation, "load_policy_contract", lambda *a: pytest.fail("loaded model")
    )
    assert live.main(cli_args(tmp_path, *extra)) == 1
    summary = json.loads((tmp_path / "evaluation" / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["hardware_opened"] is False


def test_end_to_end_report_uses_live_policy_without_target_input(tmp_path, monkeypatch):
    path = write_recording(tmp_path, "piper", "evaluation_source")
    provider = RecordedObservationProvider(
        path, OfflineReplayClock(), allow_synthetic_calibration=True
    )
    timing = ActionTiming(30, 9, 2, 11)
    adapter = PiperObservationAdapter(
        provider.metadata, allow_synthetic_calibration=True
    )
    calls = []

    class ObservationOnlyModel:
        def predict_from_observations(self, batches, *, num_inference_steps):
            batch = batches[EMBODIMENT]
            assert set(batch) == {STATE, *CAMERA_KEYS.values()}
            assert ACTION not in batch
            calls.append(num_inference_steps)
            # A deterministic hold-current policy must match the baseline, with
            # no knowledge of the recorded command labels.
            action = batch[STATE][:, None].repeat(1, timing.output_chunk_length, 1)
            return {f"{DOMAIN}_{ACTION}": action}

    policy = PiperPolicy(ObservationOnlyModel(), adapter, timing)
    metadata_path = path / ("zarr.json" if (path / "zarr.json").exists() else ".zattrs")
    contract = {
        "audit": {
            "episodes": [
                {"episode": str(path), "metadata": {"path": str(metadata_path)}}
            ]
        }
    }
    monkeypatch.setattr(
        evaluation, "load_policy_contract", lambda *a: (contract, policy, None)
    )
    real_provider, real_loader = (
        evaluation.RecordedObservationProvider,
        evaluation.PiperDataset,
    )
    monkeypatch.setattr(
        evaluation,
        "RecordedObservationProvider",
        lambda *a: real_provider(*a, allow_synthetic_calibration=True),
    )
    monkeypatch.setattr(
        evaluation,
        "PiperDataset",
        lambda *a: real_loader(*a, allow_synthetic_calibration=True),
    )
    monkeypatch.setattr(
        live, "run_session", lambda *a: pytest.fail("opened live devices")
    )
    assert (
        live.main(
            cli_args(
                tmp_path,
                "--execute-steps",
                "5",
                "--eval-samples-per-episode",
                "2",
                "--eval-plots",
                "1",
            )
        )
        == 0
    )
    output = tmp_path / "evaluation"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "completed" and summary["pipeline_checks"] == "passed"
    assert summary["sample_count"] == 2 and summary["command_prefix_length"] == 5
    assert summary["prediction_parity_samples"] == 2 and len(calls) == 4
    assert summary["prediction_parity_max_abs"] < 2e-6
    for side in evaluation.SIDES:
        for unit in evaluation.UNITS:
            value = summary["metrics"]["raw_model"][side][unit]["mean"]
            baseline = summary["metrics"]["hold_current_model"][side][unit]["mean"]
            assert value == pytest.approx(baseline, abs=0.002)
    assert (output / "report.md").exists()
    assert (output / "error_by_horizon.png").stat().st_size > 1000
    assert len(list((output / "plots").glob("*.png"))) == 1
    samples = list((output / "samples").glob("*.npz"))
    assert len(samples) == 2
    with np.load(samples[0]) as saved:
        assert saved["predicted_camera"].shape == (11, 14)
        assert saved["command_base"].shape == (5, 14)
        assert saved["command_offsets_s"][-1] == pytest.approx(4 / 30)


def test_existing_or_source_output_cannot_be_overwritten(tmp_path):
    args = live.build_parser().parse_args(cli_args(tmp_path))
    with pytest.raises(FileExistsError):
        evaluation.EvaluationLog(tmp_path, args)
    with pytest.raises(ValueError, match="Zarr"):
        evaluation.EvaluationLog(tmp_path / "source.zarr" / "eval", args)
