"""History is a strict portable inference contract, including RTC warmup."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pi05/src"))

from egoverse_pi05.assets import (  # noqa: E402
    load_inference_assets,
    token_config,
    validate_resume_contract,
)
from egoverse_pi05.inference import BimanualInferencePolicy, warmup_policy  # noqa: E402
from egoverse_pi05.rtc import RtcConfig  # noqa: E402

from egomimic.robot.piper_pico.pi05_data import canonical_hash  # noqa: E402
from egomimic.robot.piper_pico.state_history import make_history_contract  # noqa: E402
from tests.unit.test_pi05_jax_config import (  # noqa: E402
    portable_checkpoint as _portable_checkpoint,
)
from tests.unit.test_pi05_rtc_server import Baseline, request  # noqa: E402


@pytest.fixture
def portable_checkpoint(tmp_path):
    return _portable_checkpoint.__wrapped__(tmp_path)


def _enable_history(checkpoint, offsets):
    path = checkpoint / "assets/egoverse/contract.json"
    contract = json.loads(path.read_text())
    history = make_history_contract(offsets)
    contract["state_history"] = contract["manifest"]["state_history"] = history
    manifest = contract["manifest"]
    for record in manifest["episodes"]:
        descriptor = {"status": "available", "schema": "fixture"}
        record.update(
            feedback_history=descriptor,
            feedback_history_sha256=canonical_hash(descriptor),
            feedback_available_until_sha256="d" * 64,
            eligible_indices=list(range(record["total_frames"])),
        )
    manifest["data_identity_sha256"] = canonical_hash(
        [
            {
                key: record[key]
                for key in (
                    "metadata_sha256",
                    "numeric_sha256",
                    "total_frames",
                    "feedback_history_sha256",
                    "feedback_available_until_sha256",
                    "eligible_indices",
                )
            }
            for record in manifest["episodes"]
        ]
    )
    manifest["manifest_sha256"] = canonical_hash(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    )
    contract["training"]["max_token_len"] = 384
    contract["model_input"] = token_config(history, 384)
    path.write_text(json.dumps(contract))
    return path, contract


def test_legacy_checkpoint_keeps_200_tokens_and_no_history(portable_checkpoint):
    checkpoint, _ = portable_checkpoint
    path = checkpoint / "assets/egoverse/contract.json"
    value = json.loads(path.read_text())
    value.pop("model_input")
    value["training"].pop("max_token_len")
    path.write_text(json.dumps(value))
    assets = load_inference_assets(checkpoint)
    assert assets.metadata["model_input"]["max_token_len"] == 200
    assert assets.metadata["model_input"]["overflow"] == "upstream_truncate"
    assert "state_history" not in assets.metadata
    expected = json.loads(json.dumps(value))
    expected["model_input"] = token_config()
    expected["training"]["max_token_len"] = 200
    validate_resume_contract(value, expected)
    expected["model_input"] = token_config(max_token_len=256)
    expected["training"]["max_token_len"] = 256
    with pytest.raises(ValueError, match="Resume contract differs"):
        validate_resume_contract(value, expected)


@pytest.mark.parametrize("offsets", [[-0.025], [-0.11, -0.05, -0.025]])
def test_history_checkpoint_portable_and_bound_to_manifest(
    portable_checkpoint, offsets
):
    checkpoint, _ = portable_checkpoint
    path, contract = _enable_history(checkpoint, offsets)
    metadata = load_inference_assets(checkpoint).metadata
    assert metadata["state_history"]["offsets_s"] == offsets
    assert metadata["model_input"]["max_token_len"] == 384
    assert metadata["model_input"]["overflow"] == "error"
    assert metadata["action_dim"] == 14 and metadata["model_action_dim"] == 32
    for change in (
        {"state_history": make_history_contract([-0.01])},
        {"model_input": token_config(contract["state_history"], 512)},
        {"model_input": None},
    ):
        changed = {**contract, **change}
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="state_history|model_input"):
            load_inference_assets(checkpoint)


def test_history_resume_rejects_changed_offsets_and_token_budget(portable_checkpoint):
    checkpoint, _ = portable_checkpoint
    _, saved = _enable_history(checkpoint, [-0.07, -0.025])
    validate_resume_contract(saved, copy.deepcopy(saved))
    for field, replacement in (
        ("state_history", make_history_contract([-0.075, -0.025])),
        ("model_input", token_config(saved["state_history"], 512)),
    ):
        changed = {**saved, field: replacement}
        with pytest.raises(ValueError, match="Resume contract differs"):
            validate_resume_contract(saved, changed)


def test_history_rpc_baseline_rtc_shape_missing_and_warmup(portable_checkpoint):
    checkpoint, _ = portable_checkpoint
    _enable_history(checkpoint, [-0.07, -0.025])
    assets = load_inference_assets(checkpoint)
    baseline, rtc = Baseline(), Baseline()

    class Runner:
        def infer(self, obs, request, **kw):
            return rtc.infer(obs, **kw)

    policy = BimanualInferencePolicy(
        baseline, assets.metadata, rtc_runner=Runner(), rtc_config=RtcConfig()
    )
    warmup_policy(policy, assets)
    assert len(baseline.calls) == 1 and len(rtc.calls) == 2
    obs = baseline.calls[0][0]
    assert obs["observation.state_history"].shape == (2, 14)
    for sent, _ in rtc.calls:
        np.testing.assert_array_equal(
            sent["observation.state_history"], obs["observation.state_history"]
        )
    for history in (np.zeros((1, 14)), np.zeros((2, 12)), np.full((2, 14), np.nan)):
        with pytest.raises(ValueError, match="2 x 14 finite"):
            policy.infer({**obs, "observation.state_history": history})
    missing = copy.deepcopy(obs)
    missing.pop("observation.state_history")
    with pytest.raises(ValueError, match="exactly"):
        policy.infer(missing)
    assert policy.infer({**obs, "rtc": request()})["actions"].shape == (100, 14)


def test_real_openpi_history_preprocessing():
    interpreter = os.environ.get("EGOVERSE_TEST_PI05_PYTHON")
    if not interpreter:
        pytest.skip("Set EGOVERSE_TEST_PI05_PYTHON for actual OpenPI CPU transforms")
    command = [interpreter, str(ROOT / "tests/integration/check_pi05_state_history.py")]
    if upstream := os.environ.get("EGOVERSE_TEST_PI05_OPENPI_ROOT"):
        command.extend(["--openpi-root", upstream])
    if tokenizer := os.environ.get("EGOVERSE_TEST_PI05_TOKENIZER"):
        command.extend(["--tokenizer-path", tokenizer])
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
