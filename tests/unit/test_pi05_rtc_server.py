"""CPU-only protocol checks; optional JAX analytic checks use an existing env."""

import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pi05/src"))

from egoverse_pi05.inference import BimanualInferencePolicy, warmup_policy  # noqa: E402
from egoverse_pi05.rtc import (  # noqa: E402
    SCHEMA,
    RtcConfig,
    padded_mask,
    validate_request,
)


def request():
    return {
        "schema": SCHEMA,
        "previous_actions": np.zeros((100, 14), np.float32),
        "weights": np.ones(100, np.float32),
        "request_id": 12,
    }


def observation():
    return {
        "observation.state": np.zeros(14),
        "prompt": "Pick",
        "observation.images.front_1": np.zeros((3, 4, 3), np.uint8),
    }


class Baseline:
    def __init__(self):
        self.calls = []

    def infer(self, obs, **kwargs):
        self.calls.append((obs, kwargs))
        return {"actions": np.zeros((100, 14), np.float32)}

    def reset(self):
        pass


def policy(enabled=True):
    # Bimanual validation requires all three named cameras.
    from egoverse_pi05.assets import CAMERAS

    metadata = {"image_shapes": {c: (3, 4, 3) for c in CAMERAS}}
    baseline = Baseline()

    class Runner:
        def __init__(self):
            self.requests = []

        def infer(self, obs, value, **kwargs):
            self.requests.append(value)
            return {"actions": np.ones((100, 14), np.float32)}

    runner = Runner()
    p = BimanualInferencePolicy(
        baseline,
        metadata,
        rtc_runner=runner if enabled else None,
        rtc_config=RtcConfig() if enabled else None,
    )
    obs = {"observation.state": np.zeros(14), "prompt": "Pick"}
    obs.update(
        {
            f"observation.images.{c}": np.zeros(s, np.uint8)
            for c, s in metadata["image_shapes"].items()
        }
    )
    return p, baseline, runner, obs


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "other"},
        {"request_id": True},
        {"request_id": -1},
        {"request_id": 1.5},
        {"weights": np.zeros(99)},
        {"weights": np.full(100, np.nan)},
        {"weights": np.full(100, 1.01)},
        {"weights": np.full(100, -0.1)},
        {"previous_actions": np.zeros((100, 32))},
        {"previous_actions": np.full((100, 14), np.inf)},
        {"unexpected": 1},
    ],
)
def test_reject_bad_rtc_metadata(change):
    with pytest.raises(ValueError):
        validate_request(request() | change)


@pytest.mark.parametrize("value", [False, 0, -1, float("nan"), float("inf")])
def test_bad_guidance_rejected(value):
    with pytest.raises(ValueError):
        RtcConfig(value)


def test_mask_unused_model_dimensions_are_zero():
    weights = np.linspace(1, 0, 100, dtype=np.float32)
    mask = padded_mask(weights)
    assert mask.shape == (100, 32)
    np.testing.assert_array_equal(
        mask[:, :14], np.broadcast_to(weights[:, None], (100, 14))
    )
    assert np.count_nonzero(mask[:, 14:]) == 0


def test_baseline_unchanged_and_rtc_echoes_request():
    p, baseline, runner, obs = policy()
    assert "rtc" in p.metadata
    assert p.infer(obs)["actions"].sum() == 0
    assert len(baseline.calls) == 1 and not runner.requests
    sent = request()
    original = copy.deepcopy(sent)
    result = p.infer(dict(obs, rtc=sent))
    assert result["rtc"] == {"schema": SCHEMA, "request_id": 12}
    assert result["actions"].sum() == 1400
    assert len(baseline.calls) == 1
    np.testing.assert_array_equal(
        original["previous_actions"], sent["previous_actions"]
    )
    np.testing.assert_array_equal(original["weights"], sent["weights"])
    assert "rtc" not in obs


def test_rtc_requires_explicit_server_enable_and_rejects_null():
    p, baseline, _, obs = policy(False)
    assert "rtc" not in p.metadata
    with pytest.raises(ValueError, match="not enabled"):
        p.infer(dict(obs, rtc=request()))
    assert not baseline.calls
    p, baseline, runner, obs = policy()
    with pytest.raises(ValueError, match="schema"):
        p.infer(dict(obs, rtc=None))
    assert not baseline.calls and not runner.requests


def test_warmup_compiles_guided_and_exercises_zero_mask():
    p, baseline, runner, _ = policy()

    class Assets:
        metadata = {**p.metadata, "manifest": {"episodes": [{"prompt": "Pick"}]}}
        norm_stats = {"state": {"q01": [-1] * 14, "q99": [1] * 14}}

    warmup_policy(p, Assets())
    assert len(baseline.calls) == 1
    assert len(runner.requests) == 2
    assert np.all(runner.requests[0]["weights"] == 1)
    assert np.all(runner.requests[1]["weights"] == 0)


def test_analytic_jax_guidance_in_existing_environment():
    python = os.environ.get("EGOVERSE_TEST_PI05_PYTHON")
    if not python:
        pytest.skip("Set EGOVERSE_TEST_PI05_PYTHON to an existing JAX environment")
    result = subprocess.run(
        [python, str(ROOT / "pi05/scripts/check_pi05_rtc.py"), "--cpu-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RTC ANALYTIC CHECKS PASSED" in result.stdout
