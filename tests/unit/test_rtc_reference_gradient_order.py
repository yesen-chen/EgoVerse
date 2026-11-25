"""CPU proof of the Jacobian distinction, independent of learned weights."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


def reference_order(x, slope, t, previous, mask):
    # Preserve the exact ordering in local LeRobot RTCProcessor.denoise_step:
    # clone/detach -> denoiser -> requires_grad_(True) -> denoised estimate.
    x = x.clone().detach()
    v = slope * x + 0.1
    x.requires_grad_(True)
    clean = x - t * v
    error = (previous - clean) * mask
    correction = torch.autograd.grad(clean, x, error.detach())[0]
    return error.detach(), correction.detach()


@pytest.mark.parametrize("t,slope", [(0.5, 2.0), (0.5, 1.0), (0.1, 2.0)])
def test_late_requires_grad_removes_the_denoiser_input_jacobian(t, slope):
    x = torch.tensor([0.3])
    # Network parameters having gradients does not restore the missing input
    # dependency after the forward pass has already completed.
    coefficient = torch.tensor(slope, requires_grad=True)
    previous = torch.tensor([0.7])
    mask = torch.ones_like(x)
    error, correction = reference_order(x, coefficient, t, previous, mask)
    torch.testing.assert_close(correction, error)
    y = x.clone().requires_grad_(True)
    clean = y - t * (coefficient * y + 0.1)
    full = torch.autograd.grad(clean, y, (previous - clean).detach())[0]
    torch.testing.assert_close(full, (1 - t * slope) * error)


def test_diagnostic_helper_uses_mask_once_and_restores_production(monkeypatch):
    # A NumPy fake makes this contract test independent of the optional JAX
    # environment; the production scale itself is covered by RTC unit tests.
    import sys
    from types import ModuleType

    package = ModuleType("egoverse_pi05")
    rtc = ModuleType("egoverse_pi05.rtc")
    original = object()
    rtc.guided_velocity = original
    rtc.guidance_scale = lambda t, maximum: min(
        maximum, (t * t + (1 - t) ** 2) / (t * (1 - t))
    )
    package.rtc = rtc
    monkeypatch.setitem(sys.modules, "egoverse_pi05", package)
    monkeypatch.setitem(sys.modules, "egoverse_pi05.rtc", rtc)
    path = (
        Path(__file__).resolve().parents[2]
        / "pi05/src/egoverse_pi05/rtc_diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location("rtc_identity_probe_tested", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    x = np.array([0.3, 0.3, 0.3])
    previous = np.array([0.7, 0.7, 0.7])
    mask = np.array([1.0, 0.5, 0.0])

    def v(value, t):
        return 2 * value + 0.1

    expected = v(x, 0.5) - 2 * (previous - (x - 0.5 * v(x, 0.5))) * mask
    with pytest.raises(RuntimeError, match="probe ended"):
        with helper.identity_guidance_for_diagnostic():
            assert rtc.guided_velocity is helper.identity_guided_velocity
            np.testing.assert_allclose(
                rtc.guided_velocity(v, x, 0.5, previous, mask, 20), expected
            )
            raise RuntimeError("probe ended")
    assert rtc.guided_velocity is original
