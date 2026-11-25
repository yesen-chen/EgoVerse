"""Diagnostic-only RTC Jacobian ablation; production full VJP is unchanged.

Use a fresh process and keep the context open across construction AND every
inference call: JAX traces lazily and compiled samplers retain the traced rule.
This does not hard-clamp a prefix and is not a deployment recommendation.
"""

from contextlib import contextmanager


def identity_guided_velocity(velocity, x, t, previous, mask, maximum=5.0):
    """Ablate only the denoised Jacobian; preserve mask, sign and time scale."""
    from egoverse_pi05.rtc import guidance_scale

    v = velocity(x, t)
    clean = x - t * v
    error = (previous - clean) * mask
    return v - guidance_scale(t, maximum) * error


@contextmanager
def identity_guidance_for_diagnostic():
    """Temporary rule for an entire fresh-process diagnostic policy lifetime."""
    from egoverse_pi05 import rtc

    original = rtc.guided_velocity
    rtc.guided_velocity = identity_guided_velocity
    try:
        yield
    finally:
        rtc.guided_velocity = original
