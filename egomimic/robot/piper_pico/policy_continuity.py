"""Execution-side trajectory conditioning, separate from model-side RTC.

The complete predicted horizon is available, so within-chunk denoising uses a
centred window instead of a causal EMA with accumulated tracking lag. A quintic
transition joins two trajectories at the SAME absolute time. It retains the
old value/velocity/acceleration at entry and the new ones at exit wherever the
underlying interpolants are differentiable. No claim is made about task success
or the robot's physical tracking. Gripper commands are never filtered or blended.
"""

import math

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_pico.policy_streaming import ChunkPlan


def smooth_pose_actions(actions, timing, window_s):
    """Denoise xyz and SO(3) quaternion components; retain all 100 timestamps."""
    window_s = float(window_s)
    if not math.isfinite(window_s) or window_s < 0:
        raise ValueError("RTC smoothing duration must be finite and nonnegative")
    result = np.array(actions, dtype=np.float64, copy=True)
    if (
        result.shape != (timing.output_chunk_length, 14)
        or not np.isfinite(result).all()
    ):
        raise ValueError("Trajectory must contain finite checkpoint-shaped actions")
    if window_s == 0:
        return result
    offsets = np.asarray(timing.output_offsets_s)
    interval = float(offsets[1] - offsets[0])
    if not np.allclose(np.diff(offsets), interval, rtol=1e-6, atol=1e-9):
        raise ValueError("RTC smoothing requires a uniformly sampled model horizon")
    if window_s >= timing.effective_span_s:
        raise ValueError("RTC smoothing window must be shorter than the horizon")
    # Approximately requested duration, rounded to an odd model-grid window.
    window = max(5, int(round(window_s / interval)) + 1)
    window += 1 - window % 2
    window = min(window, len(result) if len(result) % 2 else len(result) - 1)
    if window < 5:
        raise ValueError("RTC smoothing needs at least five model points")
    for k in (0, 7):
        result[:, k : k + 3] = savgol_filter(result[:, k : k + 3], window, 3, axis=0)
        quats = Rotation.from_rotvec(result[:, k + 3 : k + 6]).as_quat()
        # Quaternion signs are equivalent physically, but must agree before a
        # component filter. This also handles +/-pi rotation-vector branches.
        signs = np.where(np.sum(quats[1:] * quats[:-1], axis=1) < 0, -1, 1)
        quats[1:] *= np.cumprod(signs)[:, None]
        quats = savgol_filter(quats, window, 3, axis=0)
        norms = np.linalg.norm(quats, axis=1)
        if np.any(norms < 1e-8):
            raise ValueError("RTC rotation smoothing produced a degenerate quaternion")
        result[:, k + 3 : k + 6] = Rotation.from_quat(
            quats / norms[:, None]
        ).as_rotvec()
    return result


class ConstantPosePlan:
    """Held measured pose used only to enter the first plan after a resume."""

    def __init__(self, pose, end_walltime):
        self.pose = np.array(pose, dtype=np.float64, copy=True)
        self.end_walltime = float(end_walltime)
        if self.pose.shape != (14,) or not np.isfinite(self.pose).all():
            raise ValueError("Held pose must be finite with shape (14,)")
        if not math.isfinite(self.end_walltime):
            raise ValueError("Held pose horizon must be finite")

    def sample(self, walltime):
        return self.pose.copy()


class ContinuousPlan:
    """Quintic pose-only transition, followed by the new time-aligned plan.

    Instances are immutable after construction and may be sampled by the
    control thread and the RTC prefix worker concurrently. The next prefix must
    sample THIS executed plan, including its transition, not the raw prediction.
    """

    def __init__(self, current, previous, start_walltime, duration_s):
        duration = float(duration_s)
        start = float(start_walltime)
        if not np.isfinite([duration, start]).all() or duration <= 0:
            raise ValueError("RTC transition duration must be finite and positive")
        end = start + duration
        if start < current.anchor or start < getattr(previous, "anchor", -math.inf):
            raise ValueError("RTC transition starts before valid old/new plan overlap")
        if end >= min(previous.end_walltime, current.end_walltime):
            raise ValueError("RTC transition exceeds valid old/new plan overlap")
        # No unbounded chain: completed older transitions are irrelevant for
        # future sampling. Never mutate a plan shared with the inference worker.
        while isinstance(previous, ContinuousPlan) and start >= previous.transition_end:
            previous = previous.current
        self.current, self.previous = current, previous
        self.transition_start, self.transition_end = start, end
        self.duration_s = duration
        self.anchor, self.span = current.anchor, current.span
        self.offsets, self.actions = current.offsets, current.actions

    @property
    def end_walltime(self):
        return self.current.end_walltime

    def offset(self, walltime):
        return float(walltime) - self.anchor

    def sample(self, walltime):
        new = self.current.sample(walltime)
        if walltime >= self.transition_end:
            return new
        old = self.previous.sample(walltime)
        u = float(np.clip((walltime - self.transition_start) / self.duration_s, 0, 1))
        weight = u**3 * (10 + u * (-15 + 6 * u))
        for k in (0, 7):
            new[k : k + 3] = old[k : k + 3] + weight * (new[k : k + 3] - old[k : k + 3])
            r0 = Rotation.from_rotvec(old[k + 3 : k + 6])
            delta = (r0.inv() * Rotation.from_rotvec(new[k + 3 : k + 6])).as_rotvec()
            new[k + 3 : k + 6] = (r0 * Rotation.from_rotvec(weight * delta)).as_rotvec()
        return new


def conditioned_plan(actions, timing, anchor, smoothing_s):
    return ChunkPlan(smooth_pose_actions(actions, timing, smoothing_s), timing, anchor)


def sample_plan(plan, times):
    """Vectorized worker-side audit/prefix sampling; no per-point Python SLERP."""
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("Plan sample times must be a finite vector")
    if not len(times):
        return np.empty((0, 14), dtype=np.float64)
    if isinstance(plan, ConstantPosePlan):
        return np.repeat(plan.pose[None], len(times), axis=0)
    if isinstance(plan, ContinuousPlan):
        new = sample_plan(plan.current, times)
        active = times < plan.transition_end
        if np.any(active):
            old = sample_plan(plan.previous, times[active])
            u = np.clip((times[active] - plan.transition_start) / plan.duration_s, 0, 1)
            w = (u**3 * (10 + u * (-15 + 6 * u)))[:, None]
            for k in (0, 7):
                new[active, k : k + 3] = old[:, k : k + 3] + w * (
                    new[active, k : k + 3] - old[:, k : k + 3]
                )
                r0 = Rotation.from_rotvec(old[:, k + 3 : k + 6])
                delta = (
                    r0.inv() * Rotation.from_rotvec(new[active, k + 3 : k + 6])
                ).as_rotvec()
                new[active, k + 3 : k + 6] = (
                    r0 * Rotation.from_rotvec(w * delta)
                ).as_rotvec()
        return new
    if isinstance(plan, ChunkPlan):
        offsets = np.clip(times - plan.anchor, 0, plan.span)
        out = np.empty((len(times), 14))
        for k, slerp in zip((0, 7), plan._slerps):
            for col in (k, k + 1, k + 2, k + 6):
                out[:, col] = np.interp(offsets, plan.offsets, plan.actions[:, col])
            out[:, k + 3 : k + 6] = slerp(offsets).as_rotvec()
        return out
    return np.array([plan.sample(t) for t in times])


def transition_disagreement(current, previous, start, duration):
    """Inspect the entire transition, so a large error cannot hide after entry."""
    # Both grids' breakpoints plus control-independent 100 Hz checks. This is
    # a diagnostic acceptance sample, not a continuous-time collision bound.
    end = start + duration
    times = [np.linspace(start, end, max(2, math.ceil(duration / 0.01) + 1))]
    for plan in (current, previous):
        if hasattr(plan, "offsets"):
            grid = plan.anchor + plan.offsets
            times.append(grid[(grid >= start) & (grid <= end)])
    times = np.unique(np.concatenate(times))
    old = sample_plan(previous, times)
    new = sample_plan(current, times)
    return {
        side: {
            "max_position_m": float(
                np.linalg.norm(new[:, k : k + 3] - old[:, k : k + 3], axis=1).max()
            ),
            "max_rotation_rad": float(
                (
                    Rotation.from_rotvec(old[:, k + 3 : k + 6]).inv()
                    * Rotation.from_rotvec(new[:, k + 3 : k + 6])
                )
                .magnitude()
                .max()
            ),
        }
        for side, k in (("left", 0), ("right", 7))
    }
