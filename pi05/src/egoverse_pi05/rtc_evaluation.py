"""Offline, time-aligned handoff metrics using the production RTC conditioner.

All poses are SDK base-frame xyz metres / rotation vectors radians / gripper
metres. ``switch`` is the disagreement of two plans at the SAME wall-clock
instant; it is not a robot step. Step and second-difference metrics are sampled
at ``control_frequency`` and have units mm/sample and mm/sample**2 respectively,
not mm/s or mm/s**2. Angular increments use spatial relative rotations, avoiding
rotation-vector branch cuts. These metrics describe commands, not robot motion.
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_pico.policy_continuity import (
    ContinuousPlan,
    conditioned_plan,
    sample_plan,
)
from egomimic.robot.piper_pico.policy_streaming import ChunkPlan


class BaseCameraAdapter:
    """Geometry-only equivalent of PiperObservationAdapter's two pose methods.

    ``extrinsics[side]`` is T_base_camera. This small offline adapter avoids
    importing camera, Aria, torch or device dependencies in the JAX environment.
    It retains float64 precision and the production intrinsic-ZYX convention.
    Gripper feedback/commands outside the nominal interval are not clipped.
    """

    def __init__(self, extrinsics, gripper_open_m):
        self.gripper_open_m = float(gripper_open_m)
        if not np.isfinite(self.gripper_open_m) or self.gripper_open_m <= 0:
            raise ValueError("gripper_open_m must be finite and positive")
        self.extrinsics = {}
        for side in ("left", "right"):
            value = np.array(extrinsics[side], dtype=np.float64, copy=True)
            if (
                value.shape != (4, 4)
                or not np.isfinite(value).all()
                or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(
                    value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6, rtol=0
                )
                or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-6, rtol=0)
            ):
                raise ValueError(
                    f"{side} extrinsic must be a finite rigid T_base_camera"
                )
            self.extrinsics[side] = value

    def base_to_camera_state(self, state):
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("base_state must be finite shape (14,)")
        result = np.empty(14, dtype=np.float64)
        for side, offset in (("left", 0), ("right", 7)):
            row = state[offset : offset + 7]
            base_tcp = np.eye(4)
            base_tcp[:3, :3] = Rotation.from_rotvec(row[3:6]).as_matrix()
            base_tcp[:3, 3] = row[:3]
            camera_tcp = np.linalg.inv(self.extrinsics[side]) @ base_tcp
            result[offset : offset + 3] = camera_tcp[:3, 3]
            result[offset + 3 : offset + 6] = Rotation.from_matrix(
                camera_tcp[:3, :3]
            ).as_euler("ZYX")
            result[offset + 6] = row[6] / self.gripper_open_m
        return result

    def camera_to_base_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if (
            actions.ndim != 2
            or actions.shape[1] != 14
            or not len(actions)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("camera actions must be finite (M,14)")
        result = np.empty_like(actions)
        for side, offset in (("left", 0), ("right", 7)):
            camera_tcp = np.broadcast_to(np.eye(4), (len(actions), 4, 4)).copy()
            camera_tcp[:, :3, 3] = actions[:, offset : offset + 3]
            camera_tcp[:, :3, :3] = Rotation.from_euler(
                "ZYX", actions[:, offset + 3 : offset + 6]
            ).as_matrix()
            base_tcp = self.extrinsics[side][None] @ camera_tcp
            result[:, offset : offset + 3] = base_tcp[:, :3, 3]
            result[:, offset + 3 : offset + 6] = Rotation.from_matrix(
                base_tcp[:, :3, :3]
            ).as_rotvec()
            result[:, offset + 6] = actions[:, offset + 6] * self.gripper_open_m
        return result


def _errors(actual, expected, k):
    return (
        np.linalg.norm(actual[:, k : k + 3] - expected[:, k : k + 3], axis=1) * 1000,
        np.rad2deg(
            (
                Rotation.from_rotvec(expected[:, k + 3 : k + 6]).inv()
                * Rotation.from_rotvec(actual[:, k + 3 : k + 6])
            ).magnitude()
        ),
    )


def _motion_metrics(actions, k):
    xyz = actions[:, k : k + 3] * 1000
    rotations = Rotation.from_rotvec(actions[:, k + 3 : k + 6])
    # Spatial (base-coordinate) increments can be differenced without treating
    # an orientation's equivalent +/-pi representation as a physical jump.
    angular_steps = np.rad2deg((rotations[1:] * rotations[:-1].inv()).as_rotvec())
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    d2 = np.linalg.norm(np.diff(xyz, n=2, axis=0), axis=1)
    angular_d2 = np.linalg.norm(np.diff(angular_steps, axis=0), axis=1)
    return {
        "max_step_mm": float(steps.max()),
        "step_p95_mm": float(np.percentile(steps, 95)),
        "d2_p95_mm": float(np.percentile(d2, 95)),
        "d2_max_mm": float(d2.max()),
        "max_step_deg": float(np.linalg.norm(angular_steps, axis=1).max()),
        "d2_p95_deg": float(np.percentile(angular_d2, 95)),
        "max_gripper_step_mm": float(np.abs(np.diff(actions[:, k + 6])).max() * 1000),
    }


def evaluate_handoff(
    old_plan,
    new_base_actions,
    timing,
    new_anchor,
    activation,
    *,
    smoothing_s=0.150,
    transition_s=0.300,
    control_frequency=30,
    reference_base_actions=None,
    current_base_state=None,
):
    """Compare a raw/SG/transitioned candidate against the committed old plan.

    The reference, when supplied, uses the same shape, time origin and timing as
    ``new_base_actions``. Its metrics cover identical post-activation control
    samples for all candidates. This distinguishes conservative smooth commands
    from accurate moving commands. ``old_plan`` may itself be ContinuousPlan.

    The transition trace includes two old control samples before activation and
    at least two samples after the transition ends. All compared samples must
    be within BOTH horizons: production's endpoint clamp is never counted as
    measured overlap. A short/invalid overlap raises ValueError. ``metrics`` and
    ``sampling`` are JSON-ready; numpy arrays in ``traces`` can be saved to NPZ.
    No input or previously committed plan is modified.
    """
    scalars = np.asarray(
        [new_anchor, activation, smoothing_s, transition_s, control_frequency],
        dtype=float,
    )
    if not np.isfinite(scalars).all() or control_frequency <= 0 or transition_s < 0:
        raise ValueError(
            "Handoff times/durations must be finite and frequency positive"
        )
    new_anchor, activation = float(new_anchor), float(activation)
    interval = 1.0 / float(control_frequency)
    raw = ChunkPlan(new_base_actions, timing, new_anchor)
    smoothed = conditioned_plan(new_base_actions, timing, new_anchor, smoothing_s)
    overlap_start = max(new_anchor, float(getattr(old_plan, "anchor", -math.inf)))
    overlap_end = min(raw.end_walltime, float(old_plan.end_walltime))
    # Include at least two complete intervals after the end, also when the end
    # falls between control ticks. Do not resample onto a different grid.
    after_steps = math.ceil(float(transition_s) / interval) + 2
    relative_steps = np.arange(-2, after_steps + 1)
    times = activation + relative_steps * interval
    if times[0] < overlap_start - 1e-9 or times[-1] > overlap_end + 1e-9:
        raise ValueError(
            "Insufficient valid old/new overlap for two pre-activation and "
            "two post-transition control intervals"
        )
    if transition_s:
        executed = ContinuousPlan(smoothed, old_plan, activation, transition_s)
    else:
        executed = smoothed
    old_trace = sample_plan(old_plan, times)
    raw_trace = sample_plan(raw, times)
    smoothed_trace = sample_plan(smoothed, times)
    executed_trace = sample_plan(executed, times)
    # The candidate is not executed before activation, even without blending.
    executed_trace[relative_steps < 0] = old_trace[relative_steps < 0]
    raw_handoff = raw_trace.copy()
    smoothed_handoff = smoothed_trace.copy()
    raw_handoff[relative_steps < 0] = old_trace[relative_steps < 0]
    smoothed_handoff[relative_steps < 0] = old_trace[relative_steps < 0]

    model_times = raw.anchor + raw.offsets
    prefix_times = np.unique(
        np.r_[
            overlap_start,
            model_times[(model_times >= overlap_start) & (model_times <= activation)],
            activation,
        ]
    )
    old_prefix = sample_plan(old_plan, prefix_times)
    raw_prefix = sample_plan(raw, prefix_times)
    smoothed_prefix = sample_plan(smoothed, prefix_times)
    internal_times = (
        raw.anchor + np.arange(math.floor(raw.span / interval + 1e-9) + 1) * interval
    )
    if len(internal_times) < 3:
        raise ValueError("Horizon must contain at least three control samples")
    raw_internal = sample_plan(raw, internal_times)
    smoothed_internal = sample_plan(smoothed, internal_times)
    post = relative_steps >= 0
    reference = (
        None
        if reference_base_actions is None
        else sample_plan(
            ChunkPlan(reference_base_actions, timing, new_anchor), times[post]
        )
    )
    if current_base_state is not None:
        current_base_state = np.asarray(current_base_state, dtype=float)
        if (
            current_base_state.shape != (14,)
            or not np.isfinite(current_base_state).all()
        ):
            raise ValueError("Current measured state must be finite shape (14,)")

    metrics = {}
    at_activation = int(np.flatnonzero(relative_steps == 0)[0])
    for side, k in (("left", 0), ("right", 7)):
        values = {}
        for name, trace, prefix in (
            ("raw", raw_trace, raw_prefix),
            ("smoothed", smoothed_trace, smoothed_prefix),
            ("executed", executed_trace, None),
        ):
            pos, angle = _errors(
                trace[at_activation : at_activation + 1],
                old_trace[at_activation : at_activation + 1],
                k,
            )
            values[f"{name}_switch_mm"] = float(pos[0])
            values[f"{name}_switch_deg"] = float(angle[0])
            if prefix is not None:
                pos, angle = _errors(prefix, old_prefix, k)
                values[f"{name}_prefix_max_mm"] = float(pos.max())
                values[f"{name}_prefix_max_deg"] = float(angle.max())
            if reference is not None:
                pos, angle = _errors(trace[post], reference, k)
                values[f"{name}_target_position_rmse_mm"] = float(
                    np.sqrt(np.mean(pos**2))
                )
                values[f"{name}_target_rotation_rmse_deg"] = float(
                    np.sqrt(np.mean(angle**2))
                )
                values[f"{name}_target_gripper_rmse_mm"] = float(
                    np.sqrt(np.mean((trace[post, k + 6] - reference[:, k + 6]) ** 2))
                    * 1000
                )
        for name, trace in (
            ("raw_handoff", raw_handoff),
            ("smoothed_handoff", smoothed_handoff),
            ("transition", executed_trace),
            ("raw_internal", raw_internal),
            ("smoothed_internal", smoothed_internal),
        ):
            values.update(
                {
                    f"{name}_{key}": value
                    for key, value in _motion_metrics(trace, k).items()
                }
            )
        if current_base_state is not None:
            measured = np.broadcast_to(current_base_state, raw_internal.shape).copy()
            pos, angle = _errors(raw_internal, measured, k)
            values["raw_max_from_current_mm"] = float(pos.max())
            values["raw_max_from_current_deg"] = float(angle.max())
        metrics[side] = values
    traces = {
        "times_s": times,
        "old_base_actions": old_trace,
        "raw_base_actions": raw_trace,
        "smoothed_base_actions": smoothed_trace,
        "executed_base_actions": executed_trace,
        "prefix_times_s": prefix_times,
        "old_prefix_base_actions": old_prefix,
        "raw_prefix_base_actions": raw_prefix,
        "smoothed_prefix_base_actions": smoothed_prefix,
        "internal_times_s": internal_times,
        "raw_internal_base_actions": raw_internal,
        "smoothed_internal_base_actions": smoothed_internal,
    }
    if reference is not None:
        traces["reference_times_s"] = times[post]
        traces["reference_base_actions"] = reference
    return {
        "metrics": metrics,
        "sampling": {
            "control_frequency_hz": float(control_frequency),
            "new_anchor_s": new_anchor,
            "activation_s": activation,
            "smoothing_s": float(smoothing_s),
            "transition_s": float(transition_s),
            "window_start_s": float(times[0]),
            "window_end_s": float(times[-1]),
            "control_samples": len(times),
            "prefix_samples": len(prefix_times),
            "internal_samples": len(internal_times),
            "second_difference_units": "mm per control sample squared; deg per control sample squared",
            "switch_definition": "old/new command disagreement at the same activation time",
            "target_error_window": "post-activation control samples through two intervals after transition",
        },
        "traces": traces,
    }
