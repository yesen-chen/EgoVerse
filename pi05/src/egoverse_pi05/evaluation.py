"""Offline motion metrics in base-frame metres/rotation vectors; no devices."""

import numpy as np
from scipy.spatial.transform import Rotation


def describe(values):
    x = np.asarray(values, dtype=float).reshape(-1)
    if not len(x):
        return None
    if not np.isfinite(x).all():
        raise ValueError("Non-finite metric")
    return dict(
        count=len(x),
        mean=float(x.mean()),
        median=float(np.median(x)),
        p95=float(np.percentile(x, 95)),
        max=float(x.max()),
    )


def trajectory_metrics(predicted, target, state, side):
    """Per-point errors and per-window drift, without crossing chunk boundaries."""
    k = side * 7
    p, t = np.asarray(predicted)[:, k : k + 7], np.asarray(target)[:, k : k + 7]
    s = np.asarray(state)[k : k + 7]
    if p.shape != t.shape or p.ndim != 2 or len(p) < 3:
        raise ValueError("Matching trajectories with at least three points required")
    if not all(np.isfinite(v).all() for v in (p, t, s)):
        raise ValueError("Non-finite trajectory")
    rot = Rotation.from_rotvec(p[:, 3:6])
    target_rot = Rotation.from_rotvec(t[:, 3:6])
    steps = np.diff(p[:, :3], axis=0) * 1000
    lengths = np.linalg.norm(steps, axis=1)
    eligible = (lengths[:-1] > 0.25) & (lengths[1:] > 0.25)
    reversal = (np.sum(steps[:-1] * steps[1:], axis=1) < 0) & eligible
    from_state = np.linalg.norm(p[:, :3] - s[:3], axis=1) * 1000
    rot_state = np.rad2deg((Rotation.from_rotvec(s[3:6]).inv() * rot).magnitude())
    return {
        "position_error_mm": np.linalg.norm(p[:, :3] - t[:, :3], axis=1) * 1000,
        "rotation_error_deg": np.rad2deg((target_rot.inv() * rot).magnitude()),
        "gripper_error_mm": np.abs(p[:, 6] - t[:, 6]) * 1000,
        "step_mm": lengths,
        "step_deg": np.rad2deg((rot[:-1].inv() * rot[1:]).magnitude()),
        "second_difference_mm": np.linalg.norm(np.diff(steps, axis=0), axis=1),
        "gripper_step_mm": np.abs(np.diff(p[:, 6])) * 1000,
        "max_from_current_mm": [from_state.max()],
        "first_from_current_mm": [from_state[0]],
        "end_from_current_mm": [from_state[-1]],
        "max_from_first_mm": [np.linalg.norm(p[:, :3] - p[0, :3], axis=1).max() * 1000],
        "max_rotation_from_current_deg": [rot_state.max()],
        "max_gripper_from_current_mm": [np.abs(p[:, 6] - s[6]).max() * 1000],
        "reversals": int(reversal.sum()),
        "eligible": int(eligible.sum()),
    }


def aggregate_metrics(items):
    if not items:
        return {"windows": 0}
    result = {
        key: describe(np.concatenate([np.atleast_1d(m[key]) for m in items]))
        for key in items[0]
        if key not in ("reversals", "eligible")
    }
    result.update(
        windows=len(items),
        reversals=sum(m["reversals"] for m in items),
        eligible=sum(m["eligible"] for m in items),
    )
    result["reversal_fraction"] = (
        result["reversals"] / result["eligible"] if result["eligible"] else None
    )
    return result


def static_window(
    arrays,
    side,
    start,
    horizon,
    open_m,
    position_mm=2.0,
    rotation_deg=1.0,
    gripper_mm=1.0,
):
    """Both measured and commanded motion must be small in a full native window.

    Also require initial feedback/command agreement within 5 mm / 2 degrees /
    2 mm width, so a stopped target still being approached is not called holding.
    Thresholds are screening definitions, not model acceptance criteria.
    """
    result = {"complete": start + horizon <= len(arrays[f"{side}.obs_ee_pose"])}
    first = {}
    for source in ("cmd", "obs"):
        pose = arrays[f"{side}.{source}_ee_pose"][start : start + horizon]
        grip = arrays[f"{side}.{source}_gripper"][start : start + horizon, 0] * open_m
        rot = Rotation.from_quat(pose[:, [4, 5, 6, 3]])
        first[source] = (pose[0, :3], rot[0], grip[0])
        result[source] = {
            "max_position_from_first_mm": float(
                np.linalg.norm(pose[:, :3] - pose[0, :3], axis=1).max() * 1000
            ),
            "max_rotation_from_first_deg": float(
                np.rad2deg((rot[0].inv() * rot).magnitude()).max()
            ),
            "max_gripper_from_first_mm": float(np.abs(grip - grip[0]).max() * 1000),
        }
    c, o = first["cmd"], first["obs"]
    result["initial_tracking_gap"] = {
        "position_mm": float(np.linalg.norm(c[0] - o[0]) * 1000),
        "rotation_deg": float(np.rad2deg((c[1].inv() * o[1]).magnitude())),
        "gripper_mm": float(abs(c[2] - o[2]) * 1000),
    }
    result["static"] = bool(
        result["complete"]
        and all(
            all(
                v <= lim
                for v, lim in zip(
                    result[src].values(), (position_mm, rotation_deg, gripper_mm)
                )
            )
            for src in ("cmd", "obs")
        )
        and all(
            v <= lim
            for v, lim in zip(result["initial_tracking_gap"].values(), (5.0, 2.0, 2.0))
        )
    )
    return result


def repeat_metrics(chunks, side):
    """Independent predictions for unchanged input; not a closed-loop rollout."""
    p = np.asarray(chunks)[:, :, 7 * side : 7 * side + 3] * 1000
    center = p.mean(axis=0)
    return {
        "position_std_rms_mm": float(
            np.sqrt(np.mean(np.sum((p - center) ** 2, axis=-1)))
        ),
        "endpoint_std_rms_mm": float(
            np.sqrt(np.mean(np.sum((p[:, -1] - center[-1]) ** 2, axis=-1)))
        ),
        "successive_prediction_difference_mm": describe(
            np.linalg.norm(np.diff(p, axis=0), axis=-1)
        ),
        "fixed_input_end_to_next_start_mm": describe(
            np.linalg.norm(p[1:, 0] - p[:-1, -1], axis=-1)
        ),
    }


def select_static_repeats(windows, per_side):
    """Spread static repeats across episodes independently for both arms."""
    selected = set()
    for side in ("left", "right"):
        first_per_episode = {}
        for i, window in enumerate(windows):
            if window["static"][side]["static"]:
                first_per_episode.setdefault(window["episode"], i)
        candidates = list(first_per_episode.values())
        if candidates and per_side > 0:
            selected.update(
                candidates[i]
                for i in np.linspace(
                    0, len(candidates) - 1, min(per_side, len(candidates)), dtype=int
                )
            )
    return sorted(selected)
