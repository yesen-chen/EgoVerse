"""Add raw feedback sidecars to existing Piper Zarr episodes, without image I/O.

The explicit recording root is checked against each episode's stored tick
timestamps, observed state and action provenance. Existing Zarr arrays and
metadata are never rewritten; old manifest hashes therefore stay valid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_pico.state_history import write_feedback_sidecar


def _verify_episode(source_data, episode, source_index, ends):
    metadata = dict(episode.attrs)
    if metadata.get("schema") != "piper_base_v3":
        raise ValueError("backfill requires a piper_base_v3 episode")
    count = metadata.get("total_frames")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("invalid converted total_frames")
    provenance = metadata.get("conversion", {})
    if provenance.get("source_episode") != source_index:
        raise ValueError("converted source_episode disagrees with its directory name")
    if source_index < 0 or source_index >= len(ends):
        raise ValueError("source episode does not exist in recording")
    start = 0 if source_index == 0 else int(ends[source_index - 1])
    end = int(ends[source_index])
    rows = np.asarray(episode["source.action_index"][:count])
    if (
        rows.shape != (count,)
        or rows.dtype.kind not in "iu"
        or np.any(np.diff(rows) != 1)
        or rows[0] < start
        or rows[-1] >= end
    ):
        raise ValueError(
            "converted source row indices do not belong to the raw episode"
        )
    selection = slice(int(rows[0]), int(rows[-1]) + 1)
    if not np.array_equal(
        episode["timestamp"][:count], source_data["timestamp"][selection]
    ):
        raise ValueError("raw and converted observation timestamps disagree")
    for index, side in enumerate(("left", "right")):
        recorded_pose = np.asarray(episode[f"{side}.obs_ee_pose"][:count])
        source_xyz = source_data[f"robot{index}_eef_pos"][selection]
        source_rotation = Rotation.from_rotvec(
            source_data[f"robot{index}_eef_rot_axis_angle"][selection]
        )
        stored_rotation = Rotation.from_quat(recorded_pose[:, [4, 5, 6, 3]])
        if not np.allclose(
            recorded_pose[:, :3], source_xyz, atol=1e-10, rtol=0
        ) or not np.allclose(
            (stored_rotation.inv() * source_rotation).magnitude(), 0, atol=1e-9, rtol=0
        ):
            raise ValueError(f"{side}: raw and converted measured pose disagree")
        if not np.array_equal(
            episode[f"{side}.obs_gripper_width_m"][:count],
            source_data[f"robot{index}_gripper_width"][selection],
        ):
            raise ValueError(f"{side}: raw and converted gripper feedback disagree")
        if not np.array_equal(
            episode[f"source.{side}.robot_timestamp"][:count],
            source_data[f"robot{index}_source_timestamp"][selection],
        ):
            raise ValueError(
                f"{side}: raw and converted available feedback cutoff disagree"
            )


def backfill_feedback_history(source, converted_root, *, episodes=None):
    """Populate only sidecars; return paths. Repeating the same call is harmless."""
    source, converted_root = Path(source).resolve(), Path(converted_root).resolve()
    recording = zarr.open_group(str(source / "replay_buffer.zarr"), mode="r")
    ends = np.asarray(recording["meta/episode_ends"][:])
    if (
        ends.ndim != 1
        or not len(ends)
        or ends.dtype.kind not in "iu"
        or np.any(np.diff(np.r_[0, ends]) <= 0)
    ):
        raise ValueError("invalid source recording episode boundaries")
    paths = sorted(converted_root.glob("episode_*.zarr"))
    if episodes is not None:
        ids = list(episodes)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("episodes must be a nonempty unique list")
        paths = [converted_root / f"episode_{index:06d}.zarr" for index in ids]
    if not paths:
        raise ValueError("no converted episodes found")
    # Check provenance for all selected episodes before adding anything.
    selected = []
    for path in paths:
        source_index = int(path.stem.removeprefix("episode_"))
        episode = zarr.open_group(str(path), mode="r")
        _verify_episode(recording["data"], episode, source_index, ends)
        video_dir = source / "videos" / str(source_index)
        for index in range(2):
            raw = video_dir / f"robot{index}_state.npz"
            if not raw.is_file():
                raise FileNotFoundError(f"missing raw feedback: {raw}")
        selected.append((video_dir, path))
    for video_dir, path in selected:
        write_feedback_sidecar(video_dir, path, allow_missing=False)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Raw recording batch containing replay_buffer.zarr and videos/",
    )
    parser.add_argument(
        "--converted-root",
        type=Path,
        required=True,
        help="Converted batch directory containing episode_*.zarr",
    )
    parser.add_argument("--episodes", type=int, nargs="+")
    args = parser.parse_args(argv)
    for path in backfill_feedback_history(
        args.source, args.converted_root, episodes=args.episodes
    ):
        print(f"Feedback history verified: {path}")


if __name__ == "__main__":
    main()
