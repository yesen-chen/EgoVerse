#!/usr/bin/env python3
"""Plot Eva zarr episodes the way data_visualization.ipynb plots HDF5 demos.

For each arm, overlay commanded and measured xyz, ZYX ypr, and gripper.
Also export every camera frame as PNG plus a 3-frame contact sheet.

Pose arrays in the zarr are ``[x, y, z, qw, qx, qy, qz]`` after
``eva_to_zarr.rot_orientation``. Ypr is converted back to the collector's
ZYX euler so the curves match ``actions/eepose`` and ``observations/eepose``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import simplejpeg
import zarr
from scipy.spatial.transform import Rotation as R

# Same matrix eva_to_zarr applies before writing the quaternion.
R_T_E = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)

ARMS = ("left", "right")
CAMERAS = ("images.front_1", "images.left_wrist", "images.right_wrist")


def _episode_dirs(dataset_path: Path) -> list[Path]:
    if (dataset_path / "zarr.json").is_file():
        return [dataset_path]
    episodes = sorted(
        p for p in dataset_path.iterdir() if p.is_dir() and (p / "zarr.json").is_file()
    )
    return [
        p
        for p in episodes
        if json.loads((p / "zarr.json").read_text())
        .get("attributes", {})
        .get("embodiment", "")
        .startswith("eva")
    ]


def _total_frames(group) -> int:
    attrs = dict(group.attrs)
    if "total_frames" in attrs:
        return int(attrs["total_frames"])
    return int(group["left.obs_ee_pose"].shape[0])


def _read(group, key: str, n: int) -> np.ndarray:
    return np.asarray(group[key][:n])


def pose_to_xyz_ypr(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(T, 7)`` xyzwxyz -> xyz and collector-frame ZYX ypr."""
    xyz = pose[:, :3]
    wxyz = pose[:, 3:7]
    xyzw = np.concatenate([wxyz[:, 1:4], wxyz[:, :1]], axis=1)
    stored = R.from_quat(xyzw).as_matrix()
    collector = np.einsum("ij,tjk->tik", R_T_E.T, stored)
    ypr = R.from_matrix(collector).as_euler("ZYX", degrees=False)
    return xyz, ypr


def _plot_arm(
    out_path: Path,
    arm: str,
    cmd_xyz: np.ndarray,
    obs_xyz: np.ndarray,
    cmd_ypr: np.ndarray,
    obs_ypr: np.ndarray,
    cmd_grip: np.ndarray,
    obs_grip: np.ndarray,
    title: str,
) -> None:
    series = [
        ("x", cmd_xyz[:, 0], obs_xyz[:, 0]),
        ("y", cmd_xyz[:, 1], obs_xyz[:, 1]),
        ("z", cmd_xyz[:, 2], obs_xyz[:, 2]),
        ("yaw", cmd_ypr[:, 0], obs_ypr[:, 0]),
        ("pitch", cmd_ypr[:, 1], obs_ypr[:, 1]),
        ("roll", cmd_ypr[:, 2], obs_ypr[:, 2]),
        ("gripper", cmd_grip.reshape(-1), obs_grip.reshape(-1)),
    ]
    t = np.arange(cmd_xyz.shape[0])
    fig, axs = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(title)
    for ax, (label, cmd, obs) in zip(axs, series):
        ax.plot(t, cmd, label="cmd")
        ax.plot(t, obs, label="measured")
        ax.set_ylabel(label)
        ax.legend(loc="upper right")
    axs[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _decode_jpeg(store, index: int) -> np.ndarray:
    # Scalar index on the vlen array returns a pointer object, not JPEG bytes.
    value = store[index : index + 1][0]
    return simplejpeg.decode_jpeg(bytes(value), colorspace="RGB")


def _export_images(group, n: int, out_dir: Path, task: str) -> None:
    preview_times = [0, n // 2, n - 1]
    for cam in CAMERAS:
        if cam not in group:
            continue
        cam_dir = out_dir / "frames" / cam.split(".", 1)[1]
        cam_dir.mkdir(parents=True, exist_ok=True)
        sheet = []
        store = group[cam]
        for i in range(n):
            rgb = _decode_jpeg(store, i)
            plt.imsave(cam_dir / f"{i:05d}.png", rgb)
            if i in preview_times:
                sheet.append((i, rgb))
        fig, axs = plt.subplots(1, len(sheet), figsize=(4 * len(sheet), 4))
        if len(sheet) == 1:
            axs = [axs]
        for ax, (i, rgb) in zip(axs, sheet):
            ax.imshow(rgb)
            ax.set_title(f"{cam} @ {i}")
            ax.axis("off")
        fig.suptitle(task)
        fig.tight_layout()
        fig.savefig(out_dir / f"preview_{cam.split('.', 1)[1]}.png", dpi=120)
        plt.close(fig)


def plot_episode(episode: Path, out_root: Path) -> Path:
    group = zarr.open_group(episode, mode="r")
    attrs = dict(group.attrs)
    n = _total_frames(group)
    task = attrs.get("task_name", episode.name)
    out_dir = out_root / episode.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for arm in ARMS:
        cmd_xyz, cmd_ypr = pose_to_xyz_ypr(_read(group, f"{arm}.cmd_ee_pose", n))
        obs_xyz, obs_ypr = pose_to_xyz_ypr(_read(group, f"{arm}.obs_ee_pose", n))
        _plot_arm(
            out_dir / f"{arm}_cmd_vs_obs.png",
            arm,
            cmd_xyz,
            obs_xyz,
            cmd_ypr,
            obs_ypr,
            _read(group, f"{arm}.cmd_gripper", n),
            _read(group, f"{arm}.obs_gripper", n),
            f"{episode.name}  {task}  {arm}  cmd vs measured",
        )
    _export_images(group, n, out_dir, f"{episode.name} {task}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="One episode directory, or a folder of Eva episodes.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output root. Default: <dataset-path>/viz or the episode's parent/viz.",
    )
    args = parser.parse_args()
    episodes = _episode_dirs(args.dataset_path)
    if not episodes:
        raise SystemExit(f"No Eva zarr episodes under {args.dataset_path}")
    if args.out is not None:
        out_root = args.out
    elif (args.dataset_path / "zarr.json").is_file():
        out_root = args.dataset_path.parent / "viz"
    else:
        out_root = args.dataset_path / "viz"
    for episode in episodes:
        dest = plot_episode(episode, out_root)
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
