"""Inspect physical state/action batches after the new training transforms.

No calibration fitting or robot connection. Synthetic Piper calibration is
opt-in and visibly labelled. This checks loader geometry, not hardware accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset

from egomimic.rldb.action_timing import ActionTiming
from egomimic.rldb.embodiment.pico_ego import PicoEgo, PicoEgoDataset
from egomimic.rldb.embodiment.piper import Piper
from egomimic.rldb.zarr.piper_dataset import PiperDataset


def project(xyz, intrinsics):
    xyz = np.asarray(xyz)
    homogeneous = xyz @ np.asarray(intrinsics)[:3, :3].T
    uv = np.full((len(xyz), 2), np.nan)
    valid = (xyz[:, 2] > 0.01) & np.isfinite(homogeneous).all(axis=1)
    uv[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:3]
    return uv


def inspect_batch(
    episode,
    output,
    *,
    kind,
    timing,
    indices=None,
    gripper_mode="zero",
    expected_axis_convention="piper_x",
    allow_synthetic_calibration=False,
):
    episode, output = Path(episode).resolve(), Path(output).resolve()
    if episode == output or episode in output.parents:
        raise ValueError("Output must be outside the source episode")
    if output.exists():
        raise FileExistsError(output)
    if kind == "piper":
        embodiment = Piper
        dataset = PiperDataset(
            episode,
            Piper.get_keymap(timing=timing),
            Piper.get_transform_list(timing=timing),
            allow_synthetic_calibration=allow_synthetic_calibration,
        )
        gripper_mode = "width"
    elif kind == "pico_ego":
        embodiment = PicoEgo
        dataset = PicoEgoDataset(
            episode,
            PicoEgo.get_keymap(timing=timing, gripper_mode=gripper_mode),
            PicoEgo.get_transform_list(timing=timing, gripper_mode=gripper_mode),
            gripper_mode=gripper_mode,
            expected_axis_convention=expected_axis_convention,
        )
    else:
        raise ValueError(f"Unknown kind: {kind}")
    indices = (
        list(indices)
        if indices is not None
        else [0, len(dataset) // 2, len(dataset) - 1]
    )
    if not indices or any(i < 0 or i >= len(dataset) for i in indices):
        raise ValueError("Sample indices must be inside the valid episode length")
    batch = next(
        iter(
            DataLoader(Subset(dataset, indices), batch_size=len(indices), num_workers=0)
        )
    )
    projection_available = (
        "intrinsics" in batch and np.isfinite(batch["intrinsics"].numpy()).all()
    )
    synthetic = (
        dataset.metadata.get("calibration_status") == "synthetic"
        or dataset.metadata.get("calibration", {})
        .get("data", {})
        .get("synthetic", False)
        is True
    )
    title = (
        "SYNTHETIC CALIBRATION: geometry test only"
        if synthetic
        else "Current front_1 camera; no calibration refit"
    )
    output.mkdir(parents=True)
    width = 7 if gripper_mode == "width" else 6
    offsets = np.asarray(timing.output_offsets_s)
    for row, index in enumerate(indices):
        state = batch["observations.state.ee_pose"][row].numpy()
        action = batch["actions_cartesian"][row].numpy()
        image = batch[embodiment.VIZ_IMAGE_KEY][row].numpy().transpose(1, 2, 0)
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes[0, 0].imshow(image)
        for hand, color, label in ((0, "limegreen", "L"), (1, "darkorange", "R")):
            xyz = action[:, hand * width : hand * width + 3]
            current = state[hand * width : hand * width + 3]
            if projection_available:
                k = batch["intrinsics"][row].numpy()
                uv, current_uv = project(xyz, k), project(current[None], k)
                axes[0, 0].plot(
                    uv[:, 0], uv[:, 1], color=color, label=f"{label} future"
                )
                axes[0, 0].scatter(*current_uv[0], c=color, marker="x", s=60)
            axes[0, 1].plot(xyz[:, 0], xyz[:, 2], color=color, label=f"{label} future")
            axes[0, 1].scatter(current[0], current[2], c=color, marker="x")
            for coordinate, style in enumerate(("-", "--", ":")):
                axes[1, 0].plot(
                    offsets,
                    xyz[:, coordinate],
                    linestyle=style,
                    color=color,
                    label=f"{label} {'xyz'[coordinate]}",
                )
            if width == 7:
                axes[1, 1].plot(
                    offsets, action[:, hand * width + 6], color=color, label=label
                )
                axes[1, 1].scatter(0, state[hand * width + 6], c=color, marker="x")
        axes[0, 0].set_xlim(0, image.shape[1])
        axes[0, 0].set_ylim(image.shape[0], 0)
        axes[0, 0].set_title("Projection (cross = current observation)")
        if not projection_available:
            axes[0, 0].set_title("Intrinsics absent: projection unavailable")
        axes[0, 1].set(xlabel="camera x (m)", ylabel="camera z (m)")
        axes[1, 0].set(xlabel="action offset (s)", ylabel="camera position (m)")
        axes[1, 1].set(xlabel="action offset (s)", ylabel="fixed width ratio")
        if width == 6:
            axes[1, 1].text(
                0.1, 0.5, "zero mode: shared head inserts zero after normalization"
            )
        for ax in axes.flat:
            if ax.lines:
                ax.legend(fontsize=8)
        fig.suptitle(f"{episode.name} frame {index} | {kind}/{gripper_mode}\n{title}")
        fig.tight_layout()
        fig.savefig(output / f"frame_{index:06d}.png", dpi=130)
        plt.close(fig)
    report = {
        "episode": str(episode),
        "kind": kind,
        "gripper_mode": gripper_mode,
        "synthetic_calibration": synthetic,
        "sample_indices": indices,
        "timing": timing.to_dict(),
        "coordinate_frame": "current_front_1_optical_camera",
        "state_shape": list(batch["observations.state.ee_pose"].shape),
        "action_shape": list(batch["actions_cartesian"].shape),
        "values": "physical units before statistical normalization",
        "projection_available": bool(projection_available),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kind", choices=("piper", "pico_ego"), required=True)
    parser.add_argument("--gripper-mode", choices=("zero", "width"), default="zero")
    parser.add_argument(
        "--expected-axis-convention", choices=("piper_x", "piper"), default="piper_x"
    )
    parser.add_argument("--allow-synthetic-calibration", action="store_true")
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--source-fps", type=float, default=30)
    parser.add_argument("--source-horizon", type=int, default=30)
    parser.add_argument("--action-stride", type=int, default=1)
    parser.add_argument("--output-chunk-length", type=int, default=100)
    args = parser.parse_args()
    timing = ActionTiming(
        args.source_fps,
        args.source_horizon,
        args.action_stride,
        args.output_chunk_length,
    )
    print(
        json.dumps(
            inspect_batch(
                args.episode,
                args.output,
                kind=args.kind,
                timing=timing,
                indices=args.indices,
                gripper_mode=args.gripper_mode,
                expected_axis_convention=args.expected_axis_convention,
                allow_synthetic_calibration=args.allow_synthetic_calibration,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
