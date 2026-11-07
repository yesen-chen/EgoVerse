"""Strict LoRA/full checkpoint restore and real-dataset inference, with no robot."""

import argparse
import json
import time
from pathlib import Path

from _bootstrap import setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup(openpi_root=args.openpi_root)
    import numpy as np
    from egoverse_pi05.assets import load_inference_assets
    from egoverse_pi05.inference import create_policy
    from scipy.spatial.transform import Rotation

    from egomimic.robot.piper_pico.pi05_data import Pi05Dataset

    assets = load_inference_assets(args.checkpoint_path, args.num_inference_steps)
    dataset = Pi05Dataset(args.manifest, split=args.split, include_images=True)
    if (
        dataset.manifest["manifest_sha256"]
        != assets.metadata["manifest"]["manifest_sha256"]
    ):
        raise ValueError(
            "Inference regression requires the exact training/validation manifest"
        )
    if not 0 <= args.sample_index < len(dataset):
        raise ValueError("sample-index is outside the selected split")
    sample = dataset[args.sample_index]
    target = sample.pop("actions")
    policy = create_policy(assets)
    started = time.monotonic()
    predicted = policy.infer(sample)["actions"]
    first_seconds = time.monotonic() - started
    started = time.monotonic()
    repeated = policy.infer(sample)["actions"]
    warm_seconds = time.monotonic() - started
    metrics = {}
    for side, offset in (("left", 0), ("right", 7)):
        position = (
            np.linalg.norm(
                predicted[:, offset : offset + 3] - target[:, offset : offset + 3],
                axis=-1,
            )
            * 1000
        )
        rotation = (
            Rotation.from_euler("ZYX", predicted[:, offset + 3 : offset + 6]).inv()
            * Rotation.from_euler("ZYX", target[:, offset + 3 : offset + 6])
        ).magnitude()
        metrics[side] = {
            "mean_position_error_mm": float(position.mean()),
            "mean_rotation_error_deg": float(np.rad2deg(rotation).mean()),
            "mean_gripper_error_normalized": float(
                np.abs(predicted[:, offset + 6] - target[:, offset + 6]).mean()
            ),
            "mean_position_second_difference_mm": float(
                np.linalg.norm(
                    np.diff(predicted[:, offset : offset + 3], n=2, axis=0), axis=-1
                ).mean()
                * 1000
            ),
            "same_input_resample_position_difference_mm": float(
                np.linalg.norm(
                    predicted[:, offset : offset + 3]
                    - repeated[:, offset : offset + 3],
                    axis=-1,
                ).mean()
                * 1000
            ),
        }
    report = {
        "passed": True,
        "checkpoint": str(assets.checkpoint_path),
        "split": args.split,
        "sample_index": args.sample_index,
        "actions_shape": list(predicted.shape),
        "first_inference_seconds_including_compilation": first_seconds,
        "warm_inference_seconds": warm_seconds,
        "metrics": metrics,
        "scope": "strict model/LoRA restore and finite physical actions; no task-quality acceptance threshold",
    }
    if args.output.exists():
        raise FileExistsError("Inference report already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez(
        args.output.with_suffix(".npz"),
        predicted=predicted,
        repeated=repeated,
        target=target,
        state=sample["observation.state"],
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
