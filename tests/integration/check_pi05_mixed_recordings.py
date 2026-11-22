"""Read-only acceptance over real converted data; no model, GPU, or robot SDK.

Checks independent geometric endpoint oracles, source grouping, eligibility,
image decoding and mixture coverage. Use train_pi05.py --dry-run separately
to validate actual OpenPI normalization/tokenization/collation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "pi05/src")]

from egomimic.robot.piper_pico.pi05_data import canonical_hash  # noqa: E402
from egomimic.robot.piper_pico.pi05_training_data import (  # noqa: E402
    SCHEMA,
    TrainingDataset,
    domain_datasets,
    load_training_manifest,
    load_training_stats,
)


def matrix(pose):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def check(manifest_path, stats_dir=None, seed=42):
    manifest = load_training_manifest(manifest_path)
    if manifest["schema"] != SCHEMA:
        raise ValueError("Expected a mixed manifest")
    datasets = domain_datasets(manifest_path, split="all", include_images=True)
    rng = np.random.default_rng(seed)
    report = {"schema": "pi05_mixed_recording_acceptance_v1", "domains": {}}
    for domain, dataset in datasets.items():
        offset, checked, pixel_checks = 0, 0, 0
        xyz_error, rotation_error, grip_error = 0.0, 0.0, 0.0
        source_groups = {}
        for episode, record in enumerate(dataset.episodes):
            arrays = dataset._numeric[episode]
            count = record["total_frames"]
            starts = record.get("eligible_starts", count)
            before = canonical_hash(
                {key: value.tolist() for key, value in arrays.items()}
            )
            if domain == "ego":
                group, split = record["source_recording_sha256"], record["split"]
                if group in source_groups and source_groups[group] != split:
                    raise AssertionError("Source recording leaks across splits")
                source_groups[group] = split
                if record["requires_full_window"]:
                    assert starts == count - dataset.timing.source_horizon + 1
                else:
                    assert starts == count
            locals_ = sorted(
                {0, starts // 2, starts - 1, *rng.integers(0, starts, 2).tolist()}
            )
            for local in locals_:
                sample = dataset[offset + local]
                assert sample["actions"].shape == (100, 14)
                assert sample["observation.state"].shape == (14,)
                assert np.isfinite(sample["actions"]).all()
                cameras = (
                    ("front_1",)
                    if domain == "ego"
                    else ("front_1", "left_wrist", "right_wrist")
                )
                for camera in cameras:
                    pixels = sample[f"observation.images.{camera}"]
                    assert (
                        pixels.dtype == np.uint8
                        and list(pixels.shape) == record["image_shapes"][camera]
                    )
                    pixel_checks += 1
                for arm, side in enumerate(("left", "right")):
                    transform = (
                        matrix(arrays["obs_head_pose"][local])
                        if domain == "ego"
                        else np.asarray(record["contract"]["extrinsics"][side])
                    )
                    inverse = np.linalg.inv(transform)
                    # Both future endpoints use exactly this CURRENT camera transform.
                    cases = [
                        (
                            "state",
                            local,
                            "obs",
                            sample["observation.state"][arm * 7 : arm * 7 + 7],
                        )
                    ]
                    target_field = "obs" if domain == "ego" else "cmd"
                    for point, support in (
                        (0, 0),
                        (-1, dataset.timing.support_indices[-1]),
                    ):
                        row = min(local + support, count - 1)
                        cases.append(
                            (
                                "action",
                                row,
                                target_field,
                                sample["actions"][point, arm * 7 : arm * 7 + 7],
                            )
                        )
                    for _, row, field, actual in cases:
                        expected = inverse @ matrix(
                            arrays[f"{side}.{field}_ee_pose"][row]
                        )
                        xyz_error = max(
                            xyz_error,
                            float(np.max(np.abs(expected[:3, 3] - actual[:3]))),
                        )
                        rotation_error = max(
                            rotation_error,
                            float(
                                (
                                    Rotation.from_matrix(expected[:3, :3]).inv()
                                    * Rotation.from_euler("ZYX", actual[3:6])
                                ).magnitude()
                            ),
                        )
                        gripper = (
                            0.0
                            if domain == "ego" and dataset.gripper_mode == "zero"
                            else float(arrays[f"{side}.{field}_gripper"][row, 0])
                        )
                        grip_error = max(grip_error, abs(gripper - float(actual[6])))
                checked += 1
            assert (
                canonical_hash({key: value.tolist() for key, value in arrays.items()})
                == before
            )
            offset += starts
        assert xyz_error < 2e-6 and rotation_error < 2e-6 and grip_error < 2e-6
        report["domains"][domain] = {
            "episodes": len(dataset.episodes),
            "source_rows": sum(r["total_frames"] for r in dataset.episodes),
            "eligible_starts": len(dataset),
            "checked_windows": checked,
            "decoded_images": pixel_checks,
            "max_position_error_m": xyz_error,
            "max_rotation_error_rad": rotation_error,
            "max_gripper_error": grip_error,
            "source_span_s": dataset.timing.effective_span_s,
            "source_horizon": dataset.timing.source_horizon,
            "model_points": 100,
            "source_arrays_unchanged": True,
        }
    mixture = TrainingDataset(manifest_path, include_images=False)
    counts = Counter(domain for domain, _ in mixture.schedule)
    for domain, dataset in mixture.datasets.items():
        assert {i for d, i in mixture.schedule if d == domain} == set(
            range(len(dataset))
        )
    weights = manifest["sampling"]["weights"]
    assert counts["piper"] * weights["ego"] == counts["ego"] * weights["piper"]
    report["training_schedule"] = dict(counts)
    report["unique_training_starts"] = {
        d: len(ds) for d, ds in mixture.datasets.items()
    }
    report["source_span_ratio_piper_to_ego"] = (
        report["domains"]["piper"]["source_span_s"]
        / report["domains"]["ego"]["source_span_s"]
    )
    if stats_dir:
        stats = load_training_stats(manifest, stats_dir)
        report["separate_statistics_verified"] = sorted(stats)
    report["passed"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = check(args.manifest, args.stats_dir, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
