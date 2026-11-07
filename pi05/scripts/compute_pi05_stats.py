"""Compute fresh 14-D statistics using training episodes only, before model padding."""

import argparse
import json
from pathlib import Path

from _bootstrap import setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument(
        "--max-windows",
        type=int,
        help="Deterministic subset for smoke tests only; omit for production statistics",
    )
    args = parser.parse_args()
    setup(cpu=True, openpi_root=args.openpi_root)
    from egoverse_pi05.assets import validate_stats

    from egomimic.robot.piper_pico.pi05_data import compute_norm_stats, load_manifest

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            "Statistics output already exists; use a fresh output-dir"
        )
    manifest = load_manifest(args.manifest)
    stats = {
        "norm_stats": compute_norm_stats(args.manifest, max_windows=args.max_windows)
    }
    validate_stats(stats)
    output.mkdir(parents=True)
    (output / "norm_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    report = {
        "manifest_sha256": manifest["manifest_sha256"],
        "data_identity_sha256": manifest["data_identity_sha256"],
        "split": "train",
        "action_horizon": 100,
        "action_dim": 14,
        "max_windows": args.max_windows,
        "smoke_test_statistics": args.max_windows is not None,
        "normalization": "OpenPI q01/q99; physical units before 32-D padding; no clipping",
    }
    (output / "stats_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PI05 TRAIN-ONLY STATS PASSED: {output}")


if __name__ == "__main__":
    main()
