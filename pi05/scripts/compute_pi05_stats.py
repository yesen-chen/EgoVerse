"""Compute fresh 14-D statistics using training episodes only, before model padding."""

import argparse
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
    from egomimic.robot.piper_pico.pi05_training_data import (
        load_training_manifest,
        load_training_stats,
        write_training_stats,
    )

    output = args.output_dir.expanduser().resolve()
    report = write_training_stats(args.manifest, output, max_windows=args.max_windows)
    load_training_stats(load_training_manifest(args.manifest), output)
    print(f"PI05 TRAIN-ONLY STATS PASSED: {output}")
    for domain, detail in report["domains"].items():
        print(
            f"{domain}: {detail['sampled_windows']}/{detail['eligible_training_windows']} eligible training windows; {detail['filename']}"
        )


if __name__ == "__main__":
    main()
