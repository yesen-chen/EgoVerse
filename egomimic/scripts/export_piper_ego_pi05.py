"""Compose existing Piper and PICO Ego manifests without retiming source data."""

import argparse
from pathlib import Path

from egomimic.robot.piper_pico.pi05_training_data import (
    export_mixed_manifest,
    load_training_manifest,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piper-manifest", type=Path, required=True)
    parser.add_argument("--ego-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--piper-weight", type=int, default=1)
    parser.add_argument("--ego-weight", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        path = export_mixed_manifest(
            args.piper_manifest,
            args.ego_manifest,
            args.output,
            piper_weight=args.piper_weight,
            ego_weight=args.ego_weight,
            seed=args.seed,
        )
        manifest = load_training_manifest(path)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Mixed Pi0.5 export failed: {exc}\n")
    print(f"Mixed Pi0.5 manifest: {path}")
    for domain, timing in manifest["action_timings"].items():
        print(
            f"{domain}: {timing['source_horizon']} source rows -> 100 points; source span {timing['effective_span_s']:.6f} s"
        )
    print("Source timestamps are unchanged; robot deployment uses Piper timing.")


if __name__ == "__main__":
    main()
