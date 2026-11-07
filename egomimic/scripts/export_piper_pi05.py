"""Reference already-converted Piper episodes for Pi0.5, without copying images."""

from __future__ import annotations

import argparse
from pathlib import Path

from egomimic.robot.piper_pico.pi05_data import export_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Converted Zarr episode or root containing episodes",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New manifest JSON path"
    )
    parser.add_argument("--source-fps", type=float, default=30)
    parser.add_argument("--source-horizon", type=int, default=30)
    parser.add_argument("--action-stride", type=int, default=1)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", help="Override each episode's task_description")
    args = parser.parse_args(argv)
    try:
        if not args.input.is_dir():
            raise FileNotFoundError(args.input)
        paths = (
            [args.input]
            if args.input.suffix == ".zarr"
            else sorted(args.input.rglob("episode_*.zarr"))
        )
        result = export_manifest(
            paths,
            args.output,
            source_fps=args.source_fps,
            source_horizon=args.source_horizon,
            action_stride=args.action_stride,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            prompt=args.prompt,
        )
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.exit(1, f"Pi0.5 export failed: {exc}\n")
    print(
        f"Pi0.5 manifest: {result}; native rows retained; training interpolates each window to 100 points"
    )


if __name__ == "__main__":
    main()
