"""Reference converted PICO Ego episodes for Pi0.5 without retiming raw data."""

from __future__ import annotations

import argparse
from pathlib import Path

from egomimic.robot.piper_pico.pi05_ego_data import (
    export_ego_manifest,
    load_ego_manifest,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Converted Ego episode or root"
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
    parser.add_argument("--gripper-mode", choices=("zero", "width"), default="zero")
    parser.add_argument(
        "--state-history-offsets-ms",
        type=float,
        nargs="*",
        default=[],
        help="Negative offsets in milliseconds, oldest first; empty disables history",
    )
    parser.add_argument(
        "--allow-repeat-after-trim",
        action="store_true",
        help="Opt in to repeated terminal supervision after UI trimming",
    )
    parser.add_argument(
        "--allow-repeat-after-inactivity",
        action="store_true",
        help="Opt in to repeated terminal supervision after inactive/tracking/video gaps",
    )
    args = parser.parse_args(argv)
    try:
        if not args.input.is_dir():
            raise FileNotFoundError(args.input)
        paths = (
            [args.input]
            if args.input.suffix == ".zarr"
            else sorted(args.input.rglob("episode_*.zarr"))
        )
        output = export_ego_manifest(
            paths,
            args.output,
            source_fps=args.source_fps,
            source_horizon=args.source_horizon,
            action_stride=args.action_stride,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            prompt=args.prompt,
            gripper_mode=args.gripper_mode,
            state_history_offsets_s=tuple(
                x / 1000 for x in args.state_history_offsets_ms
            ),
            require_full_window_after_trim=not args.allow_repeat_after_trim,
            require_full_window_after_inactivity=not args.allow_repeat_after_inactivity,
        )
        manifest = load_ego_manifest(output)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.exit(1, f"Ego Pi0.5 export failed: {exc}\n")
    timing = manifest["action_timing"]
    starts = sum(r["eligible_starts"] for r in manifest["episodes"])
    print(
        f"Ego Pi0.5 manifest: {output}\n{len(paths)} episodes; {starts} eligible starts; source {timing['effective_span_s']:.6f} s -> 100 model points; original timestamps unchanged"
    )


if __name__ == "__main__":
    main()
