"""Print/serialize an explicit action timing YAML without data, models or hardware."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from egomimic.rldb.action_timing import parse_action_timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, help="New JSON report path; never overwrite"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    node = OmegaConf.select(cfg, "action_timing")
    if node is None:
        parser.error("config must contain an explicit action_timing mapping")
    specs = parse_action_timing(OmegaConf.to_container(node, resolve=True))
    if not specs:
        parser.error("action_timing must not be empty")
    report = {name: spec.to_dict() for name, spec in specs.items()}
    for name, spec in specs.items():
        print(
            f"{name}: {spec.source_fps:g} Hz, H={spec.source_horizon}, "
            f"stride={spec.action_stride}, output={spec.output_chunk_length}; "
            f"source span={spec.source_span_s:.9f} s, "
            f"effective span={spec.effective_span_s:.9f} s; starts at current sample"
        )
    if args.output:
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
