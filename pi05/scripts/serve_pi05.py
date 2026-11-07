"""Serve a portable Pi0.5 checkpoint; no original training dataset is required."""

import argparse
import json
from pathlib import Path

from _bootstrap import setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.host.strip() or not 1 <= args.port <= 65535:
        parser.error("A nonempty host and port 1..65535 are required")
    setup(cpu=args.check_only, openpi_root=args.openpi_root)
    from egoverse_pi05.assets import inference_summary, load_inference_assets

    assets = load_inference_assets(args.checkpoint_path, args.num_inference_steps)
    print(
        json.dumps(inference_summary(assets.metadata), indent=2, ensure_ascii=False),
        flush=True,
    )
    if args.check_only:
        print("PI05 PORTABLE ASSETS PASSED; no model weights loaded, no server started")
        return
    from egoverse_pi05.inference import create_policy, warmup_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    policy = create_policy(assets, require_single_gpu=True)
    result = warmup_policy(policy, assets)
    print(
        f"PI05 WARMUP PASSED: {result['actions'].shape}; serving {args.host}:{args.port}",
        flush=True,
    )
    WebsocketPolicyServer(
        policy, host=args.host, port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
