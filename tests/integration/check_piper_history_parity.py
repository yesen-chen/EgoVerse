"""Compare training samples with the actual robot client request, without devices.

Uses converted observations and raw feedback sidecars, the production replay
reader and Pi05RemotePolicy. A local in-memory echo replaces only the model RPC;
this checks preprocessing, not model quality. No sockets or GPU are opened.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from egomimic.robot.piper_pico.pi05_data import Pi05Dataset
from egomimic.robot.piper_pico.pi05_remote import (
    Pi05RemotePolicy,
    pack_message,
    unpack_message,
)
from egomimic.robot.piper_pico.policy_rollout import OfflineReplayClock
from egomimic.scripts.replay_piper_policy import RecordedObservationProvider


class EchoRPC:
    """Exercise the wire encoding but never send anything to a robot/server."""

    def __init__(self, manifest):
        self.metadata = {
            "protocol_version": 1,
            "policy_type": "pi05",
            "action_dim": 14,
            "action_horizon": 100,
            "num_inference_steps": 10,
            "manifest": copy.deepcopy(manifest),
            **{
                key: copy.deepcopy(manifest[key])
                for key in (
                    "action_timing",
                    "action_contract",
                    "training_contracts",
                    "image_shapes",
                )
            },
        }
        if "state_history" in manifest:
            self.metadata["state_history"] = copy.deepcopy(manifest["state_history"])
        self.request = None

    def infer(self, request):
        self.request = unpack_message(pack_message(request))
        return {
            "actions": np.repeat(self.request["observation.state"][None], 100, axis=0)
        }

    def close(self):
        pass


def check_parity(manifest, samples=12):
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    dataset = Pi05Dataset(manifest, split="all", include_images=True)
    if not len(dataset):
        raise ValueError("manifest has no eligible observations")
    providers, policies, maxima, locations = {}, {}, {}, []
    selected = np.unique(
        np.linspace(0, len(dataset) - 1, min(samples, len(dataset)), dtype=int)
    )
    try:
        for index in selected:
            episode, row = dataset.source_location(int(index))
            if episode not in providers:
                provider = RecordedObservationProvider(
                    dataset._paths[episode],
                    OfflineReplayClock(),
                    allow_synthetic_calibration=dataset.manifest[
                        "allow_synthetic_calibration"
                    ],
                    state_history_offsets_s=dataset.state_history_offsets_s,
                )
                rpc = EchoRPC(dataset.manifest)
                policy = Pi05RemotePolicy(
                    rpc,
                    provider.metadata,
                    task_description=dataset.episodes[episode]["prompt"],
                )
                providers[episode], policies[episode] = provider, policy
            sample = dataset[int(index)]
            # Targets are never part of the live observation or request.
            expected = {key: value for key, value in sample.items() if key != "actions"}
            policy = policies[episode]
            policy.predict(providers[episode].read_frame(row))
            request = policy.rpc.request
            if set(request) != set(expected):
                raise AssertionError(
                    f"Training/client fields disagree: {set(expected)} vs {set(request)}"
                )
            for key, value in expected.items():
                if key == "prompt":
                    if value != request[key]:
                        raise AssertionError("Training/client task text disagrees")
                    continue
                if value.shape != request[key].shape:
                    raise AssertionError(f"{key}: shape mismatch")
                error = float(
                    np.max(np.abs(value.astype(float) - request[key].astype(float)))
                )
                maxima[key] = max(maxima.get(key, 0.0), error)
                tolerance = 0.0 if key.startswith("observation.images.") else 2e-6
                if error > tolerance:
                    raise AssertionError(
                        f"Episode {episode} row {row}: {key} error {error} > {tolerance}"
                    )
            locations.append(
                {"dataset_index": int(index), "episode": episode, "source_row": row}
            )
    finally:
        for policy in policies.values():
            policy.close()
    return {
        "passed": True,
        "scope": "training_vs_production_client_preprocessing_only_no_model_or_robot",
        "manifest": str(Path(manifest).resolve()),
        "history_offsets_s": list(dataset.state_history_offsets_s),
        "checked_samples": len(selected),
        "maximum_absolute_errors": maxima,
        "locations": locations,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    report = check_parity(args.manifest, args.samples)
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
