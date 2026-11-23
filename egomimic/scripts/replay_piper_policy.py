"""Replay a trusted local Piper checkpoint against local converted observations.

No CAN, cameras, network, environment construction, or physical commands.
"""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import simplejpeg
import zarr
from scipy.spatial.transform import Rotation

from egomimic.rldb.zarr.piper_dataset import preflight_piper_episode
from egomimic.robot.piper_pico.policy_execution import RecordingActionSink, SafetyLimits
from egomimic.robot.piper_pico.policy_inference import (
    CAMERA_KEYS,
    ObservationFrame,
    PiperPolicy,
)
from egomimic.robot.piper_pico.policy_rollout import OfflineReplayClock, run_sync_policy


def _local_path(value, *, directory):
    text = str(value)
    if "://" in text:
        raise ValueError("Only local filesystem paths are supported")
    path = Path(value).expanduser().resolve(strict=True)
    if (directory and not path.is_dir()) or (not directory and not path.is_file()):
        raise ValueError(
            f"Expected local {'directory' if directory else 'file'}: {path}"
        )
    return path


class RecordedObservationProvider:
    """Latest recorded tick observation on a virtual timeline.

    Recorded camera and arm source timestamps retain their original offsets
    from the observation time. An episode without them uses the observation
    time explicitly as an offline proxy.
    """

    def __init__(
        self,
        episode,
        clock,
        *,
        start_frame=0,
        allow_synthetic_calibration=False,
        state_history_offsets_s=(),
    ):
        from egomimic.robot.piper_pico.state_history import (
            load_feedback_streams,
            validate_history_offsets,
        )

        self.state_history_offsets_s = validate_history_offsets(state_history_offsets_s)
        self.path = _local_path(episode, directory=True)
        self.metadata = preflight_piper_episode(
            self.path,
            allow_synthetic_calibration=allow_synthetic_calibration,
        )
        self.store = zarr.open_group(str(self.path), mode="r")
        self.feedback_streams = (
            load_feedback_streams(self.path) if self.state_history_offsets_s else None
        )
        if self.state_history_offsets_s:
            for side in ("left", "right"):
                if f"source.{side}.robot_timestamp" not in self.store:
                    raise ValueError(
                        "History replay requires recorded feedback availability timestamps"
                    )
        self.timestamps = np.asarray(
            self.store["timestamp"][: self.metadata["total_frames"]], dtype=float
        )
        if (
            isinstance(start_frame, bool)
            or not isinstance(start_frame, int)
            or not 0 <= start_frame < len(self.timestamps)
        ):
            raise ValueError("start_frame must be a valid recording frame index")
        self.clock, self.next_index = clock, start_frame
        self.origin = float(self.timestamps[start_frame])
        self.virtual_origin = clock.wall_time()
        self.last_record = {}

    def latest(self):
        recorded_now = self.origin + self.clock.wall_time() - self.virtual_origin
        index = max(
            self.next_index,
            int(np.searchsorted(self.timestamps, recorded_now, side="right") - 1),
        )
        if index >= len(self.timestamps):
            raise StopIteration
        original = float(self.timestamps[index])
        stamp = self.virtual_origin + original - self.origin
        self.clock.sleep(max(0.0, stamp - self.clock.wall_time()))
        self.next_index = index + 1
        frame = self.read_frame(index)
        source = dict(frame.source_timestamps)
        self.last_record = {
            "frame_index": index,
            "original_observation_timestamp": original,
            "original_source_timestamps": source,
            "source_timestamp_semantics": self.source_timestamp_semantics,
            "virtual_observation_timestamp": stamp,
        }
        if "source.decision_timestamp" in self.store:
            self.last_record["original_decision_timestamp"] = float(
                self.store["source.decision_timestamp"][index]
            )
        return ObservationFrame(
            stamp,
            frame.images,
            frame.base_state,
            {name: stamp + (value - original) for name, value in source.items()},
            image_space="processed",
            base_state_history=frame.base_state_history,
            state_history_offsets_s=frame.state_history_offsets_s,
        )

    def read_frame(self, index):
        """Random-access recorded input with ORIGINAL time; no clock advancement.

        Evaluation reads observation fields only here. Command labels must be
        loaded separately and are never part of the policy's ObservationFrame.
        """
        if (
            isinstance(index, bool)
            or not isinstance(index, (int, np.integer))
            or not 0 <= index < len(self.timestamps)
        ):
            raise IndexError("recorded frame index is outside the episode")
        original = float(self.timestamps[index])
        images = {}
        for name in CAMERA_KEYS:
            # Zarr v3 VariableLengthBytes integer reads can nest 0-D arrays.
            # Slice first, as ZarrEpisode.read does, to obtain actual JPEG bytes.
            payload = self.store[f"images.{name}"][index : index + 1][0]
            images[name] = simplejpeg.decode_jpeg(payload, colorspace="RGB")
        open_m = float(self.metadata["gripper_encoding"]["open_m"])
        state = []
        for side in ("left", "right"):
            pose = np.asarray(self.store[f"{side}.obs_ee_pose"][index])
            state.extend(pose[:3])
            state.extend(Rotation.from_quat(pose[[4, 5, 6, 3]]).as_rotvec())
            state.append(
                float(np.asarray(self.store[f"{side}.obs_gripper"][index]).item())
                * open_m
            )
        source, semantics = {}, {}
        for name in (*CAMERA_KEYS, "left", "right"):
            camera = name in CAMERA_KEYS
            key = (
                f"source.{name}.timestamp"
                if camera
                else f"source.{name}.robot_timestamp"
            )
            if key in self.store:
                source[name] = float(self.store[key][index])
                semantics[name] = (
                    "recorded_corrected_camera_timestamp"
                    if camera
                    else "recorded_arm_state_timestamp"
                )
            else:
                source[name] = original
                semantics[name] = (
                    "offline_observation_time_proxy_not_measured_source_time"
                )
        self.source_timestamp_semantics = semantics
        history = None
        if self.state_history_offsets_s:
            from egomimic.robot.piper_pico.state_history import sample_state_history

            history = sample_state_history(
                self.feedback_streams,
                original,
                self.state_history_offsets_s,
                available_until=[source[side] for side in ("left", "right")],
            )
        return ObservationFrame(
            original,
            images,
            np.asarray(state),
            source,
            image_space="processed",
            base_state_history=history,
            state_history_offsets_s=self.state_history_offsets_s,
        )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trusted local Lightning checkpoint",
    )
    parser.add_argument(
        "--episode", type=Path, required=True, help="Local converted Piper episode.zarr"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="New directory; existing paths are refused"
    )
    parser.add_argument("--allow-synthetic-calibration", action="store_true")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--control-frequency", type=float, default=30.0)
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=1,
        help="Resampled prefix; default one target, not whole trajectory",
    )
    parser.add_argument(
        "--workspace-min",
        type=float,
        nargs=6,
        required=True,
        metavar="M",
        help="Explicit left xyz then right xyz in metres, each arm base",
    )
    parser.add_argument(
        "--workspace-max", type=float, nargs=6, required=True, metavar="M"
    )
    parser.add_argument("--max-translation-step-m", type=float, default=0.05)
    parser.add_argument("--max-rotation-step-rad", type=float, default=0.35)
    parser.add_argument("--gripper-min-m", type=float, default=0.0)
    parser.add_argument("--gripper-max-m", type=float, default=0.08)
    parser.add_argument("--max-observation-age-s", type=float, default=0.5)
    parser.add_argument("--future-timestamp-tolerance-s", type=float, default=0.05)
    parser.add_argument("--max-schedule-slip-s", type=float, default=0.1)
    parser.add_argument("--command-freshness-s", type=float, default=0.1)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    checkpoint = _local_path(args.checkpoint, directory=False)
    clock = OfflineReplayClock()
    provider = RecordedObservationProvider(
        args.episode,
        clock,
        start_frame=args.start_frame,
        allow_synthetic_calibration=args.allow_synthetic_calibration,
    )
    limits = SafetyLimits(
        workspace_min=np.asarray(args.workspace_min).reshape(2, 3),
        workspace_max=np.asarray(args.workspace_max).reshape(2, 3),
        max_translation_step_m=args.max_translation_step_m,
        max_rotation_step_rad=args.max_rotation_step_rad,
        gripper_min_m=args.gripper_min_m,
        gripper_max_m=args.gripper_max_m,
        max_observation_age_s=args.max_observation_age_s,
        future_timestamp_tolerance_s=args.future_timestamp_tolerance_s,
    )
    output = args.output_dir or Path("diagnostics") / (
        "piper_policy_replay_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    output = output.expanduser().resolve()
    if output == provider.path or provider.path in output.parents:
        raise ValueError("Output directory must be outside the source episode")
    output.mkdir(parents=True, exist_ok=False)
    sink = RecordingActionSink()
    report = {
        "mode": "offline_recorded_replay_no_CAN",
        "checkpoint": str(checkpoint),
        "episode": str(provider.path),
        "device": "cpu",
        "clock": "virtual offline scheduling; CPU inference does not advance virtual time",
        "freshness_scope": "recorded source timestamps on a virtual clock; not a hardware freshness certification",
        "inference_timing": "measured actual CPU wall duration; excluded from virtual scheduling; not a real-time feasibility test",
        "limits": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in asdict(limits).items()
        },
        "allow_synthetic_calibration": args.allow_synthetic_calibration,
        "calibration_status": provider.metadata.get("calibration_status"),
    }
    try:
        policy = PiperPolicy.from_checkpoint(
            checkpoint,
            provider.metadata,
            device="cpu",
            inference_steps=args.inference_steps,
            allow_synthetic_calibration=args.allow_synthetic_calibration,
        )
        report["timing"] = policy.timing.to_dict()
        if hasattr(policy, "gripper_contract"):
            report["gripper_contract"] = policy.gripper_contract
        if hasattr(policy, "adapter"):
            report["image_shapes"] = policy.adapter.image_shapes
        report["rollout"] = run_sync_policy(
            policy,
            provider,
            sink,
            limits,
            max_chunks=args.max_chunks,
            control_frequency=args.control_frequency,
            execute_steps=args.execute_steps,
            monotonic=clock.monotonic,
            wall_time=clock.wall_time,
            sleep=clock.sleep,
            max_schedule_slip_s=args.max_schedule_slip_s,
            command_freshness_s=args.command_freshness_s,
            observation_to_send_s=getattr(policy, "observation_to_send_s", None),
        )
        report["status"] = report["rollout"]["status"]
    except Exception as exc:
        report.update(
            status="failed", error={"type": type(exc).__name__, "message": str(exc)}
        )
        sink.hold("failed")
    report["hold_reasons"] = sink.hold_reasons
    report["sent_action_count"] = len(sink.records)
    with (output / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (output / "sent_actions.jsonl").open("x") as handle:
        for record in sink.records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    with (output / "scheduled_actions.jsonl").open("x") as handle:
        for cycle_index, cycle in enumerate(
            report.get("rollout", {}).get("cycles", [])
        ):
            sent = {
                row["index"]: row
                for row in cycle.get("execution", {}).get("samples", [])
            }
            for row in cycle.get("candidate_schedule", []):
                record = dict(row, cycle_index=cycle_index, sent=row["index"] in sent)
                if record["sent"]:
                    record["execution"] = sent[row["index"]]
                handle.write(json.dumps(record, allow_nan=False) + "\n")
    print(f"{report['status']}: {output / 'report.json'}")
    return 0 if report["status"] in ("completed", "exhausted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
