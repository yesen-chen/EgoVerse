"""Paired, fixed-observation RTC recovery diagnostics. Never opens a device.

No recorded observation is advanced using counterfactual actions. Matching an
old plan, closing earlier, or approaching a predicted grasp pose is not success.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "diagnostics/rtc_recovery_followup_20261124"
CHECKPOINT = ROOT / "ckpt/egoverse_1107_pick_h50_full_bs16_l20_19999/19999"
SESSIONS = [
    "rtc_ab_A_20261124_231926",
    "rtc_ab_A_20261124_231501",
    "rtc_ab_B_20261124_231733",
    "rtc_ab_B_20261124_232009",
]


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def variant_weights(weights, grid, entry, variant):
    """Change only weights; never extend into invalid/zero-filled old support."""
    w = np.asarray(weights, dtype=np.float32).copy()
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all():
        raise ValueError("Invalid action clock")
    if w.shape != grid.shape or not np.isfinite(w).all() or np.any((w < 0) | (w > 1)):
        raise ValueError("Invalid weights")
    if not np.all(np.diff(grid) > 0) or not np.isfinite(entry):
        raise ValueError("Invalid action clock")
    if variant == "original":
        return w
    if variant == "unguided":
        return np.zeros_like(w)
    if variant == "hard_only":
        return (w >= 1 - 1e-6).astype(np.float32)
    if variant in {"hold150_release300", "hold200_release300"}:
        hold_s = 0.150 if variant == "hold150_release300" else 0.200
        hard = np.flatnonzero(w >= 1 - 1e-6)
        if not len(hard):
            raise ValueError("Missing committed prefix")
        right = int(np.searchsorted(grid, max(entry + hold_s, grid[hard[-1]])))
        if right >= len(w) or w[right] <= 0:
            raise ValueError("Hold has no valid old-plan support")
        release_end = grid[right] + 0.300
        last = int(np.searchsorted(grid, release_end))
        if last >= len(w) or w[last] <= 0:
            raise ValueError("300ms release has no valid old-plan support")
        u = np.clip((grid - grid[right]) / 0.300, 0.0, 1.0)
        result = (1 - u**3 * (10 + u * (-15 + 6 * u))).astype(np.float32)
        result[w <= 0] = 0
        return result
    if variant != "overlap300":
        raise ValueError(variant)
    support = np.flatnonzero(w > 0)
    right = int(np.searchsorted(grid, entry + 0.3))
    if not len(support) or right >= len(w) or w[right] <= 0:
        raise ValueError("300ms constraint has no valid old-plan support")
    end = int(support[-1]) + 1
    d = right + 1
    result = np.zeros_like(w)
    result[:d] = 1
    if end > d:
        indices = np.arange(d, end)
        c = (end - indices) / (end - d + 1)
        result[indices] = c * np.expm1(c) / np.expm1(1.0)
    result[w <= 0] = 0
    return result


def xyz(actions, times, grid, k):
    times = np.atleast_1d(times)
    if times.min() < grid[0] - 1e-9 or times.max() > grid[-1] + 1e-9:
        raise ValueError("Cannot measure extrapolated trajectory")
    return np.column_stack(
        [np.interp(times, grid, actions[:, j]) for j in range(k, k + 3)]
    )


def close_event(actions, grid, entry, next_switch_offset, threshold_mm):
    """Saved-grid event, not contact; a missing event stays null."""
    indices = np.flatnonzero((grid >= entry) & (actions[:, 13] * 1000 < threshold_mm))
    if not len(indices):
        return None
    i = int(indices[0])
    return {
        "offset_s": float(grid[i]),
        "after_switch_s": float(grid[i] - entry),
        "before_next_recorded_switch": (
            None if next_switch_offset is None else bool(grid[i] < next_switch_offset)
        ),
        "xyz_mm": (actions[i, 7:10] * 1000).tolist(),
        "rotvec_rad": actions[i, 10:13].tolist(),
        "gripper_mm": float(actions[i, 13] * 1000),
    }


def measure(actions, smoothed, reference, grid, entry, next_switch_offset):
    result = {"arms": {}, "right_closure": {}}
    scan = entry + np.linspace(0, 0.3, 301)
    for side, k in (("left", 0), ("right", 7)):
        item = {}
        for label, new in (("raw", actions), ("sg", smoothed)):
            old_p = xyz(reference, scan, grid, k)
            new_p = xyz(new, scan, grid, k)
            gaps = np.linalg.norm(new_p - old_p, axis=1) * 1000
            item[f"{label}_entry_mm"] = float(gaps[0])
            item[f"{label}_window_max_mm"] = float(gaps.max())
            for delta in (0, 0.15, 0.3):
                stamp = entry + delta
                pair = [stamp - 1 / 60, stamp + 1 / 60]
                vn = np.diff(xyz(new, pair, grid, k), axis=0)[0] * 30
                vo = np.diff(xyz(reference, pair, grid, k), axis=0)[0] * 30
                suffix = f"{round(delta*1000)}ms"
                item[f"{label}_velocity_gap_{suffix}_m_s"] = float(
                    np.linalg.norm(vn - vo)
                )
            rn = Slerp(grid, Rotation.from_rotvec(new[:, k + 3 : k + 6]))(scan)
            ro = Slerp(grid, Rotation.from_rotvec(reference[:, k + 3 : k + 6]))(scan)
            item[f"{label}_entry_deg"] = float(
                np.rad2deg((ro[0].inv() * rn[0]).magnitude())
            )
            item[f"{label}_window_max_deg"] = float(
                np.rad2deg((ro.inv() * rn).magnitude()).max()
            )
        u = np.linspace(0, 1, 301)
        blend = 6 * u**5 - 15 * u**4 + 10 * u**3
        old_p = xyz(reference, scan, grid, k)
        new_p = xyz(smoothed, scan, grid, k)
        target = old_p * (1 - blend[:, None]) + new_p * blend[:, None]
        item["blended_target_peak_speed_m_s"] = float(
            np.linalg.norm(np.diff(target, axis=0) / 0.001, axis=1).max()
        )
        result["arms"][side] = item
    for threshold in (20, 30, 40):
        result["right_closure"][str(threshold)] = close_event(
            actions, grid, entry, next_switch_offset, threshold
        )
    future = entry + np.linspace(0, 0.6, 61)
    p = xyz(smoothed, future, grid, 7)
    result["right_future"] = {
        "xyz_at_0_150_300_600ms_mm": (
            xyz(smoothed, entry + np.array([0, 0.15, 0.3, 0.6]), grid, 7) * 1000
        ).tolist(),
        "max_upward_excursion_600ms_mm": float(
            np.max(p[:, 2] - np.minimum.accumulate(p[:, 2])) * 1000
        ),
        "lowest_xyz_600ms_mm": (p[np.argmin(p[:, 2])] * 1000).tolist(),
        "minimum_gripper_future_mm": float(actions[grid >= entry, 13].min() * 1000),
    }
    return result


class CaptureRPC:
    def __init__(self, metadata):
        self.metadata = metadata

    def infer(self, observation):
        self.captured = copy.deepcopy(observation)
        return {
            "actions": np.asarray(self.response["prediction"]["camera_actions"]),
            "rtc": {k: observation["rtc"][k] for k in ("schema", "request_id")},
        }

    def close(self):
        pass


def prepare():
    import cv2

    sys.path[:0] = [str(ROOT), str(ROOT / "pi05/src")]
    from egoverse_pi05.rtc_evaluation import BaseCameraAdapter

    from egomimic.robot.piper_pico.pi05_remote import Pi05RemotePolicy
    from egomimic.robot.piper_pico.policy_inference import ObservationFrame

    bundle = HERE / "bundle"
    if bundle.exists() and any(bundle.iterdir()):
        raise FileExistsError(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    records, sessions = [], {}
    for name in SESSIONS:
        path = ROOT / "diagnostics" / name
        manifest = json.loads((path / "manifest.json").read_text())
        parity = {
            key: digest(ROOT / "egomimic/robot/piper_pico" / key) == value
            for key, value in manifest["recording_contract"]["pipeline_sha256"].items()
        }
        assert all(
            parity[k]
            for k in ("image_geometry.py", "policy_inference.py", "pi05_remote.py")
        )
        rpc = CaptureRPC(manifest["server_metadata"])
        client = Pi05RemotePolicy(
            rpc,
            manifest["runtime_metadata"],
            task_description=manifest["task_description"],
        )
        adapter = BaseCameraAdapter(
            client.adapter.extrinsics, client.adapter.gripper_open_m
        )
        requests = rows(path / "requests.jsonl")
        starts = {r["request_id"]: r for r in requests if r["phase"] == "started"}
        ends = {r["request_id"]: r for r in requests if r["phase"] == "completed"}
        cycles = rows(path / "cycles.jsonl")
        maps = {
            cam: {
                int(r["request_id"]): r
                for r in rows(path / "recording/cameras" / cam / "requests.jsonl")
            }
            for cam in ("front_1", "left_wrist", "right_wrist")
        }
        commands = rows(path / "commands.jsonl")
        close_stamps = [
            min(r["host_send_timestamp"] for r in c["device_receipt"]["receipts"])
            for c in commands
            if c["action_base"][13] < 0.02
        ]
        first_close = min(close_stamps) if close_stamps else None
        sessions[name] = {
            "path": str(path),
            "pipeline_hash_parity": parity,
            "manifest_sha256": digest(path / "manifest.json"),
            "requests_sha256": digest(path / "requests.jsonl"),
            "cycles_sha256": digest(path / "cycles.jsonl"),
            "first_sent_close_walltime": first_close,
            "extrinsics": {k: v.tolist() for k, v in client.adapter.extrinsics.items()},
            "gripper_open_m": client.adapter.gripper_open_m,
            "t0": cycles[0]["first_target_walltime"],
            "action_timing": manifest["action_timing"],
        }
        for i, c in enumerate(cycles):
            if c["start"] != "rtc" or (
                first_close is not None and c["switch_walltime"] > first_close + 0.6
            ):
                continue
            request_id = c["audit_request_id"]
            row, end = starts[request_id], ends[request_id]
            images, image_checks = {}, {}
            for cam, ref in row["image_refs"].items():
                saved = maps[cam][request_id]
                assert (
                    saved["status"] == "saved" and saved["frame_id"] == ref["frame_id"]
                )
                assert saved["timestamp"] == ref["timestamp"]
                image_path = path / "recording/cameras" / cam / saved["file"]
                images[cam] = cv2.cvtColor(
                    cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB
                )
                sha = hashlib.sha256(images[cam].tobytes()).hexdigest()
                assert sha == saved["sha256_rgb"]
                image_checks[cam] = {
                    "file": str(image_path),
                    "sha256_rgb": sha,
                    "frame_id": saved["frame_id"],
                }
            frame = ObservationFrame(
                timestamp=row["observation_timestamp"],
                images=images,
                base_state=np.asarray(row["base_state"]),
                source_timestamps=row["source_timestamps"],
                image_space=row["image_space"],
                image_frame_ids={
                    k: v["frame_id"] for k, v in row["image_refs"].items()
                },
                image_receive_timestamps=row["image_receive_timestamps"],
            )
            rtc = copy.deepcopy(row["rtc"])
            rtc["base_actions"] = np.asarray(rtc["base_actions"])
            rtc["weights"] = np.asarray(rtc["weights"])
            rpc.response = end
            actual = client.predict(frame, rtc=rtc)
            expected = np.asarray(end["prediction"]["base_actions"])
            np.testing.assert_array_equal(actual["base_actions"], expected)
            np.testing.assert_allclose(
                adapter.camera_to_base_actions(end["prediction"]["camera_actions"]),
                expected,
                atol=1e-12,
                rtol=0,
            )
            captured = rpc.captured
            arrays = {k: np.asarray(v) for k, v in captured.items() if k != "rtc"}
            arrays.update(
                {f"rtc.{k}": np.asarray(v) for k, v in captured["rtc"].items()}
            )
            arrays["logged.prefix_base"] = rtc["base_actions"]
            arrays["logged.base_actions"] = expected
            arrays["logged.base_state"] = frame.base_state
            next_switch = (
                cycles[i + 1]["switch_walltime"] if i + 1 < len(cycles) else None
            )
            anchor = c["switch_walltime"] - c["start_offset_s"]
            filename = f"{name}_{request_id:04d}.npz"
            np.savez_compressed(bundle / filename, **arrays)
            records.append(
                {
                    "session": name,
                    "request_id": request_id,
                    "file": filename,
                    "failed_session": name == SESSIONS[0],
                    "pre_actual_close": first_close is None
                    or c["switch_walltime"] < first_close,
                    "entry_s": c["start_offset_s"],
                    "anchor_walltime": anchor,
                    "switch_walltime": c["switch_walltime"],
                    "next_switch_offset_s": None
                    if next_switch is None
                    else next_switch - anchor,
                    "images": image_checks,
                    "input_sha256": digest(bundle / filename),
                }
            )
    index = {
        "checkpoint": str(CHECKPOINT),
        "contract_sha256": digest(CHECKPOINT / "assets/egoverse/contract.json"),
        "sessions": sessions,
        "records": records,
        "seeds": [42, 43, 44],
        "num_steps": 10,
        "guidance": 20,
        "selection": "All 12 normal failure takeovers; three successful basket-stage controls selected before metrics, until first actual sent width<20mm plus 600ms.",
        "limitations": [
            "Original online noise was not saved. Paired explicit seeds do not reproduce the historical random sample.",
            "Fixed actual observations and old prefixes: one-step counterfactuals, not closed-loop rollouts.",
            "No true grasp-region, basket-pose, or contact labels; continuity and earlier closure are not recovery success.",
            "Counterfactual predictions do not generate the next observation or prefix.",
        ],
    }
    save(bundle / "index.json", index)
    print(
        json.dumps(
            {
                "prepared": len(records),
                "per_session": {
                    s: sum(r["session"] == s for r in records) for s in sessions
                },
            }
        ),
        flush=True,
    )


def load_record(path):
    with np.load(path, allow_pickle=False) as f:
        observation = {k: f[k].copy() for k in f.files if k.startswith("observation.")}
        observation["prompt"] = str(f["prompt"].item())
        observation["rtc"] = {
            "schema": str(f["rtc.schema"].item()),
            "request_id": int(f["rtc.request_id"].item()),
            "previous_actions": f["rtc.previous_actions"].copy(),
            "weights": f["rtc.weights"].copy(),
        }
        return observation, f["logged.prefix_base"].copy()


def run(mode, variants=None, run_name=None):
    index = json.loads((HERE / "bundle/index.json").read_text())
    if run_name is not None and (
        Path(run_name).name != run_name or run_name in {".", ".."}
    ):
        raise ValueError("run_name must be a directory name")
    output = HERE / (run_name or f"replay_{mode}")
    output.mkdir(exist_ok=False)
    sys.path[:0] = [str(ROOT / "pi05/scripts"), str(ROOT)]
    from _bootstrap import setup

    setup(openpi_root=ROOT / "pi05/openpi")
    from egoverse_pi05.assets import load_inference_assets
    from egoverse_pi05.inference import create_policy
    from egoverse_pi05.rtc import RtcConfig
    from egoverse_pi05.rtc_diagnostics import identity_guidance_for_diagnostic
    from egoverse_pi05.rtc_evaluation import BaseCameraAdapter

    from egomimic.rldb.action_timing import ActionTiming
    from egomimic.robot.piper_pico.policy_continuity import smooth_pose_actions

    assert (
        digest(CHECKPOINT / "assets/egoverse/contract.json") == index["contract_sha256"]
    )
    assets = load_inference_assets(CHECKPOINT, 10)
    grid = np.asarray(assets.metadata["action_timing"]["output_offsets_s"])
    timing = ActionTiming(
        **{
            k: assets.metadata["action_timing"][k]
            for k in (
                "source_fps",
                "source_horizon",
                "action_stride",
                "output_chunk_length",
                "includes_current",
            )
        }
    )
    variants = variants or (
        ["original", "hard_only", "overlap300"]
        + (["unguided"] if mode == "vjp" else [])
    )
    save(
        output / "arguments.json",
        {
            "mode": mode,
            "variants": variants,
            "script_sha256": digest(__file__),
            "index_sha256": digest(HERE / "bundle/index.json"),
            "seeds": index["seeds"],
            "limitations": index["limitations"],
            "fresh_process": True,
        },
    )
    context = (
        identity_guidance_for_diagnostic() if mode == "identity" else nullcontext()
    )
    count = 0
    with context:
        print("Loading", mode, flush=True)
        policy = create_policy(
            assets, require_single_gpu=True, rtc_config=RtcConfig(20)
        )
        print("Ready", mode, flush=True)
        for record in index["records"]:
            path = HERE / "bundle" / record["file"]
            assert digest(path) == record["input_sha256"]
            current, reference = load_record(path)
            metadata = index["sessions"][record["session"]]
            adapter = BaseCameraAdapter(
                metadata["extrinsics"], metadata["gripper_open_m"]
            )
            entry = record["entry_s"]
            original_weights = current["rtc"]["weights"]
            assert grid[np.flatnonzero(original_weights > 0)[-1]] > entry + 0.3 + 1 / 60
            for seed in index["seeds"]:
                noise = (
                    np.random.default_rng(seed)
                    .standard_normal((100, 32))
                    .astype(np.float32)
                )
                for variant in variants:
                    obs = copy.deepcopy(current)
                    obs["rtc"]["weights"] = variant_weights(
                        original_weights, grid, entry, variant
                    )
                    start = time.monotonic()
                    prediction = np.asarray(policy.infer(obs, noise=noise)["actions"])
                    elapsed = time.monotonic() - start
                    base = adapter.camera_to_base_actions(prediction)
                    smoothed = smooth_pose_actions(base, timing, 0.150)
                    metrics = measure(
                        base,
                        smoothed,
                        reference,
                        grid,
                        entry,
                        record["next_switch_offset_s"],
                    )
                    filename = f"{record['session']}_{record['request_id']:04d}_{seed}_{variant}.npz"
                    np.savez_compressed(
                        output / filename,
                        camera_actions=prediction,
                        base_actions=base,
                        smoothed_base_actions=smoothed,
                        noise=noise,
                        weights=obs["rtc"]["weights"],
                    )
                    item = {
                        **record,
                        "mode": mode,
                        "variant": variant,
                        "seed": seed,
                        "elapsed_s": elapsed,
                        "output_file": filename,
                        "metrics": metrics,
                    }
                    with (output / "samples.jsonl").open("a") as f:
                        f.write(json.dumps(item, allow_nan=False) + "\n")
                    count += 1
            print(
                f"{mode} {record['session']} request={record['request_id']} done={count}",
                flush=True,
            )
    save(output / "complete.json", {"samples": count, "mode": mode})
    print("Completed", mode, count, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare", "vjp", "identity"])
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[
            "original",
            "hard_only",
            "overlap300",
            "unguided",
            "hold150_release300",
            "hold200_release300",
        ],
    )
    parser.add_argument("--run-name")
    args = parser.parse_args()
    HERE = args.output_dir.resolve()
    prepare() if args.operation == "prepare" else run(
        args.operation, args.variants, args.run_name
    )
