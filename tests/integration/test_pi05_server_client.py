"""Opt-in real JAX Pi0.5 server / robot-environment client test; no hardware.

Run from the existing robot environment after setting all four paths:

  EGOVERSE_TEST_PI05_PYTHON=/existing/jax/environment/bin/python
  EGOVERSE_TEST_PI05_OPENPI_ROOT=/existing/pinned/openpi
  EGOVERSE_TEST_PI05_CHECKPOINT=/existing/checkpoints/experiment/4
  EGOVERSE_TEST_PI05_MANIFEST=/existing/export/manifest.json
  python -m pytest --integration -m gpu \
      tests/integration/test_pi05_server_client.py -q -s

Expose one GPU to the subprocess using the caller's CUDA_VISIBLE_DEVICES. This
test never creates an environment, installs dependencies, starts CAN or opens a
camera. Server warmup is bounded at 300 s; all stages share a 600 s deadline.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest
import zarr
from scipy.spatial.transform import Rotation

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

REPO = Path(__file__).resolve().parents[2]
ENV_PREFIX = "EGOVERSE_TEST_PI05_"
TOTAL_TIMEOUT_S = 600.0
STARTUP_TIMEOUT_S = 300.0
MAX_LOG_BYTES = 8 * 1024 * 1024
MAX_LOG_TAIL_BYTES = 64 * 1024
LOG_TRUNCATION_MARKER = (
    b"\n[server log truncated; middle omitted, latest 64 KiB follows]\n"
)


def _paths():
    names = ("PYTHON", "OPENPI_ROOT", "CHECKPOINT", "MANIFEST")
    missing = [
        ENV_PREFIX + key for key in names if not os.environ.get(ENV_PREFIX + key)
    ]
    if missing:
        pytest.skip("Set explicit integration paths: " + ", ".join(missing))
    paths = {}
    for key in names:
        path = Path(os.environ[ENV_PREFIX + key]).expanduser()
        # Preserve the venv's executable symlink. Resolving it to the shared
        # base Python would silently discard the JAX environment's site-packages.
        paths[key] = path.absolute() if key == "PYTHON" else path.resolve()
    for key, path in paths.items():
        assert path.exists(), f"{ENV_PREFIX + key} does not exist: {path}"
    assert paths["PYTHON"].is_file() and os.access(paths["PYTHON"], os.X_OK)
    assert paths["CHECKPOINT"].is_dir() and paths["MANIFEST"].is_file()
    assert (paths["OPENPI_ROOT"] / "src/openpi/models/pi0_config.py").is_file()
    return paths


def _remaining(deadline, maximum):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(
            "Pi0.5 server/client integration exceeded its 600 s deadline"
        )
    return min(float(maximum), remaining)


def _ephemeral_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _server_environment():
    """Do not export the robot client's OpenCV loader paths into the JAX venv."""
    environment = os.environ.copy()
    removed = {}
    cv2_root = Path(cv2.__file__).resolve().parent
    client_cv2_library = (cv2_root / "../../lib64").resolve()
    library_path = environment.get("LD_LIBRARY_PATH")
    if library_path is not None:
        entries = library_path.split(os.pathsep)
        leaked = [
            item
            for item in entries
            if item and Path(item).resolve() == client_cv2_library
        ]
        if leaked:
            environment["LD_LIBRARY_PATH"] = os.pathsep.join(
                item for item in entries if item not in leaked
            )
            removed["LD_LIBRARY_PATH"] = leaked
    for name in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        value = environment.get(name)
        if value and Path(value).resolve().is_relative_to(cv2_root):
            removed[name] = [environment.pop(name)]
    # Keep caller GPU selection, XLA settings and system library paths intact.
    # This only affects the owned child; it never mutates the robot environment.
    return environment, removed


class _ServerLog:
    """Drain output continuously, preserving its prefix and latest error tail."""

    def __init__(self, stream, path):
        self.stream, self.path = stream, path
        self.bytes_read = 0
        self.truncated = False
        self.error = None
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self):
        tail = bytearray()
        prefix_bytes = MAX_LOG_BYTES - MAX_LOG_TAIL_BYTES - len(LOG_TRUNCATION_MARKER)
        try:
            with self.path.open("wb") as output:
                while block := self.stream.read(8192):
                    self.bytes_read += len(block)
                    tail.extend(block)
                    del tail[:-MAX_LOG_TAIL_BYTES]
                    if self.bytes_read <= MAX_LOG_BYTES:
                        output.write(block)
                    else:
                        # Keep draining after truncation and refresh the tail
                        # on disk, including while the child is still running.
                        # An oversized startup manifest must not hide its later
                        # traceback. The marker is included in the file cap.
                        output.seek(prefix_bytes)
                        output.write(LOG_TRUNCATION_MARKER)
                        output.write(tail)
                        output.truncate()
                        self.truncated = True
                    output.flush()
        except (OSError, ValueError) as exc:
            self.error = str(exc)

    def finish(self):
        self.thread.join(timeout=3)
        if not self.thread.is_alive():
            self.stream.close()


def _wait_ready(process, port, deadline, log_path):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    ready_deadline = min(deadline, time.monotonic() + STARTUP_TIMEOUT_S)
    while time.monotonic() < ready_deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"Owned Pi0.5 server exited with {code}; see {log_path}")
        try:
            with opener.open(
                f"http://127.0.0.1:{port}/healthz",
                timeout=_remaining(ready_deadline, 1),
            ) as response:
                if response.status == 200 and response.read(64).strip() == b"OK":
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(min(0.2, _remaining(ready_deadline, 0.2)))
    raise TimeoutError(f"Pi0.5 server did not finish warmup in 300 s; see {log_path}")


def _recorded_frame(dataset):
    """Build client input independently from the dataset's camera-frame transform."""
    from egomimic.robot.piper_pico.policy_inference import (
        CAMERA_KEYS,
        SOURCE_NAMES,
        ObservationFrame,
    )

    first = dataset.episodes[0]
    path = (dataset.manifest_path.parent / first["path"]).resolve()
    store = zarr.open_group(str(path), mode="r")
    metadata = dict(store.attrs)
    opening = float(metadata["gripper_encoding"]["open_m"])
    stamp = float(store["timestamp"][0])
    physical_state = []
    for side in ("left", "right"):
        # These are measured base poses, not model labels or commanded poses.
        pose = np.asarray(store[f"{side}.obs_ee_pose"][0], dtype=float)
        physical_state.extend(pose[:3])
        physical_state.extend(Rotation.from_quat(pose[[4, 5, 6, 3]]).as_rotvec())
        width_key = f"{side}.obs_gripper_width_m"
        if width_key in store:
            width = float(np.asarray(store[width_key][0]).item())
        else:
            width = float(np.asarray(store[f"{side}.obs_gripper"][0]).item()) * opening
        physical_state.append(width)
    images = {}
    for name in CAMERA_KEYS:
        encoded = store[f"images.{name}"][0:1][0]
        if isinstance(encoded, np.ndarray):
            encoded = encoded.item()
        bgr = cv2.imdecode(np.frombuffer(bytes(encoded), np.uint8), cv2.IMREAD_COLOR)
        assert bgr is not None, f"Bad stored JPEG for {name} in {path}"
        images[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sources = {name: stamp for name in SOURCE_NAMES}
    for name in CAMERA_KEYS:
        key = f"source.{name}.timestamp"
        if key in store:
            sources[name] = float(np.asarray(store[key][0]).item())
    for side in ("left", "right"):
        key = f"source.{side}.robot_timestamp"
        if key in store:
            sources[side] = float(np.asarray(store[key][0]).item())
    return (
        metadata,
        ObservationFrame(
            stamp, images, np.asarray(physical_state), sources, image_space="processed"
        ),
        path,
    )


class _AuditedRPC:
    def __init__(self, rpc, expected):
        self.rpc, self.metadata, self.expected = rpc, rpc.metadata, expected
        self.requests = 0
        self.closed = False
        self.parity = None

    def infer(self, request):
        # Pi05Dataset computes this expectation independently of the deployed
        # adapter, using stored quaternion poses and per-episode calibration.
        assert set(request) == set(self.expected) - {"actions"}
        actual = np.asarray(request["observation.state"])
        desired = np.asarray(self.expected["observation.state"])
        np.testing.assert_allclose(actual, desired, rtol=0, atol=3e-6)
        assert actual.dtype == np.float32 and actual.shape == (14,)
        assert request["prompt"] == self.expected["prompt"]
        hashes = {}
        for key in sorted(request):
            if not key.startswith("observation.images."):
                continue
            image = np.asarray(request[key])
            np.testing.assert_array_equal(image, self.expected[key])
            assert image.dtype == np.uint8 and image.shape[-1] == 3
            hashes[key] = hashlib.sha256(image.tobytes()).hexdigest()
        self.requests += 1
        self.parity = {
            "state_max_abs_error": float(np.max(np.abs(actual - desired))),
            "processed_rgb_exact_match": True,
            "processed_rgb_sha256": hashes,
            "prompt": request["prompt"],
        }
        return self.rpc.infer(request)

    def close(self):
        self.closed = True
        self.rpc.close()


def test_real_jax_server_with_robot_environment_client(tmp_path):
    paths = _paths()
    pytest.importorskip("websocket")
    pytest.importorskip("msgpack")
    from egomimic.robot.piper_pico.pi05_data import Pi05Dataset
    from egomimic.robot.piper_pico.pi05_remote import Pi05RemotePolicy, WebSocketRPC
    from egomimic.robot.piper_pico.policy_execution import resample_action_block

    began = time.monotonic()
    deadline = began + TOTAL_TIMEOUT_S
    jax_before = "jax" in sys.modules
    log_path = tmp_path / "pi05_server.log"
    report_path = tmp_path / "pi05_server_client_report.json"
    report = {
        "schema": "egoverse_pi05_server_client_integration_v1",
        "status": "starting",
        "paths": {key: str(value) for key, value in paths.items()},
        "client_python": sys.executable,
        "server_log": str(log_path),
        "hardware": "none; recorded observations and RPC only",
        "timeout_s": TOTAL_TIMEOUT_S,
    }
    process = reader = policy = None
    connections = []
    try:
        dataset = Pi05Dataset(paths["MANIFEST"], split="train")
        expected = dataset[0]
        metadata, observation, episode_path = _recorded_frame(dataset)
        report["episode"] = str(episode_path)
        port = _ephemeral_port()
        report["port"] = port
        command = [
            str(paths["PYTHON"]),
            "-u",
            str(REPO / "pi05/scripts/serve_pi05.py"),
            "--checkpoint-path",
            str(paths["CHECKPOINT"]),
            "--openpi-root",
            str(paths["OPENPI_ROOT"]),
            "--num-inference-steps",
            "2",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        _remaining(deadline, 1)
        server_environment, removed_environment = _server_environment()
        report["server_environment_removed_client_paths"] = removed_environment
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=server_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        report["owned_server_pid"] = process.pid
        reader = _ServerLog(process.stdout, log_path)
        _wait_ready(process, port, deadline, log_path)
        report["server_ready_elapsed_s"] = time.monotonic() - began
        url = f"ws://127.0.0.1:{port}"

        def audited_factory(address, *, timeout_s):
            wrapped = _AuditedRPC(WebSocketRPC(address, timeout_s=timeout_s), expected)
            connections.append(wrapped)
            return wrapped

        policy = Pi05RemotePolicy.connect(
            url,
            metadata,
            timeout_s=_remaining(deadline, 30),
            task_description=expected["prompt"],
            rpc_factory=audited_factory,
        )
        server = policy.server_metadata
        # A raced/occupied port must never be mistaken for the owned checkpoint.
        assert Path(server["checkpoint"]).resolve() == paths["CHECKPOINT"]
        assert (
            server["manifest"]["data_identity_sha256"]
            == dataset.manifest["data_identity_sha256"]
        )
        predicted = policy.predict(observation)
        for name in ("camera_actions", "base_actions"):
            values = np.asarray(predicted[name])
            assert values.shape == (100, 14) and np.isfinite(values).all()
        plan = resample_action_block(
            predicted["base_actions"], policy.timing, 30
        ).to_report()
        assert (
            plan["full_point_count"] == 30
        ), "This integration case expects the 30 Hz / 30-row training recipe"
        assert policy.timing.effective_span_s == pytest.approx(29 / 30)
        assert connections[0].requests == 1
        report.update(
            request_parity=connections[0].parity,
            server_manifest_sha256=server["manifest"]["manifest_sha256"],
            data_identity_sha256=server["manifest"]["data_identity_sha256"],
            num_inference_steps=predicted["inference_steps"],
            rpc_inference_elapsed_s=predicted["inference_elapsed_s"],
            action_shape=list(predicted["base_actions"].shape),
            action_timing=policy.timing.to_dict(),
            resampling=plan,
        )
        policy.close()
        policy = None

        wrong = copy.deepcopy(metadata)
        wrong["extrinsics"]["left"][0][3] += 1000.0
        wrong["calibration"]["data"]["extrinsics"] = copy.deepcopy(wrong["extrinsics"])
        with pytest.raises(ValueError, match="no matching calibration"):
            Pi05RemotePolicy.connect(
                url,
                wrong,
                timeout_s=_remaining(deadline, 30),
                task_description=expected["prompt"],
                rpc_factory=audited_factory,
            )
        assert connections[-1].closed and connections[-1].requests == 0
        assert "jax" not in sys.modules or jax_before, "The robot client imported JAX"
        report.update(bad_calibration_rejected_before_inference=True, status="passed")
    except BaseException as exc:
        report.update(
            status="failed",
            error={"type": type(exc).__name__, "message": str(exc)[:3000]},
        )
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        cleanup_errors = []
        if policy is not None:
            try:
                policy.close()
            except Exception as exc:
                cleanup_errors.append(f"policy close: {exc}")
        for connection in connections:
            try:
                connection.close()
            except Exception as exc:
                cleanup_errors.append(f"RPC close: {exc}")
        # Only this Popen object's process is signalled; never use pkill/killall.
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            except Exception as exc:
                cleanup_errors.append(f"owned server cleanup: {exc}")
            report["owned_server_returncode"] = process.returncode
        if reader is not None:
            reader.finish()
            report["server_log_bytes_seen"] = reader.bytes_read
            report["server_log_truncated"] = reader.truncated
            report["server_log_error"] = reader.error
        if cleanup_errors:
            report.update(status="failed", cleanup_errors=cleanup_errors)
        report["elapsed_s"] = time.monotonic() - began
        report["client_imported_jax"] = "jax" in sys.modules and not jax_before
        if report["status"] == "failed" and log_path.exists():
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size - 3000))
                report["server_log_tail"] = stream.read().decode(
                    "utf-8", errors="replace"
                )
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Pi0.5 server/client integration report: {report_path}", flush=True)
        if cleanup_errors and not active_error:
            pytest.fail("Pi0.5 integration cleanup failed; see " + str(report_path))
