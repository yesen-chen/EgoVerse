"""Synchronous OpenPI-compatible Pi0.5 RPC, without a JAX client dependency.

The server sends metadata immediately after the WebSocket connection opens.
Requests and responses use OpenPI's msgpack NumPy representation; no pickle or
automatic reconnect is used. Runtime calibration is checked before devices open.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Mapping
from urllib.parse import urlsplit

import numpy as np

from egomimic.rldb.action_timing import ActionTiming
from egomimic.rldb.zarr.piper_dataset import validate_piper_metadata
from egomimic.robot.piper_pico.live_policy_contract import _profile, _same
from egomimic.robot.piper_pico.policy_inference import (
    CAMERA_KEYS,
    EMBODIMENT,
    STATE,
    PiperObservationAdapter,
)

MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def _pack_numpy(value):
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.kind not in "biuf":
            raise ValueError(f"Unsupported wire dtype: {value.dtype}")
        if isinstance(value, np.ndarray):
            return {
                b"__ndarray__": True,
                b"data": value.tobytes(),
                b"dtype": value.dtype.str,
                b"shape": value.shape,
            }
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"Unsupported wire value: {type(value).__name__}")


def _unpack_numpy(value):
    if b"__ndarray__" not in value and b"__npgeneric__" not in value:
        return value
    dtype = np.dtype(value[b"dtype"])
    if dtype.kind not in "biuf":
        raise ValueError(f"Unsupported wire dtype: {dtype}")
    if b"__npgeneric__" in value:
        return dtype.type(value[b"data"])
    shape, data = value[b"shape"], value[b"data"]
    if (
        not isinstance(shape, (list, tuple))
        or any(type(size) is not int or size < 0 for size in shape)
        or not isinstance(data, bytes)
        or math.prod(shape) * dtype.itemsize != len(data)
    ):
        raise ValueError("Malformed wire ndarray shape or byte length")
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def pack_message(value):
    import msgpack

    return msgpack.packb(value, default=_pack_numpy, use_bin_type=True)


def unpack_message(payload):
    import msgpack

    if isinstance(payload, str):
        raise RuntimeError(f"Pi0.5 server error: {payload[:2000]}")
    if (
        not isinstance(payload, bytes)
        or not payload
        or len(payload) > MAX_MESSAGE_BYTES
    ):
        raise ValueError("Pi0.5 expected a nonempty bounded binary msgpack message")
    result = msgpack.unpackb(payload, object_hook=_unpack_numpy, raw=False)
    if not isinstance(result, dict):
        raise ValueError("Pi0.5 message must be a mapping")
    if "error" in result:
        raise RuntimeError(f"Pi0.5 server error: {str(result['error'])[:2000]}")
    return result


class WebSocketRPC:
    """A finite-timeout connection using the existing websocket-client library."""

    def __init__(self, url, *, timeout_s=10.0):
        parts = urlsplit(url)
        if parts.scheme not in ("ws", "wss") or not parts.hostname:
            raise ValueError("--server-url must be an explicit ws:// or wss:// URL")
        if (
            isinstance(timeout_s, bool)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("--server-timeout must be finite and positive")
        self.url, self.timeout_s, self.socket = url, float(timeout_s), None
        try:
            import websocket
        except ImportError as exc:
            raise RuntimeError(
                "Pi0.5 RPC requires websocket-client and msgpack in the client environment; "
                "no JAX or OpenPI installation is needed on the robot client."
            ) from exc
        try:
            self.socket = websocket.create_connection(
                url, timeout=self.timeout_s, enable_multithread=True
            )
            self.metadata = unpack_message(self.socket.recv())
        except BaseException:
            self.close()
            raise

    def infer(self, observation):
        if self.socket is None:
            raise RuntimeError("Pi0.5 connection is closed; restart the session")
        try:
            self.socket.send_binary(pack_message(observation))
            return unpack_message(self.socket.recv())
        except BaseException:
            self.close()
            raise

    def close(self):
        connection, self.socket = self.socket, None
        if connection is not None:
            # Never let a peer that stopped responding delay local cleanup.
            try:
                connection.close(timeout=min(self.timeout_s, 1.0))
            except Exception:
                connection.shutdown()


def _server_contract(server, metadata):
    from egomimic.robot.piper_pico.pi05_data import ACTION_CONTRACT, validate_manifest

    if not isinstance(server, Mapping):
        raise ValueError("Pi0.5 server metadata must be a mapping")
    for key, expected in {
        "protocol_version": 1,
        "policy_type": "pi05",
        "action_dim": 14,
        "action_horizon": 100,
    }.items():
        if type(server.get(key)) is not type(expected) or server[key] != expected:
            raise ValueError(f"Pi0.5 server {key} must be {expected!r}")
    if server.get("action_contract") != ACTION_CONTRACT:
        raise ValueError("Pi0.5 server action representation/units disagree")
    manifest = validate_manifest(copy.deepcopy(server.get("manifest")))
    for key in ("action_timing", "training_contracts", "image_shapes"):
        if key not in server:
            raise ValueError(f"Pi0.5 server lacks {key}")
        _same(server[key], manifest[key], f"server.{key} vs training manifest")
    values = server["action_timing"]
    timing = ActionTiming(
        **{
            name: values[name]
            for name in (
                "source_fps",
                "source_horizon",
                "action_stride",
                "output_chunk_length",
                "includes_current",
            )
        }
    )
    _same(values, timing.to_dict(), "Pi0.5 action_timing")
    if timing.output_chunk_length != 100:
        raise ValueError("Pi0.5 requires 100 predicted actions")
    if not math.isclose(timing.source_fps, metadata["fps"], abs_tol=1e-9, rel_tol=0):
        raise ValueError("Pi0.5 source_fps differs from selected training data")
    steps = server.get("num_inference_steps")
    if type(steps) is not int or steps < 1:
        raise ValueError("Pi0.5 server num_inference_steps must be a positive integer")
    contracts = server["training_contracts"]
    if not isinstance(contracts, list) or not contracts:
        raise ValueError("Pi0.5 training_contracts must be a nonempty list")
    mismatches = []
    for contract in contracts:
        try:
            if not isinstance(contract, Mapping):
                raise ValueError("training contract must be a mapping")
            candidate = copy.deepcopy(metadata)
            candidate.update(copy.deepcopy(contract))
            if candidate["calibration"]["data"].get(
                "translation_unit", "metre"
            ) not in (
                "metre",
                "meter",
                "m",
            ):
                raise ValueError("Pi0.5 training calibration must use metres")
            validate_piper_metadata(candidate)
            _same(
                _profile(candidate), _profile(metadata), "Pi0.5 training rig geometry"
            )
            _same(
                candidate.get("tool_frame"),
                metadata.get("tool_frame"),
                "Pi0.5 tool_frame",
            )
        except (ValueError, TypeError, KeyError) as exc:
            mismatches.append(str(exc))
        else:
            break
    else:
        raise ValueError(
            "Pi0.5 server training manifest has no matching calibration/image/gripper "
            f"contract for this rig: {mismatches[0]}"
        )
    expected_shapes = {
        name: metadata["features"][f"images.{name}"]["shape"] for name in CAMERA_KEYS
    }
    _same(server["image_shapes"], expected_shapes, "Pi0.5 processed image shapes")
    return timing, steps


class Pi05RemotePolicy:
    """Physical 14-D camera-frame actions from a validated remote Pi0.5 model."""

    def __init__(self, rpc, metadata, *, task_description=None):
        self.rpc = rpc
        try:
            self.server_metadata = copy.deepcopy(rpc.metadata)
            self.timing, self.inference_steps = _server_contract(
                self.server_metadata, metadata
            )
            prompt = (
                metadata.get("task_description")
                if task_description is None
                else task_description
            )
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    "Pi0.5 needs a nonempty --task-description or training task_description"
                )
            self.prompt = prompt.strip()
            self.adapter = PiperObservationAdapter(metadata)
            self.observation_to_send_s = self.adapter.observation_to_send_s
            self.checkpoint_path = None
            self.gripper_contract = {
                "open_m": self.adapter.gripper_open_m,
                "source": "server_training_manifest_checked_against_runtime_metadata",
                "checkpoint_open_m_verified": True,
            }
        except BaseException:
            self.close()
            raise

    @classmethod
    def connect(
        cls, url, metadata, *, timeout_s=10.0, task_description=None, rpc_factory=None
    ):
        rpc = (rpc_factory or WebSocketRPC)(url, timeout_s=timeout_s)
        return cls(rpc, metadata, task_description=task_description)

    def require_rtc(self):
        """Check server capability before opening any robot devices."""
        capability = self.server_metadata.get("rtc")
        if (
            not isinstance(capability, Mapping)
            or capability.get("schema") != "egoverse_rtc_v1"
            or isinstance(capability.get("max_guidance"), bool)
            or not isinstance(capability.get("max_guidance"), (int, float))
            or not math.isfinite(capability["max_guidance"])
            or capability["max_guidance"] <= 0
        ):
            raise ValueError(
                "RTC requires a compatible Pi0.5 server started with --rtc; "
                "restart the server with RTC enabled."
            )
        return copy.deepcopy(capability)

    def _rtc_request(self, rtc):
        self.require_rtc()
        expected = {"schema", "base_actions", "weights", "request_id"}
        if not isinstance(rtc, Mapping) or set(rtc) != expected:
            raise ValueError(f"RTC client prefix requires exactly {sorted(expected)}")
        if rtc["schema"] != "egoverse_rtc_v1":
            raise ValueError("Unsupported RTC prefix schema")
        if type(rtc["request_id"]) is not int or rtc["request_id"] < 0:
            raise ValueError("RTC request_id must be a nonnegative integer")
        previous = np.asarray(rtc["base_actions"])
        weights = np.asarray(rtc["weights"])
        if (
            previous.dtype.kind not in "iuf"
            or previous.shape != (100, 14)
            or not np.isfinite(previous).all()
            or weights.dtype.kind not in "iuf"
            or weights.shape != (100,)
            or not np.isfinite(weights).all()
            or np.any((weights < 0) | (weights > 1))
        ):
            raise ValueError(
                "RTC needs finite (100,14) base actions and 100 weights in [0,1]"
            )
        # Prefixes use the same calibrated camera XYZ/YPR and normalized-width
        # representation as training actions. Quantile normalization stays on
        # the server; a base rotvec or a metre width must never enter that model.
        camera = np.stack(
            [self.adapter.base_to_camera_state(row) for row in previous]
        ).astype(np.float32)
        return {
            "schema": rtc["schema"],
            "request_id": rtc["request_id"],
            "previous_actions": camera,
            "weights": weights.astype(np.float32, copy=True),
        }

    def predict(self, frame, *, rtc=None):
        if self.rpc is None:
            raise RuntimeError("Pi0.5 connection is closed; restart the session")
        try:
            batch = self.adapter.prepare(frame)[EMBODIMENT]
            request = {
                "observation.state": batch[STATE][0].numpy(),
                "prompt": self.prompt,
            }
            for name, key in CAMERA_KEYS.items():
                request[f"observation.images.{name}"] = np.ascontiguousarray(
                    batch[key][0].numpy().transpose(1, 2, 0)
                )
            if rtc is not None:
                request["rtc"] = self._rtc_request(rtc)
            started = time.perf_counter()
            response = self.rpc.infer(request)
            if not isinstance(response, Mapping) or "error" in response:
                raise ValueError("Malformed Pi0.5 inference response")
            if rtc is not None:
                receipt = response.get("rtc")
                if (
                    not isinstance(receipt, Mapping)
                    or receipt.get("schema") != "egoverse_rtc_v1"
                    or type(receipt.get("request_id")) is not int
                    or receipt["request_id"] != rtc["request_id"]
                ):
                    raise ValueError("Missing or mismatched RTC response request_id")
            actions = np.asarray(response.get("actions"))
            if (
                actions.dtype.kind not in "iuf"
                or actions.shape != (100, 14)
                or not np.isfinite(actions).all()
            ):
                raise ValueError(
                    "Pi0.5 actions must be finite physical values with shape (100,14)"
                )
            camera = actions.copy()
            base = self.adapter.camera_to_base_actions(camera)
            return {
                "camera_actions": camera,
                "base_actions": base,
                "observation_timestamp": frame.timestamp,
                "inference_elapsed_s": time.perf_counter() - started,
                "inference_steps": self.inference_steps,
                **({"rtc": copy.deepcopy(response["rtc"])} if rtc is not None else {}),
            }
        except BaseException:
            self.close()
            raise

    def close(self):
        rpc, self.rpc = self.rpc, None
        if rpc is not None:
            rpc.close()
