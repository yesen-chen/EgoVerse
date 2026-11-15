"""Pi0.5 RPC/geometry parity with fake policies and an optional local WebSocket."""

from __future__ import annotations

import asyncio
import copy
import queue
import threading
from contextlib import contextmanager

import msgpack
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_live_policy_contract import fixture as metadata_files  # noqa: F401
from test_live_policy_contract import prepare

from egomimic.rldb.action_timing import ActionTiming
from egomimic.robot.piper_pico import live_policy as live
from egomimic.robot.piper_pico.pi05_data import ACTION_CONTRACT, canonical_hash
from egomimic.robot.piper_pico.pi05_remote import (
    Pi05RemotePolicy,
    WebSocketRPC,
    pack_message,
    unpack_message,
)
from egomimic.robot.piper_pico.policy_execution import resample_action_block
from egomimic.robot.piper_pico.policy_inference import (
    CAMERA_KEYS,
    EMBODIMENT,
    SOURCE_NAMES,
    STATE,
    ObservationFrame,
    PiperObservationAdapter,
)


@pytest.fixture
def runtime(metadata_files):  # noqa: F811
    value = prepare(metadata_files)
    metadata = value["metadata"]
    metadata["task_description"] = "Pick up the object."
    metadata["tool_frame"] = "piper_sdk_end_pose"
    for side, shift in (("left", 0.3), ("right", -0.4)):
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("ZYX", [shift, 0.2, -0.1]).as_matrix()
        transform[:3, 3] = [shift, 0.2, 0.7]
        metadata["extrinsics"][side] = transform.tolist()
    metadata["calibration"]["data"]["extrinsics"] = copy.deepcopy(
        metadata["extrinsics"]
    )
    return value


def server_metadata(metadata, *, profiles=None):
    def contract(value):
        result = {
            key: copy.deepcopy(value.get(key))
            for key in (
                "calibration",
                "calibration_status",
                "extrinsics",
                "intrinsics",
                "gripper_encoding",
                "image_processing",
                "image_space",
                "tool_frame",
            )
        }
        result["calibration"] = {"data": result["calibration"]["data"]}
        return result

    profiles = profiles or [metadata]
    contracts = [contract(value) for value in profiles]
    shapes = {
        name: metadata["features"][f"images.{name}"]["shape"] for name in CAMERA_KEYS
    }
    timing = ActionTiming(30, 30, 1, 100).to_dict()
    records = []
    for index, candidate in enumerate(contracts):
        records.append(
            {
                "path": f"recording/episode_{index:06d}.zarr",
                "split": "train",
                "total_frames": 60,
                "prompt": "Pick up the object.",
                "image_shapes": shapes,
                "metadata_sha256": f"metadata-{index}",
                "numeric_sha256": f"numeric-{index}",
                "contract": {
                    **copy.deepcopy(candidate),
                    "time_alignment": {"observation_to_send_s": 0.005},
                    "source_capture_config": metadata["source_capture_config"],
                },
            }
        )
    manifest = {
        "schema": "egoverse_pi05_manifest_v1",
        "action_timing": timing,
        "action_contract": copy.deepcopy(ACTION_CONTRACT),
        "training_contract": contracts[0],
        "training_contracts": contracts,
        "image_shapes": shapes,
        "episodes": records,
        "allow_synthetic_calibration": False,
        "data_identity_sha256": canonical_hash(
            [
                {
                    key: item[key]
                    for key in ("metadata_sha256", "numeric_sha256", "total_frames")
                }
                for item in records
            ]
        ),
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    return {
        "protocol_version": 1,
        "policy_type": "pi05",
        "action_dim": 14,
        "action_horizon": 100,
        "action_timing": timing,
        "action_contract": copy.deepcopy(ACTION_CONTRACT),
        "training_contracts": contracts,
        "image_shapes": shapes,
        "num_inference_steps": 10,
        "manifest": manifest,
    }


class FakeRPC:
    def __init__(self, metadata, response=None):
        self.metadata, self.response = metadata, response
        self.requests, self.closed = [], False

    def infer(self, request):
        self.requests.append(request)
        if isinstance(self.response, BaseException):
            raise self.response
        if self.response is not None:
            return self.response
        return {"actions": np.repeat(request["observation.state"][None], 100, axis=0)}

    def close(self):
        self.closed = True


def frame():
    state = np.array(
        [
            0.2,
            0.1,
            0.4,
            0.4,
            -0.2,
            0.1,
            0.04,
            -0.1,
            0.5,
            0.3,
            -0.5,
            0.2,
            0.3,
            0.06,
        ]
    )
    rng = np.random.default_rng(14)
    return ObservationFrame(
        1000.0,
        {
            name: rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            for name in CAMERA_KEYS
        },
        state,
        {name: 1000.0 for name in SOURCE_NAMES},
    )


def test_request_matches_existing_adapter_and_actions_inverse_roundtrip(runtime):
    metadata = runtime["metadata"]
    rpc = FakeRPC(server_metadata(metadata))
    policy = Pi05RemotePolicy(rpc, metadata)
    observation = frame()
    prediction = policy.predict(observation)
    prepared = PiperObservationAdapter(metadata).prepare(observation)[EMBODIMENT]
    request = rpc.requests[0]
    assert request["prompt"] == metadata["task_description"]
    np.testing.assert_array_equal(
        request["observation.state"], prepared[STATE][0].numpy()
    )
    assert request["observation.state"].shape == (14,)
    assert request["observation.state"].dtype == np.float32
    for name, key in CAMERA_KEYS.items():
        actual = request[f"observation.images.{name}"]
        assert actual.dtype == np.uint8 and actual.shape == (16, 16, 3)
        np.testing.assert_array_equal(
            actual, prepared[key][0].numpy().transpose(1, 2, 0)
        )
    np.testing.assert_allclose(
        prediction["base_actions"],
        np.repeat(observation.base_state[None], 100, axis=0),
        atol=2e-7,
    )
    assert prediction["camera_actions"].shape == (100, 14)
    plan = resample_action_block(
        prediction["base_actions"], policy.timing, 30
    ).to_report()
    assert plan["full_point_count"] == 30
    assert policy.timing.effective_span_s == pytest.approx(29 / 30)
    policy.close()
    assert rpc.closed


def test_processed_images_are_not_rectified_twice_and_prompt_override(runtime):
    metadata = runtime["metadata"]
    first = frame()
    adapter = PiperObservationAdapter(metadata)
    images = {
        name: adapter.image_processors[name].apply(
            first.images[name], image_space="raw"
        )
        for name in CAMERA_KEYS
    }
    processed = ObservationFrame(
        first.timestamp,
        images,
        first.base_state,
        first.source_timestamps,
        image_space="processed",
    )
    rpc = FakeRPC(server_metadata(metadata))
    policy = Pi05RemotePolicy(rpc, metadata, task_description="  Place in the bin.  ")
    policy.predict(processed)
    assert rpc.requests[0]["prompt"] == "Place in the bin."
    for name in CAMERA_KEYS:
        np.testing.assert_array_equal(
            rpc.requests[0][f"observation.images.{name}"], images[name]
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("protocol_version", 2),
        ("protocol_version", True),
        ("policy_type", "other"),
        ("action_dim", 32),
        ("action_horizon", 50),
        ("num_inference_steps", 0),
    ],
)
def test_bad_handshake_closes_before_inference(runtime, key, value):
    metadata = runtime["metadata"]
    remote = server_metadata(metadata)
    remote[key] = value
    rpc = FakeRPC(remote)
    with pytest.raises(ValueError):
        Pi05RemotePolicy(rpc, metadata)
    assert rpc.closed and not rpc.requests


def test_manifest_must_match_current_rig_but_may_contain_other_rigs(runtime):
    metadata = runtime["metadata"]
    other = copy.deepcopy(metadata)
    other["extrinsics"]["left"][0][3] += 0.15
    other["calibration"]["data"]["extrinsics"] = copy.deepcopy(other["extrinsics"])
    wrong = FakeRPC(server_metadata(metadata, profiles=[other]))
    with pytest.raises(ValueError, match="no matching calibration"):
        Pi05RemotePolicy(wrong, metadata)
    assert wrong.closed
    compatible = FakeRPC(server_metadata(metadata, profiles=[other, metadata]))
    policy = Pi05RemotePolicy(compatible, metadata)
    assert not compatible.closed
    policy.close()


def test_manifest_checksum_and_declared_timing_cannot_be_bypassed(runtime):
    metadata = runtime["metadata"]
    remote = server_metadata(metadata)
    remote["manifest"]["manifest_sha256"] = "wrong"
    rpc = FakeRPC(remote)
    with pytest.raises(ValueError, match="checksum"):
        Pi05RemotePolicy(rpc, metadata)
    assert rpc.closed
    remote = server_metadata(metadata)
    remote["action_timing"] = ActionTiming(30, 45, 1, 100).to_dict()
    rpc = FakeRPC(remote)
    with pytest.raises(ValueError, match="manifest"):
        Pi05RemotePolicy(rpc, metadata)
    assert rpc.closed


def test_server_training_calibration_units_are_checked(runtime):
    metadata = runtime["metadata"]
    other = copy.deepcopy(metadata)
    other["calibration"]["data"]["translation_unit"] = "millimetre"
    rpc = FakeRPC(server_metadata(other))
    with pytest.raises(ValueError, match="metres"):
        Pi05RemotePolicy(rpc, metadata)
    assert rpc.closed


def test_missing_prompt_rejected_and_report_only_calibration_changes_allowed(runtime):
    metadata = runtime["metadata"]
    rpc = FakeRPC(server_metadata(metadata))
    with pytest.raises(ValueError, match="nonempty"):
        Pi05RemotePolicy(rpc, metadata, task_description=" ")
    assert rpc.closed
    other = copy.deepcopy(metadata)
    other["calibration"]["data"]["notes"] = "Report reworded without geometry changes"
    other["calibration"]["data"]["calibration_id"] = "renamed-equivalent"
    Pi05RemotePolicy(FakeRPC(server_metadata(other)), metadata).close()


@pytest.mark.parametrize(
    "response",
    [
        {"actions": np.zeros((50, 14))},
        {"actions": np.zeros((100, 32))},
        {"actions": np.full((100, 14), np.nan)},
        {"actions": np.full((100, 14), np.inf)},
        {"actions": np.zeros((100, 14), dtype=bool)},
        {"error": "server failed"},
        {},
        TimeoutError("test receive timeout"),
    ],
)
def test_bad_response_or_timeout_closes_connection(runtime, response):
    rpc = FakeRPC(server_metadata(runtime["metadata"]), response=response)
    policy = Pi05RemotePolicy(rpc, runtime["metadata"])
    with pytest.raises((ValueError, TimeoutError)):
        policy.predict(frame())
    assert rpc.closed
    with pytest.raises(RuntimeError, match="closed"):
        policy.predict(frame())


def test_numpy_wire_format_is_compatible_and_rejects_object_dtype():
    value = np.arange(24, dtype=np.float32).reshape(2, 12)
    wire = pack_message({"actions": value, "steps": np.int64(10)})
    encoded = msgpack.unpackb(wire, raw=False)
    assert encoded["actions"][b"__ndarray__"] is True
    assert encoded["actions"][b"dtype"] == value.dtype.str
    assert encoded["actions"][b"shape"] == [2, 12]
    assert encoded["actions"][b"data"] == value.tobytes()
    np.testing.assert_array_equal(unpack_message(wire)["actions"], value)
    with pytest.raises(ValueError, match="dtype"):
        pack_message({"actions": np.array([object()], dtype=object)})
    with pytest.raises(RuntimeError, match="server error"):
        unpack_message("model failed")


def cli_args(backend="pi05"):
    flags = [
        "--policy-backend",
        backend,
        "--calibration",
        "calibration.json",
        "--training-data",
        "data",
        "--camera-config",
        "camera.json",
    ]
    if backend == "pi05":
        flags += ["--server-url", "ws://127.0.0.1:8000"]
    else:
        flags += ["--checkpoint", "model.ckpt"]
    return live.build_parser().parse_args(flags)


def test_parser_keeps_hpt_default_and_pi05_does_not_require_checkpoint():
    args = cli_args()
    assert args.checkpoint is None and args.policy_backend == "pi05"
    assert args.server_timeout == 10.0 and not args.execute
    args = live.build_parser().parse_args(
        [
            "--checkpoint",
            "model.ckpt",
            "--calibration",
            "c.json",
            "--training-data",
            "data",
            "--camera-config",
            "cameras.json",
        ]
    )
    assert args.policy_backend == "hpt" and args.execute_steps == 30


def test_live_rejects_handshake_before_device_construction(runtime, monkeypatch):
    import egomimic.robot.piper_pico.live_policy_contract as contract_module

    monkeypatch.setattr(contract_module, "prepare_live_contract", lambda *a: runtime)
    remote = server_metadata(runtime["metadata"])
    remote["action_horizon"] = 50
    rpc = FakeRPC(remote)
    monkeypatch.setattr(
        "egomimic.robot.piper_pico.pi05_remote.WebSocketRPC", lambda *a, **kw: rpc
    )
    monkeypatch.setattr(live.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        live, "_run_device_session", lambda *a: pytest.fail("device construction")
    )
    with pytest.raises(ValueError, match="action_horizon"):
        live.run_session(cli_args(), object())
    assert rpc.closed


def test_live_closes_remote_when_device_session_fails(runtime, monkeypatch):
    rpc = FakeRPC(server_metadata(runtime["metadata"]))
    policy = Pi05RemotePolicy(rpc, runtime["metadata"])
    monkeypatch.setattr(live.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        live, "load_policy_contract", lambda *a: (runtime, policy, None)
    )

    def fail(*args):
        raise RuntimeError("test startup failed")

    monkeypatch.setattr(live, "_run_device_session", fail)
    with pytest.raises(RuntimeError, match="startup failed"):
        live.run_session(cli_args(), object())
    assert rpc.closed


@contextmanager
def loopback_server(metadata, *, stall=False):
    web = pytest.importorskip("aiohttp.web")
    pytest.importorskip("websocket")
    loop = asyncio.new_event_loop()
    ready = queue.Queue()
    requests = []

    async def handler(request):
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.send_bytes(pack_message(metadata))
        async for item in socket:
            if item.type == web.WSMsgType.BINARY:
                observation = unpack_message(item.data)
                requests.append(observation)
                if stall:
                    await asyncio.sleep(0.3)
                else:
                    result = {
                        "actions": np.repeat(
                            observation["observation.state"][None], 100, axis=0
                        )
                    }
                    if "rtc" in observation:
                        result["actions"] = observation["rtc"]["previous_actions"]
                        result["rtc"] = {
                            "schema": observation["rtc"]["schema"],
                            "request_id": observation["rtc"]["request_id"],
                        }
                    await socket.send_bytes(pack_message(result))
        return socket

    async def setup():
        application = web.Application()
        application.router.add_get("/", handler)
        runner = web.AppRunner(application)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        try:
            await site.start()
        except BaseException as exc:
            ready.put(exc)
            await runner.cleanup()
            return False
        ready.put((runner, site._server.sockets[0].getsockname()[1]))
        return True

    def work():
        asyncio.set_event_loop(loop)
        if loop.run_until_complete(setup()):
            loop.run_forever()

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    result = ready.get(timeout=5)
    if isinstance(result, BaseException):
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()
        raise result
    runner, port = result
    try:
        yield f"ws://127.0.0.1:{port}", requests
    finally:
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(timeout=5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


def test_real_loopback_transport_and_receive_timeout(runtime):
    metadata = runtime["metadata"]
    with loopback_server(server_metadata(metadata)) as (url, requests):
        policy = Pi05RemotePolicy.connect(url, metadata, timeout_s=1)
        prediction = policy.predict(frame())
        assert len(requests) == 1 and prediction["base_actions"].shape == (100, 14)
        policy.close()
    import websocket

    with loopback_server(server_metadata(metadata), stall=True) as (url, requests):
        policy = Pi05RemotePolicy.connect(url, metadata, timeout_s=0.05)
        with pytest.raises(websocket.WebSocketTimeoutException):
            policy.predict(frame())
        assert policy.rpc is None and len(requests) == 1


@pytest.mark.parametrize(
    "url,timeout",
    [("http://localhost", 1), ("ws://localhost", 0), ("ws://localhost", float("inf"))],
)
def test_transport_parameters_rejected_without_connecting(url, timeout):
    with pytest.raises(ValueError):
        WebSocketRPC(url, timeout_s=timeout)


def test_connect_and_handshake_have_a_timeout_and_never_retry(monkeypatch):
    websocket = pytest.importorskip("websocket")
    calls = []

    def refused(url, **kwargs):
        calls.append((url, kwargs))
        raise ConnectionRefusedError("test unavailable server")

    monkeypatch.setattr(websocket, "create_connection", refused)
    with pytest.raises(ConnectionRefusedError):
        WebSocketRPC("ws://127.0.0.1:8000", timeout_s=0.2)
    assert len(calls) == 1 and calls[0][1]["timeout"] == 0.2

    class Socket:
        closed = False

        def recv(self):
            raise TimeoutError("test handshake timeout")

        def close(self, **kwargs):
            self.closed = True

    socket = Socket()
    monkeypatch.setattr(websocket, "create_connection", lambda *a, **kw: socket)
    with pytest.raises(TimeoutError):
        WebSocketRPC("ws://127.0.0.1:8000", timeout_s=0.2)
    assert socket.closed
