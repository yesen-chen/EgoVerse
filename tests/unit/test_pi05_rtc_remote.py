"""RTC wire/geometry/entry-point regression without any hardware or GPU."""

# ruff: noqa: F811 -- imported pytest fixtures are injected by parameter name.

import copy

import numpy as np
import pytest
from test_pi05_remote import (  # noqa: F401
    FakeRPC,
    cli_args,
    frame,
    loopback_server,
    metadata_files,
    runtime,
    server_metadata,
)

from egomimic.robot.piper_pico import live_policy as live
from egomimic.robot.piper_pico.pi05_remote import Pi05RemotePolicy


def prefix(observation):
    return {
        "schema": "egoverse_rtc_v1",
        "request_id": 7,
        "base_actions": np.repeat(observation.base_state[None], 100, axis=0),
        "weights": np.linspace(1, 0, 100),
    }


class RTCFakeRPC(FakeRPC):
    def infer(self, request):
        result = super().infer(request)
        if "rtc" in request:
            result["actions"] = request["rtc"]["previous_actions"]
            result["rtc"] = {
                "schema": "egoverse_rtc_v1",
                "request_id": request["rtc"]["request_id"],
            }
        return result


def rtc_policy(runtime, rpc_class=RTCFakeRPC):
    metadata = server_metadata(runtime["metadata"])
    metadata["rtc"] = {"schema": "egoverse_rtc_v1", "max_guidance": 5.0}
    rpc = rpc_class(metadata)
    return Pi05RemotePolicy(rpc, runtime["metadata"]), rpc


def test_rtc_prefix_uses_same_nonidentity_calibration_and_gripper_units(runtime):
    policy, rpc = rtc_policy(runtime)
    obs = frame()
    previous = prefix(obs)
    before = copy.deepcopy(previous)
    prediction = policy.predict(obs, rtc=previous)
    request = rpc.requests[-1]
    # State and constant prefix describe the same physical pose in different
    # frames; catches accidentally forwarding base rotvec or width in metres.
    np.testing.assert_allclose(
        request["rtc"]["previous_actions"],
        np.repeat(request["observation.state"][None], 100, axis=0),
    )
    np.testing.assert_allclose(
        prediction["base_actions"], previous["base_actions"], atol=2e-7
    )
    np.testing.assert_array_equal(previous["base_actions"], before["base_actions"])
    np.testing.assert_array_equal(previous["weights"], before["weights"])
    assert prediction["rtc"]["request_id"] == 7
    assert request["rtc"]["previous_actions"].dtype == np.float32


@pytest.mark.parametrize(
    "capability",
    [
        None,
        {},
        {"schema": "old", "max_guidance": 5},
        {"schema": "egoverse_rtc_v1", "max_guidance": float("nan")},
    ],
)
def test_rtc_requires_server_capability(runtime, capability):
    policy, rpc = rtc_policy(runtime)
    policy.server_metadata["rtc"] = capability
    with pytest.raises(ValueError, match="server started with --rtc"):
        policy.require_rtc()
    # Baseline inference remains compatible with ordinary servers.
    assert policy.predict(frame())["base_actions"].shape == (100, 14)
    assert "rtc" not in rpc.requests[-1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_actions", np.zeros((30, 14))),
        ("base_actions", np.full((100, 14), np.nan)),
        ("weights", np.ones(30)),
        ("weights", np.full(100, 1.1)),
        ("weights", np.full(100, np.nan)),
        ("request_id", True),
        ("request_id", -1),
        ("schema", "old"),
    ],
)
def test_bad_prefix_is_never_sent(runtime, field, value):
    policy, rpc = rtc_policy(runtime)
    data = prefix(frame())
    data[field] = value
    with pytest.raises(ValueError):
        policy.predict(frame(), rtc=data)
    assert not rpc.requests and rpc.closed


@pytest.mark.parametrize(
    "receipt",
    [
        None,
        {"schema": "egoverse_rtc_v1", "request_id": 6},
        {"schema": "egoverse_rtc_v1", "request_id": True},
    ],
)
def test_mismatched_reply_fails_closed(runtime, receipt):
    class WrongRPC(RTCFakeRPC):
        def infer(self, request):
            result = super().infer(request)
            result["rtc"] = receipt
            return result

    policy, rpc = rtc_policy(runtime, WrongRPC)
    with pytest.raises(ValueError, match="request_id"):
        policy.predict(frame(), rtc=prefix(frame()))
    assert rpc.closed


def test_actuator_wrapper_forwards_rtc_and_preserves_reply(runtime):
    policy, rpc = rtc_policy(runtime)

    class Log:
        def event(self, *a, **kw):
            pytest.fail("in-range gripper was unexpectedly clipped")

    wrapped = live.ActuatorPolicy(policy, policy.adapter.gripper_open_m, Log())
    prediction = wrapped.predict(frame(), rtc=prefix(frame()))
    assert prediction["rtc"]["request_id"] == 7
    assert rpc.requests[-1]["rtc"]["request_id"] == 7


@pytest.mark.parametrize(
    "backend,streaming,match", [("hpt", False, "Pi0.5"), ("pi05", True, "不能同时")]
)
def test_rtc_incompatible_modes_rejected_before_io(backend, streaming, match):
    args = cli_args(backend)
    args.rtc, args.async_inference = True, streaming
    with pytest.raises(ValueError, match=match):
        live.load_policy_contract(args, object())


def test_rtc_disabled_by_default():
    args = cli_args()
    assert not args.rtc and args.rtc_delay_ms == 200


def test_rtc_numpy_prefix_survives_actual_websocket_roundtrip(runtime):
    metadata = server_metadata(runtime["metadata"])
    metadata["rtc"] = {"schema": "egoverse_rtc_v1", "max_guidance": 5.0}
    obs = frame()
    data = prefix(obs)
    with loopback_server(metadata) as (url, requests):
        policy = Pi05RemotePolicy.connect(url, runtime["metadata"], timeout_s=2)
        try:
            result = policy.predict(obs, rtc=data)
            np.testing.assert_allclose(
                result["base_actions"], data["base_actions"], atol=2e-7
            )
            np.testing.assert_allclose(requests[0]["rtc"]["weights"], data["weights"])
            assert result["rtc"]["request_id"] == data["request_id"]
        finally:
            policy.close()


def test_rtc_missing_capability_rejected_before_devices(runtime, monkeypatch):
    import egomimic.robot.piper_pico.live_policy_contract as contracts

    monkeypatch.setattr(contracts, "prepare_live_contract", lambda *a: runtime)
    rpc = FakeRPC(server_metadata(runtime["metadata"]))
    monkeypatch.setattr(
        "egomimic.robot.piper_pico.pi05_remote.WebSocketRPC", lambda *a, **kw: rpc
    )
    monkeypatch.setattr(live.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        live, "_run_device_session", lambda *a: pytest.fail("devices opened")
    )
    args = cli_args()
    args.rtc, args.execute_steps = True, 12
    with pytest.raises(ValueError, match="server started with --rtc"):
        live.run_session(args, object())
    assert rpc.closed


def test_enabled_rtc_contract_records_mode_without_changing_100_point_timing(runtime):
    policy, _ = rtc_policy(runtime)
    args = cli_args()
    args.rtc, args.execute_steps = True, 12

    class Log:
        def write_manifest(self, value):
            self.manifest = value

    log = Log()
    live._finish_policy_contract(args, log, runtime, policy, {"policy_backend": "pi05"})
    assert log.manifest["rtc"]["rtc_delay_s"] == pytest.approx(0.2)
    assert log.manifest["streaming"] is None
    assert log.manifest["timing_mode"].startswith("rtc;")
    assert log.manifest["action_timing"]["output_chunk_length"] == 100
    assert log.manifest["schedule"]["full_point_count"] == 30
    args.execute_steps = 30
    with pytest.raises(ValueError, match="RTC needs"):
        live._finish_policy_contract(
            args, log, runtime, policy, {"policy_backend": "pi05"}
        )


def test_rtc_pause_counts_survive_disabled_detailed_logging(tmp_path):
    args = cli_args()
    args.log = False
    log = live.SessionLog(tmp_path / "run", args)
    try:
        for reason in ("rtc_deadline_missed", "rtc_deadline_missed", "schedule_slip"):
            log.cycle(
                {"status": "paused", "timing_mode": "rtc", "pause_reason": reason}
            )
        assert log.summary["rtc_pause_count"] == 3
        assert log.summary["rtc_pause_reasons"]["rtc_deadline_missed"] == 2
        assert log.summary["schedule_pause_count"] == 1
    finally:
        log.close()
    assert not list(log.directory.glob("*.jsonl"))
