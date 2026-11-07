"""Latency measurements reuse capture modes but never subtract old compensation."""

import json
from types import SimpleNamespace

import pytest

from egomimic.robot.piper_pico import camera_latency as latency


@pytest.fixture
def rig(tmp_path):
    spec = {
        "name": "front_1",
        "type": "uvc",
        "device": "/dev/fake-front",
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "fourcc": "MJPG",
        "latency": 0.123,
        "latency_status": "measured",
        "controls": {"focus_automatic_continuous": 0, "focus_absolute": 0},
        "expected_identity": {"ID_SERIAL": "fake-front"},
    }
    path = tmp_path / "rig.json"
    path.write_text(json.dumps({"schema": "camera_rig_v1", "cameras": [spec]}))
    return path, spec


def test_cli_selects_native_front_settings_and_keeps_original_compensation_in_report(
    monkeypatch, tmp_path, rig
):
    path, spec = rig
    made, reports = [], []
    monkeypatch.setattr(latency, "UvcFeed", lambda **kw: made.append(kw))
    monkeypatch.setattr(latency, "run_measurement", lambda *a: reports.append(a[-1]))
    latency.main(
        [
            "--output",
            str(tmp_path / "report"),
            "uvc",
            "--camera-config",
            str(path),
            "--camera-name",
            "front_1",
        ]
    )
    assert made[0]["width"] == 1920
    assert made[0]["height"] == 1080
    expected = {**spec, "controls": {"brightness": 48, **spec["controls"]}}
    assert made[0]["camera_spec"] == expected
    assert reports[0]["capture_camera_spec"] == expected
    assert reports[0]["measurement_receive_latency_s"] == 0
    assert reports[0]["display_latency_sec"] == 0


def test_worker_receives_verified_settings_with_zero_measurement_compensation(
    monkeypatch, rig
):
    _, spec = rig
    made = []
    process = SimpleNamespace(ring=object(), start=lambda: None)
    monkeypatch.setattr(
        latency, "CameraProcess", lambda *a, **kw: made.append(kw) or process
    )
    feed = latency.UvcFeed(
        spec["device"], 1920, 1080, 30, "MJPG", 60, 10, camera_spec=spec
    )
    feed._open()
    assert made[0]["latency"] == 0
    assert made[0]["controls"] == spec["controls"]
    assert made[0]["expected_identity"] == spec["expected_identity"]
    assert spec["latency"] == 0.123


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--camera-name", "absent"],
        ["--camera-name", "front_1", "--width", "1280"],
        ["--camera-name", "front_1", "--device", "/dev/other"],
    ],
)
def test_invalid_selection_rejected_before_hardware(monkeypatch, rig, extra):
    monkeypatch.setattr(
        latency, "UvcFeed", lambda **kw: pytest.fail("hardware allocation")
    )
    with pytest.raises(SystemExit) as exc:
        latency.main(["uvc", "--camera-config", str(rig[0]), *extra])
    assert exc.value.code == 2


def test_camera_name_without_config_rejected():
    with pytest.raises(SystemExit):
        latency.main(["uvc", "--camera-name", "front_1"])
