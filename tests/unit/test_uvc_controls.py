"""No USB/CAN access: UVC settings must be applied and independently read back."""

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from egomimic.robot.piper_pico import uvc_controls as controls


def calibration(values):
    return {"camera": {"controls": {"available": True, "values": values}}}


def test_legacy_settings_do_not_require_external_tools(monkeypatch):
    run = Mock(side_effect=AssertionError("legacy settings must not probe hardware"))
    monkeypatch.setattr(controls.subprocess, "run", run)
    result = controls.verify_controls("0")
    assert result["status"] == "not_requested"
    assert controls.verify_identity("0")["status"] == "not_requested"
    assert (
        controls.verify_controls("0", calibration=calibration({}))["calibration_lens"][
            "status"
        ]
        == "unknown"
    )
    run.assert_not_called()


@pytest.mark.parametrize("actual_after_set", [48, 32])
def test_brightness_is_written_and_readback_must_match(monkeypatch, actual_after_set):
    monkeypatch.setattr(controls.shutil, "which", lambda name: f"/usr/bin/{name}")
    commands = []

    def run(args, **kwargs):
        commands.append(args)
        actual = 32 if len(commands) == 1 else actual_after_set
        return SimpleNamespace(
            returncode=0, stderr="",
            stdout=f"brightness 0x00 (int) : min=0 max=64 step=1 default=32 value={actual}\n",
        )

    monkeypatch.setattr(controls.subprocess, "run", run)
    if actual_after_set != 48:
        with pytest.raises(RuntimeError, match="brightness expected 48, read back 32"):
            controls.verify_controls("0", {"brightness": 48}, apply=True)
    else:
        result = controls.verify_controls("0", {"brightness": 48}, apply=True)
        assert result["requested"] == {"brightness": 48}
        assert result["readback"]["values"]["brightness"] == 48
    assert commands[1] == ["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=brightness=48"]
    assert commands[-1][-1] == "--list-ctrls"


def test_controls_apply_auto_before_manual_and_verify_aliases(monkeypatch):
    monkeypatch.setattr(controls.shutil, "which", lambda name: f"/usr/bin/{name}")
    commands = []

    def run(args, **kwargs):
        assert kwargs["timeout"] == 3.0
        commands.append(args)
        if args[-1] == "--list-ctrls":
            focus = 80 if len(commands) == 1 else 120
            stdout = (
                "focus_automatic_continuous 0x00 (bool) : default=1 value=0\n"
                f"focus_absolute 0x01 (int) : min=0 max=250 step=1 default=0 value={focus}\n"
                "zoom_absolute 0x02 (int) : min=100 max=400 default=100 value=100\n"
            )
        else:
            stdout = ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(controls.subprocess, "run", run)
    result = controls.verify_controls(
        "0",
        {"focus_absolute": 120, "focus_auto": 0},
        calibration({"focus_auto": 0, "focus_absolute": 120, "zoom_absolute": 100}),
        apply=True,
    )
    assert [args[-1] for args in commands if args[-1].startswith("--set-ctrl=")] == [
        "--set-ctrl=focus_absolute=120"
    ]
    assert commands[-1][-1] == "--list-ctrls"
    assert result["calibration_lens"]["status"] == "verified"


def test_apply_skips_write_when_readback_already_matches(monkeypatch):
    monkeypatch.setattr(controls.shutil, "which", lambda name: f"/usr/bin/{name}")
    commands = []

    def run(args, **kwargs):
        commands.append(args)
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "focus_automatic_continuous 0x00 (bool) : default=1 value=0\n"
                "focus_absolute 0x01 (int) : min=0 max=250 step=1 default=0 value=0\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(controls.subprocess, "run", run)
    result = controls.verify_controls(
        "0",
        {"focus_automatic_continuous": 0, "focus_absolute": 0},
        apply=True,
    )
    assert all("--set-ctrl=" not in " ".join(args) for args in commands)
    assert result["status"] == "verified"


def test_apply_accepts_protocol_error_when_readback_matches(monkeypatch):
    monkeypatch.setattr(controls.shutil, "which", lambda name: f"/usr/bin/{name}")
    listed = []

    def run(args, **kwargs):
        if args[-1].startswith("--set-ctrl=focus_absolute="):
            return SimpleNamespace(
                returncode=1, stdout="", stderr="focus_absolute: Protocol error"
            )
        listed.append(len(listed))
        focus = 5 if len(listed) == 1 else 0
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "focus_automatic_continuous 0x00 (bool) : default=1 value=0\n"
                f"focus_absolute 0x01 (int) : min=0 max=15 step=1 default=0 value={focus}\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(controls.subprocess, "run", run)
    result = controls.verify_controls(
        "0",
        {"focus_automatic_continuous": 0, "focus_absolute": 0},
        apply=True,
    )
    assert result["status"] == "verified"


@pytest.mark.parametrize("actual", [{"focus_absolute": 123}, {}, {"focus_auto": 0}])
def test_known_calibrated_lens_value_never_accepts_mismatch_or_unknown(
    monkeypatch, actual
):
    monkeypatch.setattr(
        controls, "read_controls", lambda device: {"available": True, "values": actual}
    )
    with pytest.raises(RuntimeError, match="focus_absolute"):
        controls.verify_controls(
            "/dev/fake", calibration=calibration({"focus_absolute": 120})
        )


def test_unreadable_known_control_fails(monkeypatch):
    monkeypatch.setattr(
        controls,
        "read_controls",
        lambda device: {
            "available": False,
            "values": {},
            "reason": "timeout",
        },
    )
    with pytest.raises(RuntimeError, match="timeout"):
        controls.verify_controls(
            "/dev/fake", calibration=calibration({"zoom_absolute": 100})
        )


def test_requested_and_calibrated_conflict_fails_without_touching_device(monkeypatch):
    read = Mock()
    monkeypatch.setattr(controls, "read_controls", read)
    with pytest.raises(ValueError, match="conflicts with calibration"):
        controls.verify_controls(
            "/dev/fake", {"focus_absolute": 140}, calibration({"focus_absolute": 120})
        )
    read.assert_not_called()


@pytest.mark.parametrize("returned", [None, "wrong-camera"])
def test_identity_mismatch_or_unknown_is_rejected(monkeypatch, returned):
    monkeypatch.setattr(controls.shutil, "which", lambda name: f"/usr/bin/{name}")
    run = Mock(
        return_value=SimpleNamespace(
            returncode=0,
            stdout=f"ID_SERIAL={returned}\n" if returned else "ID_MODEL=webcam\n",
            stderr="",
        )
    )
    monkeypatch.setattr(controls.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        controls.verify_identity("0", {"ID_SERIAL": "camera-a"})
    assert run.call_args.args[0][-1] == "--name=/dev/video0"
    assert run.call_args.kwargs["timeout"] == 3.0


def test_identity_expected_properties_are_verified_exactly(monkeypatch):
    monkeypatch.setattr(controls.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        controls.subprocess,
        "run",
        Mock(
            return_value=SimpleNamespace(
                returncode=0,
                stdout="ID_SERIAL=abc\nID_PATH=usb-1\n",
                stderr="",
            )
        ),
    )
    result = controls.verify_identity(
        "/dev/fake", {"ID_SERIAL": "abc", "ID_PATH": "usb-1"}
    )
    assert result["status"] == "verified"
    assert result["actual"]["ID_PATH"] == "usb-1"


def test_calibration_serial_is_checked_automatically_without_requiring_original_port(
    monkeypatch,
):
    monkeypatch.setattr(controls.shutil, "which", lambda name: f"/usr/bin/{name}")
    run = Mock(
        return_value=SimpleNamespace(
            returncode=0,
            stdout="ID_SERIAL=abc\nID_PATH=new-port\n",
            stderr="",
        )
    )
    monkeypatch.setattr(controls.subprocess, "run", run)
    calibrated = {"camera": {"ID_SERIAL": "abc", "ID_PATH": "old-port"}}
    result = controls.verify_identity("00", calibration=calibrated)
    assert result["expected"] == {"ID_SERIAL": "abc"}
    assert result["actual"]["device"] == "/dev/video0"
    assert run.call_args.args[0][-1] == "--name=/dev/video0"
    result = controls.verify_identity(
        "00", {"ID_PATH": "new-port"}, calibration=calibrated
    )
    assert result["expected"] == {"ID_SERIAL": "abc", "ID_PATH": "new-port"}
    run.return_value.stdout = "ID_SERIAL=abc\nID_PATH=old-port\n"
    with pytest.raises(RuntimeError, match="identity mismatch.*ID_PATH"):
        controls.verify_identity(
            "00", {"ID_PATH": "new-port"}, calibration=calibrated
        )
    run.return_value.stdout = "ID_SERIAL=wrong\n"
    with pytest.raises(RuntimeError, match="identity mismatch"):
        controls.verify_identity("00", calibration=calibrated)


def test_calibration_path_fallback_and_explicit_conflict(monkeypatch):
    monkeypatch.setattr(controls.shutil, "which", lambda name: f"/usr/bin/{name}")
    run = Mock(
        return_value=SimpleNamespace(returncode=0, stdout="ID_PATH=port-a\n", stderr="")
    )
    monkeypatch.setattr(controls.subprocess, "run", run)
    result = controls.verify_identity(
        "0", calibration={"camera": {"ID_PATH": "port-a"}}
    )
    assert result["expected"] == {"ID_PATH": "port-a"}
    run.reset_mock()
    with pytest.raises(ValueError, match="conflicts with calibration"):
        controls.verify_identity(
            "0", {"ID_SERIAL": "wrong"}, calibration={"camera": {"ID_SERIAL": "abc"}}
        )
    run.assert_not_called()


def test_calibrated_lens_rejects_enabled_autofocus_even_when_absolute_value_matches(
    monkeypatch,
):
    monkeypatch.setattr(
        controls,
        "read_controls",
        lambda device: {
            "available": True,
            "values": {"focus_absolute": 120, "focus_auto": 1},
        },
    )
    with pytest.raises(RuntimeError, match="autofocus is enabled"):
        controls.verify_controls("0", calibration=calibration({"focus_absolute": 120}))


@pytest.mark.parametrize("operation", ["controls", "identity"])
def test_external_commands_have_bounded_timeout_and_fail_closed(monkeypatch, operation):
    monkeypatch.setattr(controls.shutil, "which", lambda name: f"/usr/bin/{name}")
    run = Mock(side_effect=subprocess.TimeoutExpired("tool", 3))
    monkeypatch.setattr(controls.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="cannot verify"):
        if operation == "controls":
            controls.verify_controls("0", {"focus_absolute": 120})
        else:
            controls.verify_identity("0", {"ID_SERIAL": "abc"})
    assert run.call_args.kwargs["timeout"] == 3.0


@pytest.mark.parametrize(
    "requested",
    [
        {"focus_absolute": True},
        {"focus_absolute": 1.5},
        {"focus_absolute,other": 1},
        {"focus_auto": 0, "focus_automatic_continuous": 1},
    ],
)
def test_controls_reject_non_integer_or_ambiguous_settings(requested):
    with pytest.raises(ValueError):
        controls.validate_controls(requested)
