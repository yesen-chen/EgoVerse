"""Power-check parser and verdict tests: no adb/device or XR SDK access."""

from __future__ import annotations

import argparse
import json
import subprocess
from types import SimpleNamespace

import pytest

from egomimic.scripts import check_pico_awake as awake


def _power(state="Awake", sleep=123, wake=200):
    return (
        "POWER MANAGER (dumpsys power)\n"
        f"  mWakefulness={state}\n"
        f"  mLastSleepTime={sleep} (1000 ms ago)\n"
        f"  mLastWakeTime={wake} (900 ms ago)\n"
        "  mLastSleepReason=timeout\n"
    )


def test_parse_global_fields_and_ignore_display_group_duplicates():
    snapshot = awake.parse_power_snapshot(_power() + _power("Asleep", 99, 100))
    assert snapshot == awake.PowerSnapshot("Awake", 123, 200, "timeout", 1100)


def test_uptime_falls_back_to_sleep_age_and_is_optional_for_older_output():
    snapshot = awake.parse_power_snapshot(_power().replace(" (900 ms ago)", ""))
    assert snapshot.device_uptime_ms == 1123
    legacy = awake.parse_power_snapshot(
        _power().replace(" (900 ms ago)", "").replace(" (1000 ms ago)", "")
    )
    assert legacy == awake.PowerSnapshot("Awake", 123, 200, "timeout")
    check = awake.PowerCheck()
    check.observe(legacy)
    check.observe(legacy)
    assert check.status == "PASS"


def test_reboot_with_zero_sleep_and_wake_timestamps_cannot_become_pass():
    check = awake.PowerCheck()
    for age in (92856, 97856):
        snapshot = awake.parse_power_snapshot(
            _power(sleep=0, wake=0).replace("900 ms ago", f"{age} ms ago")
        )
        assert snapshot.device_uptime_ms == age
        check.observe(snapshot)
    assert check.status == "PASS"
    for age in (1000, 6000):
        check.observe(awake.parse_power_snapshot(
            _power(sleep=0, wake=0).replace("900 ms ago", f"{age} ms ago")
        ))
    assert check.status == "UNKNOWN"
    assert any("uptime went backwards" in reason for reason in check.unknowns)


def test_quick_sleep_wake_between_polls_fails_even_when_both_reads_awake():
    check = awake.PowerCheck()
    check.observe(awake.parse_power_snapshot(_power(sleep=100, wake=200)))
    check.observe(awake.parse_power_snapshot(_power(sleep=300, wake=400)))
    assert check.status == "FAIL"
    assert "sleep occurred between reads" in check.failures[0]


@pytest.mark.parametrize("initial", ["Asleep", "Dozing"])
def test_initial_sleep_is_failure_even_after_recovery(initial):
    check = awake.PowerCheck()
    check.observe(awake.parse_power_snapshot(_power(initial)))
    check.observe(awake.parse_power_snapshot(_power()))
    assert check.status == "FAIL"


def test_missing_fields_disconnect_and_clock_reset_cannot_become_pass():
    for bad in ("  mWakefulness=Awake\n", _power(sleep=1, wake=2)):
        check = awake.PowerCheck()
        check.observe(awake.parse_power_snapshot(_power()))
        check.observe(awake.parse_power_snapshot(bad))
        assert check.status == "UNKNOWN"
    check = awake.PowerCheck()
    check.observe(awake.parse_power_snapshot(_power()))
    check.unknown("adb disconnected")
    check.observe(awake.parse_power_snapshot(_power()))
    assert check.status == "UNKNOWN"


def test_device_selection_requires_explicit_serial_for_multiple_devices():
    listing = "List of devices attached\npico123 device product:pico\npico456 offline\n"
    with pytest.raises(ValueError, match="--serial"):
        awake.select_device(listing)
    assert awake.select_device(listing, "pico123") == "pico123"
    with pytest.raises(ValueError, match="online/authorized"):
        awake.select_device(listing, "pico456")
    assert awake.validate_serial("192.168.1.5:5555") == "192.168.1.5:5555"
    for bad in ("-s", "pico;reboot", "pico\nother"):
        with pytest.raises(argparse.ArgumentTypeError):
            awake.validate_serial(bad)
    for bad in ("nan", "inf", "0", "-1"):
        with pytest.raises(argparse.ArgumentTypeError):
            awake.positive_seconds(bad)


def test_adb_timeout_saves_partial_output_and_never_uses_shell(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        assert command == ["adb", "-s", "pico123", "shell", "dumpsys", "power"]
        assert kwargs["shell"] is False
        assert kwargs["timeout"] == 10.0
        raise subprocess.TimeoutExpired(command, 10, output=b"partial power output")

    monkeypatch.setattr(awake.subprocess, "run", fake_run)
    raw_path = tmp_path / "raw.txt"
    with pytest.raises(RuntimeError, match="adb failed"):
        awake.run_adb(["-s", "pico123", "shell", "dumpsys", "power"], raw_path)
    assert "partial power output" in raw_path.read_text()


@pytest.mark.parametrize("failure", [None, "disconnect", "interrupt"])
def test_cli_saves_summary_with_only_allowed_read_commands(monkeypatch, tmp_path, failure):
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds):
        if failure == "interrupt":
            raise KeyboardInterrupt
        clock.now += seconds

    monkeypatch.setattr(awake, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    commands = []
    power_reads = 0

    def fake_adb(arguments, raw_path):
        nonlocal power_reads
        commands.append(arguments)
        raw_path.write_text("mocked raw evidence")
        if arguments == ["devices", "-l"]:
            return "List of devices attached\npico123 device product:pico\n"
        assert arguments[:3] == ["-s", "pico123", "shell"]
        if arguments[3] == "getprop":
            assert arguments[4] in awake.PROPERTIES
            return "test property value\n"
        assert arguments[3:] == ["dumpsys", "power"]
        power_reads += 1
        if failure == "disconnect" and power_reads == 2:
            raise RuntimeError("adb disconnected")
        return _power()

    monkeypatch.setattr(awake, "run_adb", fake_adb)
    output = tmp_path / "check"
    result = awake.main(["--seconds", "6", "--output-dir", str(output)])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == ("PASS" if failure is None else "UNKNOWN")
    assert result == (0 if failure is None else 2)
    assert set(summary["properties"]) == set(awake.PROPERTIES)
    assert "pose" in summary["limitation"]
    assert power_reads == (1 if failure == "interrupt" else 3)
    assert all("setprop" not in command and "input" not in command for command in commands)
