"""Read-only PICO power-state check; never initializes an XR or robot SDK.

Example::

    python -m egomimic.scripts.check_pico_awake --seconds 90

Only ``adb devices``, five named ``getprop`` reads, and ``dumpsys power`` are
used. No settings, wake keys, or controller commands are sent. Exit status is
0 for PASS, 1 for FAIL, and 2 for UNKNOWN (including an interrupted check).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import signal
import subprocess
import time


PROPERTIES = (
    "persist.psensor.screenoff.delay",
    "persist.psensor.sleep.delay",
    "persist.pvr.sleep_by_static",
    "ro.product.model",
    "ro.build.display.id",
)
POLL_SECONDS = 5.0
COMMAND_TIMEOUT_SECONDS = 10.0
LIMITATION = "This checks headset power only; PASS does not prove that pose tracking never drops."


@dataclass
class PowerSnapshot:
    wakefulness: str | None
    last_sleep_time: int | None
    last_wake_time: int | None
    last_sleep_reason: str | None
    device_uptime_ms: int | None = None


def parse_power_snapshot(text: str) -> PowerSnapshot:
    """Read the first/global value, before any per-display power-group copies."""
    def value(name: str) -> str | None:
        match = re.search(rf"(?m)^\s*{name}\s*[=:]\s*([^\r\n]+)", text)
        return match.group(1).strip() if match else None

    def number(name: str) -> int | None:
        raw = value(name)
        match = re.match(r"^(\d+)(?:\s|$)", raw or "")
        return int(match.group(1)) if match else None

    def uptime(name: str) -> int | None:
        # dumpsys prints an uptime event timestamp followed by its age. Their
        # sum also detects a reboot when both last-event timestamps remain 0.
        raw = value(name)
        match = re.match(r"^(\d+)\s+\((\d+)\s+ms\s+ago\)", raw or "")
        return int(match.group(1)) + int(match.group(2)) if match else None

    device_uptime_ms = uptime("mLastWakeTime")
    if device_uptime_ms is None:
        device_uptime_ms = uptime("mLastSleepTime")
    return PowerSnapshot(
        wakefulness=value("mWakefulness"),
        last_sleep_time=number("mLastSleepTime"),
        last_wake_time=number("mLastWakeTime"),
        last_sleep_reason=value("mLastSleepReason"),
        device_uptime_ms=device_uptime_ms,
    )


class PowerCheck:
    """Conservative verdict over the whole observation window."""

    def __init__(self):
        self.failures: list[str] = []
        self.unknowns: list[str] = []
        self.samples = 0
        self._sleep_time: int | None = None
        self._wake_time: int | None = None
        self._uptime_ms: int | None = None

    def unknown(self, reason: str) -> None:
        if reason not in self.unknowns:
            self.unknowns.append(reason)

    def observe(self, snapshot: PowerSnapshot) -> None:
        self.samples += 1
        required = ("wakefulness", "last_sleep_time", "last_wake_time", "last_sleep_reason")
        missing = [name for name in required if getattr(snapshot, name) is None]
        if missing:
            self.unknown("Incomplete dumpsys power fields: " + ", ".join(missing))
        if snapshot.wakefulness in ("Asleep", "Dozing"):
            self.failures.append(f"Sample {self.samples}: mWakefulness={snapshot.wakefulness}")
        elif snapshot.wakefulness not in (None, "Awake"):
            self.unknown(f"Unverified wakefulness state: {snapshot.wakefulness}")
        current = snapshot.last_sleep_time
        if current is not None:
            if self._sleep_time is not None and current > self._sleep_time:
                self.failures.append(
                    f"Sample {self.samples}: mLastSleepTime increased "
                    f"from {self._sleep_time} to {current} (sleep occurred between reads)"
                )
            elif self._sleep_time is not None and current < self._sleep_time:
                self.unknown("mLastSleepTime went backwards; possible headset reboot or clock reset")
            self._sleep_time = current
        current = snapshot.last_wake_time
        if current is not None:
            if self._wake_time is not None and current < self._wake_time:
                self.unknown("mLastWakeTime went backwards; possible headset reboot or clock reset")
            self._wake_time = current
        current = snapshot.device_uptime_ms
        if current is not None:
            if self._uptime_ms is not None and current < self._uptime_ms:
                self.unknown("Device uptime went backwards; possible headset reboot or clock reset")
            self._uptime_ms = current

    @property
    def status(self) -> str:
        if self.failures:
            return "FAIL"
        if self.unknowns or self.samples < 2:
            return "UNKNOWN"
        return "PASS"


def validate_serial(serial: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", serial):
        raise argparse.ArgumentTypeError("serial must be a device identifier or host:port")
    return serial


def positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seconds must be a positive finite number") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("seconds must be a positive finite number")
    return seconds


def select_device(devices_text: str, requested: str | None = None) -> str:
    devices = {}
    for line in devices_text.splitlines():
        match = re.match(r"^(\S+)\s+(device|offline|unauthorized|recovery|sideload|bootloader)(?:\s|$)", line)
        if match:
            devices[match.group(1)] = match.group(2)
    if requested is None:
        if len(devices) != 1:
            raise ValueError("Specify --serial when multiple devices are listed; no usable device if list is empty")
        requested = next(iter(devices))
    if devices.get(requested) != "device":
        raise ValueError(f"Device {requested!r} is not online/authorized: {devices.get(requested, 'not listed')}")
    return validate_serial(requested)


def _text(value) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")


def run_adb(arguments: list[str], raw_path: Path) -> str:
    """Save even failed/partial command output, with a bounded subprocess wait."""
    command = ["adb", *arguments]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS,
            check=False, shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raw_path.write_text(
            f"ERROR: {exc}\nSTDOUT:\n{_text(getattr(exc, 'stdout', ''))}"
            f"\nSTDERR:\n{_text(getattr(exc, 'stderr', ''))}", encoding="utf-8",
        )
        raise RuntimeError(f"adb failed: {exc}") from exc
    raw_path.write_text(
        f"RETURN_CODE: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"adb returned {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def _termination_requested(signum, frame) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", type=validate_serial, help="required when adb lists multiple devices")
    parser.add_argument("--seconds", type=positive_seconds, default=90.0)
    parser.add_argument("--output-dir", type=Path, help="new or empty directory for raw snapshots and summary.json")
    args = parser.parse_args(argv)
    output = args.output_dir or Path("diagnostics") / datetime.now().strftime("pico_awake_check_%Y%m%d_%H%M%S_%f")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output-dir must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    check = PowerCheck()
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "serial": args.serial,
        "requested_seconds": args.seconds,
        "poll_seconds": POLL_SECONDS,
        "properties": {},
        "limitation": LIMITATION,
    }
    records = []
    started = None
    elapsed = 0.0
    completed = False
    old_sigterm = signal.signal(signal.SIGTERM, _termination_requested)
    try:
        devices = run_adb(["devices", "-l"], output / "adb_devices.txt")
        serial = select_device(devices, args.serial)
        metadata["serial"] = serial
        for prop in PROPERTIES:
            try:
                metadata["properties"][prop] = run_adb(
                    ["-s", serial, "shell", "getprop", prop], output / f"getprop_{prop}.txt"
                ).strip()
            except RuntimeError as exc:
                check.unknown(str(exc))
                metadata["properties"][prop] = None
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Read-only power check for {serial}, {args.seconds:g}s; logs: {output.resolve()}", flush=True)
        started = time.monotonic()
        deadline = started + args.seconds
        while True:
            index = len(records)
            raw_name = f"power_{index:04d}.txt"
            record = {"index": index, "raw_file": raw_name}
            try:
                raw = run_adb(["-s", serial, "shell", "dumpsys", "power"], output / raw_name)
                snapshot = parse_power_snapshot(raw)
                check.observe(snapshot)
                record.update(asdict(snapshot))
            except RuntimeError as exc:
                check.unknown(str(exc))
                record["error"] = str(exc)
            elapsed = time.monotonic() - started
            record["elapsed_s"] = elapsed
            records.append(record)
            print(
                f"  {elapsed:6.1f}s state={record.get('wakefulness', 'UNKNOWN')} "
                f"last_sleep={record.get('last_sleep_time')} verdict={check.status}",
                flush=True,
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                completed = True
                break
            time.sleep(min(POLL_SECONDS, remaining))
    except KeyboardInterrupt:
        check.unknown("Interrupted before the observation window completed")
    except (RuntimeError, ValueError, OSError, argparse.ArgumentTypeError) as exc:
        check.unknown(str(exc))
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)
        if started is not None:
            elapsed = time.monotonic() - started
        if not completed:
            check.unknown("Observation window did not complete")
        if check.samples < 2:
            check.unknown("Fewer than two parsed power snapshots")
        summary = {
            **metadata,
            "status": check.status,
            "completed": completed,
            "elapsed_s": elapsed,
            "failures": check.failures,
            "unknowns": check.unknowns,
            "snapshots": records,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{check.status}: {output.resolve() / 'summary.json'}")
    for reason in check.failures + check.unknowns:
        print(f"  {reason}")
    print(LIMITATION)
    print("本检查仅验证头显电源状态；PASS 不代表 pose 没有丢失。")
    return {"PASS": 0, "FAIL": 1, "UNKNOWN": 2}[check.status]


if __name__ == "__main__":
    raise SystemExit(main())
