"""Pico process.

The child opens the SDK. The ring stores raw controller samples. The clutch
runs in the main loop, so this process does not turn poses into arm targets.
``receive_timestamp`` is the local polling time, not evidence of a new SDK
packet. Source diagnostics observe changes to the SDK packet timestamp using
the local monotonic clock; they do not measure headset-to-PC latency.
"""

from __future__ import annotations

import time
from multiprocessing.managers import SharedMemoryManager

import numpy as np

from egomimic.robot.piper_pico.shm import SharedMemoryRingBuffer, join_process


def pico_examples() -> dict:
    return {
        "left_pos": np.zeros(3, dtype=np.float64),
        "left_quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "left_grip": np.float64(0.0),
        "left_trigger": np.float64(0.0),
        "right_pos": np.zeros(3, dtype=np.float64),
        "right_quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "right_grip": np.float64(0.0),
        "right_trigger": np.float64(0.0),
        "button_a": np.float64(0.0),
        "button_b": np.float64(0.0),
        "button_x": np.float64(0.0),
        "button_y": np.float64(0.0),
        "receive_timestamp": np.float64(0.0),
        "source_timestamp_ns": np.int64(0),
        "source_age_s": np.float64(np.nan),
        "source_stall_count": np.int64(0),
        "source_max_gap_s": np.float64(0.0),
        "head_pos": np.full(3, np.nan, dtype=np.float64),
        "head_quat_xyzw": np.full(4, np.nan, dtype=np.float64),
    }


def _pose_or_zero(read) -> np.ndarray:
    pose = np.asarray(read(), dtype=np.float64).reshape(-1)
    if pose.shape[0] < 7:
        raise RuntimeError(f"Pico pose has {pose.shape[0]} values, expected 7")
    return pose[:7]


class SourceTimestampMonitor:
    """Count unchanged SDK-stamp intervals, without comparing clock domains.

    One interval contributes at most one stall, when an unchanged positive
    timestamp first persists for 100 ms. Missing/zero timestamps mean unknown,
    not a stalled source. A changed stamp (including a reset backwards) starts
    a new interval. This observes SDK packet freshness, not pose validity.
    """

    def __init__(self, stall_threshold_s: float = 0.1):
        self.stall_threshold_s = float(stall_threshold_s)
        self._stamp = None
        self._changed_at = None
        self._stalled = False
        self.stall_count = 0
        self.max_gap_s = 0.0

    def update(self, stamp: int, now_monotonic: float) -> dict:
        now = float(now_monotonic)
        if stamp <= 0:
            self._stamp = None
            self._changed_at = None
            self._stalled = False
            age = np.nan
        elif self._stamp is None:
            self._stamp = stamp
            self._changed_at = now
            age = 0.0
        else:
            gap = max(0.0, now - self._changed_at)
            self.max_gap_s = max(self.max_gap_s, gap)
            if stamp != self._stamp:
                self._stamp = stamp
                self._changed_at = now
                self._stalled = False
                age = 0.0
            else:
                age = gap
                if age >= self.stall_threshold_s and not self._stalled:
                    self.stall_count += 1
                    self._stalled = True
        return {
            "source_timestamp_ns": np.int64(max(0, stamp)),
            "source_age_s": np.float64(age),
            "source_stall_count": np.int64(self.stall_count),
            "source_max_gap_s": np.float64(self.max_gap_s),
        }


def read_pico_sample(xrt) -> dict:
    """Read SDK getters without initializing hardware.

    These are separate getter calls, NOT an atomic SDK packet snapshot: a new
    packet may arrive between poses, buttons, head, and timestamp. The source
    stamp is sampled last and is only a freshness diagnostic. Optional head or
    timestamp getters failing/unavailable must not interrupt controller reads.
    """
    left = _pose_or_zero(xrt.get_left_controller_pose)
    right = _pose_or_zero(xrt.get_right_controller_pose)
    sample = {
        "left_pos": left[:3],
        "left_quat_xyzw": left[3:7],
        "left_grip": np.float64(xrt.get_left_grip()),
        "left_trigger": np.float64(xrt.get_left_trigger()),
        "right_pos": right[:3],
        "right_quat_xyzw": right[3:7],
        "right_grip": np.float64(xrt.get_right_grip()),
        "right_trigger": np.float64(xrt.get_right_trigger()),
        "button_a": np.float64(bool(xrt.get_A_button())),
        "button_b": np.float64(bool(xrt.get_B_button())),
        "button_x": np.float64(bool(xrt.get_X_button())),
        "button_y": np.float64(bool(xrt.get_Y_button())),
    }
    try:
        head = _pose_or_zero(xrt.get_headset_pose)
    except Exception:
        head = np.full(7, np.nan, dtype=np.float64)
    try:
        stamp = np.int64(max(0, int(xrt.get_time_stamp_ns())))
    except Exception:
        stamp = np.int64(0)
    sample.update(
        head_pos=head[:3],
        head_quat_xyzw=head[3:7],
        source_timestamp_ns=stamp,
        receive_timestamp=np.float64(time.time()),
    )
    return sample


def pico_process_loop(ring: SharedMemoryRingBuffer, stop_event, ready_event, frequency: float) -> None:
    import xensevr_pc_service_sdk as xrt

    period = 1.0 / float(frequency)
    xrt.init()
    try:
        time.sleep(0.5)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            left = _pose_or_zero(xrt.get_left_controller_pose)
            right = _pose_or_zero(xrt.get_right_controller_pose)
            if np.any(np.abs(left) > 1e-6) and np.any(np.abs(right) > 1e-6):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Pico poses stayed at zero. Restart the VR client and PC service.")
        next_tick = time.monotonic()
        source_monitor = SourceTimestampMonitor()
        while not stop_event.is_set():
            sample = read_pico_sample(xrt)
            sample.update(source_monitor.update(int(sample["source_timestamp_ns"]), time.monotonic()))
            ring.put(sample)
            if not ready_event.is_set():
                ready_event.set()
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            elif delay < -period:
                next_tick = time.monotonic()
    finally:
        try:
            xrt.close()
        except Exception as exc:
            print(f"[pico] failed to close the SDK: {exc}")


class PicoProcess:
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        frequency: float = 90.0,
    ):
        import multiprocessing as mp

        self.name = "pico"
        self.ring = SharedMemoryRingBuffer.create_from_examples(
            shm_manager,
            pico_examples(),
            get_max_k=8,
            get_time_budget=0.01,
            put_desired_frequency=frequency,
        )
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.process = mp.Process(
            target=pico_process_loop,
            args=(self.ring, self.stop_event, self.ready_event, frequency),
            daemon=False,
        )

    def start(self) -> None:
        self.process.start()

    def get_latest(self) -> dict:
        return self.ring.get()

    def stop(self, timeout: float = 3.0) -> None:
        self.stop_event.set()
        join_process(self.process, timeout)
