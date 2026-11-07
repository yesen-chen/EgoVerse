"""CPU-only camera acceptance tests: fake workers, clocks and saved video probes."""

import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.scripts import check_camera_rig as diag


def camera_spec(name="left_wrist", device="/dev/video0"):
    return {
        "name": name,
        "device": device,
        "type": "uvc",
        "width": 16,
        "height": 12,
        "fps": 10.0,
        "fourcc": "MJPG",
        "latency": 0.03,
        "latency_status": "unmeasured",
    }


def config(tmp_path, specs=None):
    path = tmp_path / "rig.json"
    path.write_text(
        json.dumps(
            {
                "schema": "camera_rig_v1",
                "cameras": specs
                or [camera_spec(), camera_spec("right_wrist", "/dev/video2")],
            }
        )
    )
    return path


class FakeRig:
    def __init__(self, *, failure=None):
        self.now = 1000.0
        self.failure = failure
        self.events, self.workers, self.managers = [], [], []
        self.saved = {}

    def sleep(self, seconds):
        self.now += seconds

    def manager(self):
        owner = self

        class Manager:
            def start(self):
                owner.events.append("manager.start")

            def shutdown(self):
                owner.events.append("manager.shutdown")

        value = Manager()
        self.managers.append(value)
        return value

    def camera(self, manager, **kwargs):
        owner = self

        class Camera:
            def __init__(self):
                self.spec = kwargs
                self.name = kwargs["name"]
                self.started = False
                self.ready_event, self.stop_event = Event(), Event()
                self.process = SimpleNamespace(
                    exitcode=None, is_alive=lambda: self.started
                )
                self.recording_start = None

            def start(self):
                owner.events.append(f"{self.name}.start")
                self.started = True
                if owner.failure == "start" and self.name == "right_wrist":
                    raise RuntimeError("start failure")
                if owner.failure != "not_ready":
                    self.ready_event.set()

            def get_last(self, k):
                owner.events.append(f"{self.name}.get")
                if owner.failure == "missing":
                    raise RuntimeError("no frames")
                stale = 2.0 if owner.failure == "stale" else 0.0
                return {
                    "camera_receive_timestamp": np.array([owner.now - stale]),
                    "color": np.zeros((1, 12, 16, 3), dtype=np.uint8),
                }

            def start_recording(self, path, timeout):
                owner.events.append(f"{self.name}.record")
                self.path, self.recording_start = Path(path), owner.now
                if owner.failure == "record_ack" and self.name == "right_wrist":
                    raise TimeoutError("lost start acknowledgement")

            def stop_recording(self, timeout):
                owner.events.append(f"{self.name}.record_stop")
                stamps = np.arange(self.recording_start, owner.now + 1e-7, 0.1)
                if owner.failure == "short_recording":
                    stamps = stamps[:2]
                self.path.write_bytes(b"fake video; decoding is injected")
                np.save(self.path.with_suffix(".npy"), stamps - kwargs["latency"])
                np.save(self.path.with_suffix(".receive_timestamps.npy"), stamps)
                save_metadata(self.path, len(stamps))
                owner.saved[str(self.path)] = len(stamps)

            def stop(self, timeout):
                owner.events.append(f"{self.name}.stop")
                self.started = False
                self.process.exitcode = 0
                if owner.failure == "cleanup" and self.name == "left_wrist":
                    raise TimeoutError("forced termination")

        value = Camera()
        self.workers.append(value)
        return value

    def probe(self, path):
        return {"decoded_frames": self.saved[str(path)], "decoded_sizes": [[16, 12]]}

    def run(self, tmp_path, **kwargs):
        specs = kwargs.pop("specs", None)
        return diag.run_diagnostic(
            config(tmp_path, specs),
            tmp_path / "out",
            duration=1.0,
            cycles=3,
            timeout=0.2,
            poll_seconds=0.1,
            camera_factory=self.camera,
            manager_factory=self.manager,
            monotonic=lambda: self.now,
            wall_time=lambda: self.now,
            sleep=self.sleep,
            video_probe=self.probe,
            **kwargs,
        )


def test_three_cycles_reopen_every_worker_and_validate_saved_records(tmp_path):
    rig = FakeRig()
    report = rig.run(tmp_path)
    assert report["status"] == "PASS"
    assert report["completed_cycles"] == 3
    assert len(rig.workers) == 6 and len(rig.managers) == 3
    assert rig.events.count("manager.shutdown") == 3
    assert rig.events.index("right_wrist.start") < rig.events.index("left_wrist.get")
    assert all(worker.stop_event.is_set() for worker in rig.workers)
    assert report["mean_observed_fps_by_camera"] == pytest.approx(
        {"left_wrist": 10, "right_wrist": 10}
    )
    for cycle in report["cycles"]:
        assert cycle["manager_released"] and all(
            item["clean"] for item in cycle["cleanup"]
        )
        assert cycle["receive_differences"][0]["absolute_max_s"] == 0
        assert cycle["cameras"][0]["receive_minus_timestamp_mean_s"] == pytest.approx(
            0.03
        )
    assert json.loads((tmp_path / "out/report.json").read_text())["status"] == "PASS"
    assert "physical exposure" in report["limitation"]
    assert (tmp_path / "out/camera_config.json").is_file()


@pytest.mark.parametrize(
    "failure",
    [
        "start",
        "record_ack",
        "cleanup",
        "stale",
        "missing",
        "not_ready",
        "short_recording",
    ],
)
def test_failures_stop_all_constructed_workers_and_release_manager(tmp_path, failure):
    rig = FakeRig(failure=failure)
    report = rig.run(tmp_path)
    assert report["status"] == "FAIL" and report["errors"]
    assert report["completed_cycles"] == 1
    assert len(rig.workers) == 2
    assert all(not worker.process.is_alive() for worker in rig.workers)
    assert (
        rig.events.count("left_wrist.stop") == rig.events.count("right_wrist.stop") == 1
    )
    assert rig.events.count("manager.shutdown") == 1
    if failure == "record_ack":
        assert (
            "left_wrist.record_stop" in rig.events
            and "right_wrist.record_stop" in rig.events
        )


def test_controls_identity_and_latency_status_reach_the_worker(tmp_path):
    rig = FakeRig()
    spec = camera_spec()
    spec.update(
        controls={"focus_automatic_continuous": 0, "focus_absolute": 30},
        expected_identity={"ID_SERIAL": "camera_serial"},
        latency_status="measured",
    )
    report = rig.run(tmp_path, specs=[spec])
    assert report["status"] == "PASS"
    for worker in rig.workers:
        assert worker.spec["controls"] == {"brightness": 48, **spec["controls"]}
        assert worker.spec["expected_identity"] == spec["expected_identity"]
        assert worker.spec["latency_status"] == "measured"


def save_metadata(path, count):
    path.with_suffix(".stream_metadata.json").write_text(
        json.dumps(
            {
                "schema": "uvc_stream_metadata_v1",
                "frame_count": count,
                "native_frame_shape": [12, 16, 3],
                "actual_mode": {"fps": 999},
            }
        )
    )


def saved_record(tmp_path, stamps=None):
    path = tmp_path / "left_wrist.mp4"
    path.touch()
    stamps = np.asarray([1.0, 1.1, 1.2] if stamps is None else stamps)
    np.save(path.with_suffix(".npy"), stamps - 0.03)
    np.save(path.with_suffix(".receive_timestamps.npy"), stamps)
    save_metadata(path, len(stamps))
    return path


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing", "No such file"),
        ("count", "count mismatch"),
        ("nan", "finite"),
        ("duplicate", "strictly increasing"),
        ("video_count", "count mismatch"),
        ("size", "native mode"),
    ],
)
def test_recording_missing_corrupt_counts_and_sizes_fail(tmp_path, mutation, match):
    path = saved_record(tmp_path)
    count, sizes = 3, [[16, 12]]
    if mutation == "missing":
        path.with_suffix(".receive_timestamps.npy").unlink()
    if mutation == "count":
        np.save(path.with_suffix(".receive_timestamps.npy"), [1.0, 1.1])
    if mutation == "nan":
        np.save(path.with_suffix(".receive_timestamps.npy"), [1.0, np.nan, 1.2])
    if mutation == "duplicate":
        np.save(path.with_suffix(".receive_timestamps.npy"), [1.0, 1.0, 1.2])
    if mutation == "video_count":
        count = 2
    if mutation == "size":
        sizes = [[8, 6]]
    with pytest.raises((ValueError, OSError), match=match):
        diag.inspect_recording(
            path,
            camera_spec(),
            video_probe=lambda _: {"decoded_frames": count, "decoded_sizes": sizes},
        )


def test_large_gap_and_low_observed_rate_fail_despite_requested_fps(tmp_path):
    path = saved_record(tmp_path, [1.0, 1.1, 1.5])
    metric, _ = diag.inspect_recording(
        path,
        camera_spec(),
        video_probe=lambda _: {"decoded_frames": 3, "decoded_sizes": [[16, 12]]},
    )
    assert metric["status"] == "FAIL"
    assert metric["observed_mean_fps"] == 4
    assert metric["excessive_gap_count"] == 1
    assert len(metric["errors"]) == 2


def test_wrong_configured_latency_is_not_treated_as_measurement(tmp_path):
    path = saved_record(tmp_path)
    spec = camera_spec()
    spec["latency"] = 0.1
    metric, _ = diag.inspect_recording(
        path,
        spec,
        video_probe=lambda _: {"decoded_frames": 3, "decoded_sizes": [[16, 12]]},
    )
    assert metric["status"] == "FAIL"
    assert "configured compensation" in metric["errors"][0]


def test_pico_config_and_existing_output_are_rejected_before_workers(tmp_path):
    spec = {**camera_spec(), "type": "pico", "headset_ip": "192.0.2.1"}
    rig = FakeRig()
    with pytest.raises(ValueError, match="UVC cameras only"):
        diag.run_diagnostic(
            config(tmp_path, [spec]), tmp_path / "out", camera_factory=rig.camera
        )
    assert not rig.workers and not (tmp_path / "out").exists()
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        diag.run_diagnostic(config(tmp_path), output, camera_factory=rig.camera)
    assert not rig.workers


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--duration", "nan"),
        ("--duration", "0"),
        ("--timeout", "inf"),
        ("--stale-seconds", "-1"),
        ("--cycles", "1.5"),
        ("--cycles", "0"),
        ("--max-gap-seconds", "nan"),
        ("--min-fps-ratio", "-1"),
    ],
)
def test_cli_numeric_arguments_are_finite_and_positive(flag, value):
    with pytest.raises(SystemExit):
        diag.build_parser().parse_args(
            ["--camera-config", "rig.json", "--output", "out", flag, value]
        )


def test_receive_skew_is_nearest_host_arrival_not_physical_sync():
    metrics = diag.receive_differences(
        {"a": np.array([1.0, 1.1, 1.2]), "b": np.array([1.01, 1.11, 1.21])}
    )
    assert metrics[0]["signed_mean_s"] == pytest.approx(0.01)
    assert "not exposure sync" in metrics[0]["method"]
    with pytest.raises(ValueError, match="no receive-time overlap"):
        diag.receive_differences({"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])})
