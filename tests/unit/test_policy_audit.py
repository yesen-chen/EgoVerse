"""Request logging is independent of model computation and camera pixels."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.robot.piper_pico.policy_audit import CAMERAS, RecordedPolicy


class ForbiddenImages:
    def __iter__(self):
        raise AssertionError("Audit must not iterate or copy image pixels")

    def __array__(self, *args, **kwargs):
        raise AssertionError("Audit must not convert image pixels")

    def __deepcopy__(self, memo):
        raise AssertionError("Audit must not copy image pixels")


class Log:
    def __init__(self):
        self.requests, self.events = [], []

    def request(self, payload):
        self.requests.append(payload)

    def event(self, name, **fields):
        self.events.append((name, fields))


class Recorder:
    def __init__(self):
        self.calls = []

    def request(self, request_id, refs):
        self.calls.append((request_id, refs))


class Policy:
    timing = "policy timing sentinel"

    def __init__(self):
        self.calls = []
        self.prediction = {
            "base_actions": np.ones((100, 14)),
            "camera_actions": np.zeros((100, 14)),
            "inference_elapsed_s": 0.1,
        }

    def predict(self, frame, **kwargs):
        self.calls.append((frame, kwargs))
        return self.prediction


@pytest.fixture
def frame():
    return SimpleNamespace(
        timestamp=100.0,
        images=ForbiddenImages(),
        base_state=np.arange(14, dtype=float),
        source_timestamps={
            **dict.fromkeys(CAMERAS, 99.9),
            "left": 99.99,
            "right": 99.995,
        },
        image_frame_ids={name: np.int64(20 + i) for i, name in enumerate(CAMERAS)},
        image_receive_timestamps=dict.fromkeys(CAMERAS, 99.95),
        image_space="raw",
        base_state_history=np.arange(28, dtype=float).reshape(2, 14),
        state_history_offsets_s=(-0.0667, -0.0333),
    )


def test_records_exact_refs_current_history_rtc_without_touching_images(frame):
    policy, log, recorder = Policy(), Log(), Recorder()
    wrapped = RecordedPolicy(policy, log, recorder)
    rtc = {
        "schema": "egoverse_rtc_v1",
        "request_id": 31,
        "base_actions": np.arange(1400).reshape(100, 14),
        "weights": np.linspace(1, 0, 100),
    }
    result = wrapped.predict(frame, rtc=rtc)
    assert wrapped.timing == policy.timing
    assert policy.calls == [(frame, {"rtc": rtc})]
    assert result["audit_request_id"] == 1
    assert "audit_request_id" not in policy.prediction
    assert result["base_actions"] is policy.prediction["base_actions"]
    started, completed = log.requests
    assert [row["phase"] for row in log.requests] == ["started", "completed"]
    assert (
        started["image_refs"]
        == recorder.calls[0][1]
        == {
            name: {"frame_id": 20 + i, "timestamp": 99.9}
            for i, name in enumerate(CAMERAS)
        }
    )
    assert started["image_receive_timestamps"] == frame.image_receive_timestamps
    assert started["source_timestamps"]["left"] == 99.99
    assert started["image_space"] == "raw"
    assert started["missing_image_frame_ids"] == []
    np.testing.assert_array_equal(started["base_state"], frame.base_state)
    np.testing.assert_array_equal(
        started["base_state_history"], frame.base_state_history
    )
    assert started["state_history_offsets_s"] == [-0.0667, -0.0333]
    np.testing.assert_array_equal(started["rtc"]["base_actions"], rtc["base_actions"])
    np.testing.assert_array_equal(started["rtc"]["weights"], rtc["weights"])
    assert completed["prediction"]["inference_elapsed_s"] == 0.1
    assert completed["monotonic"] >= started["monotonic"]
    assert "images" not in started and "images" not in completed

    # Async audit rows remain stable after feedback, prefix, and action updates.
    frame.base_state[:] = -1
    frame.base_state_history[:] = -1
    rtc["base_actions"][:] = -1
    policy.prediction["base_actions"][:] = -1
    assert started["base_state"][1] == 1
    assert started["base_state_history"][1, 0] == 14
    assert started["rtc"]["base_actions"][0, 1] == 1
    assert completed["prediction"]["base_actions"][0, 0] == 1


def test_warmup_and_policy_share_ids_and_hpt_compatibility(frame):
    class Hpt:
        def predict(self, frame):
            return {"base_actions": np.zeros((100, 14))}

    log = Log()
    wrapper = RecordedPolicy(Hpt(), log)
    wrapper.set_phase("warmup")
    assert wrapper.predict(frame)["audit_request_id"] == 1
    wrapper.phase = "policy"
    assert wrapper.predict(frame)["audit_request_id"] == 2
    assert [row["policy_phase"] for row in log.requests] == [
        "warmup",
        "warmup",
        "policy",
        "policy",
    ]
    with pytest.raises(ValueError):
        wrapper.set_phase("invalid")


def test_missing_frame_ids_are_explicit_and_do_not_prevent_prediction(frame):
    frame.image_frame_ids = {"front_1": 7}
    policy, log, recorder = Policy(), Log(), Recorder()
    result = RecordedPolicy(policy, log, recorder).predict(frame)
    assert result["audit_request_id"] == 1 and len(policy.calls) == 1
    assert log.requests[0]["missing_image_frame_ids"] == ["left_wrist", "right_wrist"]
    assert recorder.calls[0][1]["left_wrist"] == {"frame_id": None, "timestamp": 99.9}


def test_policy_error_is_recorded_and_preserved_even_when_failure_logging_breaks(frame):
    error = RuntimeError("model failed")

    class FailedPolicy:
        def predict(self, frame):
            raise error

    log = Log()
    wrapper = RecordedPolicy(FailedPolicy(), log)
    with pytest.raises(RuntimeError) as caught:
        wrapper.predict(frame)
    assert caught.value is error
    assert [row["phase"] for row in log.requests] == ["started", "failed"]
    assert log.requests[-1]["error"]["message"] == "model failed"

    class BrokenLog:
        def request(self, payload):
            raise OSError("disk full")

        def event(self, *args, **kwargs):
            raise OSError("event logger also dead")

    with pytest.raises(RuntimeError) as caught:
        RecordedPolicy(FailedPolicy(), BrokenLog()).predict(frame)
    assert caught.value is error


@pytest.mark.parametrize("failure", ["raise", "full"])
def test_recorder_and_logger_failures_never_block_policy_call(frame, failure):
    class BrokenLog(Log):
        def request(self, payload):
            if failure == "raise":
                raise OSError("logger offline")
            return False

    class BrokenRecorder:
        def request(self, *args):
            if failure == "raise":
                raise OSError("recorder offline")
            return False

    policy, log = Policy(), BrokenLog()
    wrapper = RecordedPolicy(policy, log, BrokenRecorder())
    output = wrapper.predict(frame)
    assert output["audit_request_id"] == 1 and len(policy.calls) == 1
    assert wrapper.audit_failures == {"log.request": 2, "recorder.request": 1}
    assert len(log.events) == 3


def test_prediction_pixels_are_omitted_without_copying_them(frame):
    policy, log = Policy(), Log()
    policy.prediction["images"] = ForbiddenImages()
    policy.prediction["debug"] = {
        "observation.images.front_1": ForbiddenImages(),
        "status": "ok",
    }
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    policy.prediction["unlabelled_debug_array"] = image
    result = RecordedPolicy(policy, log).predict(frame)
    saved = log.requests[-1]["prediction"]
    assert "images" not in saved
    assert saved["debug"] == {"status": "ok"}
    assert saved["unlabelled_debug_array"]["omitted"] == "image_or_high_rank_array"
    assert result["images"] is policy.prediction["images"]
    assert result["unlabelled_debug_array"] is image


def test_input_metadata_failure_still_calls_model(frame):
    frame.image_frame_ids = None
    policy, log = Policy(), Log()
    wrapper = RecordedPolicy(policy, log)
    assert wrapper.predict(frame)["audit_request_id"] == 1
    assert len(policy.calls) == 1
    assert wrapper.audit_failures == {"input_snapshot": 1}
    assert "input_snapshot_error" in log.requests[0]


def test_per_camera_recorder_receipt_is_retained_and_rejection_is_reported(frame):
    class PartialRecorder:
        def request(self, request_id, refs):
            return {"front_1": True, "left_wrist": False, "right_wrist": True}

    policy, log = Policy(), Log()
    wrapper = RecordedPolicy(policy, log, PartialRecorder())
    wrapper.predict(frame)
    assert log.requests[-1]["image_recording_submission"] == {
        "front_1": True,
        "left_wrist": False,
        "right_wrist": True,
    }
    assert wrapper.audit_failures == {"recorder.request": 1}
    assert len(policy.calls) == 1


def test_audit_never_reads_image_property(frame):
    class FrameWithoutPixelAccess:
        @property
        def images(self):
            raise AssertionError("The audit layer must not read image storage")

    metadata_only = FrameWithoutPixelAccess()
    for name, value in vars(frame).items():
        if name != "images":
            setattr(metadata_only, name, value)
    policy, log = Policy(), Log()
    assert (
        RecordedPolicy(policy, log, Recorder()).predict(metadata_only)[
            "audit_request_id"
        ]
        == 1
    )
    assert [row["phase"] for row in log.requests] == ["started", "completed"]


def test_request_ids_are_unique_across_concurrent_calls(frame):
    wrapper, log = RecordedPolicy(Policy(), Log()), Log()
    wrapper.log = log
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: wrapper.predict(frame), range(32)))
    assert sorted(result["audit_request_id"] for result in outputs) == list(
        range(1, 33)
    )
    for request_id in range(1, 33):
        assert [
            row["phase"] for row in log.requests if row["request_id"] == request_id
        ] == ["started", "completed"]
