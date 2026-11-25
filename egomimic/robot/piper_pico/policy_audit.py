"""Best-effort policy-request audit without moving camera pixels in the caller.

The log and recorder adapters must submit to bounded queues without waiting.
This wrapper never preprocesses observations, reads frame.images, encodes JSON,
or copies image pixels. Small state/prefix/output arrays are owned snapshots so
the asynchronous writer cannot observe later model or actuator mutations.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Mapping
from threading import Lock

import numpy as np

CAMERAS = ("front_1", "left_wrist", "right_wrist")
SCHEMA = "piper_policy_request_v1"


def _snapshot(value):
    """Copy numeric audit data, omitting images without inspecting their pixels."""
    if isinstance(value, Mapping):
        return {
            key: _snapshot(item)
            for key, item in value.items()
            if not (
                isinstance(key, str)
                and (
                    key in ("image", "images") or key.startswith("observation.images.")
                )
            )
        }
    if isinstance(value, np.ndarray):
        if value.ndim >= 3:
            return {
                "omitted": "image_or_high_rank_array",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_snapshot(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    # Policies currently return mappings of NumPy arrays and scalar metadata.
    # Unknown payload objects must not invoke arbitrary conversion or GPU work.
    return {"omitted": "unsupported_object", "type": type(value).__name__}


class RecordedPolicy:
    """Record every prediction, including warmup, before actuator modification.

    ``request_id`` counts all calls for this wrapper/session and is independent
    of the RTC planner's request IDs (which can restart after pauses). The
    returned ``audit_request_id`` links execution cycles to these request rows.
    ``phase`` identifies warmup versus ordinary inference; payload ``phase``
    identifies the request lifecycle: started/completed/failed.
    """

    def __init__(self, policy, log, recorder=None):
        self.policy = policy
        self.log = log
        self.recorder = recorder
        self.phase = "policy"
        self._request_ids = itertools.count(1)
        self._id_lock = Lock()
        self.audit_failures = {}

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def set_phase(self, phase):
        if phase not in ("warmup", "policy"):
            raise ValueError("policy audit phase must be warmup or policy")
        self.phase = phase

    def _failure(self, request_id, stage, exc):
        self.audit_failures[stage] = self.audit_failures.get(stage, 0) + 1
        try:
            self.log.event(
                "policy_audit_error",
                audit_request_id=request_id,
                stage=stage,
                error={"type": type(exc).__name__, "message": str(exc)[:2000]},
            )
        except Exception:
            # A failed diagnostic channel must not recurse or mask policy work.
            pass

    def _emit(self, request_id, payload):
        try:
            if self.log.request(payload) is False:
                self._failure(
                    request_id,
                    "log.request",
                    RuntimeError("audit queue rejected request"),
                )
        except Exception as exc:
            self._failure(request_id, "log.request", exc)

    def predict(self, frame, *, rtc=None):
        with self._id_lock:
            request_id = next(self._request_ids)
        started_wall = time.time()
        started_mono = time.monotonic()
        common = {
            "schema": SCHEMA,
            "request_id": request_id,
            "policy_phase": self.phase,
            "started_walltime": started_wall,
            "started_monotonic": started_mono,
        }
        refs = {}
        try:
            frame_ids = getattr(frame, "image_frame_ids", {})
            received = getattr(frame, "image_receive_timestamps", {})
            sources = frame.source_timestamps
            refs = {
                name: {
                    "frame_id": _snapshot(frame_ids.get(name)),
                    "timestamp": _snapshot(sources.get(name)),
                }
                for name in CAMERAS
            }
            common.update(
                observation_timestamp=_snapshot(frame.timestamp),
                source_timestamps=_snapshot(sources),
                image_refs=refs,
                image_receive_timestamps={
                    name: _snapshot(received.get(name)) for name in CAMERAS
                },
                missing_image_frame_ids=[
                    name for name in CAMERAS if refs[name]["frame_id"] is None
                ],
                base_state=_snapshot(frame.base_state),
                base_state_history=_snapshot(
                    getattr(frame, "base_state_history", None)
                ),
                state_history_offsets_s=_snapshot(
                    getattr(frame, "state_history_offsets_s", ())
                ),
                image_space=getattr(frame, "image_space", None),
                rtc=_snapshot(rtc),
            )
        except Exception as exc:
            common["input_snapshot_error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:2000],
            }
            self._failure(request_id, "input_snapshot", exc)
        self._emit(
            request_id,
            {
                **common,
                "phase": "started",
                "walltime": started_wall,
                "monotonic": started_mono,
            },
        )
        if self.recorder is not None:
            try:
                # Only tiny exact-frame references cross the recorder queue.
                accepted = self.recorder.request(request_id, _snapshot(refs))
                common["image_recording_submission"] = _snapshot(accepted)
                rejected = (
                    [name for name, value in accepted.items() if value is False]
                    if isinstance(accepted, Mapping)
                    else []
                )
                if accepted is False or rejected:
                    self._failure(
                        request_id,
                        "recorder.request",
                        RuntimeError(
                            "recorder queue rejected request"
                            + (
                                f" for cameras: {', '.join(rejected)}"
                                if rejected
                                else ""
                            )
                        ),
                    )
            except Exception as exc:
                self._failure(request_id, "recorder.request", exc)
        try:
            prediction = (
                self.policy.predict(frame)
                if rtc is None
                else self.policy.predict(frame, rtc=rtc)
            )
        except BaseException as exc:
            self._emit(
                request_id,
                {
                    **common,
                    "phase": "failed",
                    "walltime": time.time(),
                    "monotonic": time.monotonic(),
                    "error": {"type": type(exc).__name__, "message": str(exc)[:2000]},
                },
            )
            raise
        try:
            completed = {
                **common,
                "phase": "completed",
                "walltime": time.time(),
                "monotonic": time.monotonic(),
                "prediction": _snapshot(prediction),
            }
            self._emit(request_id, completed)
        except Exception as exc:
            self._failure(request_id, "output_snapshot", exc)
        # Preserve every model output and array; do not mutate the policy's dict.
        # A nonmapping result remains the existing caller's validation concern.
        if isinstance(prediction, Mapping):
            return {**prediction, "audit_request_id": request_id}
        return prediction
