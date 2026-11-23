"""End-to-end dataset/replay/production-client parity with synthetic episodes."""

import numpy as np
import pytest
import zarr

from egomimic.robot.piper_pico.image_geometry import camera_image_geometry
from tests.integration.check_piper_history_parity import check_parity
from tests.unit.test_pi05_state_history_data import export_piper, piper_episode


def live_episode(root, **kwargs):
    path, transforms = piper_episode(root, **kwargs)
    group = zarr.open_group(str(path), mode="a")
    calibration = dict(group.attrs["calibration"])
    calibration["data"]["image_sizes"] = {"front_1": [16, 12]}
    group.attrs["calibration"] = calibration
    group.attrs["image_processing"] = {
        name: camera_image_geometry(name, {}, calibration["data"], [16, 12], [16, 12])
        for name in ("front_1", "left_wrist", "right_wrist")
    }
    group.attrs["image_space"] = "processed"
    return path, transforms


@pytest.mark.parametrize("offsets", [(), (-0.025,), (-0.09, -0.047, -0.021)])
def test_full_training_and_live_request_agree(tmp_path, offsets):
    episode, _ = live_episode(tmp_path / "data", first_offset=0)
    manifest = export_piper(tmp_path, episode, offsets)
    report = check_parity(manifest, samples=8)
    assert report["passed"]
    assert report["history_offsets_s"] == list(offsets)
    if offsets:
        assert "observation.state_history" in report["maximum_absolute_errors"]
        assert report["locations"][0]["source_row"] > 0
    else:
        assert "observation.state_history" not in report["maximum_absolute_errors"]
    assert max(report["maximum_absolute_errors"].values()) < 2e-6


def test_parity_checker_detects_corrupt_online_history(tmp_path, monkeypatch):
    from dataclasses import replace

    from egomimic.scripts.replay_piper_policy import RecordedObservationProvider

    episode, _ = live_episode(tmp_path / "data")
    manifest = export_piper(tmp_path, episode, (-0.05,))
    original = RecordedObservationProvider.read_frame

    def corrupt(self, row):
        frame = original(self, row)
        return replace(
            frame,
            base_state_history=frame.base_state_history + np.array([0.01] + [0.0] * 13),
        )

    monkeypatch.setattr(RecordedObservationProvider, "read_frame", corrupt)
    with pytest.raises(AssertionError, match="state_history error"):
        check_parity(manifest, 2)
