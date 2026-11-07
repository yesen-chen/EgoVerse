"""Recording identity, strict discovery and split leakage regression tests."""

from types import SimpleNamespace

import pytest
import zarr

from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.local_recordings import (
    LocalRecordingResolver,
    RecordingMultiDataset,
    recording_group,
    validate_recording_split,
)


def store(path, source, embodiment="piper_bimanual"):
    group = zarr.open_group(str(path), mode="w")
    group.attrs.update(
        {
            "features": {},
            "total_frames": 3,
            "embodiment": embodiment,
            "conversion": {"source_dataset": source},
        }
    )


def test_recursive_ids_do_not_collide_and_pins_are_strict(tmp_path):
    for batch in ("batch_a", "batch_b"):
        store(tmp_path / batch / "episode_000000.zarr", batch)
    resolver = LocalRecordingResolver(tmp_path, "piper", {})
    paths = resolver.resolve_paths(expected_embodiment="piper_bimanual")
    assert {name for _, name in paths} == {
        "batch_a/episode_000000",
        "batch_b/episode_000000",
    }
    with pytest.raises(ValueError, match="Pinned episode missing"):
        resolver.resolve_paths(DatasetFilter(episode_hashes=["episode_000000"]))
    with pytest.raises(ValueError, match="embodiment"):
        resolver.resolve_paths(expected_embodiment="human_bimanual")


def test_missing_calibration_errors_include_every_path(tmp_path):
    for batch in ("bad_a", "bad_b"):
        store(tmp_path / batch / "episode_000000.zarr", batch)
    resolver = LocalRecordingResolver(tmp_path, "piper", {})
    with pytest.raises(ValueError, match="no episodes skipped") as error:
        resolver.resolve(expected_embodiment="piper_bimanual")
    assert "bad_a/episode_000000.zarr" in str(error.value)
    assert "bad_b/episode_000000.zarr" in str(error.value)


def test_group_uses_provenance_not_output_name():
    assert recording_group(
        {"conversion": {"source_dataset": "/raw/a"}}
    ) == recording_group({"conversion": {"source_dataset": "/raw/a", "segment": 4}})
    a = {"pico_ego": {"source_tracking": {"sha256": "abc", "path": "/old/file"}}}
    b = {"pico_ego": {"source_tracking": {"sha256": "abc", "path": "/new/file"}}}
    assert recording_group(a) == recording_group(b)
    with pytest.raises(ValueError, match="provenance"):
        recording_group({})


class Leaf:
    def __init__(self, group):
        self.recording_id = group

    def __len__(self):
        return 3


def graph():
    # Two recordings, each split into multiple converted segments.
    leaves = {
        f"{group}/segment_{i}": Leaf(group) for group in ("a", "b") for i in range(3)
    }
    return SimpleNamespace(resolve=lambda *args, **kwargs: leaves)


def test_all_segments_stay_in_one_split_and_selection_is_reproducible():
    train = RecordingMultiDataset._from_resolver(graph(), mode="train", valid_ratio=0.5)
    valid = RecordingMultiDataset._from_resolver(graph(), mode="valid", valid_ratio=0.5)
    assert len(train.datasets) == len(valid.datasets) == 3
    assert train.selected_recordings.isdisjoint(valid.selected_recordings)
    assert train.selected_recordings | valid.selected_recordings == {"a", "b"}
    again = RecordingMultiDataset._from_resolver(graph(), mode="train", valid_ratio=0.5)
    assert train.datasets.keys() == again.datasets.keys()
    validate_recording_split({"piper": train}, {"piper": valid})
    with pytest.raises(ValueError, match="overlap"):
        validate_recording_split({"piper": train}, {"piper": again})


def test_one_recording_cannot_be_silently_frame_split():
    resolver = SimpleNamespace(resolve=lambda *args, **kwargs: {"a/segment": Leaf("a")})
    with pytest.raises(ValueError, match="Empty train split"):
        RecordingMultiDataset._from_resolver(resolver, mode="train")
