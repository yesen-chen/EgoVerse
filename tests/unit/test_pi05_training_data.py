"""Mixed Pi0.5 sampling and train-only statistics on synthetic recordings."""

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import zarr
from fixtures.piper_pico_training import write_recording

from egomimic.robot.piper_pico import pi05_data as piper
from egomimic.robot.piper_pico.image_geometry import make_image_geometry
from egomimic.robot.piper_pico.pi05_ego_data import export_ego_manifest
from egomimic.robot.piper_pico.pi05_training_data import (
    TrainingDataset,
    deployment_manifest,
    domain_datasets,
    export_mixed_manifest,
    load_training_manifest,
    load_training_stats,
    validate_training_manifest,
    write_training_stats,
)


def make_recordings(root):
    """Unequal domain sizes and visibly different numeric distributions."""
    paths = {"piper": [], "ego": []}
    for index, frames in enumerate((16, 20, 24)):
        path = write_recording(
            root / "piper",
            "piper",
            f"recording_{index}",
            frames=frames,
            offset=5.0 + 10 * index,
        )
        paths["piper"].append(path)
    K = [[25.0, 0.0, 16.0], [0.0, 25.0, 16.0], [0.0, 0.0, 1.0]]
    geometry = make_image_geometry((32, 32), (32, 32), K=K)
    for index, (frames, reason) in enumerate(
        zip((35, 37, 39), ("ui_trim", "inactivity", "tracking_gap"), strict=True)
    ):
        path = write_recording(
            root / "ego",
            "pico_ego",
            f"recording_{index}",
            frames=frames,
            offset=-10.0 - 10 * index,
        )
        group = zarr.open_group(str(path), mode="a")
        group.attrs.update(
            {
                "intrinsics": {"front_1": K},
                "image_geometry": {"front_1": geometry},
                "image_space": "processed",
                "preprocessing": {
                    "schema": "ego_preprocessing_v1",
                    "start_reason": "ui_trim",
                    "end_reason": reason,
                },
            }
        )
        paths["ego"].append(path)
    return paths


@pytest.fixture
def recordings(tmp_path):
    return make_recordings(tmp_path / "recordings")


def manifests(root, recordings, *, validation_ratio=0, trim=True, inactivity=True):
    kwargs = {
        "validation_ratio": validation_ratio,
        "seed": 7,
        "prompt": "Pick and place.",
    }
    real = piper.export_manifest(
        recordings["piper"],
        root / "piper.json",
        source_horizon=50,
        allow_synthetic_calibration=True,
        **kwargs,
    )
    ego = export_ego_manifest(
        recordings["ego"],
        root / "ego.json",
        source_horizon=30,
        require_full_window_after_trim=trim,
        require_full_window_after_inactivity=inactivity,
        **kwargs,
    )
    return real, ego


def mix(root, sources, *, weights=(1, 1), seed=42):
    return export_mixed_manifest(
        *sources,
        root / "mixed.json",
        piper_weight=weights[0],
        ego_weight=weights[1],
        seed=seed,
    )


def oracle_statistics(dataset):
    samples = [dataset[index] for index in range(len(dataset))]
    values = {
        "state": np.stack([sample["observation.state"] for sample in samples]).astype(
            float
        ),
        "actions": np.concatenate([sample["actions"] for sample in samples]).astype(
            float
        ),
    }
    return {
        key: {
            "mean": value.mean(axis=0),
            "std": value.std(axis=0),
            "q01": np.quantile(value, 0.01, axis=0),
            "q99": np.quantile(value, 0.99, axis=0),
        }
        for key, value in values.items()
    }


@pytest.mark.parametrize("weights", [(1, 1), (2, 1), (1, 2)])
def test_training_ratio_covers_every_eligible_start_deterministically(
    tmp_path, recordings, weights
):
    sources = manifests(tmp_path / "sources", recordings)
    path = mix(tmp_path / "mix", sources, weights=weights)
    first = TrainingDataset(path, include_images=False)
    second = TrainingDataset(path, include_images=False)
    assert first.schedule == second.schedule
    counts = Counter(domain for domain, _ in first.schedule)
    assert counts["piper"] * weights[1] == counts["ego"] * weights[0]
    assert {domain: len(dataset) for domain, dataset in first.datasets.items()} == {
        "piper": 60,
        "ego": 24,
    }
    for domain, dataset in first.datasets.items():
        assert {index for name, index in first.schedule if name == domain} == set(
            range(len(dataset))
        )
    for position in (0, 1, len(first) - 1):
        domain, local = first.schedule[position]
        actual, expected = first[position], first.datasets[domain][local]
        assert actual["domain"] == domain
        assert actual["gripper_mode"] == ("width" if domain == "piper" else "zero")
        np.testing.assert_array_equal(actual["actions"], expected["actions"])
        np.testing.assert_array_equal(
            actual["observation.state"], expected["observation.state"]
        )
    changed = TrainingDataset(
        mix(tmp_path / "seed_43", sources, weights=weights, seed=43),
        include_images=False,
    )
    assert changed.schedule != first.schedule
    assert len(changed) == len(first)
    assert Counter(domain for domain, _ in changed.schedule) == counts
    for invalid in (-1, len(first), True, 0.5):
        with pytest.raises(IndexError):
            first[invalid]


def test_validation_has_no_oversampling_and_keeps_training_sources_out(
    tmp_path, recordings
):
    path = mix(
        tmp_path / "mix",
        manifests(tmp_path / "sources", recordings, validation_ratio=0.34),
        weights=(2, 1),
    )
    train = domain_datasets(path, split="train")
    validation = TrainingDataset(path, split="validation", include_images=False)
    assert len(validation) == sum(len(ds) for ds in validation.datasets.values())
    assert len(validation.schedule) == len(set(validation.schedule))
    for domain, dataset in validation.datasets.items():
        assert Counter(name for name, _ in validation.schedule)[domain] == len(dataset)
        train_paths = {record["path"] for record in train[domain].episodes}
        valid_paths = {record["path"] for record in dataset.episodes}
        assert train_paths and valid_paths and train_paths.isdisjoint(valid_paths)


def test_stats_are_train_only_separate_and_independent_of_sampling_weights(
    tmp_path, recordings, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "pi05/src"))
    sources = manifests(tmp_path / "sources", recordings, validation_ratio=0.34)
    equal_path = mix(tmp_path / "equal", sources)
    other_path = mix(tmp_path / "other", sources, weights=(3, 1))
    reports = [
        write_training_stats(path, tmp_path / f"stats_{index}")
        for index, path in enumerate((equal_path, other_path))
    ]
    domains = domain_datasets(equal_path, split="train")
    all_domains = domain_datasets(equal_path, split="all")
    for domain, filename in (
        ("piper", "norm_stats.json"),
        ("ego", "ego_norm_stats.json"),
    ):
        first_file, second_file = (
            tmp_path / "stats_0" / filename,
            tmp_path / "stats_1" / filename,
        )
        assert first_file.read_bytes() == second_file.read_bytes()
        raw = json.loads(first_file.read_text())["norm_stats"]
        expected = oracle_statistics(domains[domain])
        including_validation = oracle_statistics(all_domains[domain])
        assert not np.allclose(
            raw["state"]["mean"], including_validation["state"]["mean"]
        )
        for key, fields in expected.items():
            for field, value in fields.items():
                np.testing.assert_allclose(raw[key][field], value, atol=1e-12)
        for report in reports:
            entry = report["domains"][domain]
            assert (
                entry["eligible_training_windows"]
                == entry["sampled_windows"]
                == len(domains[domain])
            )
            assert (
                entry["sha256"] == hashlib.sha256(first_file.read_bytes()).hexdigest()
            )
    loaded = load_training_stats(
        load_training_manifest(equal_path), tmp_path / "stats_0"
    )
    assert set(loaded) == {"piper", "ego"}
    assert not np.allclose(
        loaded["piper"]["norm_stats"]["state"]["mean"],
        loaded["ego"]["norm_stats"]["state"]["mean"],
    )
    with pytest.raises(ValueError, match="do not match"):
        load_training_stats(load_training_manifest(other_path), tmp_path / "stats_0")


@pytest.mark.parametrize(
    "trim,inactivity,expected_ego",
    [(True, True, 24), (False, True, 53), (True, False, 82), (False, False, 111)],
)
def test_boundary_counts_match_unique_statistics_and_piper_retains_terminal_repeat(
    tmp_path, recordings, trim, inactivity, expected_ego
):
    original_timestamps = [
        zarr.open_group(str(path), mode="r")["timestamp"][:]
        for path in recordings["ego"]
    ]
    sources = manifests(
        tmp_path / "sources", recordings, trim=trim, inactivity=inactivity
    )
    path = mix(tmp_path / "mix", sources)
    datasets = domain_datasets(path)
    report = write_training_stats(path, tmp_path / "stats")
    assert len(datasets["ego"]) == expected_ego
    assert report["domains"]["ego"]["eligible_training_windows"] == expected_ego
    assert report["domains"]["ego"]["sampled_windows"] == expected_ego
    assert len(datasets["piper"]) == 60
    assert report["domains"]["piper"]["sampled_windows"] == 60
    # First Piper episode ends at row15 even though its horizon is50.
    terminal = datasets["piper"][15]["actions"]
    np.testing.assert_allclose(
        terminal, np.repeat(terminal[:1], 100, axis=0), atol=1e-6
    )
    assert not np.allclose(terminal[0], datasets["piper"][16]["actions"][0])
    for path, rows, original in zip(
        recordings["ego"], (35, 37, 39), original_timestamps, strict=True
    ):
        store = zarr.open_group(str(path), mode="r")
        assert store.attrs["total_frames"] == rows
        np.testing.assert_array_equal(store["timestamp"][:], original)


def test_child_manifest_changes_and_embedded_timing_tampering_are_rejected(
    tmp_path, recordings
):
    sources = manifests(tmp_path / "sources", recordings)
    path = mix(tmp_path / "mix", sources)
    original = load_training_manifest(path)
    changed = copy.deepcopy(original)
    changed["action_timings"]["ego"]["effective_span_s"] = 49 / 30
    changed["manifest_sha256"] = piper._manifest_hash(changed)
    with pytest.raises(ValueError, match="timing disagrees"):
        validate_training_manifest(changed)
    changed_child = json.loads(sources[1].read_text())
    changed_child["episodes"][0]["prompt"] = "Different task."
    sources[1].write_text(json.dumps(changed_child))
    with pytest.raises(ValueError, match="child manifest changed"):
        domain_datasets(path)
    # Portable metadata validation does not require training files to remain.
    real = deployment_manifest(original)
    assert real == original["sources"]["piper"]["manifest"]
    assert real["action_timing"]["effective_span_s"] == pytest.approx(49 / 30)
    assert original["action_timings"]["ego"]["effective_span_s"] == pytest.approx(
        29 / 30
    )
    real["episodes"][0]["prompt"] = "Mutation of returned copy."
    assert (
        original["sources"]["piper"]["manifest"]["episodes"][0]["prompt"]
        == "Pick and place."
    )


@pytest.mark.parametrize("domain", ["piper", "ego"])
def test_stats_files_have_separate_checked_hashes(
    tmp_path, recordings, monkeypatch, domain
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "pi05/src"))
    path = mix(tmp_path / "mix", manifests(tmp_path / "sources", recordings))
    report = write_training_stats(path, tmp_path / "stats", max_windows=3)
    assert report["smoke_test_statistics"] is True
    assert {entry["sampled_windows"] for entry in report["domains"].values()} == {3}
    stats_file = tmp_path / "stats" / report["domains"][domain]["filename"]
    stats_file.write_text(stats_file.read_text() + " ")
    with pytest.raises(ValueError, match=f"{domain} normalization identity mismatch"):
        load_training_stats(load_training_manifest(path), tmp_path / "stats")


def test_piper_only_training_remains_compatible(tmp_path, recordings, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "pi05/src"))
    real, _ = manifests(tmp_path / "sources", recordings)
    original = piper.Pi05Dataset(real, include_images=False)
    wrapped = TrainingDataset(real, include_images=False)
    assert len(wrapped) == len(original) == 60
    assert wrapped.schedule == [("piper", index) for index in range(60)]
    np.testing.assert_array_equal(wrapped[59]["actions"], original[59]["actions"])
    report = write_training_stats(real, tmp_path / "stats")
    assert set(report["domains"]) == {"piper"}
    assert not (tmp_path / "stats/ego_norm_stats.json").exists()
    raw = load_training_stats(load_training_manifest(real), tmp_path / "stats")
    expected = piper.compute_norm_stats(real)
    assert raw["piper"]["norm_stats"] == expected
