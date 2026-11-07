"""Offline end-to-end geometry acceptance with small encoded synthetic videos."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from egomimic.robot.piper_pico.image_geometry import (
    RgbImageProcessor,
    make_image_geometry,
)
from egomimic.scripts import verify_uvc_pipeline as acceptance

SIZES = ((128, 72), (64, 48), (64, 48))


@pytest.fixture
def small_result(tmp_path):
    output = tmp_path / "acceptance"
    report = acceptance.run_acceptance(
        output, native_sizes=SIZES, output_size=(64, 48), frames=4
    )
    assert report["status"] == "PASS", report["errors"]
    return output, report


def test_synthetic_roundtrip_verifies_distortion_native_sources_and_numeric_semantics(
    small_result,
):
    output, report = small_result
    assert report["legacy"] == []
    case = report["synthetic"]
    assert case["synthetic"] and case["source_files_unchanged"]
    assert case["borrowed_calibration_applied"] is False
    episode = case["episodes"][0]
    assert episode["numeric"]["rows"] == 4
    # 5 global numeric fields + 9 per arm + 4 per camera; annotations are JSON.
    assert episode["numeric"]["fields_checked"] == 5 + 2 * 9 + 3 * 4
    assert episode["numeric"]["annotations_checked"] == 0
    assert episode["numeric"]["max_absolute_error"] < 1e-10
    assert [
        camera["image_geometry"]["source_size"] for camera in episode["cameras"]
    ] == [list(size) for size in SIZES]
    for camera in episode["cameras"]:
        contract = camera["image_geometry"]
        assert contract["D_raw"][0] < 0
        assert contract["operation"] == "undistort_resize"
        assert contract["D_output"] == [0.0] * 5
        assert contract["output_size"] == [64, 48]
        assert camera["source_frames"] == 4
        assert [sample["source_frame_index"] for sample in camera["samples"]] == [
            0,
            1,
            2,
        ]
        assert all(Path(sample["comparison"]).is_file() for sample in camera["samples"])
    assert json.loads((output / "report.json").read_text())["status"] == "PASS"


def test_legacy_conversion_keeps_unknown_geometry_and_source_files(tmp_path):
    source = acceptance.write_synthetic_capture(
        tmp_path / "old_capture", native_sizes=SIZES, frames=4
    )
    (source / "calibration.json").unlink()
    config_path = source / "videos/camera_config.json"
    specs = json.loads(config_path.read_text())
    for spec in specs:
        del spec["calibration"]
    config_path.write_text(json.dumps(specs))
    before = acceptance.hash_tree(source)
    report = acceptance.run_acceptance(
        tmp_path / "out",
        real_inputs=[source],
        native_sizes=SIZES,
        output_size=(64, 48),
        frames=4,
    )
    assert report["status"] == "PASS", report["errors"]
    assert acceptance.hash_tree(source) == before
    case = report["legacy"][0]
    assert case["source_calibration_present"] is False
    assert (
        case["source_files_unchanged"] and case["borrowed_calibration_applied"] is False
    )
    episode = case["episodes"][0]
    store = zarr.open_group(episode["converted"], mode="r")
    assert store.attrs["calibration_status"] == "missing"
    assert "extrinsics" not in store.attrs
    for camera in episode["cameras"]:
        contract = camera["image_geometry"]
        assert contract["operation"] == "resize"
        assert contract["K_raw"] is None and contract["K_output"] is None
        assert contract["D_raw"] is None and contract["D_output"] is None


def test_numeric_corruption_is_detected_independently(small_result):
    output, report = small_result
    converted = report["synthetic"]["episodes"][0]["converted"]
    store = zarr.open_group(converted, mode="a")
    store["left.cmd_ee_pose"][0, 0] += 0.1
    with pytest.raises(ValueError, match="source semantics: left.cmd_ee_pose"):
        acceptance.verify_numeric(output / "synthetic_raw", 0, store)


def test_pixel_metric_rejects_wrong_shapes_values_and_thresholds():
    black = np.zeros((5, 7, 3), dtype=np.uint8)
    white = np.full_like(black, 255)
    assert (
        acceptance.pixel_comparison(black, black, max_mae=0)["mean_absolute_error"] == 0
    )
    with pytest.raises(ValueError, match="MAE"):
        acceptance.pixel_comparison(black, white, max_mae=8)
    with pytest.raises(ValueError, match="matching"):
        acceptance.pixel_comparison(black, black[:2], max_mae=8)
    with pytest.raises(ValueError, match="uint8"):
        acceptance.pixel_comparison(black, black.astype(float), max_mae=8)
    with pytest.raises(ValueError, match="finite"):
        acceptance.pixel_comparison(black, black, max_mae=float("nan"))


def test_analytic_brown_oracle_matches_float_remap_without_fixed_point_rounding():
    raw = acceptance._pattern(128, 72, 0, 0)
    K = [[102.4, 0, 62.72], [0, 79.2, 36.72], [0, 0, 1]]
    D = [-0.22, 0.06, 0.001, -0.002, -0.008]
    contract = make_image_geometry((128, 72), (64, 48), K=K, D=D, alpha=0.0)
    oracle = acceptance.independent_geometry(raw, contract)
    actual = RgbImageProcessor(contract).apply(raw)
    np.testing.assert_array_equal(actual, oracle)


def test_annotation_bytes_are_verified_and_numeric_values_cannot_hide_in_annotations():
    feature = {
        "dtype": "json",
        "format": "annotation_v1",
        "shape": [0],
        "names": ["json"],
    }
    assert (
        acceptance.verify_annotations(
            np.array([], dtype=object), feature, "variable_length_bytes", frames=4
        )
        == 0
    )
    with pytest.raises(ValueError, match="variable_length_bytes"):
        acceptance.verify_annotations(
            np.array([], dtype=float), feature, "float64", frames=4
        )
    feature["shape"] = [1]
    with pytest.raises(ValueError, match="JSON bytes"):
        acceptance.verify_annotations(
            np.array([1], dtype=object), feature, "variable_length_bytes", frames=4
        )
    with pytest.raises(ValueError, match="requested task"):
        acceptance.verify_annotations(
            np.array(
                [b'{"text":"unexpected","start_idx":0,"end_idx":4}'], dtype=object
            ),
            feature,
            "variable_length_bytes",
            frames=4,
        )


def test_tampered_k_and_hash_are_rejected_by_independent_oracle(small_result):
    _, report = small_result
    contract = report["synthetic"]["episodes"][0]["cameras"][0]["image_geometry"]
    raw = np.zeros((72, 128, 3), dtype=np.uint8)
    contract["K_output"][0][0] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        acceptance.independent_geometry(raw, contract)
    contract["sha256"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in contract.items() if k != "sha256"},
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="stored K"):
        acceptance.independent_geometry(raw, contract)


def test_existing_output_and_invalid_inputs_fail_before_any_write(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not begin synthetic generation")

    monkeypatch.setattr(acceptance, "write_synthetic_capture", forbidden)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "sentinel").write_text("untouched")
    with pytest.raises(FileExistsError, match="new directory"):
        acceptance.run_acceptance(existing)
    assert sorted(path.name for path in existing.iterdir()) == ["sentinel"]
    with pytest.raises(ValueError, match="raw capture"):
        acceptance.run_acceptance(
            tmp_path / "missing_source_output", real_inputs=[tmp_path / "missing"]
        )
    assert not (tmp_path / "missing_source_output").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_frame_age": float("nan")},
        {"max_frame_age": 0},
        {"frames": 3},
        {"frames": True},
        {"native_sizes": ((127, 72), (64, 48), (64, 48))},
        {"output_size": (0, 48)},
    ],
)
def test_invalid_options_never_create_output(tmp_path, kwargs):
    with pytest.raises(ValueError):
        acceptance.run_acceptance(tmp_path / "out", **kwargs)
    assert not (tmp_path / "out").exists()
