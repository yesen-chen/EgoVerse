"""Report tests require no models, optional OpenPI libraries, or devices."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "pi05/scripts/report_pi05_rtc_history.py"
)
_SPEC = importlib.util.spec_from_file_location("rtc_history_report", _SCRIPT)
report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report)


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def fixture_runs(tmp_path):
    selection = [
        {
            "episode": episode,
            "episode_name": f"episode_{episode}.zarr",
            "previous_row": index * 50,
            "current_row": index * 50 + 27,
            "activation": float(episode * 100 + index),
        }
        for episode in range(3)
        for index in range(2)
    ]
    for model_index, name in enumerate(report.MODELS):
        folder = tmp_path / name
        folder.mkdir()
        write_json(folder / "selection.json", selection)
        write_json(
            folder / "summary.json",
            {
                "selection_sha256": report.canonical_hash(selection),
                "pairs": 6,
                "episodes": 3,
                "timing": {"source_horizon": 50, "source_fps": 30},
            },
        )
        write_json(
            folder / "arguments.json",
            {
                **dict.fromkeys(report.RUNTIME_FIELDS, 1),
                "checkpoint_path": name,
            },
        )
        rows = [
            {
                "pair_index": index,
                **pair,
                "static": {"left": True, "right": index < 2},
                "metrics": {
                    mode: {
                        side: dict.fromkeys(
                            report.CORE_METRICS, 10 + index - model_index
                        )
                        for side in report.SIDES
                    }
                    for mode in report.MODES
                },
            }
            for index, pair in enumerate(selection)
        ]
        (folder / "samples.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows)
        )
    return tmp_path


@pytest.mark.parametrize(
    "failure",
    [
        "selection_hash",
        "selection_pair",
        "sample_id",
        "runtime",
        "timing",
        "static",
        "missing",
        "duplicate",
        "nonfinite",
    ],
)
def test_refuses_unpaired_or_invalid_input(fixture_runs, failure):
    folder = fixture_runs / "history_33ms"
    selection = report.read_json(folder / "selection.json")
    summary = report.read_json(folder / "summary.json")
    args = report.read_json(folder / "arguments.json")
    rows = [
        json.loads(line) for line in (folder / "samples.jsonl").read_text().splitlines()
    ]
    if failure == "selection_hash":
        summary["selection_sha256"] = "corrupt"
    elif failure == "selection_pair":
        selection[0]["activation"] += 0.1
        rows[0]["activation"] += 0.1
        summary["selection_sha256"] = report.canonical_hash(selection)
    elif failure == "sample_id":
        rows[0]["current_row"] += 1
    elif failure == "runtime":
        args["transition_ms"] += 1
    elif failure == "timing":
        summary["timing"]["source_horizon"] = 30
    elif failure == "static":
        rows[0]["static"]["left"] = False
    elif failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        selection[1] = copy.deepcopy(selection[0])
        summary["selection_sha256"] = report.canonical_hash(selection)
    elif failure == "nonfinite":
        rows[0]["metrics"]["rtc_self"]["left"]["raw_switch_mm"] = float("nan")
    write_json(folder / "selection.json", selection)
    write_json(folder / "summary.json", summary)
    write_json(folder / "arguments.json", args)
    (folder / "samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        report.load_paired_runs(fixture_runs)


def test_bootstrap_direction_seed_and_episode_grouping():
    reference = [10, 10, 10, 10]
    candidate = [9, 9, 9, 1]
    episodes = [0, 0, 0, 1]
    stats = report.paired_episode_bootstrap(reference, candidate, episodes)
    assert stats == report.paired_episode_bootstrap(reference, candidate, episodes)
    assert stats["episodes"] == 2
    assert stats["mean_difference"] == -5  # Episode means -1, -9, not window mean -3.
    assert stats["ci95"] == [-9, -1]
    reverse = report.paired_episode_bootstrap(candidate, reference, episodes)
    assert reverse["mean_difference"] == 5
    assert reverse["ci95"] == [1, 9]


def test_bootstrap_empty_and_shape_validation():
    assert report.paired_episode_bootstrap([], [], [])["ci95"] is None
    with pytest.raises(ValueError, match="matching"):
        report.paired_episode_bootstrap([1, 2], [1], [0, 1])
    with pytest.raises(ValueError, match="invalid"):
        report.paired_episode_bootstrap([1], [np.nan], [0])


def test_comparison_static_counts_and_mean_of_window_p95(fixture_runs):
    result = report.build_comparison(report.load_paired_runs(fixture_runs), repeats=100)
    assert result["static_counts"] == {
        "left": {"pairs": 6, "episodes": 3},
        "right": {"pairs": 2, "episodes": 1},
    }
    stats = result["metrics"]["all"]["rtc_self"]["left"]["transition_d2_p95_mm"][
        "baseline"
    ]
    assert stats["mean"] == 12.5
    difference = result["paired_differences"]["all"]["rtc_self"]["left"][
        "raw_switch_mm"
    ]["history_66_33ms"]
    assert difference["mean_difference"] == -2
    assert difference["ci95"] == [-2, -2]
    assert "not a global P95" in result["definitions"]["transition_d2_p95_mm"]


def test_report_outputs(fixture_runs):
    report.main(["--root", str(fixture_runs), "--bootstrap-repeats", "10"])
    output = report.read_json(fixture_runs / "comparison.json")
    assert output["bootstrap"]["repeats"] == 10
    assert (fixture_runs / "comparison_plot.png").stat().st_size > 10000
    table = (fixture_runs / "comparison_tables.md").read_text()
    assert "mean of per-window P95" in table
    assert "Bootstrap: 10 draws" in table
    assert all(mode in table for mode in report.MODES)
