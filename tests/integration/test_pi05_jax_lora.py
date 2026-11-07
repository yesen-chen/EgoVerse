"""Opt-in real Pi0.5 LoRA regression in a user-managed JAX interpreter.

Required EGOVERSE_TEST_PI05_{PYTHON,OPENPI_ROOT,MANIFEST,STATS_DIR,TOKENIZER,PARAMS}.
The regular test process remains in emimic; each phase releases GPU memory on exit.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.gpu
def test_real_pi05_lora_train_resume_and_inference(tmp_path):
    keys = ("PYTHON", "OPENPI_ROOT", "MANIFEST", "STATS_DIR", "TOKENIZER", "PARAMS")
    values = {key: os.environ.get(f"EGOVERSE_TEST_PI05_{key}") for key in keys}
    if missing := [key for key, value in values.items() if not value]:
        pytest.skip(f"Set explicit EGOVERSE_TEST_PI05 paths: {', '.join(missing)}")
    common = [
        "--manifest",
        values["MANIFEST"],
        "--stats-dir",
        values["STATS_DIR"],
        "--tokenizer-path",
        values["TOKENIZER"],
        "--params-path",
        values["PARAMS"],
        "--openpi-root",
        values["OPENPI_ROOT"],
        "--run-dir",
        str(tmp_path / "lora"),
    ]
    script = ROOT / "pi05/scripts/check_pi05_lora.py"
    for phase in (["--inspect"], ["--steps", "2"], ["--steps", "4", "--resume"]):
        subprocess.run(
            [values["PYTHON"], str(script), *common, *phase],
            cwd=ROOT,
            check=True,
            timeout=1800,
        )
    result = json.loads((tmp_path / "lora/lora_result.json").read_text())
    assert result["initial_state"]["step"] == 2
    assert result["final_state"]["step"] == 4
    assert (
        result["initial_state"]["frozen_samples"]
        == result["final_state"]["frozen_samples"]
    )
    checkpoint = tmp_path / "lora/checkpoints/egoverse_pi05_lora/lora_probe/3"
    output = tmp_path / "inference.json"
    subprocess.run(
        [
            values["PYTHON"],
            str(ROOT / "pi05/scripts/check_pi05_inference.py"),
            "--checkpoint-path",
            str(checkpoint),
            "--manifest",
            values["MANIFEST"],
            "--openpi-root",
            values["OPENPI_ROOT"],
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        timeout=900,
    )
    report = json.loads(output.read_text())
    assert report["passed"] is True
    assert report["actions_shape"] == [100, 14]
