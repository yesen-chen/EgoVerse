"""Optional real OpenPI CPU preprocessing regression, isolated from Torch tests."""

import os
import subprocess
from pathlib import Path

import pytest


def test_real_openpi_mixed_input_contracts():
    interpreter = os.environ.get("EGOVERSE_TEST_PI05_PYTHON")
    if not interpreter:
        pytest.skip("Set EGOVERSE_TEST_PI05_PYTHON to run optional CPU OpenPI checks")
    root = Path(__file__).resolve().parents[2]
    command = [interpreter, str(root / "tests/integration/check_pi05_mixed_inputs.py")]
    if upstream := os.environ.get("EGOVERSE_TEST_PI05_OPENPI_ROOT"):
        command.extend(["--openpi-root", upstream])
    result = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        timeout=120,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
