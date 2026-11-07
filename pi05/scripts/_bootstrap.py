"""Import source trees without installing anything into the robot environment."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def setup(*, cpu=False, openpi_root=None):
    sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent)]
    upstream = (
        Path(openpi_root or os.environ.get("EGOVERSE_OPENPI_ROOT", ROOT / "openpi"))
        .expanduser()
        .resolve()
    )
    if not (upstream / "src/openpi/models/pi0_config.py").is_file():
        raise FileNotFoundError(
            "Set --openpi-root / EGOVERSE_OPENPI_ROOT to the pinned OpenPI checkout; see pi05/README.md"
        )
    from egoverse_pi05 import OPENPI_REVISION

    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != OPENPI_REVISION:
        raise ValueError(
            f"Expected OpenPI revision {OPENPI_REVISION}, found {revision}"
        )
    sys.path[:0] = [
        str(upstream),
        str(upstream / "src"),
        str(upstream / "packages/openpi-client/src"),
    ]
    os.environ["JAX_PLATFORMS"] = "cpu" if cpu else "cuda"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    return upstream
