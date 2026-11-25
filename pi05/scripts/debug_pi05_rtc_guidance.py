"""Run one offline diagnostic with the reference implementation's Jacobian rule.

Usage: debug_pi05_rtc_guidance.py sensitivity|chain <experiment arguments>.
The normal server and robot client are never invoked or changed.
"""

import json
import runpy
import sys
from pathlib import Path

from _bootstrap import setup


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"sensitivity", "chain"}:
        raise SystemExit(__doc__)
    experiment = sys.argv[1]
    script = Path(__file__).with_name(f"debug_pi05_rtc_{experiment}.py")
    setup()
    from egoverse_pi05.rtc_diagnostics import identity_guidance_for_diagnostic

    sys.argv = [str(script), *sys.argv[2:]]
    if "--output-dir" not in sys.argv:
        raise SystemExit("--output-dir is required")
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    with identity_guidance_for_diagnostic():
        runpy.run_path(str(script), run_name="__main__")
    (output / "experimental_gradient_mode.json").write_text(
        json.dumps(
            {
                "mode": "identity_denoised_jacobian",
                "scope": "diagnostic only, fresh process; production default unchanged",
                "correction": "(previous - (x - t*v)) * mask; same sign, scale, weights",
                "argv": sys.argv,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
