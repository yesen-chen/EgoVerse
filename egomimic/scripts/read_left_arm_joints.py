"""Read the left Piper arm's joint feedback without sending a control command.

``HOME_QPOS`` is radians. ``send_joints`` multiplies that vector by
``JOINT_FACTOR`` before it reaches the firmware, whose native joint unit is
0.001 degree. This script opens the CAN socket, waits for a joint feedback
frame, and prints the radian vector that belongs in ``HOME_QPOS``. It does
not enable motors, query firmware, or send motion or gripper commands.

Example::

    python -m egomimic.scripts.read_left_arm_joints --can can0
    python -m egomimic.scripts.read_left_arm_joints --can can0 --write-home
"""

from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path

import numpy as np

from egomimic.robot.piper_pico.units import joints_sdk_to_rad

_HOME_LINE = re.compile(
    r"^HOME_QPOS = np\.array\(\[[^\n]*\]\, dtype=np\.float64\)$",
    re.MULTILINE,
)
_ARM_PROCESS = (
    Path(__file__).resolve().parents[1] / "robot" / "piper_pico" / "arm_process.py"
)


def _joint_message_ready(message) -> bool:
    try:
        stamp = float(getattr(message, "time_stamp", math.nan))
        rate = float(getattr(message, "Hz", 0.0))
    except (TypeError, ValueError):
        return False
    return math.isfinite(stamp) and stamp > 0.0 and math.isfinite(rate) and rate > 0.0


def _raw_joint_counts(message) -> list[int]:
    state = message.joint_state
    return [getattr(state, f"joint_{index}") for index in range(1, 7)]


def read_joint_radians(can_port: str, timeout: float, interface_factory) -> np.ndarray:
    """Listen for one complete joint feedback sample and return it in radians."""
    if timeout <= 0.0:
        raise ValueError("timeout must be positive")
    arm = interface_factory(str(can_port))
    arm.ConnectPort(can_init=False, piper_init=False, start_thread=True)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            message = arm.GetArmJointMsgs()
            if _joint_message_ready(message):
                # The six joints arrive in three CAN frames. One extra sample
                # lets the last frame refresh the cached message.
                time.sleep(0.05)
                message = arm.GetArmJointMsgs()
                if _joint_message_ready(message):
                    return joints_sdk_to_rad(_raw_joint_counts(message))
            time.sleep(0.02)
        raise TimeoutError(
            f"no joint feedback on {can_port} within {timeout:g}s; "
            "the arm must already be publishing, and this script will not enable it"
        )
    finally:
        arm.DisconnectPort()


def format_home_qpos(radians: np.ndarray) -> str:
    values = np.asarray(radians, dtype=np.float64).reshape(6)
    if not np.isfinite(values).all():
        raise ValueError("joint sample contains a non-finite value")
    rendered = ", ".join(f"{value:.8f}" for value in values)
    return f"HOME_QPOS = np.array([{rendered}], dtype=np.float64)"


def write_home_qpos(path: Path, radians: np.ndarray) -> None:
    source = path.read_text(encoding="utf-8")
    replacement = format_home_qpos(radians)
    updated, count = _HOME_LINE.subn(replacement, source, count=1)
    if count != 1:
        raise RuntimeError(f"expected exactly one HOME_QPOS assignment in {path}")
    path.write_text(updated, encoding="utf-8")


def _print_sample(radians: np.ndarray) -> None:
    values = np.asarray(radians, dtype=np.float64).reshape(6)
    degrees = np.degrees(values)
    print("HOME_QPOS 单位是弧度。固件原始反馈是 0.001 度，采集代码用 JOINT_FACTOR 换成弧度。")
    print("radians: " + " ".join(f"{value:.8f}" for value in values))
    print("degrees: " + " ".join(f"{value:.4f}" for value in degrees))
    print(format_home_qpos(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0", help="left arm SocketCAN interface")
    parser.add_argument("--timeout", type=float, default=3.0, help="seconds to wait for feedback")
    parser.add_argument(
        "--write-home",
        action="store_true",
        help="replace HOME_QPOS in arm_process.py with this radian sample",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from piper_sdk import C_PiperInterface_V2

    radians = read_joint_radians(args.can, args.timeout, C_PiperInterface_V2)
    _print_sample(radians)
    if args.write_home:
        write_home_qpos(_ARM_PROCESS, radians)
        print(f"wrote {_ARM_PROCESS}")


if __name__ == "__main__":
    main()
