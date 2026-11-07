"""Piper bimanual + Pico 4 Ultra collection inside EgoVerse.

The control processes and camera-time alignment follow the UMI bimanual
collector. Arm commands are absolute cartesian targets for Piper MOVE P.
"""

__all__ = ["PiperPicoEnv"]


def __getattr__(name):
    # Offline policy replay must not import device/process implementations.
    if name == "PiperPicoEnv":
        from egomimic.robot.piper_pico.env import PiperPicoEnv

        return PiperPicoEnv
    raise AttributeError(name)
