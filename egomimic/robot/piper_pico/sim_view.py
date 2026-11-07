"""3D view of the two simulated arms.

Each end-effector is the cartesian target sitting in that arm's base frame.
A red cross is a step larger than 0.1 m, which is the command the real arm
would have received.
"""

from __future__ import annotations

from collections import deque
import time

import numpy as np
from scipy.spatial.transform import Rotation

LEFT_BASE = np.array([0.0, 0.28, 0.0], dtype=np.float64)
RIGHT_BASE = np.array([0.0, -0.28, 0.0], dtype=np.float64)
JUMP_M = 0.1
_COLORS = {"left": "#1f77b4", "right": "#ff7f0e"}


def _world(base: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return base + np.asarray(pose, dtype=np.float64).reshape(6)[:3]


class SimBimanualView:
    def __init__(self, refresh_hz: float = 30.0):
        refresh_hz = float(refresh_hz)
        if not np.isfinite(refresh_hz) or refresh_hz <= 0.0:
            raise ValueError("refresh_hz must be positive and finite")
        self._draw_interval = 1.0 / refresh_hz
        self._next_draw_at = float("-inf")
        self.jump_count = 0
        self._prev = {"left": None, "right": None}
        self._jump_points: list[np.ndarray] = []
        self._trails = {name: deque(maxlen=180) for name in _COLORS}
        self._ok = False
        self._fig = None
        self._sticks = {}
        self._tools = {}
        self._points = {}
        self._trails_art = {}
        self._jumps = None
        self._hud = None
        try:
            import matplotlib.pyplot as plt
            from matplotlib import font_manager

            self._plt = plt
            for family in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
                if any(font.name.startswith(family) for font in font_manager.fontManager.ttflist):
                    plt.rcParams["font.family"] = family
                    break
            plt.ion()
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            xx, yy = np.meshgrid(np.linspace(-0.2, 0.8, 2), np.linspace(-0.8, 0.8, 2))
            ax.plot_surface(xx, yy, np.zeros_like(xx), color="0.75", alpha=0.25, linewidth=0)
            for name, base in (("left", LEFT_BASE), ("right", RIGHT_BASE)):
                color = _COLORS[name]
                self._sticks[name] = ax.plot([], [], [], color=color, lw=2.5)[0]
                self._tools[name] = ax.plot([], [], [], color=color, lw=1.5)[0]
                self._points[name] = ax.plot([], [], [], "o", color=color, ms=7)[0]
                self._trails_art[name] = ax.plot([], [], [], color=color, lw=1.0, alpha=0.45)[0]
                ax.scatter([base[0]], [base[1]], [base[2]], color=color, s=20)
            self._jumps = ax.plot([], [], [], "x", color="#d62728", ms=9, mew=2)[0]
            self._hud = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
            ax.set_xlim(-0.4, 1.0)
            ax.set_ylim(-0.9, 0.9)
            ax.set_zlim(-0.6, 0.8)
            ax.set_xlabel("x forward")
            ax.set_ylabel("y left")
            ax.set_zlabel("z up")
            ax.set_title("仿真双臂    红叉 = 末端指令单帧超过 0.1 m    终端按 A 退出")
            fig.tight_layout()
            fig.canvas.draw_idle()
            self._fig = fig
            self._ok = True
        except Exception as exc:
            print(f"[sim] 3D view unavailable ({exc}). Jump lines still print in the terminal.")

    def update(self, env) -> bool:
        poses = {
            "left": np.asarray(env.arms[0].get_state()["ee_pose"], dtype=np.float64),
            "right": np.asarray(env.arms[1].get_state()["ee_pose"], dtype=np.float64),
        }
        steps = {}
        for name, base in (("left", LEFT_BASE), ("right", RIGHT_BASE)):
            world = _world(base, poses[name])
            previous = self._prev[name]
            step = 0.0 if previous is None else float(np.linalg.norm(world - previous))
            steps[name] = step
            if previous is not None and step > JUMP_M:
                self.jump_count += 1
                self._jump_points.append(world.copy())
            self._prev[name] = world
            self._trails[name].append(world)
        if self._ok:
            # Sampling and jump detection above run at every control tick.
            # Only matplotlib work is limited, so a jump between redraws is
            # still retained in the trail and in the cumulative red markers.
            if not self._plt.fignum_exists(self._fig.number):
                return False
            now = time.monotonic()
            if now + 1e-9 < self._next_draw_at:
                return True
            self._next_draw_at = now + self._draw_interval
            for name, base in (("left", LEFT_BASE), ("right", RIGHT_BASE)):
                self._draw_arm(name, base, self._prev[name], poses[name])
            self._draw_jumps_and_hud(steps)
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
            return bool(self._plt.fignum_exists(self._fig.number))
        return True

    def _draw_arm(self, name: str, base: np.ndarray, world: np.ndarray, pose: np.ndarray) -> None:
        self._sticks[name].set_data_3d([base[0], world[0]], [base[1], world[1]], [base[2], world[2]])
        tool = world + Rotation.from_euler("xyz", pose[3:]).apply([0.0, 0.0, 0.08])
        self._tools[name].set_data_3d([world[0], tool[0]], [world[1], tool[1]], [world[2], tool[2]])
        self._points[name].set_data_3d([world[0]], [world[1]], [world[2]])
        trail = np.stack(self._trails[name], axis=0)
        self._trails_art[name].set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])

    def _draw_jumps_and_hud(self, steps: dict) -> None:
        if self._jump_points:
            pts = np.stack(self._jump_points, axis=0)
            self._jumps.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
        self._hud.set_text(
            f"left {steps['left'] * 100:.1f} cm    right {steps['right'] * 100:.1f} cm    "
            f"jumps {self.jump_count}"
        )

    def close(self) -> None:
        if self._fig is not None:
            self._plt.close(self._fig)
