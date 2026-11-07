"""Load a converted Piper Zarr episode into Rerun.

Reads ``piper_base_v3`` output from ``piper_pico_to_zarr``, not the raw capture.
Poses stay in each arm's Piper base frame; this viewer does not apply
extrinsics. Camera JPEGs are the frames the converter already selected.

The Rerun layout follows SuperInference's dataset viewer: one 3D world with
trajectories and per-step camera / scalar tracks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr
import simplejpeg
import zarr

SIDES = ("left", "right")
CAMERAS = ("front_1", "left_wrist", "right_wrist")
REQUIRED_NUMERIC = tuple(
    f"{side}.{field}"
    for side in SIDES
    for field in ("obs_ee_pose", "cmd_ee_pose", "obs_gripper", "cmd_gripper")
)
ARM_COLORS = {
    "left": {
        "obs": [90, 170, 255],
        "cmd": [40, 90, 220],
    },
    "right": {
        "obs": [255, 180, 110],
        "cmd": [210, 90, 40],
    },
}


def find_episodes(path: Path) -> list[Path]:
    path = Path(path).resolve()
    if (path / "zarr.json").is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    episodes = sorted(
        child
        for child in path.iterdir()
        if child.is_dir() and (child / "zarr.json").is_file()
    )
    if not episodes:
        raise FileNotFoundError(f"No episode_*.zarr stores under {path}")
    return episodes


def total_frames(store) -> int:
    attrs = dict(store.attrs)
    if "total_frames" in attrs:
        count = int(attrs["total_frames"])
        if count < 1:
            raise ValueError("total_frames must be a positive integer")
        return count
    return int(store["left.obs_ee_pose"].shape[0])


def encoded_image(store, index: int):
    """Read one VariableLengthBytes JPEG without collapsing to a 0-D object."""
    if isinstance(store, (bytes, bytearray, memoryview)):
        raise TypeError("Expected an image array, not a single buffer")
    try:
        # arr[i] on zarr v3 VariableLengthBytes is a 0-D object, not the JPEG.
        return store[index : index + 1][0]
    except Exception as exc:
        raise TypeError(f"Cannot read encoded image from {type(store)}") from exc


def decode_jpeg(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.ndim >= 2:
            return np.asarray(value)
        value = (
            value.item()
            if value.dtype == object or value.shape == ()
            else bytes(value)
        )
    image = simplejpeg.decode_jpeg(bytes(value), colorspace="RGB")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 JPEG, got {getattr(image, 'shape', None)}")
    return image


def wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected wxyz quaternions, got {quat.shape}")
    return quat[..., [1, 2, 3, 0]]


def gripper_width_to_color(
    width: float, min_width: float = 0.0, max_width: float = 1.0
) -> list[int]:
    span = max(max_width - min_width, 1e-9)
    normalized = float(np.clip((width - min_width) / span, 0.0, 1.0))
    if normalized < 0.5:
        return [255, int(255 * normalized * 2), 0]
    return [int(255 * (1 - (normalized - 0.5) * 2)), 255, 0]


def _history_colors(base: list[int], length: int) -> list[list[int]]:
    if length <= 0:
        return []
    scales = np.linspace(0.35, 1.0, num=length)
    return [
        [min(255, int(component * scale)) for component in base] for scale in scales
    ]


def setup_world(world_path: str = "world", grid_size: float = 0.8) -> None:
    rr.log(world_path, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(f"{world_path}/origin", rr.Transform3D(), static=True)
    rr.log(
        f"{world_path}/origin/axes",
        rr.Arrows3D(
            vectors=[[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
            origins=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["X", "Y", "Z"],
        ),
        static=True,
    )
    strips = []
    for axis in np.arange(-grid_size, grid_size + 1e-9, 0.1):
        strips.append([[-grid_size, axis, 0.0], [grid_size, axis, 0.0]])
        strips.append([[axis, -grid_size, 0.0], [axis, grid_size, 0.0]])
    rr.log(
        f"{world_path}/grid",
        rr.LineStrips3D(strips=strips, colors=[[80, 80, 80]], radii=[0.001]),
        static=True,
    )


def send_blueprint() -> None:
    import rerun.blueprint as rrb

    camera_views = [
        rrb.Spatial2DView(origin=f"world/cameras/{name}", name=name)
        for name in CAMERAS
    ]
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(*camera_views),
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="world", name="bases"),
                    rrb.TimeSeriesView(origin="series", name="signals"),
                ),
            )
        )
    )


def log_static_trajectory(
    path: str, poses: np.ndarray, color: list[int], *, marker_stride: int
) -> None:
    positions = np.asarray(poses[:, :3], dtype=np.float64)
    quats = wxyz_to_xyzw(poses[:, 3:7])
    if len(positions) == 0:
        return
    colors = _history_colors(color, len(positions))
    if len(positions) > 1:
        rr.log(
            f"{path}/trajectory",
            rr.LineStrips3D(strips=[positions.tolist()], colors=colors, radii=[0.003]),
            static=True,
        )
    rr.log(
        f"{path}/points",
        rr.Points3D(positions=positions.tolist(), colors=colors, radii=[0.004]),
        static=True,
    )
    step = max(1, int(marker_stride))
    for index in range(0, len(positions), step):
        marker = f"{path}/rotation_{index}"
        rr.log(
            marker,
            rr.Transform3D(
                translation=positions[index].tolist(),
                quaternion=quats[index].tolist(),
            ),
            static=True,
        )
        rr.log(
            f"{marker}/axes",
            rr.Arrows3D(
                vectors=[[0.03, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.03]],
                origins=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                colors=[[220, 80, 80], [80, 220, 80], [80, 80, 220]],
                radii=[0.001, 0.001, 0.001],
            ),
            static=True,
        )


def log_current_pose(path: str, pose: np.ndarray, gripper: float, color: list[int]) -> None:
    pose = np.asarray(pose, dtype=np.float64)
    rr.log(
        f"{path}/current",
        rr.Transform3D(
            translation=pose[:3].tolist(),
            quaternion=wxyz_to_xyzw(pose[3:7]).tolist(),
        ),
    )
    rr.log(
        f"{path}/current/point",
        rr.Points3D(positions=[pose[:3].tolist()], colors=[color], radii=[0.008]),
    )
    rr.log(
        f"{path}/current/info",
        rr.TextLog(
            f"pos={np.round(pose[:3], 4).tolist()} grip={float(gripper):.4f}"
        ),
    )


def log_pose_scalars(path: str, pose: np.ndarray, gripper: float) -> None:
    pose = np.asarray(pose, dtype=np.float64)
    rr.log(f"{path}/pos_x", rr.Scalar(float(pose[0])))
    rr.log(f"{path}/pos_y", rr.Scalar(float(pose[1])))
    rr.log(f"{path}/pos_z", rr.Scalar(float(pose[2])))
    rr.log(f"{path}/gripper", rr.Scalar(float(gripper)))


def load_episode(path: Path) -> dict:
    store = zarr.open_group(str(path), mode="r")
    count = total_frames(store)
    missing = [key for key in (*REQUIRED_NUMERIC, "timestamp") if key not in store]
    if missing:
        raise KeyError(f"{path}: missing {missing}")
    payload = {
        "path": Path(path),
        "attrs": dict(store.attrs),
        "count": count,
        "timestamp": np.asarray(store["timestamp"][:count], dtype=np.float64),
    }
    for side in SIDES:
        for field in ("obs_ee_pose", "cmd_ee_pose"):
            payload[f"{side}.{field}"] = np.asarray(store[f"{side}.{field}"][:count])
        for field in ("obs_gripper", "cmd_gripper"):
            payload[f"{side}.{field}"] = np.asarray(
                store[f"{side}.{field}"][:count]
            ).reshape(count, -1)[:, 0]
    for camera in CAMERAS:
        key = f"images.{camera}"
        payload[key] = store[key] if key in store else None
        age_key = f"source.{camera}.age"
        if age_key in store:
            payload[age_key] = np.asarray(store[age_key][:count], dtype=np.float64)
    return payload


def log_episode(
    episode: dict,
    *,
    stride: int = 1,
    marker_stride: int = 15,
    send_layout: bool = True,
) -> int:
    if stride < 1 or marker_stride < 1:
        raise ValueError("stride and marker_stride must be positive")
    timestamps = episode["timestamp"]
    count = episode["count"]
    setup_world()
    if send_layout:
        try:
            send_blueprint()
        except Exception:
            pass
    attrs = episode["attrs"]
    rr.log(
        "world/info",
        rr.TextLog(
            json.dumps(
                {
                    "episode": str(episode["path"]),
                    "schema": attrs.get("schema"),
                    "embodiment": attrs.get("embodiment"),
                    "calibration_status": attrs.get("calibration_status"),
                    "pose_coordinates": attrs.get("pose_coordinates"),
                    "rows": count,
                    "note": "left/right poses are in independent Piper bases; extrinsics are not applied",
                },
                ensure_ascii=False,
            )
        ),
        static=True,
    )
    for side in SIDES:
        colors = ARM_COLORS[side]
        log_static_trajectory(
            f"world/{side}/obs",
            episode[f"{side}.obs_ee_pose"],
            colors["obs"],
            marker_stride=marker_stride,
        )
        log_static_trajectory(
            f"world/{side}/cmd",
            episode[f"{side}.cmd_ee_pose"],
            colors["cmd"],
            marker_stride=marker_stride,
        )

    logged = 0
    for index in range(0, count, stride):
        rr.set_time_sequence("step", index)
        rr.set_time_seconds("timestamp", float(timestamps[index]))
        for camera in CAMERAS:
            images = episode.get(f"images.{camera}")
            if images is None:
                continue
            rr.log(
                f"world/cameras/{camera}",
                rr.Image(decode_jpeg(encoded_image(images, index))),
            )
            ages = episode.get(f"source.{camera}.age")
            if ages is not None:
                rr.log(f"series/cameras/{camera}/age_s", rr.Scalar(float(ages[index])))
        for side in SIDES:
            colors = ARM_COLORS[side]
            obs_pose = episode[f"{side}.obs_ee_pose"][index]
            cmd_pose = episode[f"{side}.cmd_ee_pose"][index]
            obs_grip = float(episode[f"{side}.obs_gripper"][index])
            cmd_grip = float(episode[f"{side}.cmd_gripper"][index])
            log_current_pose(f"world/{side}/obs", obs_pose, obs_grip, colors["obs"])
            log_current_pose(f"world/{side}/cmd", cmd_pose, cmd_grip, colors["cmd"])
            log_pose_scalars(f"series/{side}/obs", obs_pose, obs_grip)
            log_pose_scalars(f"series/{side}/cmd", cmd_pose, cmd_grip)
            rr.log(
                f"world/{side}/obs/current/gripper_color",
                rr.Points3D(
                    positions=[obs_pose[:3].tolist()],
                    colors=[gripper_width_to_color(obs_grip)],
                    radii=[0.006],
                ),
            )
        logged += 1
    return logged


def init_rerun(
    app_name: str,
    *,
    spawn: bool,
    save_path: Path | None,
) -> None:
    rr.init(app_name)
    if save_path is not None:
        rr.save(str(save_path))
        return
    if spawn:
        rr.spawn()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Converted episode.zarr or a directory of converted episodes",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index in the batch")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--marker-stride",
        type=int,
        default=15,
        help="Log one static pose axis every N converted rows",
    )
    parser.add_argument("--save", type=Path, help="Write a .rrd instead of spawning the viewer")
    parser.add_argument("--no-spawn", action="store_true")
    args = parser.parse_args(argv)
    episodes = find_episodes(args.input)
    if args.episode < 0 or args.episode >= len(episodes):
        parser.exit(1, f"--episode {args.episode} is outside 0..{len(episodes) - 1}\n")
    path = episodes[args.episode]
    payload = load_episode(path)
    init_rerun(
        f"{path.parent.name}/{path.name}",
        spawn=not args.no_spawn and args.save is None,
        save_path=args.save,
    )
    logged = log_episode(
        payload, stride=args.stride, marker_stride=args.marker_stride
    )
    print(
        f"{path}: {payload['count']} rows, logged {logged} steps, "
        f"calibration={payload['attrs'].get('calibration_status')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
