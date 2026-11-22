"""Project converted PICO ego world poses onto the saved left-camera JPEGs.

This uses the converted dataset, never refits calibration or runs MediaPipe.
The independent head-to-camera chain check detects conversion mistakes; its
agreement is not an independent measurement of temporal or calibration error.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import simplejpeg
import zarr
from scipy.spatial.transform import Rotation

from egomimic.robot.piper_pico.image_geometry import validate_image_geometry

SIDES = ("left", "right")
CHAINS = (
    (0, 1),
    (1, 2, 3, 4, 5),
    (1, 6, 7, 8, 9, 10),
    (1, 11, 12, 13, 14, 15),
    (1, 16, 17, 18, 19, 20),
    (1, 21, 22, 23, 24, 25),
)
HAND_COLORS = ((80, 240, 80), (0, 170, 255))  # OpenCV BGR.
AXIS_COLORS = ((0, 0, 255), (0, 255, 0), (255, 0, 0))


def pose_matrices(pose: np.ndarray) -> np.ndarray:
    """Convert stored xyz+wxyz poses to parent-from-child SE(3) matrices."""
    pose = np.asarray(pose, dtype=float)
    if pose.ndim != 2 or pose.shape[1] != 7 or not np.isfinite(pose).all():
        raise ValueError("Expected finite (T,7) xyz+wxyz poses")
    if np.any(np.linalg.norm(pose[:, 3:], axis=1) < 1e-10):
        raise ValueError("Zero pose quaternion")
    result = np.broadcast_to(np.eye(4), (len(pose), 4, 4)).copy()
    result[:, :3, :3] = Rotation.from_quat(pose[:, [4, 5, 6, 3]]).as_matrix()
    result[:, :3, 3] = pose[:, :3]
    return result


def project_world_points(
    points: np.ndarray, camera_from_world: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project (T,K,3) world points; invalid/behind-camera UVs are NaN."""
    points = np.asarray(points, dtype=float)
    transform = np.asarray(camera_from_world, dtype=float)
    k = np.asarray(intrinsics, dtype=float)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("Expected world points with shape (T,K,3)")
    if transform.shape != (len(points), 4, 4):
        raise ValueError("Expected one camera_from_world matrix per frame")
    if k.shape == (3, 4):
        if not np.allclose(k[:, 3], 0):
            raise ValueError("Pinhole intrinsics must have a zero fourth column")
        k = k[:, :3]
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError("Expected finite (3,3) or (3,4) pinhole intrinsics")
    camera = np.einsum("tij,tkj->tki", transform[:, :3, :3], points)
    camera += transform[:, None, :3, 3]
    homogeneous = camera @ k.T
    good = np.isfinite(homogeneous).all(-1) & (camera[..., 2] > 0.01)
    uv = np.full(points.shape[:-1] + (2,), np.nan)
    np.divide(
        homogeneous[..., :2],
        homogeneous[..., 2:],
        out=uv,
        where=good[..., None],
    )
    return uv, camera[..., 2]


def episode_paths(source: Path) -> list[Path]:
    """Read only episode groups carrying this converter's provenance."""
    candidates = (
        [source] if (source / "zarr.json").is_file() else sorted(source.iterdir())
    )
    result = []
    for candidate in candidates:
        metadata = candidate / "zarr.json"
        if not candidate.is_dir() or not metadata.is_file():
            continue
        attrs = json.loads(metadata.read_text()).get("attributes", {})
        if "pico_ego" in attrs:
            result.append(candidate)
    if not result:
        raise ValueError(f"No converted PICO ego episodes found in {source}")
    return result


def _line(image, first, second, color, thickness=2):
    if not (np.isfinite(first).all() and np.isfinite(second).all()):
        return
    # Limit values before converting to OpenCV int32 and clip to the viewport.
    first = tuple(np.rint(np.clip(first, -1e6, 1e6)).astype(int))
    second = tuple(np.rint(np.clip(second, -1e6, 1e6)).astype(int))
    ok, start, end = cv2.clipLine((0, 0, image.shape[1], image.shape[0]), first, second)
    if ok:
        cv2.line(image, start, end, color, thickness, cv2.LINE_AA)


def _text(image, text, line, color=(255, 255, 255)):
    position = (8, 21 + line * 20)
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
    )


def _stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _read(group, name, n):
    if name not in group or group[name].shape[0] < n:
        raise ValueError(f"Missing or short required array: {name}")
    return np.asarray(group[name][:n])


def _rigid_transform(value, name):
    transform = np.asarray(value, dtype=float)
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(
            transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6, rtol=0
        )
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"{name} must be a finite proper rigid transform")
    return transform


def external_uvc_geometry(attrs, image_size):
    """Verify processed pixels and recover T_C_H independently from the fit.

    The frozen fit is the source of the extrinsic reference, not the converter's
    derived camera_from_head field. Rectification has R=I, so the optical frame
    stays unchanged; only the pixel camera matrix changes.
    """
    pico = attrs["pico_ego"]
    fit = pico.get("fit", {}).get("data", {})
    if fit.get("schema") != "seeded_external_uvc_spacetime_v1":
        raise ValueError("External UVC data require the frozen external spacetime fit")
    parameters = fit.get("parameters", {})
    rotation = np.asarray(parameters.get("R_CH"), dtype=float)
    translation = np.asarray(parameters.get("t_CH_m"), dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Frozen fit requires R_CH (3,3) and t_CH_m (3,)")
    camera_from_head = np.eye(4)
    camera_from_head[:3, :3] = rotation
    camera_from_head[:3, 3] = translation
    _rigid_transform(camera_from_head, "Frozen fit T_C_H")
    saved = _rigid_transform(pico.get("camera_from_head"), "camera_from_head")
    if not np.allclose(saved, camera_from_head, atol=1e-8, rtol=0):
        raise ValueError("Saved camera_from_head differs from the frozen fit T_C_H")

    geometry = attrs.get("image_geometry", {}).get("front_1")
    validate_image_geometry(geometry)
    if (
        attrs.get("image_space") != "processed"
        or geometry["operation"] != "undistort_resize"
        or geometry["D_output"] != [0.0] * 5
    ):
        raise ValueError("External UVC JPEGs must be rectified processed images")
    if list(image_size) != geometry["output_size"]:
        raise ValueError("Saved JPEG size differs from image geometry output_size")
    k = np.asarray(attrs["intrinsics"]["front_1"], dtype=float)
    if k.shape == (3, 4) and np.all(k[:, 3] == 0):
        k = k[:, :3]
    if k.shape != (3, 3) or not np.allclose(k, geometry["K_output"], atol=1e-8, rtol=0):
        raise ValueError("Saved front_1 intrinsics differ from processed K_output")
    intrinsics = fit.get("intrinsics", {})
    if intrinsics.get("mirror") is not False:
        raise ValueError("External fit must explicitly declare mirror=false")
    if intrinsics.get("image_size") != geometry["source_size"]:
        raise ValueError("Frozen fit image_size differs from raw geometry size")
    for raw_key, geometry_key in (("raw_K", "K_raw"), ("raw_D", "D_raw")):
        raw = np.asarray(intrinsics.get(raw_key), dtype=float)
        expected = np.asarray(geometry[geometry_key], dtype=float)
        if raw.shape != expected.shape or not np.allclose(
            raw, expected, atol=1e-8, rtol=0
        ):
            raise ValueError(f"Frozen fit {raw_key} differs from image geometry")
    root_distortion = attrs.get("distortion_coefficients", {}).get("front_1")
    if root_distortion is not None and root_distortion != [0.0] * 5:
        raise ValueError("Processed front_1 must declare zero output distortion")
    return camera_from_head, geometry


def render_episode(episode: Path, output_root: Path, axis_length: float = 0.06) -> Path:
    """Write one preview and geometric verification report without changing input."""
    if not np.isfinite(axis_length) or axis_length <= 0:
        raise ValueError("axis_length must be finite and positive")
    episode = episode.resolve()
    output_root = output_root.resolve()
    if output_root == episode or episode in output_root.parents:
        raise ValueError("Visualization output must be outside the input episode")
    destination = output_root / episode.name
    if destination.exists():
        raise FileExistsError(destination)
    group = zarr.open_group(str(episode), mode="r")
    attrs = dict(group.attrs)
    n = int(attrs["total_frames"])
    if n < 1:
        raise ValueError("Episode is empty")
    k = np.asarray(attrs["intrinsics"]["front_1"], dtype=float)
    camera_pose = _read(group, "obs_head_pose", n)
    raw_head_pose = _read(group, "source.head_pose", n)
    world_from_camera = pose_matrices(camera_pose)
    camera_from_world = np.linalg.inv(world_from_camera)
    timestamps = _read(group, "timestamp_ns", n).astype(np.int64)
    grid = _read(group, "source.grid_timestamp_ns", n).astype(np.int64)
    rgb_timestamps = _read(group, "obs_rgb_timestamps_ns", n).astype(np.int64)
    if not np.array_equal(timestamps, grid):
        raise ValueError("timestamp_ns differs from the regular sampling grid")
    if n > 1 and (
        np.any(np.diff(timestamps) <= 0) or np.any(np.diff(rgb_timestamps) < 0)
    ):
        raise ValueError(
            "Grid timestamps must increase strictly; image timestamps must not decrease"
        )
    jpeg = _read(group, "images.front_1", n)
    first_rgb = simplejpeg.decode_jpeg(bytes(jpeg[0]), colorspace="RGB")
    height, width = first_rgb.shape[:2]
    if width % 2 or height % 2:
        raise ValueError("Preview H.264 video requires even image dimensions")
    source_format = attrs["pico_ego"].get("source_format", "legacy_pico")
    geometry = None
    if source_format == "uvc_joint_v1":
        camera_from_head, geometry = external_uvc_geometry(attrs, [width, height])
        reference_basis = "frozen external fit parameters R_CH and t_CH_m"
    else:
        camera_from_head = _rigid_transform(
            attrs["pico_ego"]["camera_from_head"], "pico_ego.camera_from_head"
        )
        reference_basis = "saved legacy camera_from_head"
    image_time_basis = (
        "frame receive UTC" if source_format == "uvc_joint_v1" else "video PTS"
    )
    reference_chain = camera_from_head @ np.linalg.inv(pose_matrices(raw_head_pose))
    matrix_error = float(np.max(np.abs(camera_from_world - reference_chain)))
    if source_format == "uvc_joint_v1" and matrix_error > 1e-5:
        raise ValueError(
            "Saved optical camera poses disagree with frozen-fit/head chain "
            f"(max matrix error {matrix_error:.3g})"
        )
    source_frames = _read(group, "source.video_frame_index", n)
    uv, depth, axes_uv, filled, valid, grip, age, reference_error = (
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
    )
    for side in SIDES:
        points = _read(group, f"{side}.obs_xr_keypoints", n).reshape(n, 26, 3)
        uv[side], depth[side] = project_world_points(points, camera_from_world, k)
        reference_uv, _ = project_world_points(points, reference_chain, k)
        reference_error[side] = np.linalg.norm(uv[side] - reference_uv, axis=-1)
        ee = pose_matrices(_read(group, f"{side}.obs_ee_pose", n))
        # Each column of R is the corresponding local EE axis in the world.
        endpoints = ee[:, None, :3, 3] + axis_length * ee[:, :3, :3].transpose(0, 2, 1)
        endpoints = np.concatenate([ee[:, None, :3, 3], endpoints], axis=1)
        axes_uv[side], _ = project_world_points(endpoints, camera_from_world, k)
        filled[side] = _read(group, f"{side}.hand_filled", n).astype(bool).reshape(n)
        valid[side] = _read(group, f"{side}.hand_valid", n).astype(bool).reshape(n)
        grip[side] = _read(group, f"{side}.obs_gripper", n).reshape(n)
        age[side] = _read(group, f"{side}.source_age_s", n).reshape(n)

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{episode.name}.viz-", dir=output_root))
    try:
        samples = set(np.linspace(0, n - 1, min(6, n), dtype=int).tolist())
        sample_images = []
        fps = float(attrs["fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Episode fps must be finite and positive")
        time_base = Fraction(1, 1_000_000)
        with av.open(str(temporary / "overlay.mp4"), "w") as container:
            stream = container.add_stream("libx264", rate=Fraction(str(fps)))
            stream.width, stream.height = width, height
            stream.pix_fmt = "yuv420p"
            stream.time_base = time_base
            stream.codec_context.time_base = time_base
            stream.codec_context.thread_count = 1
            stream.options = {"crf": "20", "preset": "veryfast"}
            for i in range(n):
                rgb = (
                    first_rgb
                    if i == 0
                    else simplejpeg.decode_jpeg(bytes(jpeg[i]), colorspace="RGB")
                )
                if rgb.shape != first_rgb.shape:
                    raise ValueError(f"Image dimensions change at frame {i}")
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for hand_index, side in enumerate(SIDES):
                    color = (
                        (255, 80, 255) if filled[side][i] else HAND_COLORS[hand_index]
                    )
                    points = uv[side][i]
                    for chain in CHAINS:
                        for a, b in zip(chain[:-1], chain[1:]):
                            _line(image, points[a], points[b], color)
                    for point in points:
                        if (
                            np.isfinite(point).all()
                            and 0 <= point[0] < width
                            and 0 <= point[1] < height
                        ):
                            cv2.circle(
                                image,
                                tuple(np.rint(point).astype(int)),
                                2,
                                color,
                                -1,
                                cv2.LINE_AA,
                            )
                    for axis, axis_color in enumerate(AXIS_COLORS):
                        _line(
                            image,
                            axes_uv[side][i, 0],
                            axes_uv[side][i, axis + 1],
                            axis_color,
                            3,
                        )
                    label = (
                        "HELD"
                        if filled[side][i]
                        else "valid"
                        if valid[side][i]
                        else "INVALID"
                    )
                    _text(
                        image,
                        f"{side}: {label}  grip={grip[side][i]:.3f}  age={1000 * age[side][i]:.1f}ms",
                        hand_index + 1,
                        color,
                    )
                elapsed = (int(timestamps[i]) - int(timestamps[0])) / 1e9
                _text(
                    image,
                    f"front_1  {i}/{n - 1}  source={source_frames[i]}  grid t={elapsed:.3f}s",
                    0,
                )
                image_delta_ms = (int(rgb_timestamps[i]) - int(timestamps[i])) / 1e6
                _text(
                    image,
                    f"grid -> nearest {image_time_basis}  dt={image_delta_ms:+.1f}ms | XYZ=RGB",
                    3,
                )
                frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                frame.pts = round((int(timestamps[i]) - int(timestamps[0])) / 1000)
                frame.time_base = time_base
                for packet in stream.encode(frame):
                    container.mux(packet)
                if i in samples:
                    sample_name = f"sample_{i:05d}.jpg"
                    if not cv2.imwrite(str(temporary / sample_name), image):
                        raise OSError(f"Could not write {sample_name}")
                    sample_images.append(image)
            for packet in stream.encode():
                container.mux(packet)
        # Contact sheet contains only converted images and their stored poses.
        if len(sample_images) % 2:
            sample_images.append(np.zeros_like(sample_images[-1]))
        rows = [
            np.concatenate(sample_images[i : i + 2], axis=1)
            for i in range(0, len(sample_images), 2)
        ]
        if not cv2.imwrite(
            str(temporary / "contact_sheet.jpg"), np.concatenate(rows, axis=0)
        ):
            raise OSError("Could not write contact_sheet.jpg")
        summary = {
            "input": str(episode),
            "total_frames": n,
            "image_size": [width, height],
            "duration_s": (int(timestamps[-1]) - int(timestamps[0])) / 1e9,
            "video_time_basis": f"regular sampling grid timestamps, first frame rebased to zero; source images chosen by nearest {image_time_basis} may repeat",
            "image_selection_time_basis": image_time_basis,
            "source_format": source_format,
            "camera_reference_basis": reference_basis,
            "camera_matrix_chain_max_abs_error": matrix_error,
            "image_geometry": geometry,
            "grid_to_image_abs_dt_ms": _stats(
                np.abs(rgb_timestamps - timestamps) / 1e6
            ),
            "repeated_source_images": int(
                np.count_nonzero(np.diff(rgb_timestamps) == 0)
            ),
            "intrinsics": k.tolist(),
            "interpretation": f"Projection uses saved grid-time world joints, saved grid-time world optical-camera pose and JPEGs chosen by nearest {image_time_basis}. Grid-to-image phase difference is reported separately. Chain agreement checks conversion consistency only; visible overlay is not independent ground-truth synchronization or calibration accuracy. No MediaPipe and no refitting.",
            "hands": {},
        }
        for side in SIDES:
            points = uv[side]
            in_image = (
                np.isfinite(points).all(-1)
                & (points[..., 0] >= 0)
                & (points[..., 0] < width)
                & (points[..., 1] >= 0)
                & (points[..., 1] < height)
            )
            summary["hands"][side] = {
                "filled_frames": int(filled[side].sum()),
                "observed_valid_frames": int(valid[side].sum()),
                "source_age_ms": _stats(age[side] * 1000),
                "positive_depth_fraction": float(np.mean(depth[side] > 0.01)),
                "projected_in_image_fraction": float(np.mean(in_image)),
                "chain_projection_difference_px": _stats(reference_error[side]),
            }
        np.savez_compressed(
            temporary / "projections.npz",
            timestamp_ns=timestamps,
            obs_rgb_timestamps_ns=rgb_timestamps,
            left_uv=uv["left"],
            right_uv=uv["right"],
            left_depth=depth["left"],
            right_depth=depth["right"],
        )
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Converted episode or root containing episodes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New preview root outside the source episodes",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.06,
        help="Virtual EE axis length in metres",
    )
    args = parser.parse_args()
    cv2.setNumThreads(1)
    for episode in episode_paths(args.input):
        print(
            f"wrote {render_episode(episode, args.output_dir, args.axis_length)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
