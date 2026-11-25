"""Observation-only Piper policy adapter. No device or CAN imports.

Inputs are RGB uint8 images and measured SDK-end poses in each arm's base,
expressed as xyz metres + rotation vector radians + gripper width metres.
The calibration is for the current rig, not a global training constant.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from numbers import Real
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from egomimic.rldb.action_timing import ActionTiming
from egomimic.rldb.zarr.piper_dataset import (
    observation_to_send,
    validate_piper_metadata,
)
from egomimic.robot.piper_pico.image_geometry import (
    RgbImageProcessor,
    validate_image_geometry,
)
from egomimic.utils.pose_utils import _matrix_to_xyzypr, _xyzypr_to_matrix

DOMAIN = "piper_bimanual"
EMBODIMENT = 7
STATE = "observations.state.ee_pose"
ACTION = "actions_cartesian"
CAMERA_KEYS = {
    "front_1": "observations.images.front_img_1",
    "left_wrist": "observations.images.left_wrist_img",
    "right_wrist": "observations.images.right_wrist_img",
}
SOURCE_NAMES = (*CAMERA_KEYS, "left", "right")


def _finite(value, shape, label):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{label} must be finite with shape {shape}, got {a.shape}")
    return a.copy()


@dataclass(frozen=True)
class ObservationFrame:
    timestamp: float
    images: Mapping[str, np.ndarray]
    base_state: np.ndarray
    source_timestamps: Mapping[str, float]
    image_space: str = "raw"
    base_state_history: np.ndarray | None = None
    state_history_offsets_s: tuple[float, ...] = ()
    image_frame_ids: Mapping[str, int] = dataclass_field(default_factory=dict)
    image_receive_timestamps: Mapping[str, float] = dataclass_field(default_factory=dict)

    def validate(self):
        if self.image_space not in ("raw", "processed"):
            raise ValueError("image_space must be raw or processed")
        if not np.isfinite(self.timestamp):
            raise ValueError("observation timestamp must be finite")
        _finite(self.base_state, (14,), "base_state")
        from egomimic.robot.piper_pico.state_history import validate_history_offsets

        offsets = validate_history_offsets(self.state_history_offsets_s)
        if offsets:
            _finite(self.base_state_history, (len(offsets), 14), "base_state_history")
        elif self.base_state_history is not None:
            raise ValueError("base_state_history requires explicit history offsets")
        if set(self.images) != set(CAMERA_KEYS):
            raise ValueError("images must contain front_1, left_wrist, right_wrist")
        for name, image in self.images.items():
            image = np.asarray(image)
            if (
                image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[2] != 3
                or min(image.shape[:2]) < 1
            ):
                raise ValueError(f"{name} must be HWC RGB uint8")
        if set(self.source_timestamps) != set(SOURCE_NAMES):
            raise ValueError(
                "source_timestamps must identify all three cameras and both arms"
            )
        if not np.isfinite(list(self.source_timestamps.values())).all():
            raise ValueError("source timestamps must be finite")
        for label, values in (
            ("image_frame_ids", self.image_frame_ids),
            ("image_receive_timestamps", self.image_receive_timestamps),
        ):
            if not isinstance(values, Mapping):
                raise ValueError(f"{label} must be a mapping")
            if values and set(values) != set(CAMERA_KEYS):
                raise ValueError(f"{label} must identify all three cameras")
        for value in self.image_frame_ids.values():
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not 0 <= value <= np.iinfo(np.int64).max
            ):
                raise ValueError("image_frame_ids must be nonnegative int64 IDs")
        for value in self.image_receive_timestamps.values():
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError("image_receive_timestamps must be finite numbers")

    def check_freshness(self, now, max_age_s, future_tolerance_s=0.02):
        self.validate()
        if (
            not np.isfinite([now, max_age_s, future_tolerance_s]).all()
            or max_age_s <= 0
            or future_tolerance_s < 0
        ):
            raise ValueError("invalid freshness settings")
        for name, stamp in {
            "observation": self.timestamp,
            **self.source_timestamps,
        }.items():
            age = now - stamp
            if age < -future_tolerance_s or age > max_age_s:
                raise ValueError(
                    f"{name} timestamp is stale or in the future: age={age:.6f}s"
                )

    @classmethod
    def from_aligned_obs(cls, obs, *, camera_names):
        """Adapt PiperPicoEnv.get_obs, the same call the collector records.

        Source timestamps are the selected frames' stamps and each arm's newest
        sample, so a stalled camera or arm fails the freshness check.
        """
        if len(camera_names) != 3 or set(camera_names) != set(CAMERA_KEYS):
            raise ValueError(
                "camera_names must be a permutation of the three training views"
            )
        source_timestamps = {
            name: float(np.asarray(obs[f"camera{i}_timestamp"])[-1])
            for i, name in enumerate(camera_names)
        }
        for i, side in enumerate(("left", "right")):
            source_timestamps[side] = float(obs[f"robot{i}_source_timestamp"])
        state = []
        for i in range(2):
            state.extend(np.asarray(obs[f"robot{i}_eef_pos"])[-1].reshape(3))
            state.extend(np.asarray(obs[f"robot{i}_eef_rot_axis_angle"])[-1].reshape(3))
            state.append(float(np.asarray(obs[f"robot{i}_gripper_width"])[-1].item()))
        frame = cls(
            timestamp=float(np.asarray(obs["timestamp"])[-1]),
            images={
                name: np.asarray(obs[f"camera{i}_rgb"])[-1].copy()
                for i, name in enumerate(camera_names)
            },
            base_state=np.asarray(state),
            source_timestamps=source_timestamps,
            image_space="raw",
            base_state_history=(
                np.array(obs["state_history_base"], dtype=np.float64, copy=True)
                if "state_history_base" in obs
                else None
            ),
            state_history_offsets_s=tuple(obs.get("state_history_offsets_s", ())),
            image_frame_ids={
                name: np.asarray(obs[f"camera{i}_frame_id"])[-1].item()
                for i, name in enumerate(camera_names)
                if f"camera{i}_frame_id" in obs
            },
            image_receive_timestamps={
                name: np.asarray(obs[f"camera{i}_receive_timestamp"])[-1].item()
                for i, name in enumerate(camera_names)
                if f"camera{i}_receive_timestamp" in obs
            },
        )
        frame.validate()
        return frame


CAPTURE_FIELDS = ("type", "fps", "width", "height", "latency", "eye", "fourcc")


def _canonical_controls(spec):
    controls = spec.get("controls", {})
    if not isinstance(controls, Mapping):
        raise ValueError("camera controls must be a mapping")
    result = {}
    for key, value in controls.items():
        key = "focus_automatic_continuous" if key == "focus_auto" else key
        if key in result and result[key] != value:
            raise ValueError("camera controls contain conflicting autofocus aliases")
        result[key] = value
    return result


def _stable_identity(spec):
    # A USB socket/device node may move while the physical camera stays the same.
    identity = spec.get("expected_identity", {})
    if not isinstance(identity, Mapping):
        raise ValueError("expected_identity must be a mapping")
    return {
        key: value
        for key, value in identity.items()
        if key not in ("ID_PATH", "ID_PATH_TAG")
    }


def _same_numeric_geometry(left, right):
    try:
        a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        return (
            a.shape == b.shape
            and np.isfinite(a).all()
            and np.isfinite(b).all()
            and np.allclose(a, b, atol=1e-8, rtol=0)
        )
    except (TypeError, ValueError):
        return False


def _known_lens_controls(calibration):
    controls = calibration.get("camera", {}).get("controls", {})
    values = controls.get("values", controls)
    return _canonical_controls(
        {
            "controls": {
                key: value
                for key, value in values.items()
                if "focus" in key or "zoom" in key
            }
        }
    )


def _match_raw_intrinsics(name, recorded, live):
    if not isinstance(live, Mapping):
        raise ValueError(f"{name}: live camera lacks the recorded raw calibration")
    for field in ("K", "D", "image_size"):
        if not _same_numeric_geometry(recorded.get(field), live.get(field)):
            raise ValueError(f"{name}: raw calibration {field} differs from recording")
    for field in ("camera_model", "preprocessing"):
        if recorded.get(field) != live.get(field):
            raise ValueError(f"{name}: raw calibration {field} differs from recording")
    # File paths, JSON formatting and report edits may change provenance hashes
    # without changing measured geometry; compare the actual numeric contract.
    lens, live_lens = _known_lens_controls(recorded), _known_lens_controls(live)
    if any(live_lens.get(key) != value for key, value in lens.items()):
        raise ValueError(f"{name}: raw calibration focus/zoom differs from recording")
    recorded_identity = _stable_identity(
        {
            "expected_identity": {
                key: value
                for key, value in recorded.get("camera", {}).items()
                if key.startswith("ID_")
            }
        }
    )
    live_identity = _stable_identity(
        {
            "expected_identity": {
                key: value
                for key, value in live.get("camera", {}).items()
                if key.startswith("ID_")
            }
        }
    )
    if any(live_identity.get(key) != value for key, value in recorded_identity.items()):
        raise ValueError(
            f"{name}: raw calibration camera identity differs from recording"
        )


def check_capture_config(metadata, camera_specs):
    """Live cameras must match the recording's sources, including latency.

    The observation time is compared with latency-corrected frame stamps, so a
    different latency moves every selected frame relative to training.
    """
    recorded = metadata.get("source_capture_config", {}).get("cameras")
    if not isinstance(recorded, list):
        raise ValueError("Dataset metadata lacks source_capture_config.cameras")
    if len({spec.get("name") for spec in recorded}) != len(recorded):
        raise ValueError("recorded camera names must be unique")
    if len({spec.get("name") for spec in camera_specs}) != len(camera_specs):
        raise ValueError("live camera names must be unique")
    recorded = {spec.get("name"): spec for spec in recorded}
    live = {spec.get("name"): spec for spec in camera_specs}
    if set(live) != set(CAMERA_KEYS) or not set(CAMERA_KEYS) <= set(recorded):
        raise ValueError(
            f"Live cameras {sorted(live)} and recorded {sorted(recorded)} must both provide {sorted(CAMERA_KEYS)}"
        )
    for name in CAMERA_KEYS:
        for field in CAPTURE_FIELDS:
            if recorded[name].get(field) != live[name].get(field):
                raise ValueError(
                    f"{name}.{field}: live {live[name].get(field)!r} != recorded {recorded[name].get(field)!r}"
                )
        if _canonical_controls(recorded[name]) != _canonical_controls(live[name]):
            raise ValueError(
                f"{name}.controls: live camera controls differ from recording"
            )
        identity, live_identity = (
            _stable_identity(recorded[name]),
            _stable_identity(live[name]),
        )
        if any(live_identity.get(key) != value for key, value in identity.items()):
            raise ValueError(
                f"{name}.expected_identity: live camera identity differs from recording"
            )
        raw = recorded[name].get("calibration")
        if raw is not None:
            _match_raw_intrinsics(name, raw, live[name].get("calibration"))
        contract = (metadata.get("image_processing") or {}).get(name)
        if contract is not None:
            validate_image_geometry(contract)
            if contract["source_size"] != [live[name]["width"], live[name]["height"]]:
                raise ValueError(
                    f"{name}: live native size differs from image processing contract"
                )
            live_raw = live[name].get("calibration")
            if contract["D_raw"] is not None and not isinstance(live_raw, Mapping):
                raise ValueError(
                    f"{name}: image processing requires live raw calibration"
                )
            if live_raw is not None:
                for key in ("K", "D"):
                    if contract[
                        f"{key}_raw"
                    ] is not None and not _same_numeric_geometry(
                        live_raw.get(key), contract[f"{key}_raw"]
                    ):
                        raise ValueError(
                            f"{name}: live raw {key} differs from image processing contract"
                        )


class PiperObservationAdapter:
    def __init__(
        self, metadata, *, image_shapes=None, allow_synthetic_calibration=False
    ):
        self.metadata = copy.deepcopy(dict(metadata))
        self.extrinsics = validate_piper_metadata(
            self.metadata, allow_synthetic_calibration=allow_synthetic_calibration
        )
        self.observation_to_send_s = observation_to_send(self.metadata)
        encoding = self.metadata["gripper_encoding"]
        self.gripper_open_m = float(encoding.get("open_m", np.nan))
        if (
            not np.isfinite(self.gripper_open_m)
            or self.gripper_open_m <= 0
            or encoding.get("closed_m", 0) != 0
        ):
            raise ValueError("gripper encoding requires closed_m=0 and positive open_m")
        self.image_shapes = {}
        self.image_processors = {}
        processing = self.metadata.get("image_processing")
        if processing is not None and (
            not isinstance(processing, Mapping)
            or not set(CAMERA_KEYS) <= set(processing)
        ):
            raise ValueError("image_processing must provide all three camera contracts")
        for name, key in CAMERA_KEYS.items():
            if image_shapes is None:
                shape = (
                    self.metadata.get("features", {})
                    .get(f"images.{name}", {})
                    .get("shape")
                )
                shape = (
                    (shape[2], shape[0], shape[1])
                    if shape and len(shape) == 3
                    else None
                )
            else:
                shape = image_shapes.get(key)
            if (
                shape is None
                or len(shape) != 3
                or shape[0] != 3
                or any(int(v) != v or v <= 0 for v in shape)
            ):
                raise ValueError(f"Missing/invalid CHW image shape for {name}")
            self.image_shapes[name] = tuple(map(int, shape))
            if processing is not None:
                processor = RgbImageProcessor(processing[name])
                if processor.contract["output_size"] != [int(shape[2]), int(shape[1])]:
                    raise ValueError(
                        f"{name}: model image shape differs from stored image processing output"
                    )
                self.image_processors[name] = processor

    def base_to_camera_state(self, state):
        state = _finite(state, (14,), "base_state")
        result = []
        for i, side in enumerate(("left", "right")):
            row = state[7 * i : 7 * i + 7]
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_rotvec(row[3:6]).as_matrix()
            matrix[:3, 3] = row[:3]
            camera_tcp = np.linalg.inv(self.extrinsics[side]) @ matrix
            result.extend(_matrix_to_xyzypr(camera_tcp[None])[0])
            # Preserve feedback values, including small physical overshoot.
            result.append(row[6] / self.gripper_open_m)
        return np.asarray(result)

    def camera_to_base_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if (
            actions.ndim != 2
            or actions.shape[1] != 14
            or not len(actions)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("camera actions must be finite (M,14)")
        result = np.empty_like(actions)
        for i, side in enumerate(("left", "right")):
            offset = i * 7
            camera_tcp = _xyzypr_to_matrix(actions[:, offset : offset + 6])
            base_tcp = self.extrinsics[side][None] @ camera_tcp
            result[:, offset : offset + 3] = base_tcp[:, :3, 3]
            result[:, offset + 3 : offset + 6] = Rotation.from_matrix(
                base_tcp[:, :3, :3]
            ).as_rotvec()
            result[:, offset + 6] = actions[:, offset + 6] * self.gripper_open_m
        # Commands are not clipped here. The executor rejects unsafe targets.
        return result

    def prepare(self, frame):
        frame.validate()
        batch = {
            STATE: torch.from_numpy(
                self.base_to_camera_state(frame.base_state)
            ).float()[None]
        }
        for name, key in CAMERA_KEYS.items():
            _, height, width = self.image_shapes[name]
            rgb = np.asarray(frame.images[name])
            if name in self.image_processors:
                rgb = self.image_processors[name].apply(
                    rgb, image_space=frame.image_space
                )
            elif rgb.shape[:2] != (height, width):
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            batch[key] = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[
                None
            ]
        return {EMBODIMENT: batch}


def checkpoint_contract(checkpoint):
    """Fail closed for legacy/other-embodiment checkpoints with unknown units."""
    hp = checkpoint.get("hyper_parameters", {})
    cfg = OmegaConf.create(hp.get("config_tree", {}))
    contract_node = OmegaConf.select(cfg, f"data_contracts.{DOMAIN}")
    if contract_node is None:
        raise ValueError("Checkpoint lacks the supported Piper data contract")
    contract = OmegaConf.to_container(contract_node, resolve=True)
    expected = {
        "version": 1,
        "kind": "piper",
        "reference_frame": "current_front_1_optical_camera",
        "pose_output": "xyz_m+yaw_pitch_roll_rad_ZYX",
        "hand_order": ["left", "right"],
        "observation_source": "measured_pose",
        "action_source": "command",
        "gripper_mode": "width",
        "gripper_encoding": "fixed_width_ratio_no_episode_rescaling",
        "calibration": "per_episode_T_base_camera",
    }
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError("Checkpoint lacks the supported Piper data contract")
    node = OmegaConf.select(cfg, f"action_timing.{DOMAIN}")
    if node is None:
        raise ValueError("Checkpoint lacks Piper action_timing")
    node = OmegaConf.to_container(node, resolve=True)
    names = (
        "source_fps",
        "source_horizon",
        "action_stride",
        "output_chunk_length",
        "includes_current",
    )
    try:
        timing = ActionTiming(**{k: node[k] for k in names})
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid checkpoint action_timing") from exc
    if node != timing.to_dict():
        raise ValueError("Checkpoint derived timing does not match its source settings")
    state = hp.get("norm_stats_state", {})
    if state.get("norm_mode") not in ("quantile", "minmax", "zscore"):
        raise ValueError("Checkpoint lacks supported normalization")
    shapes = state.get("shapes", {}).get(EMBODIMENT, {})
    if tuple(shapes.get(STATE, ())) != (14,) or tuple(shapes.get(ACTION, ())) != (
        timing.output_chunk_length,
        14,
    ):
        raise ValueError("Checkpoint Piper state/action shapes disagree with timing")
    stats = state.get("norm_stats", {}).get(EMBODIMENT, {})
    required = {
        "quantile": ("quantile_1", "quantile_99"),
        "minmax": ("min", "max"),
        "zscore": ("mean", "std"),
    }[state["norm_mode"]]
    for key in (STATE, ACTION):
        for stat in required:
            _finite(
                stats.get(key, {}).get(stat),
                tuple(shapes[key]),
                f"normalization {key}/{stat}",
            )
        lo, hi = (np.asarray(stats[key][s]) for s in required)
        if (state["norm_mode"] == "zscore" and np.any(hi < 0)) or (
            state["norm_mode"] != "zscore" and np.any(hi < lo)
        ):
            raise ValueError("Invalid checkpoint normalization range")
    key_types = state.get("key_types", {}).get(EMBODIMENT, {})
    if (
        key_types.get(STATE) != "proprio_keys"
        or key_types.get(ACTION) != "action_keys"
        or any(key_types.get(k) != "camera_keys" for k in CAMERA_KEYS.values())
    ):
        raise ValueError(
            "Checkpoint Piper key types do not match the deployment interface"
        )
    aliases = state.get("zarr_keys", {}).get(EMBODIMENT, {})
    if any(aliases.get(key) != key for key in (STATE, ACTION, *CAMERA_KEYS.values())):
        raise ValueError(
            "Checkpoint Piper aliases differ from the supported local interface"
        )
    model = cfg.model.robomimic_model
    if model.get("_target_") != "egomimic.algo.hpt.HPT" or DOMAIN not in model.get(
        "domains", []
    ):
        raise ValueError("Deployment supports HPT Piper checkpoints only")
    # This entry point intentionally accepts the approved non-language recipe.
    # A language stem could download weights before load_state_dict executes.
    if model.get("multitask", False) or model.get("annotation_key") is not None:
        raise ValueError("Deployment currently supports the non-language Piper recipe")
    stem_groups = [model.get("shared_stem_specs", {})]
    stem_groups.extend(model.get("stem_specs", {}).values())
    for group in stem_groups:
        for stem in (group or {}).values():
            if stem.get("_target_") != "egomimic.models.hpt_nets.MLPPolicyStem":
                raise ValueError(
                    "Deployment only supports the saved MLP observation stems"
                )
    if any(
        kind == "lang_keys"
        for kinds in state.get("key_types", {}).values()
        for kind in kinds.values()
    ):
        raise ValueError("Deployment currently supports the non-language Piper recipe")
    head = model.get("head_specs", {}).get("shared", {})
    if (
        head.get("_target_") != "egomimic.models.fm_policy.FMPolicy"
        or head.get("action_horizon") != timing.output_chunk_length
        or head.get("model", {}).get("act_dim") != 14
    ):
        raise ValueError(
            "Deployment requires the shared 14-D FM head with saved timing"
        )
    return cfg, timing, copy.deepcopy(state), shapes


class PiperPolicy:
    def __init__(self, model, adapter, timing, *, inference_steps=10):
        if (
            isinstance(inference_steps, bool)
            or int(inference_steps) != inference_steps
            or inference_steps < 1
        ):
            raise ValueError("inference_steps must be a positive integer")
        self.model, self.adapter, self.timing = model, adapter, timing
        self.inference_steps = int(inference_steps)
        # Execution starts each chunk at this delay after its observation, as in training.
        self.observation_to_send_s = adapter.observation_to_send_s
        self.checkpoint_path = None
        self.gripper_contract = {
            "open_m": adapter.gripper_open_m,
            "source": "explicit_runtime_dataset_metadata",
            "checkpoint_open_m_verified": False,
            "limitation": "v1 training contracts do not store open_m; verify the same opening was used to convert training data before hardware deployment",
        }

    @classmethod
    def from_checkpoint(
        cls,
        path,
        metadata,
        *,
        device="cpu",
        inference_steps=10,
        allow_synthetic_calibration=False,
    ):
        """Load a trusted local Lightning checkpoint without downloading weights.

        The state_dict is complete and loaded strictly, so ImageNet/other
        initialization downloads are unnecessary. No training data are opened.
        """
        from egomimic.pl_utils.pl_model import ModelWrapper

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg, timing, state, shapes = checkpoint_contract(checkpoint)
        adapter = PiperObservationAdapter(
            metadata,
            image_shapes=shapes,
            allow_synthetic_calibration=allow_synthetic_calibration,
        )
        cfg = copy.deepcopy(cfg)
        cfg.model.robomimic_model.device = str(device)
        cfg.model.robomimic_model.pretrained = False
        cfg.model.robomimic_model.pretrained_checkpoint = ""
        for encoder in cfg.model.robomimic_model.encoder_specs.values():
            if encoder.get("_target_") != "egomimic.models.hpt_nets.ResNet":
                raise ValueError(
                    "Deployment currently supports the saved ResNet image encoders"
                )
            encoder.weights = None
        wrapper = ModelWrapper(
            config_tree=cfg, norm_stats_state=state, enable_grad_norm=False
        )
        wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
        wrapper.eval()
        result = cls(wrapper.model, adapter, timing, inference_steps=inference_steps)
        result.checkpoint_path = str(Path(path).resolve())
        result._wrapper = wrapper
        return result

    def predict(self, frame):
        batch = self.adapter.prepare(frame)
        started = time.perf_counter()
        output = self.model.predict_from_observations(
            batch, num_inference_steps=self.inference_steps
        )
        camera = output[f"{DOMAIN}_{ACTION}"].detach().cpu().numpy()
        if camera.shape != (1, self.timing.output_chunk_length, 14):
            raise ValueError("Policy output shape disagrees with checkpoint timing")
        base = self.adapter.camera_to_base_actions(camera[0])
        return {
            "camera_actions": camera[0],
            "base_actions": base,
            "observation_timestamp": frame.timestamp,
            "inference_elapsed_s": time.perf_counter() - started,
            "inference_steps": self.inference_steps,
        }
