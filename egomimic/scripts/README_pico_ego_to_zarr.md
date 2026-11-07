# PICO ego recording → world-frame EgoVerse Zarr

This converter is separate from `piper_pico_to_zarr.py` (real robot data). It
reads XRoboToolkit JSONL tracking and a video, applies a supplied **frozen**
fit, and exports one Zarr episode per continuous valid bimanual segment.
It does not fit parameters, control hardware, or run MediaPipe/MANO.

## Convert the existing recording

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate
python -m egomimic.scripts.pico_ego_to_zarr \
  --input ../xrobo_record_20261013_234134_zRntdg \
  --fit ../xrobo_independent_fit_20261010_212402/fit.json \
  --output-dir converted_ego \
  --fps 30 \
  --axis-convention piper_x
```

The input directory must have exactly one `trackingData_*.txt` and one
`CameraRecord_*.mp4`, or specify `--tracking` / `--video`. Output is
`OUTPUT/INPUT_NAME/episode_000000.zarr`, etc., plus `conversion_report.json`.
Existing destinations and output inside the raw recording are rejected.
The complete recording is published atomically only after all segments succeed.

Default `--video-layout stereo-sbs` saves **only the left half**. For a recording
whose video already contains the left eye alone, use `--video-layout left`.
The fit must match that camera's native resolution and optical geometry.
`eye_swap` only selects the corresponding metadata extrinsic; it never changes
the image crop. Output JPEGs are 640×480 by default; intrinsics scale with them.

## Time alignment and missing data

The first video PTS is associated with the header `timeStampNs`. Subsequent
image times use actual decoded PTS and rational time base, not frame number
divided by nominal source FPS. Epoch arithmetic remains integer nanoseconds.

Training rows are a uniform 30 Hz grid (`--fps`), anchored at the header time.
For each grid timestamp `t`:

- Image: nearest actual video timestamp, ties earlier. Repeated images are
  allowed and counted; they are not represented as independent exposures.
- Head query: `t + head_offset_ms` from the fit.
- Hand query: `t + hand_offset_ms` from the fit.
- Default `--pose-sampling interpolate`: positions/width/keypoints interpolate
  linearly; orientations use shortest-arc SLERP. Adjacent source samples must be
  valid and at most 50 ms apart by default.
- Optional `--pose-sampling nearest`: copy the closest original head/hand sample
  to its respective corrected query time, ties earlier. No interpolation or
  smoothing is performed. The closest row must be valid; the converter does not
  skip an invalid row to find a distant valid row. Its absolute time error must
  be at most `--max-pose-dt-s` (default 0.025 seconds). This offline nearest rule
  may select a sample slightly later than the query; it is not a causal filter.

Both modes retain the same 30 Hz grid, camera matching and measured fit offsets.
High-rate tracking often makes nearest sampling sufficient; interpolation
reduces the remaining temporal quantization. The fit offsets are still applied
when interpolation is disabled. Source support timestamps are retained so the
actual match can be checked. The maximum hold duration applies only to genuine
short invalid gaps, not to an otherwise valid sample exceeding the match limit.

`timestamp_ns` and `source.grid_timestamp_ns` are the grid. The separate
`obs_rgb_timestamps_ns`, `source.video_frame_index`, and `source.video_pts_ns`
describe the actual selected image. Thus a 50 fps source can have image intervals
of 20/40 ms while the **training rows remain uniformly 33.333 ms apart**. The
projection report includes the image-to-grid difference; it is not hidden in
the fitted offsets. Change the capture mode/FPS only after rechecking the fit.

Head gaps and unsupported start/end records are discarded. Hand gaps can hold
the last **world-frame** pose, keypoints, and width for at most `--max-hold-s`
(default `4/30` seconds). The current camera continues moving. A short gap must
have a valid recovery sample; the entire gap must fit the duration limit, and
each held sample must be within the age limit. A known longer gap is dropped
in its entirety, including its first four periods. No initial/terminal hand
extrapolation is used. Status-invalid or degenerate skeletons count as missing.

Long gaps split episodes, so a training action window cannot cross them. A
recovery after holding also splits when its position changes by over 0.1 m or
orientation by over 60° (configurable). Image jumps over 50 ms split episodes;
images farther than `--max-image-dt-s` (default 1/30 s) from the grid are excluded.
Segments shorter than `--min-segment-frames` (default 2) are dropped.

## Coordinates, hand axes and grip

All poses use metres and `[x,y,z,qw,qx,qy,qz]`:

| Key | Meaning |
| --- | --- |
| `left/right.obs_ee_pose` | Virtual hand end frame in the tracking world |
| `obs_head_pose` | Left **optical camera** in the tracking world, `T_world_camera` |
| `source.head_pose` | Sampled original headset pose, for verification |
| `source.head_support_timestamp_ns` | Actual two source bracket timestamps; equal for nearest |
| `left/right.obs_keypoints` | Canonical 21-point layout, world xyz (63 values), no MANO fit |
| `left/right.obs_xr_keypoints` | All 26 original XR points, world xyz (78 values) |
| `left/right.obs_pinch_distance_m` | Raw thumb-tip/index-tip distance |
| `left/right.obs_gripper_width_m` | Distance clipped to 0…0.08 m |
| `left/right.obs_gripper` | Clipped distance divided by fixed 0.08 m |
| `left/right.hand_valid` | A valid observed/interpolated sample, not a hold |
| `left/right.hand_filled` | The sample was held across a short loss |
| `left/right.source_age_s` | Interpolation: age of lower bracket; nearest: absolute time error; hold: last-valid age |
| `left/right.source_support_timestamp_ns` | Actual bracket timestamps, or identical nearest/held timestamps |
| `left/right.gripper_clipped` | Raw pinch exceeded the fixed maximum |

The virtual origin is the mean of XR wrist, index/middle/ring MCP joints
`[1,7,12,17]`, following the palm-region convention in original EgoVerse.
There is **no translation to a fingertip grasp center**. The orientation is
constructed from bones, not unverified raw joint quaternions. Forward is
wrist→palm; the normal is `(index_MCP−wrist) × (pinky_MCP−wrist)`, sign-flipped
for the left hand; the frame is orthogonalized to a proper rotation.

Default `piper_x`: Z forward, X lateral/opening, Y signed palm normal. `piper`
right-multiplies this frame by Rz(−90°), giving Y as the opening axis. These are
virtual anatomical correspondences to the official model conventions, not a
measured calibration of a human hand to a physical tool. Robot records continue
to use their existing SDK end frame. Optional `--hand-frame FILE.json` applies
an additional local proper rotation per hand, after the selected convention:

```json
{"left": [[1,0,0],[0,1,0],[0,0,1]], "right": [[1,0,0],[0,1,0],[0,0,1]]}
```

The grip mapping uses XR tip indices 5 and 10 and a fixed range across all
episodes (`--gripper-max-m`, default 0.08). It never uses per-episode min/max.
Raw widths and clipping counts remain available for auditing old recordings.

## Fit provenance and projection verification

The legacy `fit.json` keys are accepted directly: `extrinsics`, `eye_swap`,
`post_axis`, `rig_rotation`, `rig_translation_m`, `K_fx_fy_cx_cy`,
`head_offset_ms`, and `hand_offset_ms`. The complete JSON, path and SHA-256 are
embedded under `pico_ego.fit`. When its reference recording is still available,
the converter checks intrinsics, extrinsics and native per-eye image size;
otherwise the report explicitly says `reference_unavailable`.

The camera chain is
`T_camera_world = Post @ Eye_left_slot @ Rig @ inverse(T_world_head)`.
The stored `obs_head_pose` is its inverse. No fixed robot-base extrinsics are
invented for ego recordings.

```bash
python -m egomimic.scripts.data_visualization.visualize_pico_ego_zarr \
  --input converted_ego/xrobo_record_20261013_234134_zRntdg \
  --output-dir converted_ego_projection
```

This reads the **exported** JPEGs, camera trajectory, skeletons and virtual TCPs
and produces an overlay video, contact sheet, samples and numeric summary for
each episode. Red/green/blue denote local TCP X/Y/Z. Held hands are highlighted.
The saved camera transform is checked against the independent raw-head chain.
Chain equality detects implementation errors, but does not establish physical
calibration or timing accuracy; inspect the overlay against visible hands.

## Training boundary

This exports the data, not a new training pipeline. The existing Human loader
uses future `obs_ee_pose` as actions and transforms the whole chunk with the
**first/current camera frame**; that convention is retained. It currently does
not consume measured ego gripper fields, so they must be wired into the loader
before joint pose-and-gripper training. No fake `cmd_ee_pose` is created.
Validity/filled masks are retained for a later loss/window policy.

Tests: `pytest tests/unit/test_pico_ego_to_zarr.py tests/unit/test_visualize_pico_ego_zarr.py`.
Conversion decodes on CPU with one video thread. Compressed selected JPEGs are
held in memory for one source recording; use separate recordings for long runs.
