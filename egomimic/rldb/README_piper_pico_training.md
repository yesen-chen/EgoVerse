# Piper-X / PICO Ego local training

This implements stages 2–4 of the Piper/PICO data interface. The existing Eva
and Human recipes remain available. The action time contract and original
EgoVerse comparison are documented in `README_piper_pico_stage1.md`.

## Data semantics

| Domain | Stored poses | Training state | Training action |
| --- | --- | --- | --- |
| `piper_bimanual` (7) | Each arm's base frame | Measured pose + measured gripper | Command pose + commanded gripper |
| `human_bimanual` (3), PICO | Tracking world | Current virtual TCP | Future observed virtual TCP |

Both loaders produce `[xyz, yaw, pitch, roll]` using ZYX Euler convention, in
metres/radians, in the **current observation's front_1 optical camera frame**.
Piper uses its episode's inverse `T_base_camera`, separately for each arm.
PICO uses the inverse of the current `obs_head_pose` (already the left optical
camera's world pose). Every future Ego action uses that same current camera.
The loader does not reapply conversion fits or change tool axes. PICO defaults
to requiring `piper_x`; mismatched or missing axis metadata is an error.

Piper loads front_1, left_wrist and right_wrist. Ego loads only front_1.
Piper's per-episode calibration is mandatory. Synthetic calibration is rejected
unless explicitly enabled; this option is for offline development only.
Discovery preflights every selected episode, reports all bad paths together,
and never silently skips bad calibration or substitutes a random JPEG frame.

Ego `recordings.gripper_mode` controls:

- `zero`: loader state/action are 12-D. After statistical normalization, the
  shared head inserts zeros to form `[left_pose, 0, right_pose, 0]`.
- `width`: loader state/action are 14-D using the converter's saved fixed width
  ratio. No second division by maximum opening and no per-episode min/max.

Piper is always 14-D. The shared Flow Matching head is always 14-D; evaluation
of zero-mode Ego removes the two gripper coordinates before unnormalization.
This keeps the original zero-mode loss convention, without an extra loss mask.

## Time and normalization

`recordings.piper_timing` and `recordings.ego_timing` independently specify
`source_fps`, `source_horizon`, `action_stride`, and `includes_current=true`.
Both use `recordings.output_chunk_length` because the head is shared.
The shipped 30 Hz / H=30 / stride=1 / M=100 values are **editable examples**,
not chosen experiment hyperparameters. That example spans 29/30 seconds;
100 is a trajectory discretization, not a prescribed robot execution rate.
Raw conversion interpolation remains a separate setting.

Stats are computed **after** camera transforms and action-window construction,
separately for domains 7 and 3, using training recordings only. Validation reuses
training stats. This statistical normalization is distinct from converting
finger distance/gripper opening into a fixed width ratio.

New local datasets default to `recordings.sample_policy=finite`: finite,
schema-valid samples outside training quantiles remain in evaluation with their
original identity. NaN/Inf and corrupt data still fail. This intentionally avoids
the legacy sampler's random replacement of out-of-quantile samples (which can
discard a whole validation recording). For comparison with that sampling rule,
set `recordings.sample_policy=legacy_quantile`. The original Eva/Human sampler
is unchanged. The normalization formula itself is unchanged and does not clip
out-of-range normalized values.

Checkpoint hyperparameters contain `norm_stats_state`, plus resolved
`config_tree.action_timing` and `config_tree.data_contracts`. Local full resume
uses the saved statistics and checks domain/key/shape, timing and semantic
compatibility. A changed calibration also invalidates the norm cache.

## Recording identity and split

The resolver searches recursively. IDs include the source subdirectory, so
`batch_a/episode_000000` and `batch_b/episode_000000` remain distinct.
Train/validation splitting groups **whole original recordings**, including
all Ego segments and alternate conversions. A single recording cannot supply
both sides; the loader fails rather than claiming a held-out validation set.

Ego provenance uses the source tracking SHA256. Piper uses the saved original
recording directory in `conversion.source_dataset`. Export copies preserving
that metadata remain grouped. Copying raw Piper recordings to new paths and
reconverting currently cannot be detected as duplicate content; retain the
original provenance and do not put such copies on opposite sides of a split.

## Launch configuration

Run from the repository root with the existing environment. Supply calibrated
Piper exports and at least two independent source recordings per selected
domain when using a train/validation split.

```bash
source emimic/bin/activate
python -m egomimic.trainHydra --config-name train_piper_pico \
  data=piper_pico_local \
  recordings.piper_root=/path/to/piper_exports \
  recordings.pico_ego_root=/path/to/pico_ego_exports \
  recordings.gripper_mode=zero
```

Select `data=piper_local` or `data=pico_ego_local` for single-domain training.
Use `recordings.gripper_mode=width` for measured Ego width supervision.
The local trainer defaults to CPU; accelerator, model size, workers and other
training options remain ordinary Hydra settings. Encoder defaults follow the
existing model recipe; offline unit acceptance explicitly disables pretrained
weight downloads and uses a small CPU model.

Example time override, applied consistently to keymaps, transforms and model:

```text
recordings.piper_timing.source_horizon=45
recordings.ego_timing.source_horizon=45
recordings.piper_timing.action_stride=2
recordings.ego_timing.action_stride=2
recordings.output_chunk_length=100
```

## Inspect transformed batches

This CLI reads the actual training loader outputs in physical units, before
statistical normalization. It produces current-image projections, action
trajectories and width curves, plus a JSON time/shape report. It never refits
calibration or connects to hardware.

```bash
python -m egomimic.scripts.data_visualization.inspect_piper_pico_batch \
  /path/to/episode_000000.zarr \
  --kind pico_ego --gripper-mode width \
  --source-fps 30 --source-horizon 30 --action-stride 1 \
  --output-chunk-length 100 --output /path/to/new_inspection_directory
```

For Piper use `--kind piper`. Development-only mock extrinsics require
`--allow-synthetic-calibration` and are labelled on the plots. Agreement with
an independent matrix/projection calculation validates software geometry; it
does not establish physical calibration accuracy or a learned policy's success.

## Offline acceptance

Tests cover stage-1 time invariants; per-episode/per-arm SE3 transforms and
inverse round trips; moving-camera windows; held samples; zero/width dimensions;
recording splits; corruption failures; separate stats and normalization round
trips; real shared-FM optimization/validation; checkpoint restoration; config
composition; and regression tests for the original train-step matrix.

Recorded-data reports and plots from this implementation are under:

- `diagnostics/piper_stage1_20261024/`
- `diagnostics/piper_stage2_20261024/`
- `diagnostics/pico_stage3_20261024/`
- `diagnostics/piper_pico_stage4_20261024/`

Original recordings and earlier conversion outputs remain unchanged. Robot
deployment, hardware calibration, asynchronous scheduling and RTC are outside
these stages. The agreed first deployment is synchronous block execution.

The subsequent observation-only HPT interface, synchronous block executor, and
no-CAN recorded-data replay are documented in
[`README_policy_inference.md`](../robot/piper_pico/README_policy_inference.md).
That offline entry point loads the saved timing and normalization, defaults to
10 FM inference steps, and records simulated sends without constructing devices.
