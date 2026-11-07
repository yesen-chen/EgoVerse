# Piper / PICO training interface: stage 1

Stage 1 adds data identity and an explicit action time contract. It does not
implement the Piper calibration loader, PICO gripper training modes, co-training
recipes, or robot deployment. Existing Human / Eva defaults are unchanged.

Stages 2–4 are now described in `README_piper_pico_training.md`; the scope above
refers specifically to stage 1. The temporal conventions below still apply.

## Data identity

- Existing Human / Eva IDs 1–6 retain their meanings.
- `piper_bimanual` has ID 7. New robot conversion outputs use `piper_base_v3` (one row per collector control tick); the loader rejects older schemas.
- PICO ego remains `human_bimanual`; its provenance identifies the PICO source.
- Older diagnostic Piper exports labeled `eva_bimanual` are not automatically
  reinterpreted. Re-export from the raw recording to a **new output directory**.
- Neither renaming the embodiment nor a time contract supplies missing camera
  calibration. Uncalibrated exports remain diagnostic only.

## Time contract

`egomimic.rldb.action_timing.ActionTiming` requires all four numerical parameters:

| Field | Meaning |
| --- | --- |
| `source_fps` | Uniform **dataset** sample frequency, not video decoder FPS |
| `source_horizon` | Number of consecutive rows read, including the current row |
| `action_stride` | Keep indices `0, stride, 2*stride, ...` within that window |
| `output_chunk_length` | Number of points after action-window interpolation |
| `includes_current` | Explicitly `true` in stage 1; other origins are unsupported |

For H rows, stride s, dataset frequency f, and M output points:

```
support_indices = range(0, H, s)
source_span_s = (H - 1) / f
effective_span_s = support_indices[-1] / f
output_offsets_s = linspace(0, effective_span_s, M)
```

This exactly follows the current EgoVerse order: the reader starts at `idx`,
`InterpolatePose` / `InterpolateLinear` slice `[::stride]`, and interpolation
preserves the first and last **retained** point. For example, the existing Eva
H=45,s=1,f=30 window spans 44/30 seconds; the existing Aria H=30,s=3,f=30 support
ends at row 27 (0.9 seconds), not row 29. These are comparison cases, **not chosen
defaults for the new Piper / PICO experiment**.

Changing M never changes the span. Changing H or s may change it. At least two
support points and two output points are required. Episode-end repeat-last
padding retains nominal time offsets; those repeated values are not additional
measured future observations.

The anchor is the observation sample time. Human action row zero is the current
observed virtual pose; robot action row zero is the current command, which need
not equal the current measured robot pose. The contract does not claim command
send/arrival times or compensate execution delay.

Raw ego conversion's `--pose-sampling interpolate|nearest` answers how raw
tracking samples are queried when producing the regular dataset. It is separate
from **training action-window** interpolation described here. Neither changes
the meaning of the other's switch.

## Configuration, logging, and checkpoint persistence

Training accepts an optional top-level `action_timing` mapping. It is `null` by
default; old recipes continue unchanged. Example declaration for inspecting the
existing Eva temporal settings (not a new training recipe):

```yaml
action_timing:
  eva_bimanual:
    source_fps: 30
    source_horizon: 45
    action_stride: 1
    output_chunk_length: 100
    includes_current: true
```

When enabled, declare every selected training embodiment separately. Before
normalization, training checks all selected train/valid episode FPS metadata,
windowed action key horizons, and action interpolation stride/output lengths.
The actual configured HPT action heads (including active shared heads) or PI
output horizon must match the declared output point count as well. Unknown
model structures fail explicitly when the contract is enabled.
Mismatches fail instead of silently recording incorrect time information.
This validates the declared dataset grid, not physical sensor timing.

This declaration does **not** silently rewrite existing keymaps or transforms.
Stages 2/3 will build Piper/PICO pipelines directly from the contract. Until
then, opt-in declarations must match the existing pipeline. The model's internal
conditioning token count is a different setting and is not checked as an action
trajectory length.

The source window and effective spans are logged. The resolved parameters and
derived offsets are saved in:

```
checkpoint['hyper_parameters']['config_tree']['action_timing'][embodiment]
```

Root-config references are resolved before checkpoint detachment. Older
checkpoints without this field still load; absent timing is **unknown**, not an
implicit frequency or window. Deployment will consume this metadata later;
stage 1 adds no action scheduling, expiry filtering, asynchronous inference,
or RTC.

Normal `ModelWrapper.load_from_checkpoint` restores the saved contract. During
`Trainer.fit(..., ckpt_path=...)` resume, Lightning uses the current model config;
the restore hook rejects a changed or omitted contract if the checkpoint has
one. Resuming a legacy checkpoint with an explicit new contract warns that the
old timing is unknown. This is a resume guard, not a weights-transfer policy;
manually loading a state dict or explicitly overriding checkpoint constructor
hyperparameters is outside this guarantee.

To print and save a declaration without opening a dataset or model:

```bash
cd /home/yesen/pico_ego_probe/EgoVerse
source emimic/bin/activate
python -m egomimic.scripts.inspect_action_timing \
  --config /absolute/path/to/timing.yaml \
  --output /absolute/path/to/new_timing_report.json
```

The standalone inspector checks parameters and computes timing only; training
performs the additional runtime dataset checks. Reports never overwrite files.

## Acceptance

- Compare output time offsets against the **existing** pose/width interpolators
  on known position ramps and rotations, including stride-truncated endpoints.
- Verify different output point counts preserve the physical span.
- Reject invalid parameters and mismatches in any train/valid episode.
- Save/reload a real Lightning checkpoint, including a legacy checkpoint.
- Train two CPU steps through the original Eva pipeline with the contract
  enabled, save it, and reject a mismatching contract through actual resume.
- Verify old identity mappings and that newly converted Zarr is read as Piper.
- Run existing configuration and CPU train-step regression tests.
