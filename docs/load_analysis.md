# Visible-load analysis implementation

`pallet_audit.load_analysis` converts reviewed, calibrated pallet/load/box
polygons into SOP evidence. It is deliberately downstream of perception and
calibration: pixel masks are not metres, and a cropped carton dataset cannot
establish pallet overhang by itself.

## Implemented geometric evidence

| Signal | Computation | Required uncertainty/coverage | Single-view limit |
|---|---|---|---|
| overhang | largest signed distance of a visible load-boundary vertex outside the metric pallet polygon | pallet/load mask confidence, visible coverage, boundary standard deviation | hidden and rear overhang remain unknown |
| geometric centroid offset | Euclidean distance between metric polygon centroids | same boundary uncertainty and valid pose | not the mass centroid; hidden depth can bias it |
| maximum visible box rotation | PCA major axis of anisotropic box masks, folded against both pallet principal axes | per-box confidence, orientation standard deviation, usable-box fraction | square/occluded masks abstain; hidden boxes unknown |
| size-inversion risk | pairwise visible metric areas across reviewed bottom-to-top layer indices | reviewed layers and box confidence | emitted as an uncalibrated score and therefore forced to manual review until calibrated |

The evidence is fed into the same uncertainty-aware `SopPolicy` as other
signals. A metric check passes only when its full 95% interval is below the SOP
limit, fails only when its full interval is above it, and otherwise returns
`MANUAL_REVIEW`. A risk score carrying `score_not_calibrated=true` cannot be
treated as a probability by policy.

The module intentionally emits no load-height, stretch-wrap, box-damage, or
pallet-damage evidence. Those require calibrated vertical geometry and reviewed
task-specific labels/models. Missing evidence is an explicit manual-review
reason, never an implicit pass.

## Runnable image adapter

`tools/infer_frame.py` runs the three committed real checkpoints on one frame:

1. pallet/fork-hole detector;
2. structural pallet-part segmenter;
3. individual-carton segmenter.

It writes `overlay.jpg`, `frame_summary.json`, measured component latency, every
raw prediction, model hashes, and one schema-valid assessment JSON per detected
pallet. Because this repository still lacks an assignment-camera calibration
and trained pallet-keypoint checkpoint, the adapter intentionally returns
`MANUAL_REVIEW` for metric pose and all unsupported checks. This makes the
current repository runnable without making unsupported compliance claims.

Example after the release checkpoints exist:

```powershell
python tools/infer_frame.py --image path/to/frame.jpg --output runs/demo/frame-001
```

