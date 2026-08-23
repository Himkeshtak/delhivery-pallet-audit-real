# Detection model card: detect-real-v1

## Intended use

YOLO11n detects the supplied dataset's `pallet` and `hole` classes as the first
stage of structural pallet localization. It is a research baseline for the
assignment, not a warehouse safety acceptance model. It does not output metric
pose, directed face, load dimensions, damage state, or an SOP verdict.

## Training decision and cost

- Initial weights: official `yolo11n.pt` pretrained checkpoint.
- Real training data: 1,488 capture-grouped images and 61,091 instances.
- Validation: 458 images/1,960 instances; no group overlaps training.
- Configuration: 15 epochs, 320 px, batch 16, AdamW selected by Ultralytics,
  seed 42, deterministic training requested, workers 0, no image cache, first
  10 model modules frozen.
- Hardware: Intel Core Ultra 5 125U, PyTorch 2.8 CPU-only.
- Cost: 41.1 minutes measured wall time. Freezing and 320 px made CPU training
  feasible but sacrificed high-resolution small-hole detail and full-backbone
  domain adaptation.

## Evaluation

The independently evaluated 262-image/573-instance test split holds out the
2025-08-06 capture family plus separately grouped non-date images. Overall test
precision/recall/mAP50/mAP50-95 is 0.905/0.826/0.876/0.556. Per-class values and
all AP IoU thresholds are in `reports/detection_test_evaluation.json`.

For 501 confidence-0.25 predictions matched at IoU 0.50, median IoU is 0.834,
median center error is 4.38 px, and p95 center error is 13.61 px. These are
image-space localization results; they do not imply the assignment's metric
`+/-2 cm / +/-3 degrees` pose bar.

## Limits and risk controls

The public data contains source augmentations, noisy taxonomy, empty declared
classes, at least one missing negative-image annotation, and camera/domain
mismatch. Wrapped loads create hole-like texture false positives. Downstream
logic must require supporting pallet geometry, calibrate thresholds, and return
`MANUAL_REVIEW` outside the measured domain. The checkpoint must not be used to
assert damage absence or hidden-side compliance.

## Artifact

`weights/releases/detect-real-v1.pt`, 5,427,738 bytes, SHA-256
`36273b60fb817a0d869188714c90ca8156a3370bb6da1d3536d71f7359cb7662`.
