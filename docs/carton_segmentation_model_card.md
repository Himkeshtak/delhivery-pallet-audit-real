# Carton segmentation model card: carton-seg-scd-v1

## Intended use

YOLO11n-seg predicts one visible instance mask per carton. It supplies visible
box separation and 2D orientation evidence to the pallet-load analysis stage.
It is not a standalone compliance model: OSCD contains no surveyed dimensions,
pallet-to-load relationship, wrap state, damage class, metric pose, or hidden
face evidence.

## Dataset and split controls

- Source: official SCD OSCD real-image archive, CC BY-NC-SA 4.0.
- Author data: 7,401 training and 1,000 test images; the downloaded COCO JSON
  contains 168,758 annotations, ten more than the project summary table.
- Leakage control: 97 identical difference-hash groups crossed the author
  boundary, so 133 matching author-training images were excluded and the
  1,000-image author test partition remained immutable.
- Prepared images: 6,396 train, 872 validation, and 1,000 test, with zero
  cross-split visual-hash groups.
- Effective masks: 128,158 train, 18,218 validation, and 20,186 test. Four
  training duplicates and one test duplicate were exposed by the loader and
  are recorded in `reports/carton_scd_duplicate_audit.json`; the importer now
  applies the same deterministic quarantine rule.

## Training decision and cost

- Initial weights: official COCO-pretrained `yolo11n-seg.pt`.
- Configuration: 10 epochs, 320 px, batch 16, seed 42, workers 0, no image
  cache, patience 4, and first 10 model modules frozen.
- Hardware: Intel Core Ultra 5 125U, PyTorch 2.8.0 CPU-only; CUDA unavailable.
- Measured wall time: 8,460.9 seconds (141.0 minutes).
- Rationale: the nano model, frozen backbone, and 320 px input made a complete
  real-data CPU run feasible while retaining an edge-deployable checkpoint.
  The tradeoff is reduced fine-boundary accuracy and domain adaptation.

## Independent evaluation

The selected validation checkpoint reached box P/R/mAP50/mAP50-95 of
0.912/0.815/0.904/0.722 and mask P/R/mAP50/mAP50-95 of
0.913/0.810/0.895/0.646.

On the untouched 1,000-image test set, box P/R/mAP50/mAP75/mAP50-95 is
0.899/0.823/0.898/0.793/0.719. Mask P/R/mAP50/mAP75/mAP50-95 is
0.900/0.816/0.889/0.745/0.650. Mask AP from IoU 0.50 through 0.95 is
0.889, 0.871, 0.855, 0.829, 0.795, 0.745, 0.660, 0.517, 0.283, and
0.055. The steep fall at 0.90-0.95 IoU is the material boundary limitation.

A deterministic held-out qualitative example is committed under
`examples/carton_segmentation_demo`; its overlay shows the predicted masks, and
its JSON retains the 22 prediction polygons/confidences for an image containing
20 reviewed test annotations. It is supporting visual evidence, not a substitute
for the aggregate immutable-test metrics.

The engineering ceiling for this source domain is approximately 0.70-0.78
mask mAP50-95 with 640 px input, selective backbone unfreezing, longer training,
and annotation cleanup. This is explicitly an estimate, not a measured result,
and it does not transfer to warehouse SOP accuracy without assignment-domain
capture.

## Runtime and release artifact

On the declared Windows 11 CPU with eight PyTorch threads, 1,000 single-image
test runs measured 81.00 ms mean, 78.25 ms median, 116.90 ms p95, and
241.74 ms maximum end-to-end, or 12.35 FPS from total wall time. Decode and
the other pipeline models are excluded.

The committed checkpoint is `weights/releases/carton-seg-scd-v1.pt`,
5,951,652 bytes, SHA-256
`5bffd9bdcd58f588bc6ba5e4165a916f4a366089043cb8c546c14ac3c2dd9fa7`.
The prior detector and structural-segmentation releases remain unchanged.

## Risks and fail-safe behavior

OSCD is biased toward cropped internet and retail/warehouse stack imagery,
often without a visible pallet. The model cannot infer metric overhang,
centroid offset, height, stretch-wrap quality, box damage, or pallet damage
from these labels. Downstream checks therefore require calibrated pallet
geometry and reviewed evidence; absent or out-of-domain evidence returns
`MANUAL_REVIEW`, never an automatic pass.
