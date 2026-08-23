# Segmentation model card: segment-real-v1

## Intended use

YOLO11n-seg predicts the eight observed structural-part classes in the supplied
PLH-C.1 polygon export: `front`, `hole`, `hole_left`, `hole_right`, `pallet`,
`pallet_front`, `pallet_pocket`, and `wood`. It is an accuracy/feasibility
baseline for visible pallet structure, not a complete load-compliance model.

## Training decision and cost

- Initial weights: official `yolo11n-seg.pt` pretrained checkpoint.
- Real training data: 841 grouped images and 2,955 polygons.
- Validation: 106 images/363 polygons; no group overlaps training.
- Configuration: 20 epochs, 320 px, batch 16, AdamW selected by Ultralytics,
  seed 42, deterministic training requested, workers 0, no image cache, first
  10 model modules frozen.
- Hardware: Intel Core Ultra 5 125U, PyTorch 2.8 CPU-only.
- Cost: 32.7 minutes measured wall time. The low resolution/frozen backbone made
  CPU training tractable but limits thin boundaries and domain adaptation.

## Evaluation

On the independently evaluated 105-image/335-instance test set, overall box
precision/recall/mAP50/mAP50-95 is 0.563/0.444/0.442/0.254. Overall mask
precision/recall/mAP50/mAP50-95 is 0.471/0.318/0.314/0.132.

Per-class mask mAP50-95 is `front` 0.009, `hole` 0.053, `hole_left` 0.065,
`hole_right` 0.111, `pallet` 0.000, `pallet_front` 0.269,
`pallet_pocket` 0.506, and `wood` 0.045. The zero full-`pallet` result and low
recall make abstention mandatory for unsupported SOP evidence.

## Limits and risk controls

The taxonomy is not the desired `pallet`/`load` instance pair and includes
overlapping part meanings. No `load`, box, stretch-wrap, damage, or physical
dimension truth exists. This model cannot determine overhang, height, centring,
damage, or hidden-side compliance. SAM 2 remains an offline human-reviewed
annotation accelerator; it was not used to create unattended pseudo-truth.

## Runtime and artifact

Standalone CPU latency at 320 px is 59.23 ms median and 61.67 ms p95 over 105
images (16.91 FPS). The committed checkpoint is
`weights/releases/segment-real-v1.pt`, 5,961,764 bytes, SHA-256
`79aaaa6c147ef5662f624e5db607f25fce0dea209b573f75c8c67554dccdcd0c`.
