# Deployment and robustness

## Measured development hardware

The currently available runtime has PyTorch `2.8.0+cpu`, reports no CUDA device,
and exposes 12 CPU inference threads. No trained real-data weights are available
yet, so reporting end-to-end model latency now would be meaningless. The
`benchmark_yolo.py` command records raw latency samples, percentiles, throughput,
software versions, and hardware fingerprint once weights and a representative
held-out image directory exist.

This is not a Jetson Orin Nano measurement. No external Jetson benchmark is
copied into this report.

## Target path: Jetson Orin Nano, 15 W, ≥15 FPS

The deployable path is YOLO11n pose + YOLO11n segmentation, geometry/SOP logic on
CPU, EfficientAD-S only on visible crops, and ByteTrack/temporal fusion. Export
each YOLO head to fixed-shape TensorRT FP16 first. Consider INT8 only after using
representative warehouse frames for calibration and measuring the change in:

- structural-keypoint pixel error and downstream translation/yaw distributions;
- pallet/load mask IoU and downstream overhang/centroid errors;
- abstention coverage and false-pass/false-fail rates.

`export_yolo.py` hashes each artifact but deliberately leaves `accuracy_cost`
pending until the same held-out evaluator is run before and after export.

## Latency budget and degradation policy

The 66.7 ms frame budget at 15 FPS is allocated empirically, not assumed. If the
full system misses it, the change order is:

1. run detector/pose every N frames and use ByteTrack between detections;
2. run segmentation/damage only for stable tracks and cache static evidence;
3. batch visible crops for EfficientAD;
4. reduce input resolution only if downstream metric-pose and SOP distributions
   remain within policy;
5. defer Mask2Former and PatchCore—they are comparisons, not edge defaults.

## Failure contract

Bad calibration, high reprojection error, missing/weak topology, implausible
pallet dimensions, pose outside the calibrated polygon, wide covariance, blur,
occlusion, inadequate visible coverage, unstable temporal evidence, and stale
tracks must surface as explicit reason codes. Confirmed visible failures remain
failures; unknown evidence becomes `MANUAL_REVIEW`. The downstream consumer never
has to infer failure from missing JSON fields.

