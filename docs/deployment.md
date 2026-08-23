# Deployment and robustness

## Measured development hardware

The detection checkpoint was measured on Windows 11 with PyTorch `2.8.0+cpu`,
no CUDA device, eight active inference threads, and an Intel Core Ultra 5 125U.
Across all 262 test images after 10 warm-up frames, single-image end-to-end
latency was 36.36 ms mean, 36.23 ms median, 38.60 ms p95, and 51.07 ms maximum.
Measured throughput from total wall time was 27.50 FPS at 320 px. This includes
in-memory preprocessing, detector inference, and postprocessing; it excludes
decode, segmentation, pose geometry, anomaly scoring, tracking, and SOP logic.

This is not a Jetson Orin Nano measurement. No external Jetson benchmark is
copied into this report.

The structural-part segmenter measured 59.14 ms mean, 59.23 ms median,
61.67 ms p95, and 62.94 ms maximum over 105 test images, or 16.91 FPS alone.
Adding the independently measured detector and segmenter means gives a 95.5 ms
arithmetic sequential estimate (about 10.5 FPS), not a measured full-pipeline
benchmark. The naive every-frame composition therefore misses the 15 FPS goal.

## Target path: Jetson Orin Nano, 15 W, at least 15 FPS

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
5. defer Mask2Former and PatchCore - they are comparisons, not edge defaults.

## Failure contract

Bad calibration, high reprojection error, missing/weak topology, implausible
pallet dimensions, pose outside the calibrated polygon, wide covariance, blur,
occlusion, inadequate visible coverage, unstable temporal evidence, and stale
tracks must surface as explicit reason codes. Confirmed visible failures remain
failures; unknown evidence becomes `MANUAL_REVIEW`. The downstream consumer never
has to infer failure from missing JSON fields.
