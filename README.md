# Delhivery Pallet Audit - real-data implementation

> Status: both user-supplied real Roboflow exports and the official real-image
> OSCD carton-mask dataset are imported and audited. Leakage-controlled
> pallet detection, structural-part segmentation, and individual-carton
> segmentation models are trained, independently evaluated, benchmarked, and
> committed with hashed weights. No result below comes from synthetic data or
> borrowed hardware.

This project implements one explainable pallet assessment per tracked pallet:
metric floor pose with uncertainty, directed face/orientation, eight SOP checks
with confidence and `PASS` / `FAIL` / `MANUAL_REVIEW`, and overall reasoning.
It is organized in the same order as the assignment rubric.

## Approach and significant decisions

The design is split into measurable perception, geometry, evidence, and policy
stages so a disputed decision can be reconstructed. The main costs are extra
annotation layers, physical calibration/capture, multiple small models, and a
conservative abstention rate. Those costs are preferred to a confident but
untraceable end-to-end verdict.

### 1. Dataset and detection (30%)

The two assignment-listed Roboflow Universe version URLs are pinned in
`configs/data_sources.yaml`; the official OSCD source is pinned in
`configs/external_sources.yaml`. The canonical download format is COCO JSON because
it preserves category metadata, boxes, and polygons without coupling the source
archive to a trainer. `DATASET.md` records the measured counts, sourcing cost,
label rules, biases, licenses, hashes, and the supplier-split leakage finding.

Detection and localization are reported separately on source-grouped holdouts.
The final YOLO11n detector uses COCO-pretrained features, a frozen backbone, a
320 px input, batch 16, and 15 epochs. This made a full CPU run possible in 41.1
minutes, but costs small-hole detail and limits backbone domain adaptation. The
untouched test capture day is declared separately from the harder validation
capture day; neither uses the leaked supplier split.

The observed test mAP50-95 is 0.556. The estimated ceiling for this exact source
taxonomy is roughly 0.65-0.75 with corrected negative labels, 640 px training,
an unfrozen backbone, and more capture days. That is an engineering estimate,
not a measured result, and says nothing about metric pose or a new warehouse.

For individual cartons, the final deployable baseline is YOLO11n-seg fine-tuned
on 6,396 real OSCD training images. Its immutable 1,000-image author test set is
kept outside model selection. These masks support visible box separation and
orientation evidence; because OSCD often crops out the pallet, they do not by
themselves prove load balance, overhang, wrap, or damage compliance.

### 2. Pose estimation (35%)

The final design uses 8-12 human-verified structural pallet keypoints, topology
refinement, and a calibrated floor homography. It reports metric `(x, y)`,
directed yaw `theta`, visible face identity, covariance, and calibration
reprojection error. Translation and rotation errors, sensitivity to height and
camera tilt, short/long-range behavior, and the operating envelope for
`+/-2 cm / +/-3 degrees` will only be filled from a physically measured evaluation set.
Low observability or out-of-envelope inputs must return `MANUAL_REVIEW`.

### 3. SOP verification (25%)

All eight checks from the assignment are represented in the versioned JSON
contract. Each check has evidence, confidence, thresholds, pose/load weighting,
and a three-way verdict. Metric polygon analysis is implemented for visible
overhang, geometric centroid offset, visible box rotation, and reviewed-layer
size inversion. Without assignment-camera calibration these signals remain
unavailable at runtime; missing evidence never silently becomes a pass.

### 4. Deployment (10%)

The intended target is Jetson Orin Nano 15 W at at least 15 FPS. The detector
was actually measured on an Intel Core Ultra 5 125U CPU at 320 px: 36.23 ms
median and 38.60 ms p95 end-to-end per image, or 27.50 FPS over 262 images.
This is one component, not a Jetson or full-pipeline benchmark. Any TensorRT
claim remains pending an actual export and target-device measurement.
The carton segmenter was separately measured over all 1,000 OSCD test images:
78.25 ms median and 116.90 ms p95 end-to-end, or 12.35 FPS from total wall
time on the same CPU. This is also a component benchmark, not an arithmetic
claim about full-pipeline speed.
Track-ID keyed temporal confidence fusion is implemented and tested to reduce
flicker; the actual ByteTrack video adapter and tuning remain pending video
data. Fail-safe gates cover blur, occlusion, bad calibration, out-of-range pose,
and stale tracks.

## Recommended pipeline

```mermaid
flowchart LR
    A["Warehouse frame or video"] --> B["Frame quality and calibration gates"]
    B --> C["YOLO pallet / structure detector"]
    C --> D["YOLO pose: 8-12 structural keypoints"]
    D --> E["Topology + geometry refinement"]
    E --> F["Floor homography: metric pose + covariance"]
    C --> G["YOLO structural + individual-carton segmentation"]
    G -. "accuracy comparison" .-> H["Mask2Former"]
    G --> I["Visible pallet / load regions"]
    I --> J["EfficientAD visible-damage score"]
    I -. "anomaly comparison" .-> K["PatchCore"]
    L["SAM 2 offline annotation assist"] -. "human-verified masks/keypoints" .-> D
    L -. "human-verified masks" .-> G
    F --> M["ByteTrack + temporal fusion"]
    J --> M
    M --> N["Eight SOP checks + confidence policy"]
    N --> O["Versioned assessment JSON"]
    B --> P["MANUAL_REVIEW with reason"]
    F --> P
    N --> P
```

Why these roles:

- YOLO provides the deployable real-time baseline for boxes, masks, and
  structural keypoints.
- Mask2Former is an accuracy-oriented segmentation comparison, not the default
  edge model.
- SAM 2 accelerates offline annotation and video propagation; a human approves
  every training label.
- EfficientAD is the preferred lightweight visible-damage baseline; PatchCore
  is the rare-defect comparison.
- Geometry and calibration, rather than an opaque regressor, make metric pose
  traceable and expose uncertainty.

## Reproducibility

```powershell
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
# Put your Roboflow private API key in .env, then:
python tools/download_roboflow.py --config configs/data_sources.yaml
# Or, for ZIPs downloaded through the Roboflow UI:
python tools/import_local_archives.py
python tools/audit_dataset.py
python tools/prepare_yolo.py --source data/raw/pallet_detect_v1 --output data/processed/detect --task detect --report reports/detection_preparation.json
python tools/prepare_yolo.py --source data/raw/plh_c_1_v1 --output data/processed/segment --task segment --report reports/segmentation_preparation.json
python tools/prepare_scd_oscd.py --source data/raw/scd_oscd_official --output data/processed/carton_scd --report reports/carton_scd_preparation.json --archive data/raw/downloads/scd_oscd_official.zip --expected-archive-sha256 da1b8063a73a7879670724daabab3005136744284bdc159624ae191b579d5980
python tools/validate_yolo_dataset.py --input data/processed/detect --task detect --report reports/detection_validation.json
python tools/validate_yolo_dataset.py --input data/processed/segment --task segment --report reports/segmentation_validation.json
```

The commands below reproduce training and evaluation after the ignored raw data
is imported. Source archives remain excluded from Git. Compact selected weights
are committed under `weights/releases` and verified by SHA-256 reports.

```powershell
# Measured detector run:
python tools/train_yolo.py --task detect --data data/processed/detect/data.yaml --epochs 15 --image-size 320 --batch 16 --device cpu --workers 0 --cache false --freeze 10 --patience 5 --run-name detect-real-v1
python tools/evaluate_yolo.py --task detect --model runs/yolo/detect-real-v1/weights/best.pt --data data/processed/detect/data.yaml --split test --image-size 320 --output reports/detection_test_evaluation.json --overlays reports/failure_cases/detection
python tools/benchmark_yolo.py --model runs/yolo/detect-real-v1/weights/best.pt --images data/processed/detect/images/test --device cpu --image-size 320 --warmup 10 --repeat 1 --output reports/runtime_detection_cpu.json
python tools/generate_manual_review_example.py

# Measured segmenter run:
python tools/train_yolo.py --task segment --data data/processed/segment/data.yaml --epochs 20 --image-size 320 --batch 16 --device cpu --workers 0 --cache false --freeze 10 --patience 7 --run-name segment-real-v1
python tools/evaluate_yolo.py --task segment --model runs/yolo/segment-real-v1/weights/best.pt --data data/processed/segment/data.yaml --split test --image-size 320 --output reports/segmentation_test_evaluation.json

# Measured real-carton segmenter run:
python tools/train_yolo.py --task segment --data data/processed/carton_scd/data.yaml --model yolo11n-seg.pt --epochs 10 --image-size 320 --batch 16 --device cpu --workers 0 --cache false --freeze 10 --patience 4 --run-name carton-seg-scd-v1
python tools/evaluate_yolo.py --task segment --model weights/releases/carton-seg-scd-v1.pt --data data/processed/carton_scd/data.yaml --split test --image-size 320 --batch 16 --device cpu --output reports/carton_segmentation_test_evaluation.json
python tools/benchmark_yolo.py --model weights/releases/carton-seg-scd-v1.pt --images data/processed/carton_scd/images/test --device cpu --image-size 320 --warmup 10 --repeat 1 --output reports/runtime_carton_segmentation_cpu.json

# Run all three real checkpoints and emit per-pallet assessment JSON:
python tools/infer_frame.py --image path/to/warehouse-frame.jpg --output runs/demo/frame-001 --device cpu --image-size 320

# Nominal visible crops versus separately labelled damaged holdout:
python tools/train_anomaly.py --model efficientad --data data/processed/anomaly
python tools/train_anomaly.py --model patchcore --data data/processed/anomaly

# Calibration and distribution reports:
python tools/calibrate_camera.py --images data/calibration/chessboard --columns 9 --rows 6 --square-size-m 0.024
python tools/calibrate_floor_homography.py --points data/calibration/floor_points.csv --calibration-id pillar-camera-v1
python tools/evaluate_detection.py --ground-truth data/holdout/_annotations.coco.json --predictions runs/predictions.json
python tools/evaluate_pose.py --predictions data/holdout/pose_predictions.csv
```

## Results

The raw-data audit measured 2,208 detection images with 63,624 boxes and 1,052
segmentation images with 3,653 polygons. File/task integrity gates pass. The
supplier splits fail independence because 110 detection and 5 segmentation
perceptual-hash groups cross boundaries. The replacement split has zero
cross-split groups and holds out whole capture days for detection: 1,488 train,
458 validation, and 262 test images. Segmentation has 841/106/105 images.

The committed detector checkpoint is
[`weights/releases/detect-real-v1.pt`](weights/releases/detect-real-v1.pt)
(5,427,738 bytes; SHA-256
`36273b60fb817a0d869188714c90ca8156a3370bb6da1d3536d71f7359cb7662`).

| Split | Images / instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Validation (2025-08-13 group plus non-date groups) | 458 / 1,960 | 0.610 | 0.837 | 0.628 | 0.421 |
| Test (2025-08-06 group plus non-date groups) | 262 / 573 | 0.905 | 0.826 | 0.876 | 0.556 |

Test per-class results are `hole`: P 0.945, R 0.768, AP50 0.882,
mAP50-95 0.508; and `pallet`: P 0.865, R 0.885, AP50 0.871,
mAP50-95 0.604. The easier test-day distribution explains why test exceeds
validation; it is not evidence that the new warehouse domain is solved.

Localization is measured separately for 501 test predictions matched at IoU
0.50 and confidence 0.25. Matched IoU mean/p05/p50/p95 is
0.818/0.628/0.834/0.943. Center error mean/p50/p95/max is
5.63/4.38/13.61/60.87 px; normalized by ground-truth box diagonal it is
0.0217/0.0166/0.0515/0.1413. Raw distributions are committed in
`reports/detection_test_evaluation.json`.

On Windows 11, PyTorch 2.8 CPU, eight inference threads, and an Intel Core Ultra
5 125U, 262 single-image runs measured 36.36 ms mean, 36.23 ms median,
38.60 ms p95, and 51.07 ms max end-to-end (27.50 FPS from total wall time).
Decode and the rest of the perception/SOP pipeline are excluded.

### Structural-part segmentation baseline

The committed YOLO11n-seg checkpoint is
[`weights/releases/segment-real-v1.pt`](weights/releases/segment-real-v1.pt)
(5,961,764 bytes; SHA-256
`79aaaa6c147ef5662f624e5db607f25fce0dea209b573f75c8c67554dccdcd0c`).
It trained for 20 epochs/32.7 minutes on 841 real images and 2,955 polygons.

| Split | Box P/R | Box mAP50 / mAP50-95 | Mask P/R | Mask mAP50 / mAP50-95 |
|---|---:|---:|---:|---:|
| Validation, 106 images / 363 instances | 0.795 / 0.428 | 0.554 / 0.343 | 0.631 / 0.317 | 0.384 / 0.158 |
| Test, 105 images / 335 instances | 0.563 / 0.444 | 0.442 / 0.254 | 0.471 / 0.318 | 0.314 / 0.132 |

Test mask mAP50-95 ranges from 0.506 for `pallet_pocket` to 0.000 for
`pallet`; `pallet_front` reaches 0.269, while all remaining classes are at or
below 0.111. The source has no `load` class and overlapping part semantics, so
this checkpoint is evidence for a structural-part baseline only. It cannot
support load overhang, centroid, or full-pallet boundary claims.

Standalone segmentation latency over 105 test images is 59.14 ms mean,
59.23 ms median, 61.67 ms p95, and 62.94 ms max (16.91 FPS). Arithmetic
composition with the detector is about 95.5 ms mean, or 10.5 FPS; that is not a
measured end-to-end pipeline and misses the 15 FPS target. Deployment therefore
runs detection intermittently, tracks between detections, and schedules masks
only for stable pallet tracks.

### Individual-carton segmentation

The official OSCD archive contributes 8,401 real images. After excluding 133
author-training images that visually duplicate the immutable author test set,
the grouped split is 6,396 train / 872 validation / 1,000 test images. The
effective instance counts are 128,158 / 18,218 / 20,186; five duplicate
segment envelopes were explicitly audited and removed by the loader. The
committed checkpoint is
[`weights/releases/carton-seg-scd-v1.pt`](weights/releases/carton-seg-scd-v1.pt)
(5,951,652 bytes; SHA-256
`5bffd9bdcd58f588bc6ba5e4165a916f4a366089043cb8c546c14ac3c2dd9fa7`).
It trained for 10 epochs/141.0 minutes on the declared CPU.

| Split | Box P/R | Box mAP50 / mAP75 / mAP50-95 | Mask P/R | Mask mAP50 / mAP75 / mAP50-95 |
|---|---:|---:|---:|---:|
| Validation, 872 images / 18,218 masks | 0.912 / 0.815 | 0.904 / not exported / 0.722 | 0.913 / 0.810 | 0.895 / not exported / 0.646 |
| Immutable test, 1,000 images / 20,186 masks | 0.899 / 0.823 | 0.898 / 0.793 / 0.719 | 0.900 / 0.816 | 0.889 / 0.745 / 0.650 |

Mask AP across IoU 0.50-0.95 is
0.889/0.871/0.855/0.829/0.795/0.745/0.660/0.517/0.283/0.055. This
distribution shows a strong visible-carton baseline at ordinary overlap but a
sharp strict-boundary ceiling. The full training curve, test distribution,
duplicate audit, and actual CPU latency are in `reports/` and summarized in
[`docs/carton_segmentation_model_card.md`](docs/carton_segmentation_model_card.md).

On a deterministic held-out test example with 20 annotated cartons, the model
returned 22 masks at confidence 0.25. The qualitative overlay below is a model
prediction, not ground truth; exact predictions and confidences are retained in
[`examples/carton_segmentation_demo/frame_summary.json`](examples/carton_segmentation_demo/frame_summary.json).

![Held-out carton mask predictions](examples/carton_segmentation_demo/overlay.jpg)

### Runnable three-model adapter and SOP boundary

[`tools/infer_frame.py`](tools/infer_frame.py) runs all three committed real
checkpoints, preserves boxes and mask polygons with confidences/model hashes,
renders an overlay, records measured component latency, and writes one
schema-valid assessment for each detected pallet. A held-out example detected
one pallet at confidence 0.87; the structural model produced one instance and
the carton model produced none, exposing the OSCD-to-warehouse domain gap. All
eight checks correctly returned `MANUAL_REVIEW` because no assignment-camera
calibration or metric pose exists. See
[`examples/real_frame_pallet/frame_summary.json`](examples/real_frame_pallet/frame_summary.json)
and the [assessment JSON](examples/real_frame_pallet/assessment_001.json).

![Three-model held-out adapter output](examples/real_frame_pallet/overlay.jpg)

## Failure analysis - three worst cases

At fixed confidence 0.25, the worst real test image contains three labelled
instances but 25 unmatched predictions. Stretch-wrap highlights, carton faces,
labels, and load gaps resemble the dominant hole class. The fix is hard-negative
mining, class-specific calibration, and accepting holes only within supported
pallet geometry.

![Worst case 1](reports/failure_cases/detection/worst_1_pallet_detect_v1_6618b3fc7fb8aa661b22.jpg)

The second is a negative close-up of reflective stretch wrap. Horizontal glare
triggers five hole predictions, showing missing wrap-only negative coverage.

![Worst case 2](reports/failure_cases/detection/worst_2_pallet_detect_v1_0eaa0971b5c5b48902d5.jpg)

The third is numerically ranked as a false-positive case, but visual review
shows a blue plastic pallet with no ground-truth instance. This is a dataset
annotation/scope omission, so the reported false-positive count overstates model
error. The holdout negatives require adjudication before acceptance testing.

![Worst case 3](reports/failure_cases/detection/worst_3_pallet_detect_v1_539bdab44bd56d405b76.jpg)

## What I couldn't finish and why

- The supplied datasets cover detection and instance segmentation only.
  Detection boxes/polygons cannot be relabelled as pose keypoints, metric pose,
  load dimensions, wrap state, or damage truth.
- Metric pose accuracy requires ruler/tape measurements and camera calibration;
  it cannot be inferred from internet images alone.
- EfficientAD and PatchCore cannot be trained honestly from these archives:
  neither contains reviewed `good`/`bad` damage labels. Synthetic defects are
  deliberately not substituted.
- Jetson latency requires the stated Jetson hardware; other machines' numbers
  will not be presented as ours.
- The required five-minute screen recording depends on the real-data pipeline
  and is therefore pending; `docs/demo_script.md` defines the evidence to record.

## AI tool usage

Codex is being used for repository implementation, dataset auditing, tests, and
documentation. The critical error caught during review was that the earlier
prototype trained on procedurally generated labels and described the resulting
checkpoint too broadly. This repository corrects that by accepting only
traceable real data for headline training and evaluation.
