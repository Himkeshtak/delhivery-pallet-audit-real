# Contract examples

`manual_review_assessment.json` is a deterministic schema-valid example of the
required output when model detections exist but the physical calibration,
structural keypoints, and SOP-specific evidence do not. It deliberately returns
`MANUAL_REVIEW` for the overall verdict and all unresolved checks.

This file is a contract/failure-path demonstration, not a measured warehouse
result. Regenerate it with:

```powershell
python tools/generate_manual_review_example.py
```

`carton_segmentation_demo/` is a real model prediction on an immutable OSCD
test image: the overlay renders predicted masks and the summary retains their
polygons/confidences. `real_frame_pallet/` runs all three release checkpoints
on a held-out pallet frame and includes the resulting schema-valid per-pallet
assessment. Its all-manual verdict is expected because physical calibration and
metric pose evidence are absent.
