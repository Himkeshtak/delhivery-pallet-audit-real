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
