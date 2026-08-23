# Assignment traceability

This checklist prevents a polished demo from hiding an unevaluated requirement.
`PENDING` means no compliant evidence exists yet; it never means pass.

| Rubric requirement | Planned evidence | Status |
|---|---|---|
| Standard dataset format and annotation tooling | canonical COCO export, deterministic converters, CVAT/SAM 2 workflow | PENDING download |
| Source, rationale, cost, counts, split, guidelines, biases | `DATASET.md`, manifests, audit report | IN PROGRESS |
| Trained model/weights | release asset + SHA-256/model card | PENDING real labels |
| Detection vs localization on differing holdout | AP/PR and IoU/center-error distributions | TOOLING PRESENT; PENDING real evaluation |
| Training decisions and accuracy ceiling | experiment card and error taxonomy | PENDING experiment |
| Physical calibration/reprojection | calibration JSON and reprojection distribution | TOOLING PRESENT; PENDING physical capture |
| Metric `(x,y)`, directed yaw/face | pose evaluator and versioned assessment JSON | PENDING pose labels |
| Translation/rotation distributions | JSON/CSV and plots on surveyed holdout | TOOLING PRESENT; PENDING physical capture |
| Height/tilt/range/yaw sensitivity | controlled-bin report | PROTOCOL PRESENT; PENDING physical capture |
| `±2 cm / ±3°` usable envelope | empirical coverage table | DEFINITION PRESENT; PENDING physical capture |
| Reliable failure output | observability gates and `MANUAL_REVIEW` tests | IMPLEMENTED + TESTED |
| Eight SOP checks | check registry and evidence contracts | IMPLEMENTED + TESTED |
| Per-check confidence/verdict | schema and policy calibration | IMPLEMENTED; PENDING calibration data |
| Pose/load weighting | auditable confidence policy | IMPLEMENTED + TESTED |
| Actual hardware/runtime/memory | benchmark JSON with hardware fingerprint | TOOLING PRESENT; PENDING weights |
| Expected Orin changes | deployment note, no borrowed benchmark | DOCUMENTED |
| TensorRT export cost | measured only after export | EXPORT TOOLING PRESENT; PENDING weights/device |
| Multi-frame behavior | ByteTrack + temporal fusion evaluation | FUSION IMPLEMENTED + TESTED; PENDING video evaluation |
| Three worst cases/root causes | report with held-out images | PENDING evaluation |
| Could not finish/why | explicit README section | PRESENT |
| AI tools and one caught error | explicit README section | PRESENT |
