# Assignment traceability

This checklist prevents a polished demo from hiding an unevaluated requirement.
`PENDING` means no compliant evidence exists yet; it never means pass.

| Rubric requirement | Planned evidence | Status |
|---|---|---|
| Private GitHub repository | authenticated push plus unauthenticated visibility check | COMPLETE; REMOTE PUSHES AND PUBLIC API RETURNS 404 |
| Standard dataset format and annotation tooling | canonical COCO export, deterministic converters, CVAT/SAM 2 workflow | THREE COCO SOURCES + VALIDATED YOLO DETECT/PALLET-SEG/CARTON-SEG COMPLETE |
| Source, rationale, cost, counts, split, guidelines, biases | `DATASET.md`, manifests, audit report | COMPLETE FOR SUPPLIED LABEL LAYERS |
| Trained model/weights | committed selected checkpoints + SHA-256/model cards | PALLET DETECT + STRUCTURAL SEGMENT + CARTON SEGMENT COMPLETE; POSE/DAMAGE LABELS ABSENT |
| Detection vs localization on differing holdout | AP/PR and IoU/center-error distributions | DETECTOR COMPLETE ON CAPTURE-GROUPED TEST |
| Training decisions and accuracy ceiling | experiment card and error taxonomy | DETECTOR + STRUCTURAL SEGMENT + CARTON SEGMENT COMPLETE |
| Physical calibration/reprojection | calibration JSON and reprojection distribution | TOOLING PRESENT; PENDING physical capture |
| Metric `(x,y)`, directed yaw/face | pose evaluator and versioned assessment JSON | IMPLEMENTED; PENDING PHYSICAL CALIBRATION/POSE LABELS |
| Translation/rotation distributions | JSON/CSV and plots on surveyed holdout | TOOLING PRESENT; PENDING physical capture |
| Height/tilt/range/yaw sensitivity | controlled-bin report | PROTOCOL PRESENT; PENDING physical capture |
| `±2 cm / ±3°` usable envelope | empirical coverage table | DEFINITION PRESENT; PENDING physical capture |
| Reliable failure output | observability gates and `MANUAL_REVIEW` tests | IMPLEMENTED + TESTED |
| Eight SOP checks | check registry and evidence contracts | IMPLEMENTED + TESTED |
| Per-check confidence/verdict | schema and policy calibration | IMPLEMENTED; PENDING calibration data |
| Pose/load weighting | auditable confidence policy | IMPLEMENTED + TESTED |
| Actual hardware/runtime/memory | benchmark JSON with hardware fingerprint | THREE MODEL CPU LATENCIES COMPLETE; FULL PIPELINE PENDING |
| Expected Orin changes | deployment note, no borrowed benchmark | DOCUMENTED |
| TensorRT export cost | measured only after export | EXPORT TOOLING PRESENT; PENDING EXPORT/TARGET DEVICE |
| Multi-frame behavior | ByteTrack + temporal fusion evaluation | TRACK-ID FUSION IMPLEMENTED + TESTED; BYTETRACK ADAPTER/VIDEO EVALUATION PENDING |
| Three worst cases/root causes | report with held-out images | DETECTOR COMPLETE; POSE PENDING |
| Could not finish/why | explicit README section | PRESENT |
| AI tools and one caught error | explicit README section | PRESENT |
| Runnable real-checkpoint adapter | overlay, raw predictions, model hashes, assessment JSON | COMPLETE; HELD-OUT EXAMPLE COMMITTED |
| Required five-minute recording | user-recorded screen capture | SCRIPT PRESENT; RECORDING PENDING USER ACTION |
