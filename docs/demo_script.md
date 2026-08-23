# Five-minute screen-recording script

Record only after the real-data artifacts exist. Do not substitute synthetic
images or paper benchmark numbers.

1. **0:00–0:35 — problem and contract:** show one versioned JSON assessment,
   metric pose uncertainty, eight checks, and explicit manual-review reasons.
2. **0:35–1:15 — data:** show pinned sources, immutable manifests/hashes,
   counts/splits, annotation skeleton/masks, duplicate audit, and domain holdout.
3. **1:15–2:05 — models:** show YOLO detection/pose/segmentation outputs, explain
   SAM 2 human review, then EfficientAD/PatchCore visible-damage scope.
4. **2:05–2:50 — geometry:** show camera/floor calibration, reprojection errors,
   topology refinement, directed front face, covariance, and an abstention.
5. **2:50–3:40 — results:** show distributions, usable envelope, separate
   detection/localization results, and three worst cases with root causes.
6. **3:40–4:25 — video/deployment:** show ByteTrack temporal stability, measured
   current-hardware latency, export comparison, and why Jetson numbers are not
   claimed without the device.
7. **4:25–5:00 — tradeoffs:** show what remains unverifiable from one view, what
   was not finished, licensing, AI-tool disclosure, and the caught synthetic-data
   error.

