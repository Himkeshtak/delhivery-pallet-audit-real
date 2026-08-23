# Physical pose and robustness evaluation protocol

This protocol creates the ground truth the public image datasets do not contain.
Temporary calibration boards and surveyed floor points are permitted during
camera setup; no fiducial is attached to or maintained on a pallet at runtime.

## Coordinate frame and independent truth

- Fix the origin at a named slot-grid corner; +x follows the row and +y follows
  the column direction, metres at floor level.
- Measure pallet centre from two surveyed floor baselines using a tape/laser
  with documented resolution. Measure directed front-face yaw with a digital
  angle gauge or a rigid aligned edge jig.
- Repeat 10% of placements without looking at the first measurement. Report
  repeatability distributions; ground-truth uncertainty must be smaller than
  the ±2 cm / ±3° target.
- Freeze camera focus, resolution, digital stabilization, mount, and calibration
  ID. Any physical movement creates a new calibration version.

## Capture matrix

Use at least three sessions/days and group all frames from one placement into a
single split unit. Capture a balanced matrix across:

- range: 1.5–2.5 m, 2.5–3.5 m, 3.5–4.5 m, and beyond 4.5 m;
- absolute view yaw: 0–15°, 15–30°, 30–45°, 45–60°, and beyond 60°;
- loaded/unloaded, pallet type, load height, floor slot, illumination, blur,
  wrap glare, truncation, and occlusion;
- deliberate near-threshold placements around 2 cm/3° and SOP thresholds.

The final assignment-domain test set is never used for model choice, threshold
tuning, anomaly nominal training, or calibration fitting.

## Height and tilt sensitivity

The production floor homography is fitted from surveyed image↔floor
correspondences, so nominal camera height and tilt are not direct runtime inputs.
Still, characterize mount/calibration sensitivity with controlled recaptures:

- camera height offsets: −5, −2, 0, +2, +5 cm;
- downward tilt offsets: −2, −1, 0, +1, +2°;
- at least 20 placements in both a short-range and long-range bin;
- run once with the original calibration (movement sensitivity) and once after
  recalibration (recoverability).

Report translation and rotation raw errors, p50/p95/max, availability, and the
joint target rate for every offset/range cell. Also monitor fixed surveyed floor
anchors each session; drift beyond policy invalidates the calibration.

## Usable envelope

A range×view-yaw cell is inside the empirical envelope only when it contains at
least 20 independent placements, availability is at least 95%, translation p95
is ≤0.02 m, and rotation p95 is ≤3°. Publish failed cells and reasons instead of
interpolating a larger envelope. The envelope is specific to the installed
camera, calibration ID, resolution, pallet types, and model hashes.

## Failure analysis

Rank cases by `translation_error/0.02 + rotation_error/3`. For the worst three,
store the source image, overlay, keypoint/mask confidences, covariance, calibration
residual, track history, predicted/true pose, error, and a root-cause category.

