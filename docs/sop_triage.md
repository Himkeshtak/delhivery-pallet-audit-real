# SOP-PAL-03 single-view triage

The pillar camera sees one side. Therefore no rule is fully verifiable for the
entire 3D pallet under all occlusions. "Partial" below means visible evidence
can confirm some failures or support a pass only under declared coverage and
calibration assumptions.

| # | Rule | Triage | Implemented evidence and limits |
|---|---|---|---|
| 1 | overhang <=3 cm | partial | implemented in `load_analysis.py` from metric pallet/load polygons; hidden/rear overhang and mask uncertainty remain unresolved |
| 2 | height <=1.8 m | partial | calibrated vertical geometry when floor contact and load top are visible; depth/occlusion and camera extrinsic error can force review; no current training truth |
| 3 | columns aligned; rotation <=15 deg | partial | implemented for anisotropic visible metric box masks relative to directed pallet axes; hidden/square/degenerate faces are unknown |
| 4 | larger boxes below smaller | partial | visible metric mask areas and reviewed layer indices produce a risk score; policy abstains until that score is calibrated |
| 5 | stretch-wrapped | partial | intended visible-surface wrap probability and continuity; no reviewed wrap labels exist and backside wrap cannot be confirmed |
| 6 | no damaged/crushed box | partial | intended EfficientAD plus PatchCore comparison on visible box crops; no reviewed good/bad training split exists and hidden/internal damage cannot be seen |
| 7 | centroid within 10 cm | partial | implemented from visible metric pallet/load polygon centroids with propagated boundary uncertainty; mass centroid is not observable |
| 8 | pallet undamaged | partial | intended visible pallet-region anomaly/defect evidence; no reviewed damage truth exists and rear/underside/internal damage is unknown |

## Verdict policy

- A high-confidence visible failure is `FAIL` even if unrelated pose evidence is
  unavailable.
- A metric check passes only when its complete 95% uncertainty interval is
  inside the SOP limit; it fails only when the complete interval is outside.
- An interval crossing the limit is `MANUAL_REVIEW`.
- Appearance probabilities require calibrated pass/fail thresholds, an
  abstention band, and minimum visible coverage. An uncalibrated risk score is
  always `MANUAL_REVIEW`.
- By default, any unresolved check blocks an overall `PASS`. This is
  intentionally conservative because the assignment says confident wrong
  answers are worst.

## Current automation boundary

The repository now has real pallet detection, real structural-part masks, real
individual-carton masks, and tested geometry/policy code. It does not have a
physically calibrated assignment-camera dataset, load-envelope truth, reviewed
wrap/damage labels, or a trained keypoint checkpoint. Consequently the runnable
adapter may show perception outputs but must return `MANUAL_REVIEW` for the
unsupported compliance claims.

