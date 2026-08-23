# SOP-PAL-03 single-view triage

The pillar camera sees one side. Therefore no rule is fully verifiable for the
entire 3D pallet under all occlusions. “Partial” below means the visible evidence
can confirm some failures or support a pass only under declared coverage and
calibration assumptions.

| # | Rule | Triage | Implemented evidence and limits |
|---|---|---|---|
| 1 | overhang ≤3 cm | partial | pallet/load masks projected to the floor; hidden/rear overhang and mask uncertainty remain unresolved |
| 2 | height ≤1.8 m | partial | calibrated vertical geometry when floor contact and load top are visible; depth/occlusion and camera extrinsic error can force review |
| 3 | columns aligned; rotation ≤15° | partial | visible box-face axes relative to directed pallet axes; hidden boxes and perspective-degenerate faces are unknown |
| 4 | larger boxes below smaller | partial | visible instance sizes and vertical order; depth can make image size misleading, so metric/corrected estimates and coverage are required |
| 5 | stretch-wrapped | partial | visible-surface wrap probability and continuity; backside wrap cannot be confirmed |
| 6 | no damaged/crushed box | partial | EfficientAD plus PatchCore comparison on visible box crops; hidden/internal damage cannot be seen |
| 7 | centroid within 10 cm | partial | segmented visible load footprint against pallet footprint with propagated uncertainty; mass centroid is not observable, only geometric load centroid |
| 8 | pallet undamaged | partial | visible pallet-region anomaly/defect evidence; rear, underside, and internal damage remain unknown |

## Verdict policy

- A high-confidence visible failure is `FAIL` even if unrelated pose evidence is
  unavailable.
- A metric check passes only when its complete 95% uncertainty interval is
  inside the SOP limit; it fails only when the complete interval is outside.
- An interval crossing the limit is `MANUAL_REVIEW`.
- Appearance probabilities have calibrated pass/fail thresholds with an
  abstention band and minimum visible coverage.
- By default, any unresolved check blocks an overall `PASS`. This is intentionally
  conservative because the assignment says confident wrong answers are worst.

