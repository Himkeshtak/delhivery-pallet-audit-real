# One-page pallet labelling guideline

## Unit and image rules

Annotate the original real frame before augmentation. Keep the source/frame ID.
Reject corrupt frames; tag, but do not discard, blur, glare, truncation, and
occlusion. One instance is one physical pallet, including a loaded pallet.

## Detection boxes

Draw the tight visible pallet box, excluding the load. If the pallet is more than
60% occluded or no footprint edge/fork-pocket structure is identifiable, mark it
`ignore_pose=true`; it may remain useful for detection. Use the normalized class
map after the dataset audit—never merge `fork`, `hole`, or `front` from names
alone without viewing a sample.

## Instance masks

Create separate `pallet` and `load` instances. Follow the visible boundary;
do not hallucinate behind occluders. Holes remain holes in the pallet mask.
SAM 2 may propagate a reviewed seed mask across a video segment, but a human
must inspect every propagated frame, correct drift, and record approval.

## Structural keypoints

Use this ordered footprint skeleton:

1. `front_left_floor`
2. `front_right_floor`
3. `rear_left_floor`
4. `rear_right_floor`
5. `front_left_top`
6. `front_right_top`
7. `left_fork_pocket_center`
8. `right_fork_pocket_center`

Optional type-specific points 9–12 may describe rear/top structure, but the first
eight never change order. Visibility is `2` visible, `1` occluded but physically
localizable, `0` not labelable. Do not guess a floor corner from the bounding-box
corner. Front/rear identity follows the operational fork-entry face definition.

## Damage and load attributes

Polygon visible crushed/torn box regions, broken boards, and split stringers.
Tag ambiguous texture, shadow, wrap reflection, tape, and printing separately.
For anomaly training, `good` means a reviewer found no visible defect in the
crop; it does not mean the hidden pallet is globally undamaged.

## Quality control

Every pose and damage label receives a second review. The pilot reports median
and p95 reviewer disagreement in pixels, topology violations, mask IoU, time per
layer, and disagreement rate. Conflicts are adjudicated, not averaged. Split by
capture session/video/near-duplicate group before any model sees the data.

