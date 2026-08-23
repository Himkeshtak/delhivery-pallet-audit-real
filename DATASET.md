# Dataset card

Status: metadata audit in progress; image-level statistics are intentionally
blank until the authenticated exports are downloaded and hashed.

## Sources and selection rationale

| Source ID | Pinned version URL | Intended role | Verified publicly |
|---|---|---|---|
| `pallet_detect_v1` | https://universe.roboflow.com/rj-xvfw4/pallet-ff5lh-fp7tw/dataset/1 | pallet/structure detection | workspace advertises a 4.4k-image object-detection project; exact redirect and archive metadata pending |
| `plh_c_1_v1` | https://universe.roboflow.com/rj-xvfw4/plh-c-1-veibf/dataset/1 | instance-mask candidate | workspace advertises a 412-image instance-segmentation project named PLH-C.1; exact archive metadata pending |

These sources were selected by the assignment author/user because they contain
real pallet imagery. We still audit duplicates, label meaning, preprocessing,
license files, and distribution fit before combining them.

## Format and provenance policy

- Download COCO JSON as the immutable canonical export.
- Record URL, version, retrieval time, archive SHA-256, extracted-file hashes,
  license, generator preprocessing, class map, and split membership.
- Never commit images or API keys.
- Preserve source IDs through conversion and reports.
- Group near-duplicates and sequences before splitting; a group may appear in
  only one of train, validation, or test.
- The assignment-domain test capture remains separate from internet sources.

## Annotation layers

| Layer | Required truth | Tooling | Acceptance |
|---|---|---|---|
| Detection | boxes + normalized taxonomy | CVAT/Roboflow review | two-pass class and box QA |
| Pose | 8–12 named structural keypoints + visibility | CVAT keypoint skeleton | topology-valid, second-person review |
| Segmentation | pallet/load instance polygons | SAM 2 propagation + CVAT correction | human-approved boundaries |
| Damage | nominal/damaged visible-region masks and defect type | CVAT polygons/tags | reviewer agreement and ambiguous flag |
| Metric pose | camera calibration + surveyed floor pose | calibration board + tape/laser reference | repeated measurements and reprojection QA |

SAM 2 is an annotation accelerator only. Propagated masks are never accepted as
ground truth without human correction and approval.

## Counts and split

Pending authenticated download and audit. The final table will include images,
instances, class counts, unique groups, duplicates removed, train/validation/
test counts, and assignment-domain holdout counts.

## Annotation cost

Pending a timed pilot on 50 representative images. The estimate will report
minutes per box, mask, keypoint skeleton, damage review, and physical pose label,
plus reviewer time and total cost assumptions.

## Known and anticipated biases

- Public internet/curated imagery may not match the fixed assignment camera,
  warehouse lighting, floor, pallet material, load, blur, or occlusion.
- Sparse damage labels bias an anomaly model toward visible/textural defects and
  cannot validate hidden structural damage.
- Single-view metric pose degrades with calibration error, pallet height
  variation, floor non-planarity, and weak front/back evidence.
- Random image splits can leak near-duplicate video frames; grouped splitting is
  mandatory.
- Small segmentation sets may not support a reliable Mask2Former comparison.

## License and attribution

The final card will quote license metadata shipped in each archive and link to
the project page. Public search currently reports CC BY 4.0 for the related
PLH-C project; this must be confirmed for the exact two version archives before
use or redistribution.

