# Dataset card

Status: the two user-supplied, pinned COCO archives were imported, hashed, and
audited on 2026-08-24. Raw images remain gitignored. The supplier splits are
not approved for headline evaluation because perceptual duplicates cross split
boundaries; a leakage-controlled grouped split is the next preparation gate.

## Sources and selection rationale

| Source ID | Pinned version URL | Intended role | Verified publicly |
|---|---|---|---|
| `pallet_detect_v1` | https://universe.roboflow.com/rj-xvfw4/pallet-ff5lh-fp7tw/dataset/1 | pallet/fork-hole detection | 2,208 images; 63,624 boxes; COCO detection |
| `plh_c_1_v1` | https://universe.roboflow.com/rj-xvfw4/plh-c-1-veibf/dataset/1 | pallet-part instance masks | 1,052 images; 3,653 polygons; COCO instance segmentation |

These sources were selected by the assignment author/user and are the sample
projects named in the assignment. They contain real pallet imagery and existing
geometry labels, reducing acquisition and first-pass annotation cost. That
choice costs domain control: camera placement, pallet/load mix, distances, and
lighting do not match the specified pillar camera; label taxonomies are noisy;
the exported data contains augmentations and split leakage; and neither source
contains metric pose, directed face, load-SOP, or damage ground truth.

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

## Counts and supplier split audit

| Source | Train | Validation | Test | Total images | Total annotations |
|---|---:|---:|---:|---:|---:|
| Detection | 1,932 | 184 | 92 | 2,208 | 63,624 boxes |
| Segmentation | 960 | 56 | 36 | 1,052 | 3,653 polygons |

Detection instances: `hole` 40,555 and `pallet` 23,069. The declared
`pallet-hole` category has zero instances. Segmentation instances: `front` 156,
`hole` 336, `hole_left` 159, `hole_right` 158, `pallet` 167,
`pallet_front` 718, `pallet_pocket` 1,478, and `wood` 481. The declared
`warehouse-Pallets-and-palle-U3mp` category has zero instances.

All image references resolve, every box passes boundary/area validation, all
3,653 segmentation annotations contain polygons, and no byte-identical image
crosses a split. However, 110 detection perceptual-hash groups and 5
segmentation groups cross supplier split boundaries. Several filenames also
identify augmented variants of the same capture. Consequently, supplier test
metrics would be optimistic and will not be reported as the final holdout.

The processed split groups source identity, augmentation family, capture date,
and identical perceptual hashes before deterministic assignment toward an
80/10/10 target. The group—not an image—is the unit of assignment. Exact ratios
are intentionally secondary to preventing capture leakage.

| Prepared task | Train | Validation | Test | Groups | Largest group |
|---|---:|---:|---:|---:|---:|
| Detection | 1,488 | 458 | 262 | 153 | 858 images |
| Segmentation | 841 | 106 | 105 | 174 | 466 images |

Detection's 2025-07-29 capture day is train-only; 2025-08-13 is
validation-only; and 2025-08-06 is test-only. This makes the holdouts differ by
capture day, but the large day groups force a 67.4/20.7/11.9 split instead of
80/10/10. The detection train/validation/test instance totals are
61,091/1,960/573. Segmentation reaches 79.9/10.1/10.0 images and
2,955/363/335 instances. No prepared group crosses a split.

The converter removes two declared but empty categories and produces observed
class maps only: detection has `hole` and `pallet`; segmentation has `front`,
`hole`, `hole_left`, `hole_right`, `pallet`, `pallet_front`, `pallet_pocket`,
and `wood`. It preserves negative images (426 detection, 4 segmentation) rather
than silently dropping them. Split manifests preserve the raw source ID,
original split, origin family, and group ID. A separately captured
assignment-domain set from the stated fixed camera is still required for metric
pose and SOP claims.

## Sourcing and annotation cost

The archives were available under CC BY 4.0, so the direct acquisition fee was
zero. Existing boxes and polygons avoided first-pass labelling, but their
taxonomy normalization, leakage removal, and quality review are real engineering
costs. No timed labelling pilot has been performed, so a rupee/hour estimate
would be fabricated. Before commissioning pose/damage labels, the protocol
requires a timed 50-image pilot reporting median and p95 minutes for boxes,
masks, keypoint skeletons, damage review, physical pose measurement, and second
review.

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

## License, attribution, and immutable archives

Both exact archives ship `README.dataset.txt` declaring `License: CC BY 4.0`
and attribution to a Roboflow user. Project links are in the source table.
Archive SHA-256 values are:

- Detection: `ea001f9d73ce0c8a5ec2907dacf766dd3f8f3f86b8cf09855c419aea5b21e967`
- Segmentation: `191964d0fc13bfe7b8dcf65201149e3cc75a6932e38dafa96142faa3798c2b7e`

The images/archives are not redistributed through Git. Per-file hashes and
source metadata live in each ignored raw directory's `manifest.json`; the
committed aggregate audit is `reports/dataset_audit.json`.
