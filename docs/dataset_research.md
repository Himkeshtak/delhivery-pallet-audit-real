# Dataset research and selection decision

Research date: 2026-08-24. This review distinguishes a dataset that can improve
one perception component from a dataset that can substantiate the assignment's
metric and SOP claims. No public dataset found contains the fixed pillar-camera
view, surveyed floor pose, directed pallet face, individual carton and load
masks, stretch-wrap state, visible damage, and all eight SOP outcomes together.
The final assignment-domain evaluation set therefore still has to be captured
and physically measured; public data is used only for component pretraining.

## Selected source

The selected new source is the authors' original **OSCD** subset of the Stacked
Carton Dataset (SCD), not a third-party repackaging. It provides real stacked
carton images and per-instance polygons in COCO format. The official page
publishes 8,401 images, a 7,401/1,000 train/test split, and 168,748 instances.
The downloaded canonical JSON contains the same image counts and 168,758
instances; this ten-instance discrepancy is preserved and disclosed rather
than silently edited.

- Official project and download: <https://github.com/yancie-yjr/scd.github.io>
- Paper: <https://arxiv.org/abs/2102.12808>
- License: CC BY-NC-SA 4.0; academic/non-commercial use only unless the authors
  grant additional permission.
- Verified archive: 1,560,001,798 bytes; SHA-256
  `da1b8063a73a7879670724daabab3005136744284bdc159624ae191b579d5980`.
- Role: fine-tune the individual-carton instance segmenter used for visible
  box count, per-box orientation, size ordering cues, damage-crop extraction,
  and the visible load-envelope union.
- It does **not** provide pallet masks, metric pose, wrap labels, damage labels,
  floor measurements, or assignment-camera data. It cannot by itself prove any
  metric SOP threshold.

## Search results by required signal

| Candidate | Public evidence | Useful role | Decision |
|---|---|---|---|
| SCD OSCD/LSCD | 16,136 real images and about 250k carton instance masks; official COCO downloads | individual cartons | **Select OSCD**: strongest traceable mask source and manageable one-class taxonomy |
| Boxes by Carton (Roboflow) | 6,865 raw instance-segmentation images, one poorly named class | individual cartons | Do not headline: provenance and split duplication need independent audit; SCD is canonical |
| Secret Awesome Sauce (Roboflow) | 1,758 real loaded-pallet images, class `Objects-on-top-pallet` | whole visible load envelope | Candidate supplement only; it is not individual-box truth and no authenticated export was available locally |
| Pallet_with_products_v3 (Roboflow) | 873 instance-segmentation images of loaded pallets | loaded-pallet domain | Candidate supplement only; one pallet/load class cannot support box alignment or size inversion |
| LOCO | 37,988 real logistics images; 5,593 manually annotated, including pallets | warehouse pallet detection/domain negatives | Useful detector pretraining, but no carton/load masks or metric pose |
| PalLoc6D | 50,000 photorealistic synthetic Euro-pallet views with 6D pose | pose pretraining/simulation | Not used for headline results: synthetic and published errors do not establish the assignment's +/-2 cm bar |
| Staer Warehouses v0.3 | synthetic stereo RGB, depth, instance masks and scene poses | multi-task pretraining | Not used: synthetic, gated, 124 GB, and CC BY-NC-SA; does not supply real assignment-domain evidence |
| Damage Detection for Packages (Roboflow) | 176 raw images; classes include crushed, dented, torn, wet and undamaged | damage bootstrap | Reject for final training: small, inconsistent class `0`, and reported recall is too low for a safety gate |
| Corrugated Cardboard Boxes Dataset | real deliberately damaged courier parcels | damage research lead | Not incorporated: repository publishes no clear dataset license or machine-readable labels |
| Roboflow pallet keypoints | 86 images with poorly documented pallet/point labels | keypoint bootstrap | Reject for metric claims: too small and has no surveyed floor pose or calibration |

Primary references:

- LOCO: <https://github.com/tum-fml/loco>
- PalLoc6D: <https://tore.tuhh.de/entities/product/4b51fbbb-6b30-4468-80c3-fd6d25419310>
- Staer Warehouses: <https://huggingface.co/datasets/staerrobotics/warehouses>
- Corrugated damage images: <https://github.com/chanllon/corrugated-cardboard-boxes-dataset>
- Roboflow damage source: <https://universe.roboflow.com/smart-damage-detection-for-logistics-packages-using-computer-vision/damage-detection-for-packages-dspra>

## Final acquisition policy

1. Preserve the two assignment-supplied real pallet datasets and both old
   checkpoints unchanged.
2. Train a separate one-class carton segmenter on audited OSCD. Do not merge its
   class IDs into the structural-pallet checkpoint.
3. Keep the authors' 1,000-image test split untouched. Remove any training image
   sharing an exact/perceptual fingerprint with test, then form validation only
   from the remaining author-training groups.
4. Report box and mask AP distributions on OSCD as component-transfer evidence,
   not as fixed-camera pallet-load performance.
5. The final acceptance set must use the actual camera and include calibration,
   tape/laser surveyed `(x,y,theta)`, named visible pallet keypoints, individual
   box masks, a load-envelope mask, wrap/damage tags, and per-SOP adjudication.
6. Split final data by physical pallet setup and capture session/day, never by
   adjacent video frame. The acceptance test remains inaccessible during model
   selection.

