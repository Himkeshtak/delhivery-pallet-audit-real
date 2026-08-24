# Model selection and experiment roles

The final pipeline is deliberately heterogeneous. A single end-to-end network
would be simpler to demo, but it would hide calibration error, face ambiguity,
unseen surfaces, and evidence provenance—the highest-risk parts of this task.

| Component | Role | Why selected | Cost / rejection rule |
|---|---|---|---|
| YOLO11 detector | pallet and visible structural-region boxes | small real-time baseline, one maintained interface across detect/segment/pose | Ultralytics has no formal YOLO11 paper; software is AGPL-3.0 or enterprise licensed, so commercial deployment needs a license decision |
| YOLO11 pose | 8–12 named pallet structural keypoints | deployable geometry localization; topology and calibration remain inspectable | cannot train from boxes; needs a new human-reviewed skeleton dataset |
| YOLO11 segmentation | structural pallet parts and individual visible cartons | selected/trained real-time baseline; the carton model reaches 0.650 test mask mAP50-95 | neither source contains the required calibrated pallet/load envelope pair |
| Mask2Former | offline segmentation accuracy comparator | masked attention is a strong universal segmentation reference | heavier; promote only if held-out mask gains justify measured latency/memory cost |
| SAM 2 | offline video annotation propagation | reduces mask annotation interactions and maintains temporal object masks | every propagated mask is reviewed; never used as unattended ground truth or runtime compliance evidence |
| EfficientAD-S | primary visible-damage experiment | teacher–student plus autoencoder design targets fast industrial anomaly localization | nominal-only training still needs real normal crops and a separately labelled damaged holdout |
| PatchCore | visible-damage comparison | strong small-data, nominal-only memory-bank baseline | memory-bank footprint may be less suitable for edge deployment |
| ByteTrack | multi-frame identity | associates lower-confidence detections to reduce fragmented tracks | temporal persistence cannot turn repeated systematic error into truth |
| Homography + topology | metric floor pose | traceable units, covariance propagation, measurable calibration residuals | assumes locally planar floor and correctly detected footprint keypoints |

## Decision gates

1. **Complete:** train YOLO detection on the verified box source and two YOLO
   segmentation baselines on their verified polygon sources. Annotation types
   were not silently translated.
2. Build a 50-image pose-annotation pilot. Continue YOLO pose only after
   topology QA and inter-reviewer localization statistics pass.
3. Use SAM 2 only inside the CVAT review workflow. It was not used to generate
   any of the completed model's ground truth.
4. Compare YOLO segmentation and Mask2Former on the same source-grouped holdout;
   deploy YOLO unless the comparator materially improves overhang/centroid error.
5. After reviewed good/bad damage labels exist, train EfficientAD-S and
   PatchCore on identical, leakage-free visible crops;
   choose using image AUROC/F1, pixel AUROC/F1 when masks exist, abstention
   calibration, latency, and memory—not an external benchmark.
6. Tune ByteTrack and temporal fusion on complete videos, reporting ID switches,
   latency, decision flicker, and delayed failure detection.

## Primary references

- Ultralytics, [YOLO11 software documentation](https://docs.ultralytics.com/models/yolo11/).
- Cheng et al., [Masked-Attention Mask Transformer for Universal Image Segmentation](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html), CVPR 2022.
- Ravi et al., [SAM 2: Segment Anything in Images and Videos](https://proceedings.iclr.cc/paper_files/paper/2025/file/45c1f6a8cbf2da59ebf2c802b4f742cd-Paper-Conference.pdf), ICLR 2025.
- Batzner et al., [EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies](https://doi.org/10.1109/WACV57701.2024.00020), WACV 2024.
- Roth et al., [Towards Total Recall in Industrial Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf), CVPR 2022.
- Zhang et al., [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf), ECCV 2022.
- Zhang, [A Flexible New Technique for Camera Calibration](https://doi.org/10.1109/34.888718), IEEE TPAMI 2000.

Numbers reported in those papers are motivation only. They are not this
project's results and never populate the assignment metrics.
