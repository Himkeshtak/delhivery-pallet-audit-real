# Five-minute screen-recording script

Record from the current repository. Do not substitute synthetic images, paper
benchmark numbers, or a claim that pending physical evidence exists.

1. **0:00-0:35 - problem and contract:** show
   `examples/real_frame_pallet/assessment_001.json`, all eight checks, and the
   explicit calibration/pose reasons for `MANUAL_REVIEW`.
2. **0:35-1:15 - data:** show `DATASET.md`, the three pinned real sources,
   archive hashes, counts/splits, six carton-mask QA overlays, leakage exclusion,
   and the five-label duplicate audit.
3. **1:15-2:05 - models:** show the three committed checkpoint hashes, real
   training reports, the held-out adapter overlay, and explain that SAM 2 is
   only an optional human-reviewed annotation accelerator. State that no pose
   or damage model was trained without valid labels.
4. **2:05-2:45 - geometry/SOP:** show `load_analysis.py` and its tests for
   overhang, centroid, rotation, and layer-size risk. Explain why pixels cannot
   be called centimetres without physical calibration.
5. **2:45-3:40 - results:** show detection/localization distributions, carton
   box/mask AP at multiple IoUs, the three detector worst cases, and the carton
   domain gap visible in the runnable example.
6. **3:40-4:25 - deployment:** show the three actual CPU benchmark reports and
   the scheduling/degradation policy. Explicitly say Jetson/TensorRT numbers are
   pending because the target device was not available.
7. **4:25-5:00 - tradeoffs:** show the traceability table, single-view limits,
   licensing, remaining physical capture work, AI-tool disclosure, and the
   caught synthetic-data error.

Useful commands to show on screen:

```powershell
python -m pytest -p no:cacheprovider -q
python tools/infer_frame.py --image data/processed/detect/images/test/pallet_detect_v1_008b44ae079f30b0d8e7.jpg --output runs/demo/recording --device cpu --image-size 320
Get-FileHash -Algorithm SHA256 weights/releases/*.pt
```
