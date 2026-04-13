# Pipeline Comparison: Coffee Shop Ceiling-Camera Dynamic Perception

**Date:** 2026-04-12
**Context:** Shared-autonomy power wheelchair project — coffee shop testbed using a static
ceiling (overhead) camera. Baseline pipeline is `pipeline.py` / `DynamicPerceptionPipeline`.
This comparison covers two pipeline variants and three dataset/annotation strategies.

---

## Part 1 — Detection Pipeline Variants

### Pipelines Reviewed

| ID | Name | Core detector | Tracker | Motion fallback | Status |
|----|------|---------------|---------|-----------------|--------|
| V-A | Baseline YOLOv8n (COCO) | YOLOv8n pretrained on COCO, no domain adaptation | ByteTrack (Ultralytics built-in) | RAFT-Small optical flow | Implemented and tested (`pipeline.py`) |
| V-B | Fine-tuned YOLOv8n (overhead) | YOLOv8n fine-tuned on overhead pedestrian data, transfer from COCO | ByteTrack | RAFT-Small optical flow | Proposed — not yet implemented |

---

### Comparative Analysis

#### V-A vs V-B: Detailed Comparison

| Aspect | V-A: Baseline YOLOv8n (COCO) | V-B: Fine-tuned YOLOv8n (overhead) |
|--------|-------------------------------|--------------------------------------|
| **Seated person recall** | Low — confirmed by user on real coffee shop video. COCO training distribution is entirely eye-level; seated people viewed from above appear as featureless ovals with no recognizable body axis. YOLO misses these systematically. | Expected high after fine-tuning. The training set will include annotated overhead views of seated humans, directly closing the distribution gap. |
| **Standing person recall** | Moderate — tops of heads and shoulder outlines partially match COCO training distribution at high confidence thresholds. Recall degrades at default `conf=0.25` for standing people who are occluded by tables or chairs. | High — standing overhead views will be included in fine-tuning data from VIRAT and Oxford Town Centre, which contain abundant walking-person examples from overhead or elevated cameras. |
| **Fast-moving staff recall** | Low to moderate — fast-moving people between RAFT flow intervals may produce only brief YOLO detections at low confidence. ByteTrack recovers low-confidence detections, but intermittent misses create track fragmentation. | Similar to V-A for motion speed itself. Fine-tuning improves per-frame recall, reducing the frequency of tracks that rely solely on ByteTrack's low-confidence recovery. |
| **Optical flow contribution** | High — RAFT-Small optical flow is essential in V-A. The `unclassified_motion_regions` layer in the current `scene_state` is the primary mechanism for capturing seated people, who produce flow when they move (reaching, gesturing) but are not YOLO-detected. When people are completely stationary the pipeline has no fallback. | Lower but still valuable — fine-tuning eliminates YOLO's systematic seated-person miss, so optical flow shifts from a gap-filler role to a supplementary fast-motion detector. Flow remains necessary for objects YOLO cannot classify at all (non-person moving objects: rolling bags, carts). |
| **Training data required** | None — uses Ultralytics pretrained `yolov8n.pt`. | 500–2000 annotated overhead frames (see Part 2). Colab fine-tuning run (~1–2 hrs on T4). |
| **Compute at inference time** | CPU-feasible: on MacBook CPU, `run_video.py` processes at ~10 FPS for 640×480 synthetic video (tester report). Real 1080p ceiling video likely 2–5 FPS on CPU. | Identical to V-A at inference time — same architecture, same weights file size (~6 MB), same ONNX/PyTorch forward pass. No GPU required at inference. |
| **Deployment complexity** | Drop-in: `YOLO("yolov8n.pt")`. Model weights downloaded automatically by Ultralytics on first use. | Requires managing a fine-tuned `.pt` file (stored in Google Drive or repo). Single config change: `config["yolo_model"] = "yolov8n_overhead.pt"`. No code changes to `pipeline.py`. |
| **Risk if fine-tuning fails** | No risk to existing pipeline — V-A is the rollback. | If fine-tuning degrades COCO-class recall (regression), the optical flow layer still provides a safety net. Evaluation gating (Phase 2 success criterion) prevents a broken model from reaching production. |
| **Scene state completeness** | `objects` array contains YOLO-detected persons only; seated people appear only in `unclassified_motion_regions` with no class label. | `objects` array captures seated people with class labels. Enables semantically richer scene state: a navigation consumer can distinguish `person_sitting` (static obstacle, high probability of remaining in place) from `person_walking` (dynamic, trajectory-relevant). |

---

#### What worked in V-A (to preserve in V-B)

- The YOLO + ByteTrack + RAFT-Small architecture is sound and fully tested. The tester
  report confirms 17/18 unit tests pass, integration test exit code 0, correct JSONL schema.
- The flow carry-forward mechanism (last computed `_flow_mask` reused between flow intervals)
  correctly handles the static-camera assumption and avoids redundant RAFT inference.
- The `unclassified_motion_regions` layer provides a genuine safety net. Any moving object
  missed by YOLO still appears in the scene state with a bounding box and magnitude. This
  property should be preserved regardless of how much fine-tuning improves recall.
- The single `config["yolo_model"]` entry point makes swapping weights trivial — V-B requires
  no architectural change to `pipeline.py`.
- CPU-feasible inference is a confirmed property of the current pipeline. Fine-tuning YOLOv8n
  (same nano architecture) preserves this — the model size and forward-pass structure are
  unchanged.

#### What failed or underperformed in V-A

- **Seated person detection: the confirmed gap.** This is the primary motivation for V-B.
  COCO contains no overhead-view training images; seated people from above are out-of-distribution
  for every COCO-trained detector. The optical flow layer partially compensates, but only when
  seated people are actively moving. Stationary seated people are invisible to V-A.
- **No class distinction for navigation.** V-A collapses all detected persons to a single
  `"person"` class. For wheelchair navigation, a seated person (static obstacle) behaves
  differently from a walking person (dynamic obstacle with trajectory). This distinction is
  absent from V-A's `scene_state` output and requires fine-tuned class labels in V-B.
- **Trajectory prediction not wired into scene_state.** `predict_trajectory()` is called
  internally but its output is discarded (see tester report, G1/G2). This is a V-A limitation
  unrelated to fine-tuning; it should be addressed separately as a pipeline integration task.
- **Centroid variable scope bug under multi-person scenes (medium severity).** Identified by
  tester: when multiple people are tracked simultaneously, `_prev_centroids` is updated with
  the wrong centroid for all but the last person in the inner loop (pipeline.py L659). This
  corrupts velocity estimates in multi-person scenes. Fix: use `tuple(obj.centroid_px)` at L659.
  This bug affects V-A and will propagate to V-B unless fixed first.

#### Design gaps neither variant addresses

The following limitations are present in both V-A and V-B and are out of scope for fine-tuning
alone. They are documented here for prioritization in a subsequent pipeline revision:

- **No metric depth.** The pipeline operates entirely in pixel space. Distances in
  `nearest_person_px` are frame-pixel distances from the frame centre, not real-world distances.
  For wheelchair navigation, knowing that a person is 1.2 m vs. 3 m from the wheelchair's
  projected path is operationally critical and is absent from the current scene state.
- **No homography calibration.** A ceiling camera's pixel positions can be mapped to floor-plane
  real-world coordinates via a homography transform (given known camera height and intrinsics).
  Neither variant implements this. Adding homography would convert `centroid_px` to `centroid_m`
  in the scene state without any model change.
- **No spatial zone annotation.** The coffee shop scene has semantically distinct regions:
  ordering area, seating area, staff-only zone, walking corridor. Neither variant annotates
  which zone a detected person occupies. This would require a one-time zone mask image per
  camera installation.
- **Stationary seated people produce zero optical flow.** A seated customer who is completely
  still (reading, looking at phone) generates no flow signal and no YOLO detection in V-A.
  V-B's fine-tuned YOLO addresses this directly, but it remains a known hard case: the detector
  must classify a motionless overhead oval as a seated person from appearance alone.

---

## Part 2 — Dataset and Annotation Strategy Comparison

Three strategies for building the overhead pedestrian fine-tuning dataset are compared.

### Strategies Reviewed

| ID | Name | Description |
|----|------|-------------|
| S-A | Public overhead datasets only | Use VIRAT Ground Dataset, Oxford Town Centre, CCTV-Crowd, or DroneCrowd; no manual annotation of the coffee shop video |
| S-B | Semi-automatic annotation of coffee shop video | Use RAFT-Small motion masks to crop candidate regions from the coffee shop video; manually label only those crops; export in YOLO format |
| S-C | Combined (recommended) | Use S-A public data as the volume/diversity base; use S-B coffee shop annotation for domain-specific seated-person coverage |

---

### Detailed Strategy Comparison

| Aspect | S-A: Public datasets only | S-B: Coffee shop semi-auto only | S-C: Combined (recommended) |
|--------|---------------------------|---------------------------------|-----------------------------|
| **Estimated annotation effort** | 0 hrs manual annotation for walking/standing (labels already exist). 2–4 hrs for downloading, converting to YOLO format, and splitting train/val/test. | 4–8 hrs total: ~1 hr to extract motion-region crops using RAFT pipeline, ~3–6 hrs for manual review and labeling in CVAT or Roboflow. | 6–12 hrs: S-A setup (2–4 hrs) plus S-B (4–8 hrs). One-time cost. |
| **Dataset size** | Large: VIRAT contains 550+ hours of ground surveillance at 2–18m height (not pure overhead but elevated). Oxford Town Centre: ~1 hr at 4m height, ~5000 annotated frames. CCTV-Crowd: 474 scenes. DroneCrowd: 33,600 frames, aerial but variable height. Easily 10,000+ usable overhead frames. | Small to medium: Depends on coffee shop video length. A 30-minute ceiling video at 30 fps with motion-region extraction at every 10th frame yields ~5,400 candidate crops. After manual filtering (non-person regions, blur artifacts), realistically 500–1,500 labeled instances. | Large base + targeted supplementation. The public data provides diversity across lighting, crowd density, and person pose; the coffee shop data provides the domain-specific seated-person examples that public datasets lack. |
| **Domain coverage — seated persons** | Poor. VIRAT and Oxford Town Centre depict pedestrians in outdoor public spaces; seated people are rare or absent. CCTV-Crowd focuses on crowd density in stations and plazas. DroneCrowd is aerial-outdoor. None of these environments contain coffee shop seating scenarios. Fine-tuning on S-A alone will not solve the seated-person detection gap. | Good for seated persons. The coffee shop video contains exactly the seated-person examples needed. Motion-region crops will capture seated people who move (reaching, standing up, gesturing). Static seated people will not generate motion regions and must be manually selected from fixed-interval frame samples. | Excellent. Seated-person gap is closed by S-B; walking/standing diversity is provided by S-A. The combined dataset generalizes across person poses and environments. |
| **Domain coverage — overhead angle** | Mixed. VIRAT is ground surveillance (elevated, not pure overhead). Oxford Town Centre is approximately 10m height, close to ceiling-camera angle. DroneCrowd is aerial, often at steep angles. None are matched exactly to a 2–4m ceiling camera, but Oxford Town Centre is the closest available public source. | Exact. The coffee shop video is the exact deployment camera, viewpoint, and lighting. This is the strongest possible domain match. | Exact match for the coffee shop deployment, supplemented by broader distribution for generalization. |
| **Risk of distribution mismatch** | High for seated persons specifically. A model fine-tuned on S-A will likely improve detection of walking people from an elevated angle, but will still fail on seated people if the fine-tuning set contains no seated examples. The gap may narrow but not close. | Low for coffee shop seated detection, but high for generalization. A model trained only on S-B will overfit to the specific coffee shop lighting, table configuration, and crowd density. It may fail on a different camera installation or a rearranged furniture layout. | Low overall. Public data prevents overfitting; domain data closes the seated-person gap. This is the standard best practice in transfer learning literature: a diverse base with targeted in-domain supplementation. |
| **Tooling** | Roboflow Universe has pre-converted YOLO-format exports for Oxford Town Centre (search "Oxford Town Centre YOLO"). VIRAT requires manual conversion from the VIRAT annotation schema. DroneCrowd provides COCO-format labels; use `roboflow` or `labelstudio` to reformat. | CVAT (open-source, self-hosted or cloud) or Roboflow (freemium cloud, easiest YOLO export). The semi-auto pipeline: run `pipeline.py` on the coffee shop video; extract `unclassified_motion_regions` bounding boxes per frame; use these as pre-labeling suggestions in CVAT; human annotators correct and add missed instances. | Combine both toolchains. Use Roboflow Projects to merge public and custom datasets, apply augmentation uniformly, and generate the final YOLO-format train/val/test split. |
| **Recommended minimum volume** | 3,000 walking/standing overhead frames (VIRAT + Oxford Town Centre) | 300 seated-person instances + 200 standing overhead instances from the coffee shop video | 3,000 public frames + 500 domain-specific frames. Total ~3,500 unique images before augmentation. |

### Recommendation: S-C

Strategy C is recommended for the following reasons, traceable to the synthesis and pipeline comparison:

1. The indoor wheelchair perception synthesis (Lecrosnier et al., 2020; Beyer et al., 2018)
   consistently shows that models trained on public datasets without in-domain supplementation
   fail to generalize to the specific sensor configuration and environment. The same applies here:
   public overhead pedestrian data covers the overhead-angle distribution but not the
   seated-in-coffee-shop instance distribution.

2. The vision-only pipeline synthesis notes (Theme 6) that the benchmark-deployment gap is
   structural — it cannot be closed by public data alone. The coffee shop video is the only
   source of ground-truth seated-person appearances at this camera height and FOV.

3. The semi-automatic annotation pipeline (RAFT motion regions as pre-labeling suggestions)
   dramatically reduces manual annotation burden. The motion mask already flags every region
   where a person moves, so annotators review candidate crops rather than scanning raw frames.
   This is precisely the labeling strategy enabled by the existing `unclassified_motion_regions`
   output in `pipeline.py`.

4. Combining S-A and S-B provides the diversity that prevents overfitting to a single camera
   installation, which matters for the wheelchair deployment target where the camera may move
   between venues.
