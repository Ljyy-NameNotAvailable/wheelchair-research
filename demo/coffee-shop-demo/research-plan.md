# Research Plan: Overhead-View Fine-Tuning for Coffee Shop Dynamic Perception

**Date:** 2026-04-12
**Project:** Shared-autonomy power wheelchair — coffee shop ceiling-camera testbed
**Pipeline under study:** `demo/coffee-shop-demo/pipeline.py` (`DynamicPerceptionPipeline`)
**Known issue:** YOLOv8n (COCO-pretrained) fails to detect seated humans from overhead viewpoint
**Goal:** Fine-tune YOLOv8n on overhead pedestrian data; re-evaluate on the coffee shop video;
produce a drop-in replacement weight file for `pipeline.py`

---

## Overview

| Phase | Name | Primary input | Primary output | Success criterion |
|-------|------|---------------|----------------|-------------------|
| 0 | Baseline characterization | Coffee shop ceiling video | Detection gap metrics, annotated error frames | Seated-person recall < 0.50 confirmed quantitatively |
| 1 | Dataset preparation | Public datasets + coffee shop video | YOLO-format train/val dataset, augmentation config | ≥ 500 seated-person instances in training split |
| 2 | Fine-tuning | Phase 1 dataset | `yolov8n_overhead.pt` weight file | mAP@0.5 improvement over baseline on held-out ceiling clip |
| 3 | Integration and re-evaluation | Fine-tuned weights + coffee shop video | Per-class recall comparison, optical flow ablation | Seated-person recall ≥ 0.70 on annotated test frames |
| 4 | Scene understanding output | Phase 3 scene logs | Updated scene state specification for wheelchair navigation | Class-aware obstacle categorization in scene_state |

---

## Phase 0 — Baseline Characterization

**Goal:** Measure the detection gap quantitatively before touching the model. A precise baseline
number prevents later confusion about whether improvements are real.

### Step 0.1 — Manual annotation of N representative frames

Select 100–200 frames from the coffee shop ceiling video, covering:
- Dense seating periods (lunch peak, e.g. frames from 11:30–13:00 if timestamped)
- Low-occupancy periods (to establish true negatives)
- Frames with standing staff as well as seated customers
- At least 30 frames containing at least one stationary seated person

**Tool:** CVAT (free, browser-based) or Roboflow Annotate. Annotate all persons visible from
above, assigning labels `person_sitting` and `person_standing`. Export in YOLO format (class ID,
normalized x-center, y-center, width, height per line).

**Why 100–200 frames:** This is the minimum needed for a reliable per-class recall estimate with
reasonable confidence intervals. Fewer frames produce recall estimates with ± 15–20% error bars
that obscure real improvement after fine-tuning.

### Step 0.2 — Run the baseline pipeline and collect raw detections

```bash
python run_video.py \
    --input coffee_shop.mp4 \
    --output coffee_shop_baseline.mp4 \
    --log coffee_shop_baseline.jsonl \
    --device cpu
```

Parse `coffee_shop_baseline.jsonl` to extract YOLO detection boxes for each annotated frame.
Match predicted boxes to ground truth via IoU ≥ 0.5 (standard PASCAL VOC protocol).

### Step 0.3 — Compute per-class precision and recall

For each class (`person_sitting`, `person_standing`):

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

Where a TP is a predicted box with IoU ≥ 0.5 to a ground truth box of the same class, FN is a
ground truth box with no matching prediction, and FP is a predicted box with no matching ground
truth. Report at the default Ultralytics confidence threshold (0.25) and also at 0.10 to
characterize the confidence distribution of marginal detections.

**Expected finding (consistent with user observation):** `person_sitting` recall < 0.40.
`person_standing` recall likely 0.55–0.75 depending on camera height and resolution.

### Step 0.4 — Use optical flow output as an independent gap detector

The `unclassified_motion_regions` field in each JSONL line captures motion pixels with no
YOLO match. Extract these regions for every annotated frame and compute the overlap between
`unclassified_motion_regions` bounding boxes and the ground truth `person_sitting` boxes
(IoU ≥ 0.30, looser threshold since flow regions are coarser than YOLO boxes).

**Metric:** "Flow coverage rate" = fraction of missed seated-person ground truth boxes that
have an overlapping unclassified motion region. This measures how much the optical flow layer
compensates for the YOLO gap. A flow coverage rate of 0.70 means 70% of missed seated people
are at least partially visible in the motion layer, but only when they are moving. Stationary
seated people will have zero flow coverage — this is the irreducible gap that fine-tuning must
close.

### Step 0.5 — Define the detection gap quantitatively

Report the baseline gap as:
- Seated-person recall at conf=0.25 and conf=0.10
- Standing-person recall at conf=0.25 and conf=0.10
- Flow coverage rate for missed seated persons
- Fraction of missed seated persons that are completely stationary (flow magnitude < theta in
  all overlapping flow windows) — this is the non-recoverable gap in V-A

**Output:** `demo/coffee-shop-demo/baseline_metrics.md` — a two-page summary of the gap
with example failure frames (exported from the annotated video).

**Success criterion for Phase 0:** Seated-person recall below 0.50 confirmed with N ≥ 100
annotated frames. If recall is above 0.50, reconsider whether fine-tuning is the right
intervention; the gap may be camera-specific or resolvable by lowering the confidence threshold.

---

## Phase 1 — Dataset Preparation for Fine-Tuning

**Goal:** Assemble a YOLO-format training dataset covering overhead pedestrian appearances,
with specific emphasis on seated humans in indoor settings. Target: ≥ 3,000 total annotated
instances (frames × people per frame) before augmentation.

### Step 1.1 — Acquire public overhead pedestrian datasets

Candidate datasets and access notes:

| Dataset | Content | Camera angle | Seated examples | Access |
|---------|---------|--------------|-----------------|--------|
| Oxford Town Centre | ~5,000 frames, dense crowd, ~10m height, outdoor | Elevated but close to overhead | Few (benches only) | [oxfordrobotics.institute/datasets](http://oxfordrobotics.institute/datasets); also on Roboflow Universe pre-converted to YOLO format |
| VIRAT Ground Dataset | 550+ hrs, indoor/outdoor surveillance, 2–18m height | Variable, some overhead | Indoor scenes have some seated | [viratdata.org](http://viratdata.org); free registration; original annotations in VIRAT format, requires conversion |
| CCTV-Crowd | 474 surveillance scenes, high crowd density | Overhead surveillance typical | Rare (public transit) | [github.com/ee-hong/CCTV-Crowd](https://github.com/ee-hong/CCTV-Crowd) |
| DroneCrowd | 33,600 frames, drone-mounted, outdoor | Steep aerial, variable | No | [download.visinf.tu-darmstadt.de](https://download.visinf.tu-darmstadt.de/data/DroneCrowd/) |
| ShanghaiTech Part A/B | Crowd density counting, mixed camera angles | Mixed, some overhead | Minimal | Available via [shanghaitech.edu.cn](http://www.shanghaitech.edu.cn) |

**Recommended subset for Phase 1:** Oxford Town Centre (primary, closest to ceiling angle) +
VIRAT indoor scenes (secondary, for indoor lighting distribution). Exclude DroneCrowd and
ShanghaiTech unless Oxford + VIRAT yield fewer than 2,000 walking/standing instances.

**Conversion to YOLO format:** For Oxford Town Centre, use the Roboflow Universe pre-converted
export if available; otherwise convert from the original `.csv` annotation format:

```python
# Each annotation: frame_id, person_id, x_tl, y_tl, x_br, y_br
# YOLO format: class_id x_center y_center width height  (normalized 0-1)
x_center = (x_tl + x_br) / (2 * frame_width)
y_center = (y_tl + y_br) / (2 * frame_height)
width    = (x_br - x_tl) / frame_width
height   = (y_br - y_tl) / frame_height
```

Class assignment for public data: all walking/standing persons → class 0 (`person_standing`).
Do not collapse to a single `person` class at this stage — see Step 1.3 for the class-design
decision.

### Step 1.2 — Semi-automatic annotation of the coffee shop video

This is the critical step for closing the seated-person gap that public datasets do not cover.

**Sub-step 1.2a — Motion region extraction**

Run the existing pipeline on the full coffee shop video with a low flow threshold to maximize
coverage:

```bash
python run_video.py \
    --input coffee_shop.mp4 \
    --output /dev/null \
    --log coffee_shop_motion.jsonl \
    --theta 1.5 \
    --flow-interval 3 \
    --device cpu
```

Parse `coffee_shop_motion.jsonl`: for every frame where `flow_computed_this_frame` is true,
extract all entries in `unclassified_motion_regions` where `area_px2 > 400` (roughly 20×20 px
minimum, to exclude noise). Save each region as a crop (`frame[y1:y2, x1:x2]`) with its
frame index and bounding box coordinates.

Additionally, extract YOLO-detected persons from `objects` entries in the same log. These are
the standing/walking people the baseline already finds correctly — useful for balanced labeling.

**Sub-step 1.2b — Fixed-interval frame sampling for stationary seated persons**

Stationary seated people produce no optical flow and therefore do not appear in
`unclassified_motion_regions`. To capture them, extract every 30th frame from the coffee shop
video:

```python
import cv2
cap = cv2.VideoCapture("coffee_shop.mp4")
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % 30 == 0:
        cv2.imwrite(f"frames/frame_{frame_idx:06d}.jpg", frame)
    frame_idx += 1
cap.release()
```

At 30 fps, this produces one frame per second — sufficient to capture seating changes over a
30-minute video without reviewing 54,000 individual frames.

**Sub-step 1.2c — Manual labeling in CVAT or Roboflow**

Import the fixed-interval frames and the motion-region crops into CVAT
([cvat.ai](https://cvat.ai), free cloud) or Roboflow ([roboflow.com](https://roboflow.com),
freemium). Configure two label classes:
- `person_sitting` (class ID 1)
- `person_standing` (class ID 0)

For motion-region crops, use CVAT's pre-annotation feature: import the motion region bounding
boxes as polygon suggestions, then accept/correct/reject each one. This reduces per-instance
annotation time to ~5–10 seconds versus ~20–30 seconds for fully manual annotation.

Target for the coffee shop annotation pass: ≥ 300 `person_sitting` instances across ≥ 100
unique frames. This is achievable in 4–6 hours of labeling with two annotators.

**Sub-step 1.2d — Export in YOLO format**

From CVAT: Export → YOLO 1.1 format. This generates per-image `.txt` files with normalized
bounding boxes and class IDs, and a `data.yaml` pointing to train/val image directories.

From Roboflow: Export → YOLOv8 format (compatible with Ultralytics `ultralytics` train API).

### Step 1.3 — Class design decision: two-class vs. single-class

**Option A (two-class):** `person_standing` (class 0) and `person_sitting` (class 1).
- Pro: Richer scene state. Navigation consumer can treat sitting persons as static obstacles
  (lower trajectory uncertainty) and standing persons as dynamic obstacles (higher urgency).
- Pro: Directly answers the research question: does fine-tuning close the seated-person gap?
- Con: Requires consistent labeling of ambiguous postures (crouching, leaning). Annotator
  agreement protocols must be defined.
- Con: Smaller per-class training set — `person_sitting` may underfit if < 300 instances.

**Option B (single-class):** Collapse to a single `person` class.
- Pro: Larger effective training set per class. Simpler labeling.
- Pro: Compatible with the existing `pipeline.py` class tracking (currently tracks only class 0
  `person` from COCO).
- Con: Scene state loses the static/dynamic distinction. For wheelchair navigation in a coffee
  shop, this distinction directly affects path planning behavior.

**Recommendation: Two-class, with fallback to single-class if annotation volume for
`person_sitting` is below 200 instances.** The navigation use case (distinguishing static
seated obstacles from dynamic walking people) justifies the annotation effort. If the annotation
campaign yields insufficient seated-person examples, collapse to a single class and re-annotate
in a subsequent pass.

**Pipeline.py change required for two-class:** Update `_TRACK_CLASSES = [0]` to
`_TRACK_CLASSES = [0, 1]` and update the `scene_state` `class` field to propagate the
fine-tuned class name through to the JSON output.

### Step 1.4 — Combine public and coffee shop datasets in Roboflow

Merge the public dataset (Step 1.1) and the coffee shop annotation (Step 1.2) into a single
Roboflow Project. Apply a consistent 80/10/10 train/val/test split, stratifying by class to
ensure both `person_sitting` and `person_standing` are present in all splits.

**Target dataset composition (before augmentation):**
- Training: ~2,800 images, ≥ 300 `person_sitting` instances, ≥ 2,000 `person_standing`
- Validation: ~400 images (held out from the same sources)
- Test: ~200 images (held out coffee shop frames only — the deployment domain)

Keep the test split as coffee-shop-only so that Phase 2 and Phase 3 evaluation is on the
exact deployment camera, not on public dataset frames.

### Step 1.5 — Data augmentation strategy for overhead perspective

Overhead pedestrian images have different augmentation requirements than standard eye-level
images. Configure the Roboflow augmentation pipeline:

| Augmentation | Setting | Rationale |
|---|---|---|
| Rotation | ±180° (full rotation) | From a ceiling camera, a person walking north vs. south vs. east vs. west looks identical except for heading direction. COCO training does not expose the model to arbitrary heading angles. |
| Horizontal and vertical flip | Both enabled | Equivalent to rotation; reinforces orientation invariance. |
| Scale | 0.75–1.25× | Camera height varies across coffee shop installations; people at tables closer to the camera are larger. |
| Brightness jitter | ±25% | Overhead fluorescent lighting varies by time of day and scene zone. |
| Gaussian blur | Up to 3px | Fast-moving people (staff walking quickly) appear blurred in 30 fps video. |
| Mosaic | 4-image mosaic (YOLOv8 built-in, enable in training config) | Improves detection of small objects at various scales; useful for overhead-view small-person oval silhouettes. |
| Hue saturation | ±15% | Clothing color variation. |

Do NOT apply: perspective distortion (ceiling camera already has consistent overhead perspective;
adding synthetic perspective skews would introduce a different distribution from the real data),
or extreme aspect-ratio cropping (overhead people are roughly circular/elliptical, not elongated
as in eye-level views).

**Output of Phase 1:** A Roboflow-hosted (or locally stored) YOLO-format dataset with:
- `data.yaml` pointing to train/val/test image and label directories
- Augmented training set of ~8,000–10,000 images (3× augmentation factor)
- Unaugmented validation and test sets
- At least 300 `person_sitting` instances in the training split

**Success criterion for Phase 1:** Dataset assembled, `data.yaml` valid, at least one sanity
check: train split loads without error in Ultralytics `YOLO("yolov8n.pt").train(data=..., epochs=1)`.

---

## Phase 2 — Fine-Tuning YOLOv8n

**Goal:** Produce a fine-tuned weight file `yolov8n_overhead.pt` that improves recall on seated
and standing people from overhead view, while preserving CPU-feasible inference speed.

### Step 2.1 — Training environment setup (Google Colab T4)

Fine-tuning runs on Google Colab with a T4 GPU (~15 GB VRAM, ~12 hr session limit). The full
fine-tuning run for YOLOv8n is expected to complete in 1–2 hours on a T4 at 100 epochs with
a dataset of ~3,000 images, well within the session limit.

```python
# Colab Cell 1 — Install and configure
!pip install ultralytics roboflow

# Mount Google Drive to persist weights across sessions
from google.colab import drive
drive.mount('/content/drive')

# Copy dataset from Drive (or download from Roboflow)
# Option A: Roboflow export
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("overhead-pedestrian")
dataset = project.version(1).download("yolov8")
# → downloads to ./overhead-pedestrian-1/ with data.yaml inside

# Option B: Dataset already in Drive
# !cp -r /content/drive/MyDrive/overhead_dataset/ ./dataset/
```

### Step 2.2 — Training configuration

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Start from COCO pretrained weights

results = model.train(
    data="overhead-pedestrian-1/data.yaml",  # path from Step 2.1
    epochs=100,
    imgsz=640,           # see Step 2.3 for rationale
    batch=16,            # T4 VRAM allows batch=16 for 640px input
    lr0=0.001,           # conservative LR for fine-tuning
    lrf=0.01,            # final LR decay factor
    weight_decay=0.0005,
    warmup_epochs=3,
    cos_lr=True,
    freeze=0,            # fine-tune all layers (see note below)
    mosaic=1.0,          # enable 4-image mosaic augmentation
    degrees=180.0,       # full rotation augmentation (overhead-specific)
    flipud=0.5,          # vertical flip (overhead-specific)
    fliplr=0.5,          # horizontal flip
    scale=0.25,          # scale jitter ±25%
    hsv_h=0.015,         # hue jitter
    hsv_s=0.7,           # saturation jitter
    hsv_v=0.4,           # value jitter
    blur=3,              # Gaussian blur up to 3px
    project="/content/drive/MyDrive/coffee_shop_finetune",
    name="yolov8n_overhead_v1",
    save=True,
    patience=20,         # early stopping if no val improvement for 20 epochs
)
```

**Note on layer freezing:** For a small domain shift (eye-level to overhead), unfreezing all
layers with a low learning rate (lr0=0.001) tends to outperform freezing the backbone, because
the overhead view requires re-learning the feature representations of person shapes at multiple
scales. If the dataset is very small (< 500 images), freeze the first 10 backbone layers with
`freeze=10` to prevent overfitting.

### Step 2.3 — Image size decision: 640 vs. 1280

| Image size | Training batch (T4) | Training time | Inference FPS (CPU) | Rationale |
|---|---|---|---|---|
| 640×640 | 16 | ~1–1.5 hrs / 100 epochs | ~3–5 FPS (estimated on MacBook CPU) | Standard YOLOv8n size. Adequate for ceiling camera at 2–4m height where people subtend 30–80px. |
| 1280×1280 | 4 | ~4–6 hrs / 100 epochs | ~1–2 FPS (estimated on MacBook CPU) | Better for very small overhead person blobs (camera at 5–8m height) but likely exceeds the coffee shop demo FPS requirement. |

**Recommendation: 640×640 for the initial fine-tuning run.** If evaluation (Phase 3) shows
that small person recall is the limiting factor (people appear < 20px across), run a second
experiment at 1280×1280 and compare mAP@0.5 on the test set.

### Step 2.4 — Hyperparameters to tune

Run a minimal hyperparameter sweep across the two most impactful parameters:

| Parameter | Values to try | Rationale |
|---|---|---|
| `lr0` | 0.0005, 0.001, 0.01 | LR is the single most impactful fine-tuning hyperparameter; 0.001 is the recommended starting point |
| `freeze` | 0, 10 | Compare full fine-tuning vs. frozen backbone for small datasets |

Each combination is one Colab session (~1–2 hrs). Do not sweep more parameters in Phase 2;
the dataset size is the binding constraint, not the learning rate.

### Step 2.5 — Evaluation on held-out ceiling clip

At the end of training, `ultralytics` saves the best weights to:
`/content/drive/MyDrive/coffee_shop_finetune/yolov8n_overhead_v1/weights/best.pt`

Run evaluation on the test split (coffee-shop-only frames from Step 1.4):

```python
metrics = model.val(
    data="overhead-pedestrian-1/data.yaml",
    split="test",
    conf=0.25,
    iou=0.5,
)
print(metrics.box.map)    # mAP@0.5 overall
print(metrics.box.maps)   # mAP@0.5 per class
```

**Compare to baseline (V-A):** Run the same evaluation with the original `yolov8n.pt` weights
on the same test split. The fine-tuned model must show improvement in `person_sitting` mAP@0.5.

**Success criterion for Phase 2:** Fine-tuned model mAP@0.5 on `person_sitting` class ≥ 0.50,
and overall mAP@0.5 ≥ baseline overall mAP@0.5 (no regression on standing persons). If the
fine-tuned model degrades standing person recall, return to Phase 1 and increase the proportion
of standing-person examples in the training set.

**Save the weight file:**

```bash
cp /content/drive/MyDrive/coffee_shop_finetune/yolov8n_overhead_v1/weights/best.pt \
   /content/drive/MyDrive/yolov8n_overhead.pt
```

---

## Phase 3 — Integration and Re-Evaluation

**Goal:** Swap the fine-tuned weights into the existing `pipeline.py` via the single
`config["yolo_model"]` entry point. Re-run on the coffee shop video. Compare before/after
on the Phase 0 annotated frame set.

### Step 3.1 — Weight file swap (single line change)

No code changes to `pipeline.py` are required. The `config` dict accepts any `.pt` path:

```python
from pipeline import DynamicPerceptionPipeline

pipeline = DynamicPerceptionPipeline({
    "yolo_model": "yolov8n_overhead.pt",   # ← only change
    "device": "cpu",
})
```

If using the two-class model from Phase 1.3, also update `_TRACK_CLASSES` in `pipeline.py`
from `[0]` to `[0, 1]` and verify that the `class` field in `scene_state.objects` now
correctly emits `"person_standing"` and `"person_sitting"` strings from `self.yolo.names`.

### Step 3.2 — Run on the original coffee shop video

```bash
python run_video.py \
    --input coffee_shop.mp4 \
    --output coffee_shop_finetuned.mp4 \
    --log coffee_shop_finetuned.jsonl \
    --device cpu
```

Visually inspect `coffee_shop_finetuned.mp4` for:
- Green boxes appearing over seated customers (the primary success indicator)
- No new spurious detections on table surfaces, bags, or chairs with no occupant
- Track ID stability (no excessive ID switching compared to the baseline video)

### Step 3.3 — Quantitative comparison on annotated frames

Using the Phase 0 annotation set (the same 100–200 manually annotated frames), extract
YOLO detections from `coffee_shop_finetuned.jsonl` and recompute precision/recall:

```
Phase 0 baseline:      person_sitting recall = R_baseline
Phase 3 fine-tuned:    person_sitting recall = R_finetuned
```

Report the recall delta and the false positive rate (spurious detections on non-person
overhead regions). A 0.20+ recall improvement on seated persons is the target.

### Step 3.4 — Optical flow ablation

After fine-tuning, re-measure the flow coverage rate from Step 0.4:
- How many YOLO detections in `coffee_shop_finetuned.jsonl` now fall in regions that were
  previously `unclassified_motion_regions` in the baseline?
- What fraction of `unclassified_motion_regions` remain (motion without any YOLO match) after
  fine-tuning?

**Expected result:** The flow coverage rate for seated persons drops — YOLO now classifies
many of the regions the flow layer previously flagged as unclassified. The remaining
`unclassified_motion_regions` should correspond to non-person motion sources (bags, cups
sliding, door openings) rather than missed seated people.

The optical flow layer remains valuable for two categories:
1. Non-person moving objects (no YOLO class covers them)
2. Transient fast-moving objects that appear and disappear within a single `flow_interval`
   window (e.g., a staff member who passes through the FOV in < 5 frames at 30 fps)

Report whether removing RAFT-Small optical flow entirely would degrade scene state coverage.
If the fine-tuned model handles all person detection, the main remaining optical flow
contribution is non-person motion detection. This is useful to quantify for the embedded
deployment decision (optical flow is the most compute-heavy pipeline component).

### Step 3.5 — Fix known bugs before final evaluation

Apply the two medium-severity bug fixes identified by the tester report before running Phase 3:

1. **Centroid scope bug (L659 in `pipeline.py`):** Replace `centroid` with
   `tuple(obj.centroid_px)` at the `_prev_centroids` update line. This is required for
   correct velocity estimates in multi-person scenes, which is exactly the scenario the
   fine-tuned pipeline will encounter in the coffee shop.

2. **`lap` dependency:** Add `lap>=0.5.12` to `requirements.txt` for a clean
   `pip install -r requirements.txt` without runtime auto-download.

These fixes do not require approval from the designer agent — they are pure correctness repairs
identified in the existing test report.

**Success criterion for Phase 3:** Seated-person recall on the annotated test frames ≥ 0.70
(from a baseline of < 0.50 measured in Phase 0). No regression in standing-person recall.
Processing FPS on laptop CPU within 20% of the baseline (fine-tuned model has the same
inference cost as the original YOLOv8n).

---

## Phase 4 — Scene Understanding Output for Wheelchair Navigation

**Goal:** Define what the scene state should convey for wheelchair navigation in a coffee shop,
and specify how `person_sitting` vs. `person_standing` class labels affect the downstream
navigation behavior.

### Step 4.1 — Scene state requirements for coffee shop navigation

A shared-autonomy power wheelchair navigating a coffee shop needs the following information
from the perception pipeline:

| Information need | Current scene_state field | Gap / proposed addition |
|------------------|--------------------------|------------------------|
| Is my path clear? | `dynamic_pixel_count`, `unclassified_motion_regions` | Insufficient — pixel count does not distinguish a person blocking the path from peripheral motion |
| Where are people? | `objects[].centroid_px`, `objects[].bbox_px` | Present but in pixel coordinates; need floor-plane coordinates in meters |
| Are they moving? | `objects[].velocity_px_per_frame` | Present; needs conversion to m/s using homography |
| Are they sitting or standing? | `objects[].class` (currently always `"person"`) | Fixed by fine-tuning — will emit `"person_sitting"` or `"person_standing"` |
| How close is the nearest person? | `nearest_person_px` | Pixel distance only; needs metric conversion |
| Which zone is each person in? | Not present | New field: zone classification (ordering_area, seating_area, corridor, staff_zone) |
| Are approach corridors clear? | Not present | New field: per-corridor occupancy mask |

### Step 4.2 — `person_sitting` vs. `person_standing` for navigation planning

The distinction matters for two navigation decisions:

**Path inflation radius.** A standing person may move in any direction at any time; the
wheelchair's path planner should inflate the obstacle radius to account for trajectory
uncertainty. A seated person is likely to remain stationary for the duration of the wheelchair's
passage; a smaller inflation radius is appropriate. Proposed convention:
- `person_sitting`: static obstacle, inflation radius = 0.6 m (minimum ADA clearance)
- `person_standing`: dynamic obstacle, inflation radius = 0.9 m (accounts for sudden movement)

**Trajectory prediction priority.** Trajectory prediction (the `predict_trajectory()` stub in
`pipeline.py`) is only meaningful for `person_standing`. Running prediction for seated persons
adds latency with no navigational benefit. After Phase 3, wire `predict_trajectory()` into the
scene state output, but only for tracks with `class == "person_standing"`.

**Scene state additions (proposed, for subsequent pipeline revision):**

```json
{
    "objects": [{
        "id": 3,
        "class": "person_sitting",
        "bbox_px": [120, 80, 200, 220],
        "centroid_px": [160, 150],
        "centroid_m": [1.2, 0.8],
        "velocity_px_per_frame": {"x": 0.1, "y": 0.0},
        "velocity_m_per_s": {"x": 0.02, "y": 0.0},
        "is_dynamic": false,
        "obstacle_inflation_m": 0.6,
        "zone": "seating_area",
        "trajectory_predicted": null,
        "approaching_camera": false,
        "confidence": 0.82,
        "masked_from_static_bg": false
    }]
}
```

The `centroid_m` and `velocity_m_per_s` fields require a homography calibration step
(Step 4.3) before they can be populated.

### Step 4.3 — Homography calibration for metric floor coordinates

A ceiling camera's pixel coordinates can be mapped to floor-plane metric coordinates via a
homography matrix `H` (3×3). Calibrate once per camera installation:

1. Place a calibration checkerboard (or known-distance markers) on the floor under the camera.
2. Record the pixel positions of the markers in the camera image.
3. Record the real-world positions of the markers in meters (measured with a tape measure).
4. Compute `H = cv2.findHomography(pixel_pts, world_pts)`.
5. Save `H` to `camera_config.json`.

At runtime, apply `cv2.perspectiveTransform(centroid_px, H)` to convert any pixel centroid
to floor-plane meters. This converts `nearest_person_px` to `nearest_person_m`, enabling
the wheelchair's motion planner to use metric distances for safety-critical decisions.

**This is a one-time per-installation calibration, not a model change.** It does not require
fine-tuning or retraining. It should be added to `pipeline.py` as an optional post-processing
step enabled when `camera_config.json` is present.

### Step 4.4 — Zone annotation

Annotate the camera's field of view with spatial zones corresponding to coffee shop layout
semantics. This is a one-time manual annotation:

1. Capture a reference frame from the coffee shop ceiling video.
2. In an image editor or CVAT, draw polygons for each zone:
   - `ordering_area` (near the counter)
   - `seating_area` (tables and chairs)
   - `staff_only` (behind counter, kitchen access)
   - `corridor` (walking path between zones)
3. Save zone polygons to `zone_config.json`.

At runtime, test each detected person's centroid against zone polygons using
`cv2.pointPolygonTest(zone_contour, centroid, measureDist=False)`.

**Success criterion for Phase 4:** The scene state specification document (`scene_state_v2.md`)
is written and reviewed. The additions in Steps 4.2–4.4 are marked as future implementation
tasks with clear interfaces (homography matrix format, zone config format, class-conditional
trajectory prediction).

---

## Compute Budget Summary

| Phase | Primary compute | Hardware | Estimated time |
|-------|----------------|----------|----------------|
| 0 — Baseline characterization | `run_video.py` on full video | Laptop CPU | 30–60 min per 30-min video |
| 0 — Frame annotation | Manual CVAT/Roboflow | Human time | 3–5 hrs (100–200 frames) |
| 1 — Dataset conversion + augmentation | Roboflow cloud | Cloud (free tier) | 1–2 hrs |
| 1 — Coffee shop annotation | Semi-auto CVAT + manual review | Human time | 4–6 hrs |
| 2 — Fine-tuning 100 epochs | `ultralytics train` | Colab T4 GPU | 1–2 hrs |
| 2 — Hyperparameter sweep (3 configs) | `ultralytics train` × 3 | Colab T4 GPU | 3–6 hrs |
| 3 — Re-evaluation | `run_video.py` + metric script | Laptop CPU | 1 hr |
| 4 — Calibration and zone annotation | Manual (one-time) | Human time | 1–2 hrs |
| **Total** | | | **~15–25 hrs elapsed, ~5–8 hrs Colab GPU** |

All Colab GPU usage fits within free tier limits (roughly 12 hrs/week on T4). The binding
constraint is human annotation time (Phase 0 and Phase 1.2), not compute.

---

## Dependencies and Tools

| Tool | Purpose | Access |
|------|---------|--------|
| `ultralytics` | YOLOv8n training and inference | `pip install ultralytics` (already in `requirements.txt`) |
| `roboflow` | Dataset management, augmentation, YOLO format export | `pip install roboflow`; free account at roboflow.com |
| CVAT | Manual annotation of coffee shop frames | cvat.ai (free cloud) or self-hosted |
| Google Colab T4 | Fine-tuning compute | colab.research.google.com (free tier) |
| Google Drive | Persisting weights across Colab sessions | Included with Google account |
| `cv2.findHomography` | Camera calibration for metric coordinates | Already in `requirements.txt` (`opencv-python`) |
| `opencv-python` | Zone polygon testing (`cv2.pointPolygonTest`) | Already in `requirements.txt` |

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Seated-person annotation volume too low (< 200 instances) | Medium | High — fine-tuned model underfits seated class | Supplement with publicly available indoor CCTV frames; ask coffee shop for a longer video clip |
| Fine-tuned model regresses on standing persons | Low | High — overall pipeline performance drops | Keep baseline `yolov8n.pt` as rollback; gate deployment on Phase 2 success criterion |
| Colab session timeout before training completes | Low | Medium — weights not saved | Always set `project` to a Google Drive path so Ultralytics saves checkpoint `.pt` files every epoch |
| Coffee shop video has poor quality (heavy compression, motion blur) | Low | Medium — annotation accuracy degrades | Pre-process video with `ffmpeg -vf eq=contrast=1.2:brightness=0.1` to improve visibility before annotation |
| Public datasets have inconsistent annotation quality | Medium | Medium — noisy labels degrade training | Run Ultralytics `val` on the public dataset portion before merging; discard splits with mAP@0.5 < 0.40 on their own test set |
| Fine-tuned model too slow on laptop CPU | Low | Low — inference path is identical YOLOv8n architecture | The fine-tuned model has the same number of parameters as the baseline; no FPS regression expected |
| `pipeline.py` L659 centroid bug corrupts Phase 3 velocity metrics | High (bug is confirmed) | Medium — velocity estimates wrong for multi-person scenes | Fix the bug before running Phase 3 (Step 3.5) |
