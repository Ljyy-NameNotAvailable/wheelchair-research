# Coffee Shop Dynamic Perception — Minimum Viable Demo

A laptop/Colab-ready pipeline that processes video from a **static ceiling
camera** to detect, track, and characterise moving people using:

- **YOLOv8n + ByteTrack** — person detection and multi-object tracking
- **RAFT-Small** — dense optical flow for motion detection independent of
  classification
- **OpenCV** — video I/O and annotation

No ROS2, no TensorRT, no depth sensors, no hardware dependencies.

---

## Requirements

Python 3.10 or later. Install all dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` pins:

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8n detection + ByteTrack tracking |
| `torchvision` | RAFT-Small optical flow |
| `torch` | PyTorch backend |
| `opencv-python` | Video I/O, drawing |
| `numpy` | Array operations |
| `tqdm` | Progress bar in `run_video.py` |

On first run `ultralytics` will download `yolov8n.pt` (~6 MB) automatically.
RAFT-Small weights (~20 MB) are downloaded by `torchvision` on first use.

---

## Local Usage

### Basic run

```bash
python run_video.py --input video.mp4 --output out.mp4
```

### With JSONL scene log and live preview

```bash
python run_video.py \
    --input  coffee_shop.mp4 \
    --output annotated.mp4   \
    --log    scene_log.jsonl \
    --show
```

Press `q` in the preview window to stop early.

### Force CPU (no CUDA)

```bash
python run_video.py --input video.mp4 --output out.mp4 --device cpu
```

### All options

| Flag | Default | Description |
|---|---|---|
| `--input` | (required) | Path to input video |
| `--output` | `output.mp4` | Path to annotated output video |
| `--flow-interval` | `5` | Compute RAFT flow every N frames |
| `--theta` | `2.0` | Flow magnitude threshold for moving pixels (px/frame) |
| `--theta-stop` | `8.0` | Flow magnitude threshold for fast-moving pixels |
| `--log` | (none) | Path to JSONL scene log file |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--show` | (flag) | Show live preview window while processing |

---

## Google Colab Quick-Start

1. Open `colab_demo.ipynb` in Colab.
2. Set runtime to **T4 GPU** (Runtime -> Change runtime type).
3. Run **Cell 1** to install dependencies.
4. Run **Cell 2** to mount Google Drive and set `INPUT_VIDEO` path,
   OR set `USE_DRIVE = False` to upload a file directly.
5. Run **Cell 3** to clone the repo (update `REPO_URL` first).
6. Run **Cell 4** to process the full video.
7. Run **Cell 5** to view the first 5 annotated frames inline.
8. Run **Cell 6** to display the first 10 rows of the JSONL scene log as a table.

Expected throughput: **2-4 fps** on a T4 GPU for 1280x720 input.

---

## Inputs

The pipeline expects a single BGR video file readable by OpenCV
(`cv2.VideoCapture`). Common formats: MP4, AVI, MOV.

`pipeline.py` itself is I/O-free — it accepts raw `numpy.ndarray` frames and
returns a `FrameResult` dataclass. You can unit-test it without a video file:

```python
import numpy as np
from pipeline import DynamicPerceptionPipeline

pipeline = DynamicPerceptionPipeline()
dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
result = pipeline.process_frame(dummy_frame, frame_idx=0, fps=30.0)
print(result.scene_state)
```

---

## Outputs

### Annotated video (`--output`)

Each frame contains:
- **Green boxes** — tracked persons, labelled with track ID and confidence.
  A velocity arrow extends from the centroid.
- **Yellow dashed boxes** — unclassified motion regions (optical flow without
  a matching YOLO detection).
- **Red dashed boxes** — unclassified regions whose mean flow exceeds
  `theta_stop` (fast-moving objects).
- **Colour heatmap overlay** — optical flow magnitude at 30% opacity, only
  where flow exceeds `theta`.
- **Frame counter + processing FPS** in top-left corner.

### Scene log (`--log`, JSONL)

One JSON object per line. Schema:

```json
{
  "frame_idx": 42,
  "timestamp_s": 1.4,
  "objects": [
    {
      "id": 3,
      "class": "person",
      "bbox_px": [120, 80, 200, 220],
      "centroid_px": [160, 150],
      "velocity_px_per_frame": {"x": -2.1, "y": 0.8},
      "approaching_camera": false,
      "confidence": 0.87,
      "masked_from_static_bg": true
    }
  ],
  "unclassified_motion_regions": [
    {
      "bbox_px": [300, 100, 380, 180],
      "area_px2": 6400,
      "mean_flow_magnitude": 9.2,
      "above_stop_threshold": true
    }
  ],
  "flow_computed_this_frame": true,
  "dynamic_pixel_count": 8200,
  "nearest_person_px": 143.2
}
```

`nearest_person_px` is the Euclidean distance in pixels from the frame centre
to the nearest tracked person centroid (proxy for camera nadir distance).

---

## Architecture Notes

### Why no ego-motion compensation?

The camera is fixed to the ceiling. Every non-zero optical flow vector comes
from a moving object — there is no camera motion to subtract. This eliminates
the hardest part of the full wheelchair pipeline (SLAM + ego-motion removal).

### Overhead camera detection quality

YOLOv8n is trained on COCO eye-level images. People viewed from a ceiling
camera appear as small ovals rather than upright figures — detection confidence
will be measurably lower than in standard forward-facing use. The optical flow
layer compensates: even if YOLO misses a person, they will appear in
`unclassified_motion_regions`. A future improvement would be fine-tuning on an
overhead pedestrian dataset such as **VIRAT** or **Oxford Town Centre**.

### Flow cadence

RAFT-Small is run every `flow_interval` frames (default: 5). Between flow
calls the last computed `flow_mask` is carried forward unchanged. This is safe
for a static camera because the background is constant — there is no camera
motion to invalidate the previous mask.

### Trajectory prediction stub

`DynamicPerceptionPipeline.predict_trajectory(track_history)` implements
linear-velocity extrapolation. It is called internally each frame but its
output is not yet wired into `scene_state`. A Social LSTM or Social GAN model
can replace the extrapolation by implementing the same interface.

---

## Notes for Tester Agent

- **No video file is bundled.** Provide any MP4 with people visible from
  above. A 30-second clip at 720p is sufficient for a functional test.
  Free sources: VIRAT, Oxford Town Centre, or any surveillance-style
  overhead footage.

- **First-frame edge case:** Frame 0 has no previous frame, so no optical
  flow is computed. `flow_computed_this_frame` will be `false` and
  `unclassified_motion_regions` will be empty. This is expected behaviour.

- **No detections:** If YOLO finds no persons (empty scene, poor overhead
  angle), `objects` will be `[]` and `dynamic_mask` will be all-zero until
  the first flow computation at frame `flow_interval`.

- **CUDA OOM:** If CUDA runs out of memory during RAFT, the pipeline falls
  back to CPU automatically with a `warnings.warn` message. No exception
  is raised.

- **ByteTrack ID gaps:** Track IDs are assigned by Ultralytics ByteTrack.
  IDs are not guaranteed to be contiguous — gaps are normal when tracks are
  lost and re-acquired.

- **Colab cell 3 requires a valid `REPO_URL`.** Update it before running, or
  manually copy `pipeline.py` to `/content/` and set
  `sys.path.insert(0, '/content')`.

- **Suggested smoke test (CPU, no video required):**

  ```python
  import numpy as np
  from pipeline import DynamicPerceptionPipeline

  p = DynamicPerceptionPipeline({"device": "cpu", "flow_interval": 2})
  rng = np.random.default_rng(0)
  for i in range(6):
      frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
      r = p.process_frame(frame, i, fps=30.0)
      assert r.dynamic_mask.shape == (480, 640)
      assert "frame_idx" in r.scene_state
  print("Smoke test passed.")
  ```

  Flow is computed at frames 0, 2, 4 (every 2 frames per config). No YOLO
  detections expected on random noise — verifies mask shapes and schema only.
