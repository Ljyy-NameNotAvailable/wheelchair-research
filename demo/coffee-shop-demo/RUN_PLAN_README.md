# Coffee-Shop Demo: Phased Research Automation Runner

An end-to-end, multi-phase pipeline for fine-tuning a person detector for
overhead (ceiling) camera footage.  Includes human annotation gates where
Roboflow is used to label ground-truth bounding boxes before training.

## Architecture

```
run_plan.py               ← main CLI entry point
phases/
  __init__.py
  common.py               ← shared paths, state file, logging helpers
  phase0_baseline.py      ← baseline metrics + Roboflow crop upload
  phase1_dataset.py       ← Oxford Town Centre download + Roboflow export + merge
  phase2_finetune.py      ← YOLOv8n fine-tune (Apple MPS or CPU)
  phase3_evaluate.py      ← re-evaluate + comparison report
annotations/              ← Roboflow upload staging area
datasets/                 ← downloaded and merged datasets
weights/                  ← model checkpoints
reports/                  ← output reports and videos
.plan_state.json          ← auto-generated progress tracker (do not edit manually)
```

## Requirements

Install dependencies before running any phase:

```bash
pip install -r requirements.txt
```

**requirements.txt** (in this directory) includes:

```
ultralytics>=8.0
roboflow>=1.1
torch>=2.0
torchvision>=0.15
opencv-python>=4.8
numpy>=1.24
tqdm>=4.65
pyyaml>=6.0
```

Python 3.10 or later is required.

---

## Usage

### Check current status

```bash
python run_plan.py status
```

Output:
```
  [✓] Phase 0 — Baseline characterization (completed 2026-04-12 14:32)
  [ ] Phase 1 — Dataset preparation
  [ ] Phase 2 — Fine-tuning
  [ ] Phase 3 — Evaluation

  Next: run phase1
```

---

### Phase 0 — Baseline Characterization

Samples frames from your video, runs the stock YOLOv8n pipeline on every
frame, extracts optical-flow crop hints, and uploads sampled frames to
Roboflow for human labelling.

```bash
python run_plan.py phase0 \
  --video /path/to/coffee-shop-video.mp4 \
  --roboflow-key YOUR_API_KEY \
  --roboflow-workspace YOUR_WORKSPACE_SLUG \
  --sample-every 10 \
  --device auto
```

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | (required) | Path to the coffee-shop ceiling camera video |
| `--roboflow-key` | (required) | Roboflow API key |
| `--roboflow-workspace` | `""` | Roboflow workspace slug |
| `--sample-every` | `10` | Sample one frame every N frames |
| `--device` | `auto` | Compute device: `auto`, `cpu`, `mps`, `cuda` |

**Human annotation gate (required before Phase 1):**

After Phase 0 completes, open the printed Roboflow URL and label each person
in the uploaded frames with one of:
- `person_sitting` — seated at a table or chair
- `person_standing` — standing or walking

Generate at least one dataset version in the Roboflow UI before running Phase 1.

---

### Phase 1 — Dataset Preparation

Exports Roboflow annotations (YOLOv8 format), downloads the Oxford Town Centre
pedestrian dataset (~700 MB), converts it to YOLO format, and merges both into
`datasets/merged/` with an 80/20 train/val split.

```bash
python run_plan.py phase1 \
  --roboflow-key YOUR_API_KEY \
  --roboflow-project overhead-person-detection \
  --roboflow-workspace YOUR_WORKSPACE_SLUG
```

To skip the Oxford download (faster for testing):
```bash
python run_plan.py phase1 \
  --roboflow-key YOUR_API_KEY \
  --roboflow-project overhead-person-detection \
  --roboflow-workspace YOUR_WORKSPACE_SLUG \
  --skip-oxford
```

| Flag | Default | Description |
|------|---------|-------------|
| `--roboflow-key` | (required) | Roboflow API key |
| `--roboflow-project` | (required) | Roboflow project slug |
| `--roboflow-workspace` | (required) | Roboflow workspace slug |
| `--skip-oxford` | false | Skip Oxford Town Centre download |

---

### Phase 2 — Fine-Tuning

Fine-tunes YOLOv8n on the merged dataset, starting from COCO pretrained
weights.  Automatically selects the best available device (MPS > CUDA > CPU).

```bash
python run_plan.py phase2 --epochs 50 --imgsz 640
```

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `50` | Number of training epochs |
| `--imgsz` | `640` | Input image size |
| `--device` | `auto` | `auto`, `mps`, `cuda`, or `cpu` |
| `--batch` | `16` | Batch size (reduce to 8 on memory-limited CPUs) |

Expected training time:
- Apple MPS: ~45–90 min for 50 epochs / ~900 images
- CPU: ~3–6 hours

---

### Phase 3 — Evaluation

Runs both the baseline and fine-tuned pipelines on the same video, generates
a comparison report, and produces a side-by-side annotated video.

```bash
python run_plan.py phase3 --video /path/to/coffee-shop-video.mp4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | (required) | Same video used in Phase 0 |
| `--device` | `auto` | Compute device |
| `--with-gt` | false | [TODO] Ground-truth evaluation (not yet implemented) |

---

## Inputs

| Phase | Required Inputs |
|-------|----------------|
| Phase 0 | Coffee-shop video file |
| Phase 1 | Roboflow API key; at least one annotated dataset version in Roboflow UI |
| Phase 2 | `datasets/merged/data.yaml` (Phase 1 output) |
| Phase 3 | `reports/baseline_raw.jsonl` (Phase 0) + `weights/yolov8n_overhead.pt` (Phase 2) + video |

---

## Outputs

| File | Phase | Description |
|------|-------|-------------|
| `annotations/crop_hints/frames/` | 0 | Sampled frames uploaded to Roboflow |
| `annotations/crop_hints/crops/` | 0 | Optical-flow region crops (reference) |
| `reports/baseline_raw.jsonl` | 0 | Per-frame baseline scene states |
| `reports/baseline_proxy_metrics.md` | 0 | Proxy detection metrics (no ground truth) |
| `datasets/coffee_shop/` | 1 | Roboflow export in YOLOv8 format |
| `datasets/oxford/` | 1 | Oxford Town Centre in YOLO format |
| `datasets/merged/` | 1 | Merged train/val dataset + `data.yaml` |
| `reports/dataset_summary.md` | 1 | Dataset statistics |
| `weights/overhead_run/` | 2 | Ultralytics training run directory |
| `weights/yolov8n_overhead.pt` | 2 | Final fine-tuned model weights |
| `reports/training_summary.md` | 2 | mAP metrics + device info |
| `reports/finetuned_raw.jsonl` | 3 | Per-frame fine-tuned scene states |
| `reports/comparison_report.md` | 3 | Baseline vs fine-tuned metrics table |
| `reports/comparison.mp4` | 3 | Side-by-side annotated video |
| `.plan_state.json` | All | Progress tracker (auto-managed) |

---

## Notes for Tester Agent

### Resetting state
Delete `.plan_state.json` to reset all phase completion records:
```bash
rm demo/coffee-shop-demo/.plan_state.json
```

### Running without a Roboflow account
Phase 0's upload step will raise a `RuntimeError` if the API key is invalid.
All prior steps in Phase 0 (frame sampling, pipeline run, proxy metrics,
crop extraction) complete and save their outputs before the upload step runs.

Phase 1 requires a valid Roboflow export.  Without one, use `--skip-oxford`
to at least verify dataset merging logic works with a manually placed
`datasets/coffee_shop/` directory.

### Mock testing without a video
Each phase step function accepts typed inputs and can be called in isolation
without going through `run_plan.py`.  Example:

```python
from phases.phase0_baseline import compute_proxy_metrics
from pathlib import Path
compute_proxy_metrics(Path("reports/baseline_raw.jsonl"))
```

### Known edge cases and failure modes

| Situation | Behaviour |
|-----------|-----------|
| RAFT out-of-memory | Auto fallback to CPU; `warnings.warn` printed |
| Oxford Town Centre URL returns 403 | `urllib.request` raises `HTTPError`; place files manually in `datasets/oxford_raw/` |
| Roboflow project already exists | SDK `workspace.project()` reconnects; no duplicate created |
| `--with-gt` flag in Phase 3 | Raises `NotImplementedError` by design (spec marks as TODO) |
| `blur=0.1` Ultralytics parameter | Replaced with `blur_limit=1` (see `SPEC_AMBIGUITY` comment in `phase2_finetune.py`) |
| Oxford annotations with no full-body bbox | Rows with invalid float values are skipped during CSV parsing |

### Suggested smoke test (no video, no Roboflow)

```bash
# 1. Confirm CLI entry point works
python run_plan.py status

# 2. Confirm phase modules import correctly
python -c "from phases import phase0_baseline, phase1_dataset, phase2_finetune, phase3_evaluate; print('imports OK')"

# 3. Confirm common utilities
python -c "
from phases.common import load_state, save_state, mark_phase_done
s = load_state()
print('state loaded:', s)
"
```
