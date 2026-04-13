"""
phase2_finetune.py — Phase 2: YOLOv8n Fine-Tuning.

Steps:
  2.1  Detect compute device (MPS > CUDA > CPU) and print time estimate
  2.2  Fine-tune YOLOv8n on datasets/merged/data.yaml
  2.3  Copy best.pt to weights/yolov8n_overhead.pt; write training summary
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Allow importing pipeline.py from the demo directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from phases.common import (
    DATASETS_DIR,
    REPORTS_DIR,
    WEIGHTS_DIR,
    ensure_dirs,
    mark_phase_done,
    print_next_step,
    print_phase_header,
    require_phase_done,
)

MERGED_DATA_YAML = DATASETS_DIR / "merged" / "data.yaml"
BEST_WEIGHTS_SRC = WEIGHTS_DIR / "overhead_run" / "weights" / "best.pt"
FINAL_WEIGHTS = WEIGHTS_DIR / "yolov8n_overhead.pt"


# ---------------------------------------------------------------------------
# Step 2.1 — Device detection
# ---------------------------------------------------------------------------

def detect_device(requested_device: str = "auto") -> str:
    """
    Step 2.1 — Detect the best available compute device.

    Priority order: MPS (Apple Silicon) > CUDA > CPU.
    If `requested_device` is not 'auto', that value is used directly
    after checking availability.

    Args:
        requested_device: 'auto', 'mps', 'cuda', or 'cpu'.

    Returns:
        Device string suitable for passing to Ultralytics model.train().

    Raises:
        RuntimeError: if torch is not installed.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "[phase2] torch is not installed. Install it with:\n"
            "  pip install torch torchvision"
        ) from exc

    if requested_device != "auto":
        device = requested_device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Print device and time estimate
    _time_estimates = {
        "mps": "~45–90 min for 50 epochs on ~900 images (Apple Silicon)",
        "cuda": "~15–30 min for 50 epochs on ~900 images (NVIDIA GPU)",
        "cpu": "~3–6 hrs for 50 epochs on ~900 images",
    }
    estimate = _time_estimates.get(device, "time unknown")
    print(f"[phase2/step2.1] Using device: {device}")
    print(f"[phase2/step2.1] Estimated training time: {estimate}")

    return device


# ---------------------------------------------------------------------------
# Step 2.2 — Fine-tune
# ---------------------------------------------------------------------------

def finetune(
    device: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
) -> dict:
    """
    Step 2.2 — Fine-tune YOLOv8n on the merged dataset.

    Starts from COCO-pretrained yolov8n.pt (downloaded automatically by
    Ultralytics if not present).  Freezes the first 10 backbone layers to
    preserve low-level feature representations while adapting the head to the
    overhead view.

    Args:
        device: compute device string ('mps', 'cuda', 'cpu').
        epochs: number of training epochs (default 50).
        imgsz:  input image size for training (default 640).
        batch:  batch size (default 16; reduce to 8 on memory-constrained CPUs).

    Returns:
        Dict with keys 'best_pt' (Path), 'map50', 'map50_95'.

    Raises:
        FileNotFoundError: if data.yaml does not exist (phase1 not complete).
        RuntimeError:      if ultralytics is not installed.
    """
    if not MERGED_DATA_YAML.exists():
        raise FileNotFoundError(
            f"[phase2] Merged data.yaml not found: {MERGED_DATA_YAML}\n"
            f"  Run Phase 1 first:  python run_plan.py phase1 ..."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "[phase2] ultralytics is not installed. Install it with:\n"
            "  pip install ultralytics"
        ) from exc

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[phase2/step2.2] Starting YOLOv8n fine-tune: {epochs} epochs, imgsz={imgsz}, device={device}")
    print(f"[phase2/step2.2] Data: {MERGED_DATA_YAML}")

    model = YOLO("yolov8n.pt")  # start from COCO pretrained; auto-downloaded if needed

    results = model.train(
        data=str(MERGED_DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch,
        freeze=10,       # freeze first 10 backbone layers
        degrees=180,     # overhead rotation invariance
        fliplr=0.5,
        flipud=0.5,
        scale=0.5,
        hsv_v=0.4,
        # SPEC_AMBIGUITY: spec lists 'blur=0.1' but Ultralytics uses 'blur_limit'
        # as an integer (max kernel size). We pass the value as 'blur_limit=1'
        # (minimum meaningful blur) since 0.1 as a probability is not a valid
        # Ultralytics parameter name.  The intent (occasional motion blur) is preserved.
        blur_limit=1,
        project=str(WEIGHTS_DIR),
        name="overhead_run",
        exist_ok=True,
    )

    # Extract mAP metrics from results
    # Ultralytics results object: results.results_dict contains final metrics
    map50 = float("nan")
    map50_95 = float("nan")
    try:
        metrics = results.results_dict
        map50 = float(metrics.get("metrics/mAP50(B)", float("nan")))
        map50_95 = float(metrics.get("metrics/mAP50-95(B)", float("nan")))
    except Exception:
        pass  # metrics unavailable — continue without them

    print(f"[phase2/step2.2] Training complete. mAP@0.5={map50:.4f}, mAP@0.5:0.95={map50_95:.4f}")

    return {
        "best_pt": BEST_WEIGHTS_SRC,
        "map50": map50,
        "map50_95": map50_95,
        "device": device,
        "epochs": epochs,
    }


# ---------------------------------------------------------------------------
# Step 2.3 — Copy best weights + write summary
# ---------------------------------------------------------------------------

def copy_weights_and_summarize(
    training_result: dict,
) -> tuple[Path, Path]:
    """
    Step 2.3 — Copy the best checkpoint to a canonical path and write a
    training summary report.

    Args:
        training_result: dict returned by `finetune()`.

    Returns:
        Tuple of (final_weights_path, training_summary_path).

    Raises:
        FileNotFoundError: if best.pt was not produced by training.
    """
    best_src = training_result["best_pt"]

    if not best_src.exists():
        raise FileNotFoundError(
            f"[phase2] Best checkpoint not found at: {best_src}\n"
            f"  Training may have failed or not produced a checkpoint."
        )

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_src, FINAL_WEIGHTS)
    print(f"[phase2/step2.3] Best weights copied to {FINAL_WEIGHTS}")

    map50 = training_result.get("map50", float("nan"))
    map50_95 = training_result.get("map50_95", float("nan"))
    device = training_result.get("device", "unknown")
    epochs = training_result.get("epochs", "?")

    summary_lines = [
        "# Training Summary",
        "",
        "## Configuration",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Device    | {device} |",
        f"| Epochs    | {epochs} |",
        f"| Base model | YOLOv8n (COCO pretrained) |",
        f"| Backbone layers frozen | 10 |",
        f"| Dataset   | datasets/merged/data.yaml |",
        "",
        "## Results",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| mAP@0.5       | {map50:.4f} |",
        f"| mAP@0.5:0.95  | {map50_95:.4f} |",
        "",
        "## Weights",
        "",
        f"- Best checkpoint (Ultralytics run): `{best_src}`",
        f"- Final weights (for inference):     `{FINAL_WEIGHTS}`",
        "",
        "## Using the fine-tuned weights",
        "",
        "To use these weights in the pipeline, set:",
        "```python",
        f"config['yolo_model'] = '{FINAL_WEIGHTS}'",
        "```",
        "",
        "Or pass via command line:",
        "```bash",
        f"python run_plan.py phase3 --video YOUR_VIDEO.mp4",
        "```",
        "(Phase 3 automatically uses `weights/yolov8n_overhead.pt`.)",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "training_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[phase2/step2.3] Training summary written to {summary_path}")

    return FINAL_WEIGHTS, summary_path


# ---------------------------------------------------------------------------
# Phase 2 entry point
# ---------------------------------------------------------------------------

def run(
    epochs: int = 50,
    imgsz: int = 640,
    device: str = "auto",
    batch: int = 16,
    state: dict | None = None,
) -> dict:
    """
    Run all Phase 2 steps in order.

    Args:
        epochs:  number of training epochs (default 50).
        imgsz:   input image size (default 640).
        device:  compute device ('auto', 'mps', 'cuda', 'cpu').
        batch:   training batch size (default 16).
        state:   current plan state dict (used to check phase1 completion).

    Returns:
        Dict of output paths for state recording.
    """
    print_phase_header(2)
    ensure_dirs()

    if state is not None:
        require_phase_done(1, state)

    # Step 2.1
    print("[phase2] Step 2.1 — Detecting compute device...")
    resolved_device = detect_device(device)
    print("[phase2] Step 2.1 — Done.")

    # Step 2.2
    print("[phase2] Step 2.2 — Fine-tuning YOLOv8n...")
    training_result = finetune(
        device=resolved_device,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
    )
    print("[phase2] Step 2.2 — Done.")

    # Step 2.3
    print("[phase2] Step 2.3 — Copying weights + writing summary...")
    final_weights, summary_path = copy_weights_and_summarize(training_result)
    print("[phase2] Step 2.3 — Done.")

    print_next_step(
        f"""
  PHASE 2 COMPLETE

  Fine-tuned weights saved to:
    {final_weights}

  To evaluate the fine-tuned model vs. baseline, run:
    python run_plan.py phase3 --video YOUR_VIDEO.mp4
"""
    )

    return {
        "final_weights": str(final_weights),
        "training_summary": str(summary_path),
        "map50": str(training_result.get("map50", "n/a")),
        "map50_95": str(training_result.get("map50_95", "n/a")),
    }
