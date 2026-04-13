"""
phase3_evaluate.py — Phase 3: Evaluation + Comparison Report.

Steps:
  3.1  Run fine-tuned DynamicPerceptionPipeline on the same video → finetuned_raw.jsonl
  3.2  Compare baseline vs fine-tuned metrics frame-by-frame → comparison_report.md
  3.3  Produce side-by-side annotated video → comparison.mp4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Allow importing pipeline.py from the demo directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from phases.common import (
    REPORTS_DIR,
    WEIGHTS_DIR,
    ensure_dirs,
    mark_phase_done,
    print_next_step,
    print_phase_header,
    require_phase_done,
)

BASELINE_JSONL = REPORTS_DIR / "baseline_raw.jsonl"
FINETUNED_JSONL = REPORTS_DIR / "finetuned_raw.jsonl"
FINAL_WEIGHTS = WEIGHTS_DIR / "yolov8n_overhead.pt"
COMPARISON_REPORT = REPORTS_DIR / "comparison_report.md"
COMPARISON_VIDEO = REPORTS_DIR / "comparison.mp4"


# ---------------------------------------------------------------------------
# Step 3.1 — Run fine-tuned pipeline
# ---------------------------------------------------------------------------

def run_finetuned_pipeline(
    video_path: Path,
    device: str,
) -> Path:
    """
    Step 3.1 — Run DynamicPerceptionPipeline with the fine-tuned weights.

    Uses weights/yolov8n_overhead.pt.  Saves per-frame scene_state to
    reports/finetuned_raw.jsonl.

    Args:
        video_path: path to the coffee-shop ceiling-camera video.
        device:     compute device string ('auto', 'cpu', 'mps').

    Returns:
        Path to the written finetuned_raw.jsonl.

    Raises:
        FileNotFoundError: if the video or fine-tuned weights do not exist.
        RuntimeError:      if the video cannot be opened.
    """
    from pipeline import DEFAULT_CONFIG, DynamicPerceptionPipeline  # noqa: E402

    if not video_path.exists():
        raise FileNotFoundError(f"[phase3] Video not found: {video_path}")
    if not FINAL_WEIGHTS.exists():
        raise FileNotFoundError(
            f"[phase3] Fine-tuned weights not found: {FINAL_WEIGHTS}\n"
            f"  Run Phase 2 first:  python run_plan.py phase2"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[phase3] Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(
        f"[phase3/step3.1] Running fine-tuned pipeline on {total} frames "
        f"(device={device}, weights={FINAL_WEIGHTS.name})..."
    )

    config = {
        **DEFAULT_CONFIG,
        "device": device,
        "yolo_model": str(FINAL_WEIGHTS),
    }
    pipeline = DynamicPerceptionPipeline(config)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
        progress = tqdm(total=total, unit="frame", desc="Fine-tuned pipeline")
    except ImportError:
        progress = None
        print("[phase3/step3.1] tqdm not found — no progress bar.")

    with open(FINETUNED_JSONL, "w", encoding="utf-8") as fh:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = pipeline.process_frame(frame, frame_idx, src_fps=src_fps)
            fh.write(json.dumps(result.scene_state) + "\n")
            frame_idx += 1
            if progress is not None:
                progress.update(1)
            elif frame_idx % 50 == 0:
                print(f"[phase3/step3.1] {frame_idx}/{total} frames")

    if progress is not None:
        progress.close()
    cap.release()

    print(f"[phase3/step3.1] Fine-tuned log written to {FINETUNED_JSONL}")
    return FINETUNED_JSONL


# ---------------------------------------------------------------------------
# Metric extraction helpers
# ---------------------------------------------------------------------------

def _load_jsonl_metrics(jsonl_path: Path) -> dict[str, list[float]]:
    """
    Load per-frame metrics from a scene_state JSONL file.

    Returns a dict with keys:
      - 'objects_detected_per_frame'
      - 'unclassified_motion_regions_per_frame'
      - 'mean_detection_confidence'
      - 'dynamic_pixel_count'
    """
    metrics: dict[str, list[float]] = {
        "objects_detected_per_frame": [],
        "unclassified_motion_regions_per_frame": [],
        "mean_detection_confidence": [],
        "dynamic_pixel_count": [],
    }

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            state = json.loads(line)

            objs = state.get("objects", [])
            metrics["objects_detected_per_frame"].append(float(len(objs)))

            n_unclass = len(state.get("unclassified_motion_regions", []))
            metrics["unclassified_motion_regions_per_frame"].append(float(n_unclass))

            confs = [o.get("confidence", 0.0) for o in objs]
            metrics["mean_detection_confidence"].append(
                float(np.mean(confs)) if confs else float("nan")
            )

            metrics["dynamic_pixel_count"].append(
                float(state.get("dynamic_pixel_count", 0))
            )

    return metrics


def _summarise(values: list[float]) -> tuple[float, float]:
    """Return (mean, std) ignoring NaN values."""
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


# ---------------------------------------------------------------------------
# Step 3.2 — Comparison metrics
# ---------------------------------------------------------------------------

def compute_comparison_report(
    baseline_jsonl: Path,
    finetuned_jsonl: Path,
) -> Path:
    """
    Step 3.2 — Compare baseline vs fine-tuned pipeline outputs.

    Computes mean ± std for each metric across all frames and writes
    reports/comparison_report.md.

    Args:
        baseline_jsonl:  path to baseline_raw.jsonl.
        finetuned_jsonl: path to finetuned_raw.jsonl.

    Returns:
        Path to the written comparison_report.md.

    Raises:
        FileNotFoundError: if either JSONL file is missing.
    """
    for p in (baseline_jsonl, finetuned_jsonl):
        if not p.exists():
            raise FileNotFoundError(f"[phase3] Required file not found: {p}")

    print("[phase3/step3.2] Computing comparison metrics...")

    base_metrics = _load_jsonl_metrics(baseline_jsonl)
    fine_metrics = _load_jsonl_metrics(finetuned_jsonl)

    _display_names = {
        "objects_detected_per_frame": "Objects detected / frame",
        "unclassified_motion_regions_per_frame": "Unclassified motion regions / frame",
        "mean_detection_confidence": "Mean detection confidence",
        "dynamic_pixel_count": "Dynamic pixel count / frame",
    }

    rows: list[str] = []
    for key, display in _display_names.items():
        bm, bs = _summarise(base_metrics[key])
        fm, fs = _summarise(fine_metrics[key])
        rows.append(
            f"| {display} | {bm:.3f} ± {bs:.3f} | {fm:.3f} ± {fs:.3f} |"
        )

    # Interpretation: did unclassified regions decrease?
    base_unclass_mean, _ = _summarise(base_metrics["unclassified_motion_regions_per_frame"])
    fine_unclass_mean, _ = _summarise(fine_metrics["unclassified_motion_regions_per_frame"])

    if not np.isnan(base_unclass_mean) and not np.isnan(fine_unclass_mean):
        if fine_unclass_mean < base_unclass_mean:
            interp = (
                f"Unclassified motion regions decreased from "
                f"{base_unclass_mean:.2f} to {fine_unclass_mean:.2f} per frame "
                f"({100*(base_unclass_mean - fine_unclass_mean)/max(base_unclass_mean, 1e-9):.1f}% reduction). "
                f"This indicates the fine-tuned model is catching more overhead-view people."
            )
        elif fine_unclass_mean > base_unclass_mean:
            interp = (
                f"Unclassified motion regions increased slightly from "
                f"{base_unclass_mean:.2f} to {fine_unclass_mean:.2f} per frame. "
                f"This may indicate the fine-tuned model has a stricter confidence threshold "
                f"or the training data was insufficient. Consider more annotated frames."
            )
        else:
            interp = (
                f"Unclassified motion regions unchanged ({fine_unclass_mean:.2f}/frame). "
                f"Fine-tuning had no measurable effect on optical-flow misses."
            )
    else:
        interp = "Insufficient data to interpret unclassified region change."

    n_base = len(base_metrics["objects_detected_per_frame"])
    n_fine = len(fine_metrics["objects_detected_per_frame"])

    report_lines = [
        "# Comparison Report: Baseline vs Fine-Tuned",
        "",
        f"Baseline frames: {n_base} | Fine-tuned frames: {n_fine}",
        "",
        "## Metrics Summary",
        "",
        "| Metric | Baseline (mean ± std) | Fine-Tuned (mean ± std) |",
        "|--------|----------------------|------------------------|",
    ] + rows + [
        "",
        "## Interpretation",
        "",
        interp,
        "",
        "## Limitation",
        "",
        "Without ground truth bounding boxes, **recall cannot be measured**. The",
        "`unclassified_motion_regions` count is a proxy metric only:",
        "- It counts optical-flow regions with no YOLO match.",
        "- A decrease could mean the fine-tuned model detects more people, OR that",
        "  it confidently detects regions the baseline missed entirely.",
        "- True precision and recall require annotated test frames.",
        "",
        "## Next Steps",
        "",
        "To obtain ground truth evaluation:",
        "1. Annotate 30 test frames in Roboflow.",
        "2. Export as YOLOv8 format.",
        "3. Run (stub — TODO):",
        "   ```bash",
        "   python run_plan.py phase3 --with-gt --gt-dir datasets/test_gt/",
        "   ```",
        "",
        "The `--with-gt` flag is not yet implemented. See `phase3_evaluate.py`.",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[phase3/step3.2] Comparison report written to {COMPARISON_REPORT}")
    return COMPARISON_REPORT


# ---------------------------------------------------------------------------
# Step 3.3 — Side-by-side annotated video
# ---------------------------------------------------------------------------

def produce_comparison_video(
    video_path: Path,
    device: str,
) -> Path:
    """
    Step 3.3 — Produce a side-by-side comparison video.

    Left panel:  baseline detections (yolov8n.pt from pipeline DEFAULT_CONFIG)
    Right panel: fine-tuned detections (yolov8n_overhead.pt)

    Both panels are rendered by DynamicPerceptionPipeline and concatenated
    horizontally using OpenCV.  Saved to reports/comparison.mp4.

    Args:
        video_path: path to the coffee-shop ceiling-camera video.
        device:     compute device string.

    Returns:
        Path to the output comparison.mp4.

    Raises:
        FileNotFoundError: if the video or fine-tuned weights do not exist.
    """
    from pipeline import DEFAULT_CONFIG, DynamicPerceptionPipeline  # noqa: E402

    if not video_path.exists():
        raise FileNotFoundError(f"[phase3] Video not found: {video_path}")
    if not FINAL_WEIGHTS.exists():
        raise FileNotFoundError(
            f"[phase3] Fine-tuned weights not found: {FINAL_WEIGHTS}\n"
            f"  Run Phase 2 first."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[phase3] Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(
        f"[phase3/step3.3] Producing side-by-side comparison video "
        f"({total} frames, {frame_w}x{frame_h})..."
    )

    # Initialise both pipelines
    base_config = {**DEFAULT_CONFIG, "device": device}
    fine_config = {**DEFAULT_CONFIG, "device": device, "yolo_model": str(FINAL_WEIGHTS)}

    print("[phase3/step3.3] Loading baseline pipeline...")
    pipeline_base = DynamicPerceptionPipeline(base_config)
    print("[phase3/step3.3] Loading fine-tuned pipeline...")
    pipeline_fine = DynamicPerceptionPipeline(fine_config)

    # Output video: double width
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(COMPARISON_VIDEO), fourcc, src_fps, (frame_w * 2, frame_h)
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"[phase3] Cannot open video writer: {COMPARISON_VIDEO}"
        )

    try:
        from tqdm import tqdm
        progress = tqdm(total=total, unit="frame", desc="Comparison video")
    except ImportError:
        progress = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_base = pipeline_base.process_frame(frame.copy(), frame_idx, src_fps=src_fps)
        result_fine = pipeline_fine.process_frame(frame.copy(), frame_idx, src_fps=src_fps)

        # Add panel labels
        left = result_base.annotated_frame.copy()
        right = result_fine.annotated_frame.copy()

        cv2.putText(
            left, "Baseline (YOLOv8n COCO)", (10, frame_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
        )
        cv2.putText(
            right, "Fine-tuned (overhead)", (10, frame_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2,
        )

        combined = np.concatenate([left, right], axis=1)
        writer.write(combined)

        frame_idx += 1
        if progress is not None:
            progress.update(1)
        elif frame_idx % 50 == 0:
            print(f"[phase3/step3.3] {frame_idx}/{total} frames")

    if progress is not None:
        progress.close()
    cap.release()
    writer.release()

    print(f"[phase3/step3.3] Side-by-side video written to {COMPARISON_VIDEO}")
    return COMPARISON_VIDEO


# ---------------------------------------------------------------------------
# Phase 3 entry point
# ---------------------------------------------------------------------------

def run(
    video_path: Path,
    device: str = "auto",
    with_gt: bool = False,
    state: dict | None = None,
) -> dict:
    """
    Run all Phase 3 steps in order.

    Args:
        video_path:  path to the coffee-shop video (same one used in Phase 0).
        device:      compute device ('auto', 'cpu', 'mps').
        with_gt:     if True, run ground-truth evaluation (TODO — stub only).
        state:       current plan state dict.

    Returns:
        Dict of output paths for state recording.
    """
    print_phase_header(3)
    ensure_dirs()

    if state is not None:
        require_phase_done(0, state)
        require_phase_done(2, state)

    if with_gt:
        # TODO: implement ground truth evaluation when annotated test set is available.
        raise NotImplementedError(
            "[phase3] --with-gt is not yet implemented.\n"
            "  To use it: annotate 30 test frames in Roboflow, export as YOLOv8 format,\n"
            "  then implement GT comparison in phase3_evaluate.py."
        )

    # Step 3.1
    print("[phase3] Step 3.1 — Running fine-tuned pipeline...")
    run_finetuned_pipeline(video_path, device)
    print("[phase3] Step 3.1 — Done.")

    # Step 3.2
    print("[phase3] Step 3.2 — Computing comparison metrics...")
    report_path = compute_comparison_report(BASELINE_JSONL, FINETUNED_JSONL)
    print("[phase3] Step 3.2 — Done.")

    # Step 3.3
    print("[phase3] Step 3.3 — Producing side-by-side video...")
    video_out = produce_comparison_video(video_path, device)
    print("[phase3] Step 3.3 — Done.")

    print_next_step(
        f"""
  PHASE 3 COMPLETE

  Outputs:
    Comparison report: {report_path}
    Side-by-side video: {video_out}
    Fine-tuned scene log: {FINETUNED_JSONL}

  To obtain true precision/recall metrics, annotate test frames and run:
    python run_plan.py phase3 --with-gt  (not yet implemented — see TODO in phase3_evaluate.py)
"""
    )

    return {
        "finetuned_jsonl": str(FINETUNED_JSONL),
        "comparison_report": str(report_path),
        "comparison_video": str(video_out),
    }
