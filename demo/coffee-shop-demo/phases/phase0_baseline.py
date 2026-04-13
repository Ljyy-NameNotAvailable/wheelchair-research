"""
phase0_baseline.py — Phase 0: Baseline Characterization + Roboflow upload.

Steps:
  0.1  Sample frames from input video (every N frames)
  0.2  Run DynamicPerceptionPipeline on full video → baseline_raw.jsonl
  0.3  Extract optical-flow crop hints from unclassified_motion_regions
  0.4  Upload sampled frames to Roboflow for human annotation
  0.5  Compute proxy detection metrics → baseline_proxy_metrics.md
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
    CROP_FRAMES_DIR,
    CROP_REGIONS_DIR,
    DEMO_DIR,
    PIPELINE_DIR,
    REPORTS_DIR,
    ensure_dirs,
    mark_phase_done,
    print_next_step,
    print_phase_header,
)


# ---------------------------------------------------------------------------
# Step 0.1 — Sample frames
# ---------------------------------------------------------------------------

def sample_frames(video_path: Path, sample_every: int) -> list[tuple[int, Path]]:
    """
    Step 0.1 — Extract one frame every `sample_every` frames from the video.

    Saves each sampled frame as a JPEG to CROP_FRAMES_DIR.

    Args:
        video_path:   path to the coffee-shop ceiling-camera video.
        sample_every: frame sampling stride (default 10).

    Returns:
        List of (frame_idx, saved_path) tuples for all sampled frames.

    Raises:
        FileNotFoundError: if `video_path` does not exist.
        RuntimeError:      if the video cannot be opened by OpenCV.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"[phase0] Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[phase0] Cannot open video: {video_path}")

    CROP_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[phase0/step0.1] Video has {total} frames. Sampling every {sample_every}.")

    sampled: list[tuple[int, Path]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            out_path = CROP_FRAMES_DIR / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            sampled.append((frame_idx, out_path))
        frame_idx += 1

    cap.release()
    print(f"[phase0/step0.1] Saved {len(sampled)} sampled frames to {CROP_FRAMES_DIR}")
    return sampled


# ---------------------------------------------------------------------------
# Step 0.2 — Run baseline pipeline
# ---------------------------------------------------------------------------

def run_baseline_pipeline(
    video_path: Path,
    device: str,
) -> Path:
    """
    Step 0.2 — Run DynamicPerceptionPipeline on every frame of the video.

    Writes one JSON object per line (frame scene_state) to
    REPORTS_DIR/baseline_raw.jsonl.

    Args:
        video_path: path to the input video.
        device:     compute device string ("cpu", "mps", "auto").

    Returns:
        Path to the written baseline_raw.jsonl file.

    Raises:
        FileNotFoundError: if the video does not exist.
        RuntimeError:      if the video cannot be opened.
    """
    # Import pipeline from the demo directory (not from repo root)
    from pipeline import DEFAULT_CONFIG, DynamicPerceptionPipeline  # noqa: E402

    if not video_path.exists():
        raise FileNotFoundError(f"[phase0] Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[phase0] Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[phase0/step0.2] Running baseline pipeline on {total} frames (device={device})...")

    config = {**DEFAULT_CONFIG, "device": device}
    pipeline = DynamicPerceptionPipeline(config)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "baseline_raw.jsonl"

    try:
        from tqdm import tqdm
        progress = tqdm(total=total, unit="frame", desc="Baseline pipeline")
    except ImportError:
        progress = None
        print("[phase0/step0.2] tqdm not found — no progress bar.")

    with open(out_path, "w", encoding="utf-8") as fh:
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
                print(f"[phase0/step0.2] {frame_idx}/{total} frames")

    if progress is not None:
        progress.close()
    cap.release()

    print(f"[phase0/step0.2] Baseline log written to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Step 0.3 — Extract crop hints from optical flow
# ---------------------------------------------------------------------------

def extract_crop_hints(
    video_path: Path,
    sampled_indices: list[int],
    baseline_jsonl: Path,
    pad: int = 20,
) -> list[Path]:
    """
    Step 0.3 — Crop unclassified_motion_regions from sampled frames.

    For each sampled frame index, reads the corresponding scene_state from
    baseline_raw.jsonl, extracts every unclassified_motion_region bounding
    box (with `pad` pixels of padding), and saves each crop as a JPEG to
    CROP_REGIONS_DIR.

    Args:
        video_path:      path to the input video (re-opened for frame access).
        sampled_indices: list of frame indices that were sampled in step 0.1.
        baseline_jsonl:  path to baseline_raw.jsonl produced in step 0.2.
        pad:             pixel padding added to each side of the crop bbox.

    Returns:
        List of Paths to all saved crop images.

    Raises:
        FileNotFoundError: if baseline_jsonl does not exist.
        RuntimeError:      if the video cannot be opened.
    """
    if not baseline_jsonl.exists():
        raise FileNotFoundError(f"[phase0] baseline_raw.jsonl not found: {baseline_jsonl}")

    # Build idx → scene_state lookup
    print("[phase0/step0.3] Loading baseline scene states for crop extraction...")
    state_by_idx: dict[int, dict[str, Any]] = {}
    with open(baseline_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            state = json.loads(line)
            state_by_idx[state["frame_idx"]] = state

    # Re-open video to read frames at sampled indices
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[phase0] Cannot open video: {video_path}")

    CROP_REGIONS_DIR.mkdir(parents=True, exist_ok=True)

    sampled_set = set(sampled_indices)
    saved_crops: list[Path] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in sampled_set:
            scene = state_by_idx.get(frame_idx)
            if scene:
                h, w = frame.shape[:2]
                for region_idx, region in enumerate(
                    scene.get("unclassified_motion_regions", [])
                ):
                    x1, y1, x2, y2 = region["bbox_px"]
                    # Apply padding, clamped to frame bounds
                    x1c = max(0, x1 - pad)
                    y1c = max(0, y1 - pad)
                    x2c = min(w, x2 + pad)
                    y2c = min(h, y2 + pad)
                    crop = frame[y1c:y2c, x1c:x2c]
                    if crop.size == 0:
                        continue
                    crop_path = CROP_REGIONS_DIR / f"frame_{frame_idx:06d}_region_{region_idx:03d}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    saved_crops.append(crop_path)
        frame_idx += 1

    cap.release()
    print(
        f"[phase0/step0.3] Saved {len(saved_crops)} flow-region crops to {CROP_REGIONS_DIR}"
    )
    return saved_crops


# ---------------------------------------------------------------------------
# Step 0.4 — Upload to Roboflow
# ---------------------------------------------------------------------------

def upload_to_roboflow(
    api_key: str,
    workspace_name: str,
    sampled_frame_paths: list[Path],
) -> str:
    """
    Step 0.4 — Upload sampled frames to Roboflow for annotation.

    Connects to the given workspace, creates or retrieves the
    'overhead-person-detection' project, and uploads all sampled frame
    images in a batch named 'coffee-shop-baseline'.

    Args:
        api_key:              Roboflow API key.
        workspace_name:       Roboflow workspace slug.
        sampled_frame_paths:  list of JPEG paths to upload.

    Returns:
        URL of the Roboflow project page.

    Raises:
        RuntimeError:  if the roboflow package is not installed or the API
                       key is rejected.
        ValueError:    if `api_key` is empty.
    """
    if not api_key or not api_key.strip():
        raise ValueError(
            "[phase0] Roboflow API key is empty. "
            "Pass --roboflow-key YOUR_KEY on the command line."
        )

    try:
        import roboflow as rf_module
    except ImportError as exc:
        raise RuntimeError(
            "[phase0] The 'roboflow' package is not installed.\n"
            "  Install it with:  pip install roboflow"
        ) from exc

    print("[phase0/step0.4] Connecting to Roboflow...")
    try:
        rf = rf_module.Roboflow(api_key=api_key)
        workspace = rf.workspace(workspace_name)
    except Exception as exc:
        raise RuntimeError(
            f"[phase0] Failed to connect to Roboflow workspace '{workspace_name}'.\n"
            f"  Check that your API key is correct and the workspace slug matches.\n"
            f"  Original error: {exc}"
        ) from exc

    project_name = "overhead-person-detection"
    print(f"[phase0/step0.4] Connecting to project '{project_name}'...")
    try:
        project = workspace.project(project_name)
    except Exception:
        # Project does not exist — create it
        print(f"[phase0/step0.4] Project not found; creating '{project_name}'...")
        try:
            project = workspace.create_project(
                project_name=project_name,
                project_type="object-detection",
                annotation="overhead-person-detection",
                project_license="MIT",
            )
        except Exception as exc:
            raise RuntimeError(
                f"[phase0] Could not create Roboflow project '{project_name}'.\n"
                f"  Original error: {exc}"
            ) from exc

    print(f"[phase0/step0.4] Uploading {len(sampled_frame_paths)} frames to batch 'coffee-shop-baseline'...")

    try:
        from tqdm import tqdm
        iterable = tqdm(sampled_frame_paths, desc="Uploading", unit="img")
    except ImportError:
        iterable = sampled_frame_paths

    for img_path in iterable:
        try:
            project.upload(
                image_path=str(img_path),
                batch_name="coffee-shop-baseline",
                tag_names=["baseline"],
                # split="train",  # SPEC_AMBIGUITY: spec does not specify a split;
                # omitting so Roboflow assigns automatically.
            )
        except Exception as exc:
            print(
                f"[phase0/step0.4] WARNING: Could not upload {img_path.name}: {exc}",
                file=sys.stderr,
            )

    # Construct project URL — Roboflow URL format
    project_url = f"https://app.roboflow.com/{workspace_name}/{project_name}"
    print(f"[phase0/step0.4] Upload complete. Project URL: {project_url}")
    return project_url


# ---------------------------------------------------------------------------
# Step 0.5 — Baseline proxy metrics
# ---------------------------------------------------------------------------

def compute_proxy_metrics(baseline_jsonl: Path) -> Path:
    """
    Step 0.5 — Compute proxy detection metrics from baseline_raw.jsonl.

    Since ground truth is not yet available, computes:
      - Mean/std objects detected per frame
      - Frames with zero detections (count + %)
      - Mean detection confidence
      - Mean unclassified_motion_regions per frame (proxy for YOLO misses)

    Args:
        baseline_jsonl: path to baseline_raw.jsonl.

    Returns:
        Path to the written baseline_proxy_metrics.md file.

    Raises:
        FileNotFoundError: if baseline_jsonl does not exist.
    """
    if not baseline_jsonl.exists():
        raise FileNotFoundError(f"[phase0] baseline_raw.jsonl not found: {baseline_jsonl}")

    print("[phase0/step0.5] Computing proxy metrics...")

    objects_per_frame: list[int] = []
    confidences: list[float] = []
    unclassified_per_frame: list[int] = []
    zero_detection_frames = 0
    total_frames = 0

    with open(baseline_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            state = json.loads(line)
            total_frames += 1

            objs = state.get("objects", [])
            n_objs = len(objs)
            objects_per_frame.append(n_objs)

            if n_objs == 0:
                zero_detection_frames += 1

            for obj in objs:
                conf = obj.get("confidence", 0.0)
                confidences.append(float(conf))

            n_unclass = len(state.get("unclassified_motion_regions", []))
            unclassified_per_frame.append(n_unclass)

    if total_frames == 0:
        raise RuntimeError(
            "[phase0] baseline_raw.jsonl is empty — did the pipeline run correctly?"
        )

    def _mean(lst: list[float | int]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    def _std(lst: list[float | int]) -> float:
        return float(np.std(lst)) if lst else 0.0

    mean_objs = _mean(objects_per_frame)
    std_objs = _std(objects_per_frame)
    zero_pct = 100.0 * zero_detection_frames / total_frames
    mean_conf = _mean(confidences) if confidences else float("nan")
    mean_unclass = _mean(unclassified_per_frame)

    report_lines = [
        "# Baseline Proxy Metrics",
        "",
        "These metrics are computed without ground truth annotations.",
        "They serve as a proxy to characterise the baseline YOLOv8n performance",
        "on the overhead coffee-shop camera view.",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total frames processed | {total_frames} |",
        f"| Mean persons detected per frame | {mean_objs:.2f} ± {std_objs:.2f} |",
        f"| Frames with zero detections | {zero_detection_frames} ({zero_pct:.1f}%) |",
        f"| Mean detection confidence | {mean_conf:.4f} |",
        f"| Mean unclassified motion regions / frame | {mean_unclass:.2f} |",
        "",
        "## Interpretation",
        "",
        "- **Unclassified motion regions** are optical-flow regions that YOLO did not",
        "  classify as a person. A high value suggests the baseline model misses people",
        "  viewed from overhead. Fine-tuning on overhead data (Phase 2) should reduce this.",
        "",
        "- **Zero-detection frames** may indicate lighting conditions or camera angles",
        "  where YOLO fails entirely. These frames are the best candidates for annotation.",
        "",
        "- **Detection confidence** for overhead pedestrians is typically 0.2–0.5 with the",
        "  stock COCO-trained YOLOv8n model. After fine-tuning, expect 0.5–0.8.",
        "",
        "## Limitation",
        "",
        "Without ground truth bounding boxes, recall cannot be measured. The",
        "`unclassified_motion_regions` count is a proxy only. True evaluation",
        "requires annotated test frames (see Phase 3).",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "baseline_proxy_metrics.md"
    out_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[phase0/step0.5] Proxy metrics written to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Phase 0 entry point
# ---------------------------------------------------------------------------

def run(
    video_path: Path,
    roboflow_key: str,
    roboflow_workspace: str,
    sample_every: int = 10,
    device: str = "auto",
) -> dict:
    """
    Run all Phase 0 steps in order.

    Args:
        video_path:          path to coffee-shop ceiling-camera video.
        roboflow_key:        Roboflow API key.
        roboflow_workspace:  Roboflow workspace slug.
        sample_every:        frame sampling stride (default 10).
        device:              compute device for pipeline ("auto", "cpu", "mps").

    Returns:
        Dict of output paths and URLs for state recording.
    """
    print_phase_header(0)
    ensure_dirs()

    # Step 0.1
    print("[phase0] Step 0.1 — Sampling frames...")
    sampled = sample_frames(video_path, sample_every)
    sampled_indices = [idx for idx, _ in sampled]
    sampled_paths = [p for _, p in sampled]
    print("[phase0] Step 0.1 — Done.")

    # Step 0.2
    print("[phase0] Step 0.2 — Running baseline pipeline...")
    baseline_jsonl = run_baseline_pipeline(video_path, device)
    print("[phase0] Step 0.2 — Done.")

    # Step 0.3
    print("[phase0] Step 0.3 — Extracting crop hints...")
    extract_crop_hints(video_path, sampled_indices, baseline_jsonl)
    print("[phase0] Step 0.3 — Done.")

    # Step 0.4
    print("[phase0] Step 0.4 — Uploading to Roboflow...")
    project_url = upload_to_roboflow(roboflow_key, roboflow_workspace, sampled_paths)
    print("[phase0] Step 0.4 — Done.")

    # Step 0.5
    print("[phase0] Step 0.5 — Computing proxy metrics...")
    metrics_path = compute_proxy_metrics(baseline_jsonl)
    print("[phase0] Step 0.5 — Done.")

    # Print next step instructions
    print_next_step(
        f"""
  PHASE 0 COMPLETE
  ════════════════

  Next: Annotate your frames in Roboflow.
    URL: {project_url}

    Label each person as:
      • person_sitting  — seated at a table or chair
      • person_standing — standing or walking

  When done, run:
    python run_plan.py phase1 \\
      --roboflow-key YOUR_KEY \\
      --roboflow-project overhead-person-detection \\
      --roboflow-workspace {roboflow_workspace}
"""
    )

    return {
        "baseline_jsonl": str(baseline_jsonl),
        "proxy_metrics": str(metrics_path),
        "roboflow_project_url": project_url,
        "sampled_frames_dir": str(CROP_FRAMES_DIR),
        "crops_dir": str(CROP_REGIONS_DIR),
    }
