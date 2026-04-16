"""
phase1_dataset.py — Phase 1: Dataset Preparation.

Steps:
  1.1  Export annotated coffee-shop frames from Roboflow (YOLOv8 format)
  1.2  Download Oxford Town Centre video + annotations; convert to YOLO format
  1.3  Merge both datasets with 80/20 train/val split
  1.4  Write augmentation_note.md documenting training augmentation settings
"""

from __future__ import annotations

import csv
import os
import random
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

import cv2

# Allow importing pipeline.py from the demo directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from phases.common import (
    DATASETS_DIR,
    REPORTS_DIR,
    ensure_dirs,
    mark_phase_done,
    print_next_step,
    print_phase_header,
    require_phase_done,
)

# ---------------------------------------------------------------------------
# Oxford Town Centre download URLs
# ---------------------------------------------------------------------------

OXFORD_VIDEO_URL = (
    "https://www.robots.ox.ac.uk/ActiveVision/Research/Projects/"
    "2009bbenfold_headpose/Datasets/TownCentreXVID.avi"
)
OXFORD_ANNOT_URL = (
    "https://www.robots.ox.ac.uk/ActiveVision/Research/Projects/"
    "2009bbenfold_headpose/Datasets/TownCentre-groundtruth.top"
)

OXFORD_RAW_DIR = DATASETS_DIR / "oxford_raw"
OXFORD_DIR = DATASETS_DIR / "oxford"
COFFEE_SHOP_DIR = DATASETS_DIR / "coffee_shop"
MERGED_DIR = DATASETS_DIR / "merged"


# ---------------------------------------------------------------------------
# Step 1.1 — Download public overhead-person dataset from Roboflow Universe
# ---------------------------------------------------------------------------

# Public Roboflow Universe project — no project creation required.
# MIT license. 5,602 overhead-view images annotated for person detection.
# https://universe.roboflow.com/abhay-c-mkdjq/overhead-head-detection
UNIVERSE_WORKSPACE = "abhay-c-mkdjq"
UNIVERSE_PROJECT   = "overhead-head-detection"
UNIVERSE_VERSION   = 1


def download_universe_dataset(api_key: str) -> Path:
    """
    Step 1.1 — Download the public overhead-person dataset from Roboflow
    Universe in YOLOv8 format. No project creation or annotation required —
    this is a pre-annotated public dataset (MIT license).

    Source: abhay-c-mkdjq/overhead-head-detection (5,602 images, YOLOv8)
    URL:    https://universe.roboflow.com/abhay-c-mkdjq/overhead-head-detection

    Downloads to DATASETS_DIR/overhead_universe/.

    Args:
        api_key: Any valid Roboflow API key (free tier is sufficient).

    Returns:
        Path to the downloaded dataset directory.

    Raises:
        ValueError:   if api_key is empty.
        RuntimeError: if the roboflow package is missing or download fails.
    """
    if not api_key or not api_key.strip():
        raise ValueError(
            "[phase1] Roboflow API key is empty. "
            "Get a free key at https://app.roboflow.com → Settings → API Keys.\n"
            "Pass it with:  --roboflow-key YOUR_KEY"
        )

    try:
        import roboflow as rf_module
    except ImportError as exc:
        raise RuntimeError(
            "[phase1] The 'roboflow' package is not installed.\n"
            "  Install it with:  pip install roboflow"
        ) from exc

    dest = DATASETS_DIR / "overhead_universe"
    if dest.exists() and any(dest.iterdir()):
        print(f"[phase1/step1.1] Dataset already downloaded at {dest} — skipping.")
        return dest

    print(
        f"[phase1/step1.1] Downloading public Universe dataset "
        f"'{UNIVERSE_WORKSPACE}/{UNIVERSE_PROJECT}' (YOLOv8 format)..."
    )
    try:
        rf = rf_module.Roboflow(api_key=api_key)
        project = rf.workspace(UNIVERSE_WORKSPACE).project(UNIVERSE_PROJECT)
        project.version(UNIVERSE_VERSION).download("yolov8", location=str(dest))
    except Exception as exc:
        raise RuntimeError(
            f"[phase1] Download from Roboflow Universe failed: {exc}\n"
            f"\n"
            f"  Check that your API key is valid (free key from app.roboflow.com).\n"
            f"  The dataset is public — no project ownership needed.\n"
            f"  Direct URL: https://universe.roboflow.com/{UNIVERSE_WORKSPACE}/{UNIVERSE_PROJECT}"
        ) from exc

    print(f"[phase1/step1.1] Universe dataset downloaded to {dest}")
    return dest


# ---------------------------------------------------------------------------
# Step 1.2 — Download and convert Oxford Town Centre
# ---------------------------------------------------------------------------

def _download_with_progress(url: str, dest: Path) -> None:
    """
    Download `url` to `dest`, showing a tqdm progress bar.

    Skips the download if `dest` already exists and is non-empty.
    """
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[phase1] Already downloaded: {dest.name} — skipping.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm

        class _TqdmHook:
            def __init__(self):
                self._bar = None

            def __call__(self, block_num, block_size, total_size):
                if self._bar is None:
                    self._bar = tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=dest.name,
                    )
                downloaded = block_num * block_size
                self._bar.update(min(block_size, total_size - self._bar.n)
                                 if total_size > 0 else block_size)
                if downloaded >= total_size and total_size > 0:
                    self._bar.close()

        urllib.request.urlretrieve(url, str(dest), reporthook=_TqdmHook())
    except ImportError:
        print(f"[phase1] Downloading {dest.name} (no progress bar — tqdm not installed)...")
        urllib.request.urlretrieve(url, str(dest))

    print(f"[phase1] Downloaded: {dest}")


def _parse_oxford_annotations(annot_path: Path) -> dict[int, list[list[float]]]:
    """
    Parse the Oxford Town Centre ground-truth CSV file.

    CSV columns (0-indexed):
      0: personNumber
      1: frameNumber
      2: humanUpperBodyX
      3: humanUpperBodyY
      4: humanUpperBodyW
      5: humanUpperBodyH
      6: humanFullBodyX
      7: humanFullBodyY
      8: humanFullBodyW
      9: humanFullBodyH

    Uses full body bbox (columns 6–9).  The class is always 0 (person_standing).

    Returns:
        Dict mapping frameNumber → list of [x, y, w, h] full-body bboxes (pixel coords).
    """
    frames: dict[int, list[list[float]]] = {}
    with open(annot_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            row = [c.strip() for c in row]
            if len(row) < 10:
                continue
            try:
                frame_num = int(float(row[1]))
                x = float(row[6])
                y = float(row[7])
                w = float(row[8])
                h = float(row[9])
            except (ValueError, IndexError):
                continue
            frames.setdefault(frame_num, []).append([x, y, w, h])
    return frames


def download_oxford(sample_every: int = 15) -> Path:
    """
    Step 1.2 — Download Oxford Town Centre video + annotations and convert
    to YOLO format.

    Downloads:
      - Video  → datasets/oxford_raw/TownCentreXVID.avi
      - Labels → datasets/oxford_raw/TownCentre-groundtruth.top

    Converts every `sample_every`-th frame to JPEG + YOLO .txt label and saves
    to datasets/oxford/images/ and datasets/oxford/labels/.

    Class mapping: all people are class 0 (person_standing) — Oxford Town Centre
    contains only walking/standing pedestrians, no seated subjects.

    Args:
        sample_every: stride for frame sampling (default 15).

    Returns:
        Path to datasets/oxford/ directory.
    """
    OXFORD_RAW_DIR.mkdir(parents=True, exist_ok=True)

    video_path = OXFORD_RAW_DIR / "TownCentreXVID.avi"
    annot_path = OXFORD_RAW_DIR / "TownCentre-groundtruth.top"

    print("[phase1/step1.2] Downloading Oxford Town Centre dataset (~700 MB)...")
    _download_with_progress(OXFORD_VIDEO_URL, video_path)
    _download_with_progress(OXFORD_ANNOT_URL, annot_path)

    print("[phase1/step1.2] Parsing Oxford annotations...")
    annotations = _parse_oxford_annotations(annot_path)

    images_dir = OXFORD_DIR / "images"
    labels_dir = OXFORD_DIR / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[phase1] Cannot open Oxford video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"[phase1/step1.2] Converting Oxford video ({total} frames, "
        f"sampling every {sample_every})..."
    )

    try:
        from tqdm import tqdm
        progress = tqdm(total=total, unit="frame", desc="Oxford conversion")
    except ImportError:
        progress = None

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0 and frame_idx in annotations:
            h, w = frame.shape[:2]
            bboxes = annotations[frame_idx]

            # Write image
            img_name = f"oxford_{frame_idx:07d}.jpg"
            cv2.imwrite(str(images_dir / img_name), frame)

            # Write YOLO label — class 0, normalised xywh
            label_name = f"oxford_{frame_idx:07d}.txt"
            lines: list[str] = []
            for bx, by, bw, bh in bboxes:
                # Convert top-left xywh to normalised cx,cy,w,h
                cx = (bx + bw / 2.0) / w
                cy = (by + bh / 2.0) / h
                nw = bw / w
                nh = bh / h
                # Clamp to [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            (labels_dir / label_name).write_text("\n".join(lines), encoding="utf-8")
            saved += 1

        frame_idx += 1
        if progress is not None:
            progress.update(1)
        elif frame_idx % 500 == 0:
            print(f"[phase1/step1.2] {frame_idx}/{total} frames scanned, {saved} saved")

    if progress is not None:
        progress.close()
    cap.release()

    print(f"[phase1/step1.2] Oxford conversion complete: {saved} images saved to {OXFORD_DIR}")
    return OXFORD_DIR


# ---------------------------------------------------------------------------
# Step 1.3 — Merge datasets
# ---------------------------------------------------------------------------

def _collect_yolov8_pairs(source_dir: Path) -> list[tuple[Path, Path | None]]:
    """
    Walk `source_dir` and collect (image_path, label_path_or_None) pairs.

    Looks for images in any subdirectory named 'images/', with corresponding
    labels in a sibling 'labels/' directory.  Also handles a flat structure
    where images and labels are in the same directory.
    """
    pairs: list[tuple[Path, Path | None]] = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    # Search all 'images' subdirectories
    for img_dir in source_dir.rglob("images"):
        if not img_dir.is_dir():
            continue
        label_dir = img_dir.parent / "labels"
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in image_exts:
                continue
            label_path = label_dir / (img_path.stem + ".txt")
            pairs.append((img_path, label_path if label_path.exists() else None))

    # Flat fallback: if no images/ subdirectory found, check root
    if not pairs:
        label_dir = source_dir / "labels"
        for img_path in source_dir.iterdir():
            if img_path.suffix.lower() not in image_exts:
                continue
            label_path = label_dir / (img_path.stem + ".txt")
            pairs.append((img_path, label_path if label_path.exists() else None))

    return pairs


def merge_datasets(
    coffee_shop_dir: Path,
    oxford_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Path:
    """
    Step 1.3 — Merge coffee-shop and Oxford datasets into datasets/merged/.

    Creates an 80/20 train/val split, keeping coffee-shop frames proportionally
    in the val set (stratified by source).

    Writes:
      - datasets/merged/images/train/  and  /val/
      - datasets/merged/labels/train/  and  /val/
      - datasets/merged/data.yaml

    Args:
        coffee_shop_dir: path to Roboflow-exported coffee-shop dataset.
        oxford_dir:      path to converted Oxford Town Centre dataset.
        train_ratio:     fraction of images to put in train split (default 0.8).
        seed:            random seed for reproducible splits.

    Returns:
        Path to datasets/merged/.
    """
    print("[phase1/step1.3] Collecting coffee-shop image/label pairs...")
    coffee_pairs = _collect_yolov8_pairs(coffee_shop_dir)
    print(f"[phase1/step1.3]   Coffee-shop: {len(coffee_pairs)} images")

    print("[phase1/step1.3] Collecting Oxford image/label pairs...")
    oxford_pairs = _collect_yolov8_pairs(oxford_dir)
    print(f"[phase1/step1.3]   Oxford: {len(oxford_pairs)} images")

    if not coffee_pairs and not oxford_pairs:
        raise RuntimeError(
            "[phase1] No images found in either dataset. "
            "Check that coffee_shop and oxford directories are populated."
        )

    rng = random.Random(seed)

    def _split(pairs: list, ratio: float) -> tuple[list, list]:
        pairs_copy = list(pairs)
        rng.shuffle(pairs_copy)
        n_train = max(1, int(len(pairs_copy) * ratio))
        return pairs_copy[:n_train], pairs_copy[n_train:]

    coffee_train, coffee_val = _split(coffee_pairs, train_ratio)
    oxford_train, oxford_val = _split(oxford_pairs, train_ratio)

    splits: dict[str, list[tuple[Path, Path | None]]] = {
        "train": coffee_train + oxford_train,
        "val": coffee_val + oxford_val,
    }

    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, pairs in splits.items():
        img_out = MERGED_DIR / "images" / split_name
        lbl_out = MERGED_DIR / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for src_img, src_lbl in pairs:
            # Ensure unique filenames by prefixing with source directory name
            prefix = src_img.parent.parent.name  # e.g. 'coffee_shop' or 'oxford'
            dst_img = img_out / f"{prefix}_{src_img.name}"
            shutil.copy2(src_img, dst_img)

            if src_lbl is not None:
                dst_lbl = lbl_out / f"{prefix}_{src_lbl.name}"
                shutil.copy2(src_lbl, dst_lbl)
            else:
                # Create empty label file for images without annotations
                dst_lbl = lbl_out / f"{prefix}_{src_img.stem}.txt"
                dst_lbl.write_text("", encoding="utf-8")

    # Write data.yaml — use absolute path for robustness with Ultralytics
    data_yaml_content = (
        f"path: {MERGED_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 2\n"
        f"names: ['person_standing', 'person_sitting']\n"
    )
    (MERGED_DIR / "data.yaml").write_text(data_yaml_content, encoding="utf-8")

    n_train = len(splits["train"])
    n_val = len(splits["val"])
    n_total = n_train + n_val
    n_oxford = len(oxford_pairs)
    n_coffee = len(coffee_pairs)

    summary = (
        f"\nDataset merged:\n"
        f"  Oxford Town Centre: {n_oxford} images (person_standing only)\n"
        f"  Coffee shop (Roboflow): {n_coffee} images (person_standing + person_sitting)\n"
        f"  Total: {n_total} images\n"
        f"  Train: {n_train} | Val: {n_val}\n"
    )
    print(summary)

    return MERGED_DIR


# ---------------------------------------------------------------------------
# Step 1.4 — Write augmentation note
# ---------------------------------------------------------------------------

def write_augmentation_note() -> Path:
    """
    Step 1.4 — Write augmentation_note.md documenting the augmentation
    settings that will be applied by Ultralytics during training (Phase 2).

    Returns:
        Path to the written augmentation_note.md.
    """
    content = """# Augmentation Settings for Overhead Pedestrian Detection

These augmentation parameters are passed directly to `model.train()` in Phase 2
(ultralytics handles augmentation at train time, not during dataset preparation).

## Rationale for each setting

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `degrees` | 180 | Full rotation invariance — ceiling camera, people face any direction |
| `fliplr` | 0.5 | Horizontal flip valid for overhead view |
| `flipud` | 0.5 | Vertical flip also valid for overhead view |
| `scale` | 0.5 | Scale variation — camera height varies across installations |
| `hsv_v` | 0.4 | Brightness variation — café lighting changes throughout the day |
| `blur` | 0.1 | Motion blur for fast-moving subjects |

## Notes

- `degrees=180` is non-standard (default is 0). This is critical for overhead
  cameras because pedestrians can face any compass direction.
- `flipud=0.5` is also non-standard (default 0.0). At eye level, vertical flips
  are invalid; from above, they are geometrically equivalent to a 180° rotation.
- `hsv_v=0.4` (value channel jitter) simulates the wide range of café lighting
  conditions (bright daylight through windows vs. dim evening lighting).
"""
    out_path = MERGED_DIR / "augmentation_note.md"
    out_path.write_text(content, encoding="utf-8")
    print(f"[phase1/step1.4] Augmentation note written to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Dataset summary report
# ---------------------------------------------------------------------------

def write_dataset_summary(
    coffee_dir: Path,
    oxford_dir: Path,
    merged_dir: Path,
) -> Path:
    """
    Write reports/dataset_summary.md summarising the merged dataset.

    Returns:
        Path to the written dataset_summary.md.
    """
    coffee_pairs = _collect_yolov8_pairs(coffee_dir)
    oxford_pairs = _collect_yolov8_pairs(oxford_dir)

    n_coffee = len(coffee_pairs)
    n_oxford = len(oxford_pairs)
    n_total = n_coffee + n_oxford

    n_train = len(list((merged_dir / "images" / "train").glob("*"))) if (merged_dir / "images" / "train").exists() else 0
    n_val = len(list((merged_dir / "images" / "val").glob("*"))) if (merged_dir / "images" / "val").exists() else 0

    lines = [
        "# Dataset Summary",
        "",
        "## Sources",
        "",
        f"| Source | Images | Classes |",
        f"|--------|--------|---------|",
        f"| Oxford Town Centre | {n_oxford} | person_standing |",
        f"| Coffee shop (Roboflow) | {n_coffee} | person_standing, person_sitting |",
        f"| **Total** | **{n_total}** | |",
        "",
        "## Train / Val Split",
        "",
        f"| Split | Images |",
        f"|-------|--------|",
        f"| Train | {n_train} |",
        f"| Val   | {n_val} |",
        "",
        "## Location",
        "",
        f"- Merged dataset: `{merged_dir}`",
        f"- Data YAML: `{merged_dir / 'data.yaml'}`",
        "",
        "## Next Step",
        "",
        "Run Phase 2 to fine-tune YOLOv8n on this merged dataset:",
        "```bash",
        "python run_plan.py phase2 --epochs 50 --imgsz 640",
        "```",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "dataset_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[phase1] Dataset summary written to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Phase 1 entry point
# ---------------------------------------------------------------------------

def run(
    roboflow_key: str = "",
    no_roboflow: bool = False,
    skip_oxford: bool = False,
    state: dict | None = None,
) -> dict:
    """
    Run all Phase 1 steps in order.

    Args:
        roboflow_key:  Roboflow API key (only needed when no_roboflow=False).
        no_roboflow:   if True, skip Roboflow Universe download and use Oxford
                       Town Centre only — no Roboflow account required.
        skip_oxford:   if True, skip Oxford Town Centre download.
        state:         current plan state dict (used to check phase0 completion).

    Returns:
        Dict of output paths for state recording.
    """
    print_phase_header(1)
    ensure_dirs()

    if state is not None:
        require_phase_done(0, state)

    # Step 1.1 — optionally download Roboflow Universe dataset
    if no_roboflow:
        print("[phase1] Step 1.1 — Skipping Roboflow Universe download (--no-roboflow).")
        universe_dir = None
    else:
        print("[phase1] Step 1.1 — Downloading public overhead-person dataset from Roboflow Universe...")
        universe_dir = download_universe_dataset(roboflow_key)
        print("[phase1] Step 1.1 — Done.")

    # Step 1.2
    if skip_oxford:
        print("[phase1] Step 1.2 — Skipping Oxford Town Centre download (--skip-oxford).")
        oxford_dir = OXFORD_DIR
        oxford_dir.mkdir(parents=True, exist_ok=True)
        (oxford_dir / "images").mkdir(parents=True, exist_ok=True)
        (oxford_dir / "labels").mkdir(parents=True, exist_ok=True)
    else:
        print("[phase1] Step 1.2 — Downloading Oxford Town Centre (~700 MB from Oxford servers)...")
        download_oxford()
        oxford_dir = OXFORD_DIR
        print("[phase1] Step 1.2 — Done.")

    if universe_dir is None and not any((oxford_dir / "images").iterdir()):
        raise RuntimeError(
            "[phase1] No training data available — both Roboflow and Oxford Town Centre are empty."
        )

    # Step 1.3 — merge (or use whichever source is available)
    print("[phase1] Step 1.3 — Building dataset...")
    primary = universe_dir if universe_dir is not None else oxford_dir
    secondary = oxford_dir if universe_dir is not None else None
    merged_dir = merge_datasets(primary, secondary if secondary and any((secondary / "images").iterdir()) else oxford_dir)
    print("[phase1] Step 1.3 — Done.")

    # Step 1.4
    print("[phase1] Step 1.4 — Writing augmentation note...")
    write_augmentation_note()
    print("[phase1] Step 1.4 — Done.")

    # Summary report
    summary_path = write_dataset_summary(
        universe_dir if universe_dir else oxford_dir,
        oxford_dir,
        merged_dir,
    )

    print_next_step(
        """
  PHASE 1 COMPLETE

  Dataset is ready for fine-tuning.

  When ready, run:
    python run_plan.py phase2 --epochs 50 --imgsz 640

  To use a specific device:
    python run_plan.py phase2 --epochs 50 --device mps
    python run_plan.py phase2 --epochs 50 --device cpu
"""
    )

    return {
        "universe_dir": str(universe_dir),
        "oxford_dir": str(oxford_dir),
        "merged_dir": str(merged_dir),
        "data_yaml": str(merged_dir / "data.yaml"),
        "dataset_summary": str(summary_path),
    }
