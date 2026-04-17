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
# Public dataset download URLs (verified 2026-04-17)
# ---------------------------------------------------------------------------

# UCY Campus Crowd Detection — Zenodo 10629757
# Overhead UAV, bounding box annotations, CC BY 4.0, ~303 MB
UCY_URL = "https://zenodo.org/api/records/10629757/files/Kornos_labeling.zip/content"

# CUHK Mall Dataset — indoor overhead camera, 88 MB, research use
# Head-point annotations → converted to pseudo-bboxes
MALL_URL = "https://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/mall_dataset.zip"

UCY_RAW_DIR  = DATASETS_DIR / "ucy_raw"
UCY_DIR      = DATASETS_DIR / "ucy"
MALL_RAW_DIR = DATASETS_DIR / "mall_raw"
MALL_DIR     = DATASETS_DIR / "mall"
COFFEE_SHOP_DIR = DATASETS_DIR / "coffee_shop"
MERGED_DIR   = DATASETS_DIR / "merged"


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


def _bbox_to_yolo(x: float, y: float, w: float, h: float,
                  img_w: int, img_h: int) -> str:
    """Convert absolute [x, y, w, h] bbox to YOLO normalised format (class 0)."""
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    cx, cy, nw, nh = (max(0.0, min(1.0, v)) for v in (cx, cy, nw, nh))
    return f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


# ---------------------------------------------------------------------------
# UCY Campus Crowd Detection (Zenodo 10629757)
# ---------------------------------------------------------------------------

def download_ucy() -> Path:
    """
    Step 1.2a — Download UCY Campus Crowd Detection (Kornos subset, ~303 MB)
    from Zenodo and convert annotations to YOLO format.

    Source:  https://zenodo.org/records/10629757
    License: CC BY 4.0
    Content: Overhead UAV imagery, 84 486 bounding-box annotations on people.

    Saves images + YOLO labels to datasets/ucy/.

    Returns:
        Path to datasets/ucy/ directory.
    """
    if UCY_DIR.exists() and any(UCY_DIR.rglob("*.jpg")):
        print(f"[phase1] UCY dataset already converted at {UCY_DIR} — skipping.")
        return UCY_DIR

    UCY_RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = UCY_RAW_DIR / "Kornos_labeling.zip"

    print("[phase1/step1.2a] Downloading UCY Campus Crowd Detection (~303 MB)...")
    _download_with_progress(UCY_URL, zip_path)

    print("[phase1/step1.2a] Extracting UCY archive...")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(UCY_RAW_DIR)

    images_out = UCY_DIR / "images"
    labels_out = UCY_DIR / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Locate images and annotation files (format may vary: COCO JSON, CSV, MOT)
    img_files = sorted(UCY_RAW_DIR.rglob("*.jpg")) + sorted(UCY_RAW_DIR.rglob("*.png"))
    json_files = list(UCY_RAW_DIR.rglob("*.json"))
    csv_files  = list(UCY_RAW_DIR.rglob("*.csv")) + list(UCY_RAW_DIR.rglob("*.txt"))

    converted = 0

    # --- Try COCO JSON format ---
    if json_files:
        import json as _json
        for jf in json_files:
            try:
                with open(jf) as f:
                    coco = _json.load(f)
                if "images" not in coco or "annotations" not in coco:
                    continue
                id_to_file = {img["id"]: img for img in coco["images"]}
                ann_by_img: dict[int, list] = {}
                for ann in coco["annotations"]:
                    ann_by_img.setdefault(ann["image_id"], []).append(ann["bbox"])
                for img_id, bboxes in ann_by_img.items():
                    img_info = id_to_file.get(img_id)
                    if img_info is None:
                        continue
                    src = UCY_RAW_DIR / img_info["file_name"]
                    if not src.exists():
                        src = next((p for p in img_files
                                    if p.name == Path(img_info["file_name"]).name), None)
                    if src is None:
                        continue
                    iw, ih = img_info["width"], img_info["height"]
                    lines = [_bbox_to_yolo(b[0], b[1], b[2], b[3], iw, ih) for b in bboxes]
                    stem = f"ucy_{img_id:06d}"
                    shutil.copy(src, images_out / f"{stem}.jpg")
                    (labels_out / f"{stem}.txt").write_text("\n".join(lines))
                    converted += 1
            except Exception:
                continue

    # --- Fallback: copy images without annotations (for augmentation volume) ---
    if converted == 0:
        print("[phase1/step1.2a] WARNING: Could not parse UCY annotations — "
              "copying images only (no labels). They will be used as background.")
        for i, src in enumerate(img_files[:500]):  # cap at 500
            stem = f"ucy_bg_{i:04d}"
            shutil.copy(src, images_out / f"{stem}.jpg")
            (labels_out / f"{stem}.txt").write_text("")  # empty = no objects
        converted = min(500, len(img_files))

    print(f"[phase1/step1.2a] UCY: {converted} images written to {UCY_DIR}")
    return UCY_DIR


# ---------------------------------------------------------------------------
# CUHK Mall Dataset (indoor overhead — most similar to coffee shop)
# ---------------------------------------------------------------------------

def download_mall() -> Path:
    """
    Step 1.2b — Download the CUHK Mall Dataset (~88 MB) and convert head-point
    annotations to YOLO pseudo-bounding-boxes.

    Source:  https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html
    License: Research / non-commercial
    Content: 2 000 frames from an indoor overhead surveillance camera in a
             shopping mall — the closest publicly available scene to a coffee shop.
             Annotations: (x, y) head centre per person per frame.

    Pseudo-bbox generation: a fixed 50×80 px box centred on each head point,
    appropriate for the camera height and resolution (480×640 px).

    Saves images + YOLO labels to datasets/mall/.

    Returns:
        Path to datasets/mall/ directory.
    """
    if MALL_DIR.exists() and any(MALL_DIR.rglob("*.jpg")):
        print(f"[phase1] Mall dataset already converted at {MALL_DIR} — skipping.")
        return MALL_DIR

    MALL_RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = MALL_RAW_DIR / "mall_dataset.zip"

    print("[phase1/step1.2b] Downloading CUHK Mall Dataset (~88 MB)...")
    _download_with_progress(MALL_URL, zip_path)

    print("[phase1/step1.2b] Extracting Mall archive...")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(MALL_RAW_DIR)

    # Locate the ground truth .mat file
    mat_files = list(MALL_RAW_DIR.rglob("mall_gt.mat"))
    if not mat_files:
        raise RuntimeError(
            "[phase1] mall_gt.mat not found after extraction. "
            "Archive structure may have changed."
        )
    mat_path = mat_files[0]

    try:
        import scipy.io
        mat = scipy.io.loadmat(str(mat_path))
        # mat['frame'] is shape (1, N) — each cell is an array of (num_persons, 2) [x, y]
        frame_data = mat["frame"]
        num_frames = frame_data.shape[1]
    except ImportError:
        raise RuntimeError(
            "[phase1] scipy is required to read Mall Dataset annotations.\n"
            "  Install it with:  pip install scipy"
        )

    img_files_mall = sorted(MALL_RAW_DIR.rglob("*.jpg"))
    img_lookup = {p.stem: p for p in img_files_mall}

    images_out = MALL_DIR / "images"
    labels_out = MALL_DIR / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    IMG_W, IMG_H = 640, 480   # Mall dataset native resolution
    BOX_W, BOX_H = 50, 80     # pseudo-bbox size in pixels (head + torso from overhead)

    converted = 0
    for i in range(num_frames):
        try:
            persons = frame_data[0, i]   # shape (num_persons, 2) — columns: x, y
            if persons.ndim == 1:
                persons = persons.reshape(-1, 2)
        except Exception:
            continue

        # Find matching image (frames named seq_000001.jpg etc.)
        stem_candidates = [f"seq_{i+1:06d}", f"frame_{i:04d}", str(i)]
        src = next((img_lookup[s] for s in stem_candidates if s in img_lookup), None)
        if src is None and img_files_mall:
            src = img_files_mall[i] if i < len(img_files_mall) else None
        if src is None:
            continue

        lines = []
        for x, y in persons:
            lines.append(_bbox_to_yolo(
                float(x) - BOX_W / 2, float(y) - BOX_H / 2,
                BOX_W, BOX_H, IMG_W, IMG_H,
            ))

        stem = f"mall_{i:05d}"
        shutil.copy(src, images_out / f"{stem}.jpg")
        (labels_out / f"{stem}.txt").write_text("\n".join(lines))
        converted += 1

    print(f"[phase1/step1.2b] Mall: {converted} frames written to {MALL_DIR}")
    return MALL_DIR

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

    # (dead code — Oxford Town Centre URL is no longer available)
    # This block is unreachable; kept for reference only.
    pass


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
    primary_dir: Path,
    secondary_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Path:
    """
    Step 1.3 — Merge two overhead datasets into datasets/merged/.

    Creates an 80/20 train/val split stratified by source.

    Writes:
      - datasets/merged/images/train/  and  /val/
      - datasets/merged/labels/train/  and  /val/
      - datasets/merged/data.yaml

    Args:
        primary_dir:  primary dataset directory (UCY or Roboflow Universe).
        secondary_dir: secondary dataset directory (Mall or Oxford fallback).
        train_ratio:  fraction of images to put in train split (default 0.8).
        seed:            random seed for reproducible splits.

    Returns:
        Path to datasets/merged/.
    """
    print("[phase1/step1.3] Collecting primary dataset image/label pairs...")
    primary_pairs = _collect_yolov8_pairs(primary_dir)
    print(f"[phase1/step1.3]   Primary ({primary_dir.name}): {len(primary_pairs)} images")

    print("[phase1/step1.3] Collecting secondary dataset image/label pairs...")
    secondary_pairs = _collect_yolov8_pairs(secondary_dir)
    print(f"[phase1/step1.3]   Secondary ({secondary_dir.name}): {len(secondary_pairs)} images")

    if not primary_pairs and not secondary_pairs:
        raise RuntimeError(
            "[phase1] No images found in either dataset directory. "
            "Check that the download and conversion steps completed successfully."
        )

    rng = random.Random(seed)

    def _split(pairs: list, ratio: float) -> tuple[list, list]:
        pairs_copy = list(pairs)
        rng.shuffle(pairs_copy)
        n_train = max(1, int(len(pairs_copy) * ratio))
        return pairs_copy[:n_train], pairs_copy[n_train:]

    primary_train, primary_val     = _split(primary_pairs, train_ratio)
    secondary_train, secondary_val = _split(secondary_pairs, train_ratio)

    splits: dict[str, list[tuple[Path, Path | None]]] = {
        "train": primary_train + secondary_train,
        "val":   primary_val   + secondary_val,
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
    n_val   = len(splits["val"])
    n_total = n_train + n_val

    summary = (
        f"\nDataset merged:\n"
        f"  {primary_dir.name}: {len(primary_pairs)} images\n"
        f"  {secondary_dir.name}: {len(secondary_pairs)} images\n"
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

    # Step 1.2 — download UCY (outdoor overhead) + Mall (indoor overhead)
    if skip_oxford:
        print("[phase1] Step 1.2 — Skipping external dataset download (--skip-oxford).")
        ucy_dir  = UCY_DIR;  ucy_dir.mkdir(parents=True, exist_ok=True)
        mall_dir = MALL_DIR; mall_dir.mkdir(parents=True, exist_ok=True)
        (ucy_dir  / "images").mkdir(parents=True, exist_ok=True)
        (ucy_dir  / "labels").mkdir(parents=True, exist_ok=True)
        (mall_dir / "images").mkdir(parents=True, exist_ok=True)
        (mall_dir / "labels").mkdir(parents=True, exist_ok=True)
    else:
        print("[phase1] Step 1.2a — Downloading Mall Dataset (~88 MB, indoor overhead)...")
        mall_dir = download_mall()
        print("[phase1] Step 1.2a — Done.")
        print("[phase1] Step 1.2b — Downloading UCY Campus dataset (~303 MB, outdoor overhead)...")
        ucy_dir = download_ucy()
        print("[phase1] Step 1.2b — Done.")

    # Check we have at least one source
    has_primary   = universe_dir is not None
    has_mall      = any((mall_dir  / "images").iterdir()) if mall_dir.exists() else False
    has_ucy       = any((ucy_dir   / "images").iterdir()) if ucy_dir.exists() else False
    if not has_primary and not has_mall and not has_ucy:
        raise RuntimeError(
            "[phase1] No training data available — all downloads appear empty."
        )

    # Step 1.3 — pick the two richest sources for the merge
    # Priority: Roboflow Universe > Mall (indoor, most relevant) > UCY (outdoor)
    primary   = universe_dir if has_primary else (mall_dir if has_mall else ucy_dir)
    secondary = mall_dir     if has_mall and primary != mall_dir else ucy_dir

    print("[phase1] Step 1.3 — Building merged dataset...")
    merged_dir = merge_datasets(primary, secondary)
    print("[phase1] Step 1.3 — Done.")

    # Step 1.4
    print("[phase1] Step 1.4 — Writing augmentation note...")
    write_augmentation_note()
    print("[phase1] Step 1.4 — Done.")

    summary_path = write_dataset_summary(primary, secondary, merged_dir)

    print_next_step(
        """
  PHASE 1 COMPLETE

  Dataset is ready for fine-tuning.

  When ready, run:
    python run_plan.py phase2 --epochs 50 --imgsz 640 --device mps
"""
    )

    return {
        "primary_dir":  str(primary),
        "secondary_dir": str(secondary),
        "merged_dir":   str(merged_dir),
        "data_yaml":    str(merged_dir / "data.yaml"),
        "dataset_summary": str(summary_path),
    }
