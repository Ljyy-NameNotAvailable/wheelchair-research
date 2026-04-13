"""
common.py — Shared utilities for the coffee-shop demo run plan.

Provides:
  - Canonical directory paths (all relative to demo/coffee-shop-demo/)
  - State file load/save/require helpers
  - Shared logging helpers
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical paths — all relative to this file's parent (demo/coffee-shop-demo/)
# ---------------------------------------------------------------------------

DEMO_DIR = Path(__file__).parent.parent  # demo/coffee-shop-demo/

ANNOTATIONS_DIR = DEMO_DIR / "annotations"
DATASETS_DIR = DEMO_DIR / "datasets"
WEIGHTS_DIR = DEMO_DIR / "weights"
REPORTS_DIR = DEMO_DIR / "reports"

CROP_HINTS_DIR = ANNOTATIONS_DIR / "crop_hints"
CROP_FRAMES_DIR = CROP_HINTS_DIR / "frames"
CROP_REGIONS_DIR = CROP_HINTS_DIR / "crops"

# State file tracking which phases have completed
STATE_FILE = DEMO_DIR / ".plan_state.json"

# Path to pipeline.py (same directory as this demo)
PIPELINE_DIR = DEMO_DIR  # pipeline.py and run_video.py live here


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def load_state() -> dict:
    """
    Load the plan state from STATE_FILE.

    Returns an empty dict if the file does not exist yet.
    """
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"[common] State file is corrupted: {STATE_FILE}\n"
            f"  JSON error: {exc}\n"
            f"  Delete the file and re-run from phase0 to reset."
        ) from exc


def save_state(state: dict) -> None:
    """
    Persist the plan state dict to STATE_FILE.

    Creates the file if it does not exist; overwrites if it does.
    """
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def mark_phase_done(phase: int, state: dict, **outputs) -> dict:
    """
    Mark a phase as completed in `state`, record the timestamp and any
    output paths, then persist and return the updated state dict.

    Args:
        phase:   phase number (0–3).
        state:   current state dict (modified in-place).
        **outputs: keyword args become key→str(value) entries in the record.
    """
    record = {
        "completed": True,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    state[f"phase{phase}"] = record
    save_state(state)
    return state


def require_phase_done(phase: int, state: dict) -> None:
    """
    Raise a clear RuntimeError if the given phase has not been completed.

    Args:
        phase: required phase number.
        state: current state dict.

    Raises:
        RuntimeError: with a human-readable message explaining what to run next.
    """
    _phase_names = {
        0: "Baseline Characterization",
        1: "Dataset Preparation",
        2: "Fine-Tuning",
        3: "Evaluation",
    }
    key = f"phase{phase}"
    if not state.get(key, {}).get("completed"):
        name = _phase_names.get(phase, f"Phase {phase}")
        raise RuntimeError(
            f"[run_plan] Phase {phase} ({name}) has not been completed yet.\n"
            f"  Run:  python run_plan.py phase{phase} --help\n"
            f"  Then re-run this phase."
        )


# ---------------------------------------------------------------------------
# Header printer
# ---------------------------------------------------------------------------

_PHASE_TITLES = {
    0: "PHASE 0 — Baseline Characterization",
    1: "PHASE 1 — Dataset Preparation",
    2: "PHASE 2 — Fine-Tuning",
    3: "PHASE 3 — Evaluation",
}


def print_phase_header(phase: int) -> None:
    """Print a prominent phase header to stdout."""
    title = _PHASE_TITLES.get(phase, f"PHASE {phase}")
    width = max(len(title) + 4, 40)
    bar = "═" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}\n")


def print_next_step(message: str) -> None:
    """Print a 'NEXT STEP' block to stdout."""
    bar = "═" * 40
    print(f"\n{bar}")
    print("  NEXT STEP")
    print(f"{bar}")
    print(message)
    print()


# ---------------------------------------------------------------------------
# Ensure all output directories exist
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create all output directories if they do not already exist."""
    for d in [
        ANNOTATIONS_DIR,
        DATASETS_DIR,
        WEIGHTS_DIR,
        REPORTS_DIR,
        CROP_HINTS_DIR,
        CROP_FRAMES_DIR,
        CROP_REGIONS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
