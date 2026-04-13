"""
run_plan.py — Coffee-Shop Demo: Phased Research Automation Runner.

Usage:
    python run_plan.py phase0 --video PATH --roboflow-key KEY [--roboflow-workspace WS] [--sample-every N] [--device auto|cpu|mps]
    python run_plan.py phase1 --roboflow-key KEY [--skip-oxford]
    python run_plan.py phase2 [--epochs 50] [--imgsz 640] [--device mps|cpu]
    python run_plan.py phase3 --video PATH [--device cpu|mps] [--with-gt]
    python run_plan.py status

Phases:
    phase0  — Baseline characterization + Roboflow frame upload
              (human annotation gate: user labels frames in Roboflow before phase1)
    phase1  — Export Roboflow annotations + Oxford Town Centre + merge datasets
    phase2  — Fine-tune YOLOv8n on merged dataset (MPS / CUDA / CPU)
    phase3  — Evaluate fine-tuned model; produce comparison report + video
    status  — Print current progress checklist

State is persisted in demo/coffee-shop-demo/.plan_state.json between runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the phases package is importable from any working directory
sys.path.insert(0, str(Path(__file__).parent))

from phases.common import (
    load_state,
    mark_phase_done,
    print_phase_header,
    save_state,
)


# ---------------------------------------------------------------------------
# Status command
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    """Print a checklist of completed and pending phases."""
    state = load_state()

    _phases = [
        (0, "Baseline characterization"),
        (1, "Dataset preparation"),
        (2, "Fine-tuning"),
        (3, "Evaluation"),
    ]

    print()
    print("══════════════════════════════════════")
    print("  Coffee-Shop Demo — Run Plan Status")
    print("══════════════════════════════════════")
    print()

    next_phase = None
    for phase_num, phase_name in _phases:
        key = f"phase{phase_num}"
        record = state.get(key, {})
        done = record.get("completed", False)
        ts = record.get("completed_at", "")
        if done:
            marker = "[✓]"
            ts_str = f"(completed {ts})" if ts else ""
        else:
            marker = "[ ]"
            ts_str = ""
            if next_phase is None:
                next_phase = phase_num

        print(f"  {marker} Phase {phase_num} — {phase_name} {ts_str}")

    print()
    if next_phase is not None:
        print(f"  Next: run phase{next_phase}")
        _hints = {
            0: "python run_plan.py phase0 --video YOUR_VIDEO.mp4 --roboflow-key YOUR_KEY",
            1: "python run_plan.py phase1 --roboflow-key YOUR_KEY",
            2: "python run_plan.py phase2 --epochs 50 --imgsz 640",
            3: "python run_plan.py phase3 --video YOUR_VIDEO.mp4",
        }
        hint = _hints.get(next_phase)
        if hint:
            print(f"  Command: {hint}")
    else:
        print("  All phases complete.")
    print()


# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------

def cmd_phase0(args: argparse.Namespace) -> None:
    """Run Phase 0 — Baseline characterization + Roboflow upload."""
    from phases import phase0_baseline

    state = load_state()

    outputs = phase0_baseline.run(
        video_path=Path(args.video),
        roboflow_key=args.roboflow_key,
        roboflow_workspace=args.roboflow_workspace,
        sample_every=args.sample_every,
        device=args.device,
    )

    mark_phase_done(0, state, **outputs)
    print("[run_plan] Phase 0 state saved.")


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

def cmd_phase1(args: argparse.Namespace) -> None:
    """Run Phase 1 — Dataset preparation."""
    from phases import phase1_dataset

    state = load_state()

    outputs = phase1_dataset.run(
        roboflow_key=args.roboflow_key,
        skip_oxford=args.skip_oxford,
        state=state,
    )

    mark_phase_done(1, state, **outputs)
    print("[run_plan] Phase 1 state saved.")


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def cmd_download_model(args: argparse.Namespace) -> None:
    """
    Download a trained model from a Roboflow project and save it to
    weights/yolov8n_overhead.pt, then mark phases 1 and 2 as complete
    so phase3 can run immediately.
    """
    import shutil

    try:
        import roboflow as rf_module
    except ImportError:
        print("[run_plan] ERROR: pip install roboflow", file=sys.stderr)
        sys.exit(1)

    from phases.common import WEIGHTS_DIR, load_state, mark_phase_done, save_state

    api_key   = args.roboflow_key
    project   = args.roboflow_project          # e.g. coffee-shop-nhtr0
    workspace = args.roboflow_workspace or ""  # optional — uses default if blank
    version   = args.version                   # default 1

    print(f"[download-model] Connecting to Roboflow project '{project}' (version {version})...")

    try:
        rf = rf_module.Roboflow(api_key=api_key)
        ws = rf.workspace(workspace) if workspace else rf.workspace()
        proj = ws.project(project)
        ver  = proj.version(version)
    except Exception as exc:
        print(
            f"\n[run_plan] ERROR: Could not connect to project '{project}'.\n"
            f"  Make sure --roboflow-project matches the project ID shown in the\n"
            f"  Roboflow URL (e.g. app.roboflow.com/YOUR_WORKSPACE/coffee-shop-nhtr0).\n"
            f"  If your workspace slug differs from your display name, pass it with\n"
            f"  --roboflow-workspace YOUR_WORKSPACE_SLUG\n"
            f"  Original error: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = WEIGHTS_DIR / "yolov8n_overhead.pt"

    # Try direct .pt download (works on Roboflow Core/Enterprise plans)
    downloaded = False
    try:
        print(f"[download-model] Attempting .pt weight download...")
        tmp = WEIGHTS_DIR / f"{project}-v{version}"
        ver.download("yolov8", location=str(tmp))
        # Roboflow saves weights inside a nested directory
        candidates = list(tmp.rglob("*.pt"))
        if candidates:
            shutil.copy(candidates[0], dest)
            print(f"[download-model] Saved weights to {dest}")
            downloaded = True
        else:
            print("[download-model] No .pt file found in download — trying ONNX fallback.")
    except Exception as e:
        print(f"[download-model] .pt download failed ({e}) — trying ONNX fallback.")

    # Fallback: download ONNX and note that pipeline needs updating
    if not downloaded:
        try:
            tmp = WEIGHTS_DIR / f"{project}-v{version}-onnx"
            ver.download("onnx", location=str(tmp))
            candidates = list(tmp.rglob("*.onnx"))
            if candidates:
                onnx_dest = WEIGHTS_DIR / "yolov8n_overhead.onnx"
                shutil.copy(candidates[0], onnx_dest)
                print(
                    f"\n[download-model] Saved ONNX model to {onnx_dest}\n"
                    f"  Ultralytics can load ONNX directly. Update pipeline config:\n"
                    f"    config['yolo_model'] = 'weights/yolov8n_overhead.onnx'\n"
                )
                # Copy as .pt path so phase3 finds it at the standard location
                shutil.copy(candidates[0], dest.with_suffix(".onnx"))
                dest = onnx_dest
                downloaded = True
        except Exception as e2:
            print(f"[download-model] ONNX download also failed: {e2}", file=sys.stderr)

    if not downloaded:
        print(
            "\n[run_plan] ERROR: Could not download model weights.\n"
            "  Roboflow .pt download requires a Core/Enterprise plan.\n"
            "  Alternative: export the model manually from the Roboflow UI\n"
            "  (Versions → Deploy → Download) and place it at:\n"
            f"    {dest}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Mark phases 1 and 2 done so phase3 runs immediately
    state = load_state()
    mark_phase_done(1, state, source="roboflow-download", universe_dir="n/a",
                    oxford_dir="n/a", merged_dir="n/a",
                    data_yaml="n/a", dataset_summary="n/a")
    mark_phase_done(2, state, weights=str(dest), source="roboflow-download")
    print(f"[download-model] Phases 1 and 2 marked complete. Run phase3 next:\n"
          f"  python run_plan.py phase3 --video YOUR_VIDEO.mp4")


def cmd_phase2(args: argparse.Namespace) -> None:
    """Run Phase 2 — YOLOv8n fine-tuning."""
    from phases import phase2_finetune

    state = load_state()

    outputs = phase2_finetune.run(
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        state=state,
    )

    mark_phase_done(2, state, **outputs)
    print("[run_plan] Phase 2 state saved.")


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------

def cmd_phase3(args: argparse.Namespace) -> None:
    """Run Phase 3 — Evaluation + comparison report."""
    from phases import phase3_evaluate

    state = load_state()

    outputs = phase3_evaluate.run(
        video_path=Path(args.video),
        device=args.device,
        with_gt=args.with_gt,
        state=state,
    )

    mark_phase_done(3, state, **outputs)
    print("[run_plan] Phase 3 state saved.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_plan.py",
        description="Coffee-shop demo: phased research automation runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- status ----
    sub.add_parser("status", help="Print current plan status checklist.")

    # ---- phase0 ----
    p0 = sub.add_parser("phase0", help="Baseline characterization + Roboflow upload.")
    p0.add_argument(
        "--video", required=True,
        help="Path to the coffee-shop ceiling-camera video.",
    )
    p0.add_argument(
        "--roboflow-key", required=True,
        help="Roboflow API key.",
    )
    p0.add_argument(
        "--roboflow-workspace", default="",
        help="Roboflow workspace slug (required for upload).",
    )
    p0.add_argument(
        "--sample-every", type=int, default=10,
        help="Sample one frame every N frames (default: 10).",
    )
    p0.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
        help="Compute device for the pipeline (default: auto).",
    )

    # ---- phase1 ----
    p1 = sub.add_parser(
        "phase1",
        help="Download public overhead-person dataset from Roboflow Universe + Oxford Town Centre + merge.",
    )
    p1.add_argument(
        "--roboflow-key", required=True,
        help="Any valid Roboflow API key (free tier). Get one at app.roboflow.com → Settings → API Keys.",
    )
    p1.add_argument(
        "--skip-oxford", action="store_true",
        help="Skip downloading Oxford Town Centre (useful for quick testing).",
    )

    # ---- phase2 ----
    p2 = sub.add_parser("phase2", help="Fine-tune YOLOv8n.")
    p2.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50).",
    )
    p2.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size for training (default: 640).",
    )
    p2.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
        help="Compute device (default: auto → MPS > CUDA > CPU).",
    )
    p2.add_argument(
        "--batch", type=int, default=16,
        help="Training batch size (default: 16; reduce to 8 on CPU).",
    )

    # ---- download-model ----
    pdl = sub.add_parser(
        "download-model",
        help="Download a trained model from your own Roboflow project, skipping phases 1 and 2.",
    )
    pdl.add_argument("--roboflow-key", required=True, help="Roboflow API key.")
    pdl.add_argument(
        "--roboflow-project", required=True,
        help="Project ID as shown in the Roboflow URL (e.g. coffee-shop-nhtr0).",
    )
    pdl.add_argument(
        "--roboflow-workspace", default="",
        help="Workspace slug (optional — auto-detected from API key if omitted).",
    )
    pdl.add_argument(
        "--version", type=int, default=1,
        help="Dataset version number with trained weights (default: 1).",
    )

    # ---- phase3 ----
    p3 = sub.add_parser("phase3", help="Evaluate fine-tuned model; produce comparison report.")
    p3.add_argument(
        "--video", required=True,
        help="Path to the coffee-shop video (same as used in phase0).",
    )
    p3.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
        help="Compute device (default: auto).",
    )
    p3.add_argument(
        "--with-gt", action="store_true",
        help="[TODO] Enable ground-truth evaluation (not yet implemented).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _dispatch = {
        "status":         cmd_status,
        "phase0":         cmd_phase0,
        "phase1":         cmd_phase1,
        "phase2":         cmd_phase2,
        "phase3":         cmd_phase3,
        "download-model": cmd_download_model,
    }

    handler = _dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"\n[run_plan] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError as exc:
        print(f"\n[run_plan] NOT IMPLEMENTED: {exc}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n[run_plan] Interrupted by user.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
