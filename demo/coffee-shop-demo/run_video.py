"""
run_video.py — CLI entry point for the Coffee Shop Dynamic Perception Demo.

Usage:
    python run_video.py --input video.mp4 --output out.mp4 [options]

See --help for all options.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import DEFAULT_CONFIG, DynamicPerceptionPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the coffee-shop dynamic perception pipeline on a video file."
    )
    p.add_argument("--input", required=True, help="Path to input video file.")
    p.add_argument(
        "--output",
        default="output.mp4",
        help="Path to output annotated video (default: output.mp4).",
    )
    p.add_argument(
        "--flow-interval",
        type=int,
        default=DEFAULT_CONFIG["flow_interval"],
        help=(
            f"Compute optical flow every N frames "
            f"(default: {DEFAULT_CONFIG['flow_interval']})."
        ),
    )
    p.add_argument(
        "--theta",
        type=float,
        default=DEFAULT_CONFIG["theta"],
        help=(
            f"Flow magnitude threshold for moving pixels (px/frame) "
            f"(default: {DEFAULT_CONFIG['theta']})."
        ),
    )
    p.add_argument(
        "--theta-stop",
        type=float,
        default=DEFAULT_CONFIG["theta_stop"],
        help=(
            f"Flow magnitude threshold for fast-moving pixels (px/frame) "
            f"(default: {DEFAULT_CONFIG['theta_stop']})."
        ),
    )
    p.add_argument(
        "--log",
        default=None,
        help="Path to output JSONL scene log (optional). One JSON object per line.",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device: auto, cpu, or cuda (default: auto).",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show live preview window while processing. Press 'q' to quit early.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(
            f"[run_video] ERROR: Input file not found: {input_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build config from CLI arguments, overriding defaults
    config = {
        **DEFAULT_CONFIG,
        "flow_interval": args.flow_interval,
        "theta": args.theta,
        "theta_stop": args.theta_stop,
        "device": args.device,
    }

    print(f"[run_video] Opening: {input_path}")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(
            f"[run_video] ERROR: Cannot open video: {input_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"[run_video] Video: {frame_w}x{frame_h} @ {src_fps:.1f} fps, "
        f"{total_frames} frames"
    )

    # Output video writer
    out_path = Path(args.output)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (frame_w, frame_h))
    if not writer.isOpened():
        print(
            f"[run_video] ERROR: Cannot open output writer: {out_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Optional JSONL scene log
    log_file = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")
        print(f"[run_video] Scene log: {log_path}")

    # Initialise pipeline
    print("[run_video] Initialising pipeline...")
    pipeline = DynamicPerceptionPipeline(config)

    # Progress bar — graceful fallback if tqdm is not installed
    try:
        from tqdm import tqdm
        progress = tqdm(total=total_frames, unit="frame", desc="Processing")
    except ImportError:
        progress = None
        print("[run_video] tqdm not found — no progress bar.")

    frame_idx = 0
    t_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.perf_counter() - t_start
        # processing fps for overlay display
        current_fps = frame_idx / elapsed if elapsed > 0 else 0.0

        result = pipeline.process_frame(
            frame,
            frame_idx,
            fps=current_fps,
            src_fps=src_fps,
        )

        # Write annotated frame to output video
        writer.write(result.annotated_frame)

        # Append JSON scene state to JSONL log
        if log_file is not None:
            log_file.write(json.dumps(result.scene_state) + "\n")

        # Optional live preview
        if args.show:
            cv2.imshow("Coffee Shop Demo", result.annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[run_video] User pressed 'q' — stopping early.")
                break

        frame_idx += 1
        if progress is not None:
            progress.update(1)
        elif frame_idx % 50 == 0:
            print(f"[run_video] {frame_idx}/{total_frames} frames processed")

    # Cleanup
    if progress is not None:
        progress.close()
    cap.release()
    writer.release()
    if log_file is not None:
        log_file.close()
    if args.show:
        cv2.destroyAllWindows()

    total_time = time.perf_counter() - t_start
    avg_fps = frame_idx / total_time if total_time > 0 else 0.0
    print(
        f"\n[run_video] Done. Processed {frame_idx} frames in {total_time:.1f}s "
        f"({avg_fps:.1f} fps average)."
    )
    print(f"[run_video] Output video: {out_path.resolve()}")
    if args.log:
        print(f"[run_video] Scene log: {Path(args.log).resolve()}")


if __name__ == "__main__":
    main()
