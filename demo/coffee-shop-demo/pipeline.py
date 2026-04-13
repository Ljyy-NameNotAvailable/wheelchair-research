"""
Dynamic Perception Pipeline — Coffee Shop Demo
Implemented from the inline specification provided 2026-04-12.

Designed for a STATIC CEILING CAMERA.  Key simplifications vs. the full
wheelchair pipeline:
  - No ego-motion compensation: camera is fixed, so ALL optical flow comes
    from moving objects — no background subtraction needed beyond the flow
    magnitude threshold.
  - No SLAM / localization.
  - No depth estimation (2-D pixel positions only).
  - No ROS2: plain Python class interfaces.

OVERHEAD CAMERA NOTE:
YOLOv8n is trained on COCO eye-level images.  People viewed from a ceiling
camera appear as small ovals — detection confidence will be lower than in
standard use.  The optical flow layer compensates by detecting motion
independently of classification.  If YOLO misses a person they will still
appear in `unclassified_motion_regions`.  A future improvement would be to
fine-tune on an overhead pedestrian dataset (e.g. VIRAT or Oxford Town Centre).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "yolo_model": "yolov8n.pt",
    "flow_interval": 5,             # compute RAFT flow every N frames
    "flow_input_size": (512, 512),  # RAFT resize target (H, W)
    "theta": 2.0,                   # flow magnitude threshold (px/frame) — moving
    "theta_stop": 8.0,              # flow magnitude threshold — fast-moving
    "flow_opacity": 0.3,            # heatmap overlay opacity
    "device": "auto",               # "auto" → cuda if available, else cpu
    "overhead_camera": True,        # informational; future: adjust detection params
}

# YOLO class IDs used by this demo (COCO dataset numbering)
_TRACK_CLASSES = [0]       # person — tracked with ByteTrack
# SPEC_AMBIGUITY: spec says "use chairs(56)/cups(41) as static anchors" but
# does not define any downstream use for anchor detections. We define the IDs
# here for completeness but do not run a separate anchor pass — persons are
# the only tracked class and the only class in scene_state.
_ANCHOR_CLASSES = [56, 41]  # chair, cup — reserved for future use


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrackedObject:
    """Single tracked object extracted from a YOLO result."""
    track_id: int
    class_name: str
    bbox_px: list[int]    # [x1, y1, x2, y2]
    centroid_px: list[int]  # [cx, cy]
    confidence: float


@dataclass
class UnclassifiedRegion:
    """A connected-component region with optical flow but no YOLO match."""
    bbox_px: list[int]          # [x1, y1, x2, y2]
    area_px2: int
    mean_flow_magnitude: float
    above_stop_threshold: bool


@dataclass
class FrameResult:
    """All outputs produced by `DynamicPerceptionPipeline.process_frame()`."""
    annotated_frame: np.ndarray     # BGR uint8, same resolution as input
    scene_state: dict[str, Any]    # JSON-serialisable scene description
    dynamic_mask: np.ndarray       # uint8, HxW, values 0 or 255


# ---------------------------------------------------------------------------
# Helper — trajectory prediction stub
# ---------------------------------------------------------------------------

def _linear_extrapolate(
    history: list[tuple[int, int]],
    steps: int = 5,
) -> list[tuple[int, int]]:
    """
    Predict future centroids via linear velocity extrapolation.

    This is a minimal stub.  A production implementation would replace this
    with a Social LSTM or Social GAN model trained on pedestrian trajectories.

    Args:
        history: list of (cx, cy) centroids, oldest-first.
        steps:   how many future frames to predict.

    Returns:
        List of predicted (cx, cy) integer positions.

    # TODO: replace with Social LSTM plug-in:
    #   from social_lstm import SocialLSTM
    #   model = SocialLSTM.load_pretrained(...)
    #   return model.predict(history, steps)
    """
    if len(history) < 2:
        return []
    dx = history[-1][0] - history[-2][0]
    dy = history[-1][1] - history[-2][1]
    cx, cy = history[-1]
    return [(cx + dx * (i + 1), cy + dy * (i + 1)) for i in range(steps)]


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class DynamicPerceptionPipeline:
    """
    Frame-by-frame dynamic perception pipeline for a static ceiling camera.

    Usage::

        pipeline = DynamicPerceptionPipeline(config)
        result = pipeline.process_frame(frame_bgr, frame_idx=42, src_fps=30.0)
        cv2.imshow("demo", result.annotated_frame)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.cfg = cfg

        # Resolve compute device
        if cfg["device"] == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg["device"])

        print(f"[pipeline] Using device: {self.device}")

        # Load YOLOv8n with ByteTrack
        self.yolo = YOLO(cfg["yolo_model"])

        # Load RAFT-Small optical flow model (with OOM fallback)
        self._raft = self._load_raft()

        # Per-frame state
        self._prev_frame: np.ndarray | None = None       # previous BGR frame for RAFT
        self._flow_mask: np.ndarray | None = None        # carried-forward binary mask
        self._flow_magnitude: np.ndarray | None = None   # carried-forward float magnitude
        self._prev_centroids: dict[int, tuple[int, int]] = {}  # track_id → centroid
        self._prev_areas: dict[int, float] = {}           # track_id → bbox area (px^2)
        self._track_history: dict[int, list[tuple[int, int]]] = {}  # track_id → centroids

    # ------------------------------------------------------------------
    # RAFT loader
    # ------------------------------------------------------------------

    def _load_raft(self) -> torch.nn.Module:
        """Load RAFT-Small with automatic CUDA → CPU fallback on OOM."""
        weights = Raft_Small_Weights.DEFAULT
        try:
            model = raft_small(weights=weights).to(self.device).eval()
        except RuntimeError as exc:
            if "CUDA" in str(exc) or "out of memory" in str(exc).lower():
                warnings.warn(
                    f"[pipeline] CUDA OOM while loading RAFT — falling back to CPU. "
                    f"({exc})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.device = torch.device("cpu")
                model = raft_small(weights=weights).to(self.device).eval()
            else:
                raise
        return model

    # ------------------------------------------------------------------
    # Optical flow
    # ------------------------------------------------------------------

    def _compute_flow(
        self,
        prev_bgr: np.ndarray,
        curr_bgr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run RAFT-Small on a (prev, curr) BGR frame pair.

        Frames are resized to `flow_input_size` for RAFT, then flow vectors
        are rescaled back to the original resolution.  Because the camera is
        static, no ego-motion subtraction is applied.

        Returns:
            flow_mask_moving : uint8 HxW, 255 where magnitude >= theta
            flow_mask_fast   : uint8 HxW, 255 where magnitude >= theta_stop
            flow_magnitude   : float32 HxW, per-pixel flow magnitude (px/frame)
        """
        h, w = curr_bgr.shape[:2]
        H, W = self.cfg["flow_input_size"]  # (512, 512) by default
        theta = float(self.cfg["theta"])
        theta_stop = float(self.cfg["theta_stop"])

        def _to_tensor(bgr: np.ndarray) -> torch.Tensor:
            """Convert a BGR frame to a float32 NCHW tensor in [0, 255]."""
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float()
            t = TF.resize(t, [H, W], antialias=True)
            return t.unsqueeze(0).to(self.device)

        t1 = _to_tensor(prev_bgr)
        t2 = _to_tensor(curr_bgr)

        with torch.inference_mode():
            try:
                flow_preds = self._raft(t1, t2)
            except RuntimeError as exc:
                if "CUDA" in str(exc) or "out of memory" in str(exc).lower():
                    warnings.warn(
                        "[pipeline] CUDA OOM during RAFT forward — retrying on CPU.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._raft = self._raft.to("cpu")
                    self.device = torch.device("cpu")
                    t1 = t1.to("cpu")
                    t2 = t2.to("cpu")
                    flow_preds = self._raft(t1, t2)
                else:
                    raise

        # RAFT returns a list of predictions; take the last (finest-resolution)
        # Shape: (1, 2, H, W) → (H, W, 2)
        flow_small = flow_preds[-1].squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Upscale flow field to original resolution
        flow_full = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR)
        # Rescale flow magnitudes proportionally to the upscale ratio
        flow_full[:, :, 0] *= w / W
        flow_full[:, :, 1] *= h / H

        mag = np.sqrt(
            flow_full[:, :, 0] ** 2 + flow_full[:, :, 1] ** 2
        ).astype(np.float32)

        flow_mask_moving = (mag >= theta).astype(np.uint8) * 255
        flow_mask_fast = (mag >= theta_stop).astype(np.uint8) * 255

        return flow_mask_moving, flow_mask_fast, mag

    # ------------------------------------------------------------------
    # Detection mask
    # ------------------------------------------------------------------

    @staticmethod
    def _detection_mask(
        shape: tuple[int, int],
        objects: list[TrackedObject],
    ) -> np.ndarray:
        """Fill bounding boxes of all tracked persons into a binary mask."""
        mask = np.zeros(shape, dtype=np.uint8)
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox_px
            # Clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(shape[1], x2)
            y2 = min(shape[0], y2)
            mask[y1:y2, x1:x2] = 255
        return mask

    # ------------------------------------------------------------------
    # Unclassified motion regions
    # ------------------------------------------------------------------

    def _unclassified_regions(
        self,
        flow_mask: np.ndarray,
        detection_mask: np.ndarray,
        flow_magnitude: np.ndarray,
        flow_fast_mask: np.ndarray,
    ) -> list[UnclassifiedRegion]:
        """
        Find connected components that have optical flow but no YOLO detection.

        These are pixels in `flow_mask` that do NOT overlap any person bounding
        box in `detection_mask`.
        """
        # Pixels with motion but no YOLO match
        unclassified = cv2.bitwise_and(
            flow_mask, cv2.bitwise_not(detection_mask)
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            unclassified, connectivity=8
        )

        regions: list[UnclassifiedRegion] = []

        for lbl in range(1, num_labels):  # label 0 = background
            x = int(stats[lbl, cv2.CC_STAT_LEFT])
            y = int(stats[lbl, cv2.CC_STAT_TOP])
            bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            area = int(stats[lbl, cv2.CC_STAT_AREA])

            component_pixels = flow_magnitude[labels == lbl]
            mean_mag = float(np.mean(component_pixels)) if component_pixels.size > 0 else 0.0

            fast_pixels = flow_fast_mask[labels == lbl]
            above_stop = bool(np.any(fast_pixels > 0))

            regions.append(
                UnclassifiedRegion(
                    bbox_px=[x, y, x + bw, y + bh],
                    area_px2=area,
                    mean_flow_magnitude=round(mean_mag, 2),
                    above_stop_threshold=above_stop,
                )
            )

        return regions

    # ------------------------------------------------------------------
    # Velocity and approach detection
    # ------------------------------------------------------------------

    def _estimate_velocity(
        self,
        track_id: int,
        centroid: tuple[int, int],
    ) -> dict[str, float]:
        """Simple finite-difference velocity from centroid displacement (px/frame)."""
        if track_id in self._prev_centroids:
            px, py = self._prev_centroids[track_id]
            cx, cy = centroid
            return {"x": round(float(cx - px), 2), "y": round(float(cy - py), 2)}
        return {"x": 0.0, "y": 0.0}

    def _is_approaching(self, track_id: int, bbox: list[int]) -> bool:
        """Return True if bounding box area is increasing (person approaching camera)."""
        x1, y1, x2, y2 = bbox
        area = float((x2 - x1) * (y2 - y1))
        prev_area = self._prev_areas.get(track_id, area)
        return area > prev_area

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _draw_annotations(
        self,
        frame: np.ndarray,
        objects: list[TrackedObject],
        velocities: dict[int, dict[str, float]],
        unclassified: list[UnclassifiedRegion],
        flow_magnitude: np.ndarray | None,
        frame_idx: int,
        fps: float,
    ) -> np.ndarray:
        """
        Render all annotations onto a copy of `frame` and return it.

        Draws:
          - Flow magnitude heatmap at `flow_opacity` where magnitude >= theta
          - Green bboxes + track-ID labels + velocity arrows for tracked persons
          - Yellow dashed rects for unclassified motion regions
          - Red dashed rects for unclassified regions above theta_stop
          - Frame counter + FPS in top-left corner
        """
        out = frame.copy()
        theta = float(self.cfg["theta"])
        theta_stop = float(self.cfg["theta_stop"])
        opacity = float(self.cfg["flow_opacity"])

        # -- Flow heatmap overlay --
        if flow_magnitude is not None:
            norm_mag = np.clip(flow_magnitude / 20.0, 0.0, 1.0)  # 20 px/f = full colour
            heatmap = cv2.applyColorMap(
                (norm_mag * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            flow_region = flow_magnitude >= theta
            mask3 = np.stack([flow_region] * 3, axis=-1)
            blended = (
                out.astype(np.float32) * (1 - opacity)
                + heatmap.astype(np.float32) * opacity
            ).astype(np.uint8)
            out = np.where(mask3, blended, out)

        # -- Tracked persons --
        GREEN = (0, 200, 0)
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox_px
            cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, 2)
            label = f"#{obj.track_id} {obj.confidence:.2f}"
            cv2.putText(
                out, label, (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2,
            )
            vel = velocities.get(obj.track_id, {"x": 0.0, "y": 0.0})
            cx, cy = obj.centroid_px
            ARROW_SCALE = 5.0
            ex = int(cx + vel["x"] * ARROW_SCALE)
            ey = int(cy + vel["y"] * ARROW_SCALE)
            cv2.arrowedLine(out, (cx, cy), (ex, ey), GREEN, 2, tipLength=0.3)

        # -- Unclassified motion regions --
        for reg in unclassified:
            x1, y1, x2, y2 = reg.bbox_px
            # Red if above stop threshold, yellow otherwise
            color = (0, 0, 220) if reg.above_stop_threshold else (0, 220, 220)
            self._draw_dashed_rect(out, (x1, y1), (x2, y2), color, thickness=2, seg_len=8)
            mag_label = f"{reg.mean_flow_magnitude:.1f}px/f"
            cv2.putText(
                out, mag_label, (x1, max(y1 - 4, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
            )

        # -- Frame counter + FPS --
        info_text = f"Frame {frame_idx}  |  {fps:.1f} FPS"
        cv2.putText(
            out, info_text, (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        return out

    @staticmethod
    def _draw_dashed_rect(
        img: np.ndarray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
        seg_len: int,
    ) -> None:
        """Draw a dashed rectangle on `img` in-place."""
        x1, y1 = pt1
        x2, y2 = pt2
        sides = [
            ((x1, y1), (x2, y1)),
            ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)),
            ((x1, y2), (x1, y1)),
        ]
        for (ax, ay), (bx, by) in sides:
            dist = int(np.hypot(bx - ax, by - ay))
            if dist == 0:
                continue  # degenerate zero-length side — skip safely
            for i in range(0, dist, seg_len * 2):
                t0 = i / dist
                t1 = min((i + seg_len) / dist, 1.0)
                p0 = (int(ax + (bx - ax) * t0), int(ay + (by - ay) * t0))
                p1 = (int(ax + (bx - ax) * t1), int(ay + (by - ay) * t1))
                cv2.line(img, p0, p1, color, thickness)

    # ------------------------------------------------------------------
    # Trajectory prediction stub
    # ------------------------------------------------------------------

    def predict_trajectory(
        self,
        track_history: list[tuple[int, int]],
        steps: int = 5,
    ) -> list[tuple[int, int]]:
        """
        Trajectory prediction stub — linear velocity extrapolation.

        Per spec: "Include the same predict_trajectory(track_history) stub
        from the full pipeline (linear velocity extrapolation, Social LSTM
        plug-in comment). Keep it in pipeline.py as a method on
        DynamicPerceptionPipeline, called but not yet wired into the scene
        state output."

        Args:
            track_history: list of (cx, cy) centroids, oldest-first.
            steps: number of future frames to predict.

        Returns:
            List of predicted (cx, cy) positions.
        """
        return _linear_extrapolate(track_history, steps)

    # ------------------------------------------------------------------
    # Core entry point
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float = 0.0,
        src_fps: float = 0.0,
    ) -> FrameResult:
        """
        Process a single BGR frame and return a FrameResult.

        This method is I/O-free and accepts raw numpy arrays directly,
        making it independently testable without a video file.

        Args:
            frame     : BGR numpy array (H x W x 3, uint8).
            frame_idx : 0-based frame index.
            fps       : current *processing* fps (for overlay display).
            src_fps   : source video fps (used for timestamp_s calculation).
                        Falls back to `fps` if not provided.

        Returns:
            FrameResult containing annotated_frame, scene_state, dynamic_mask.

        Raises:
            RuntimeError: if YOLO or RAFT raise an unrecoverable error.
            ValueError:   if `frame` has unexpected shape or dtype.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"[pipeline] Expected a 3-channel BGR frame, got shape {frame.shape}"
            )

        h, w = frame.shape[:2]

        # Compute wall-clock timestamp using source video FPS where available
        _fps_for_ts = src_fps if src_fps > 0 else fps
        timestamp_s = round(frame_idx / _fps_for_ts, 3) if _fps_for_ts > 0 else 0.0

        # ---- Step 1: YOLO detection + ByteTrack tracking ----------------
        try:
            results = self.yolo.track(
                frame,
                persist=True,
                classes=_TRACK_CLASSES,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"[pipeline] YOLO tracking failed: {exc}") from exc

        tracked_objects: list[TrackedObject] = []
        velocities: dict[int, dict[str, float]] = {}

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue  # ByteTrack hasn't assigned an ID yet
                track_id = int(box.id.item())
                cls_id = int(box.cls.item())
                cls_name = self.yolo.names[cls_id]
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                centroid: tuple[int, int] = (cx, cy)

                obj = TrackedObject(
                    track_id=track_id,
                    class_name=cls_name,
                    bbox_px=[x1, y1, x2, y2],
                    centroid_px=[cx, cy],
                    confidence=conf,
                )
                tracked_objects.append(obj)

                # Velocity (finite difference from previous centroid)
                vel = self._estimate_velocity(track_id, centroid)
                velocities[track_id] = vel

                # Maintain centroid history for trajectory prediction stub
                hist = self._track_history.setdefault(track_id, [])
                hist.append(centroid)
                if len(hist) > 30:
                    hist.pop(0)
                # Called per spec — result intentionally unused here
                self.predict_trajectory(hist)

        # ---- Step 2: Optical flow (every flow_interval frames) ----------
        flow_computed_this_frame = False
        flow_fast_mask: np.ndarray | None = None

        if (
            self._prev_frame is not None
            and frame_idx % self.cfg["flow_interval"] == 0
        ):
            flow_computed_this_frame = True
            try:
                fm, ffm, fmag = self._compute_flow(self._prev_frame, frame)
            except Exception as exc:
                raise RuntimeError(
                    f"[pipeline] Optical flow computation failed: {exc}"
                ) from exc
            self._flow_mask = fm
            self._flow_magnitude = fmag
            flow_fast_mask = ffm

        # Carry forward last known flow masks (safe for static camera)
        current_flow_mask = self._flow_mask
        current_flow_mag = self._flow_magnitude

        # ---- Step 3: Detection mask -------------------------------------
        det_mask = self._detection_mask((h, w), tracked_objects)

        # ---- Step 4: Dynamic mask = detection | flow --------------------
        if current_flow_mask is not None:
            dynamic_mask = cv2.bitwise_or(det_mask, current_flow_mask)
        else:
            dynamic_mask = det_mask.copy()

        # ---- Step 5: Unclassified motion regions ------------------------
        unclassified: list[UnclassifiedRegion] = []
        if current_flow_mask is not None and current_flow_mag is not None:
            # Reconstruct fast mask from carried-forward magnitude if needed
            if flow_fast_mask is None:
                theta_stop = float(self.cfg["theta_stop"])
                flow_fast_mask = (current_flow_mag >= theta_stop).astype(np.uint8) * 255
            unclassified = self._unclassified_regions(
                current_flow_mask, det_mask, current_flow_mag, flow_fast_mask
            )

        # ---- Step 6: Build scene_state ----------------------------------
        objects_out: list[dict[str, Any]] = []
        for obj in tracked_objects:
            bbox = obj.bbox_px
            area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            vel = velocities.get(obj.track_id, {"x": 0.0, "y": 0.0})
            approaching = self._is_approaching(obj.track_id, bbox)

            # masked_from_static_bg: True if centroid falls in flow_mask
            cx, cy = obj.centroid_px
            in_flow = bool(
                current_flow_mask is not None
                and 0 <= cy < h
                and 0 <= cx < w
                and current_flow_mask[cy, cx] > 0
            )

            objects_out.append(
                {
                    "id": obj.track_id,
                    "class": obj.class_name,
                    "bbox_px": bbox,
                    "centroid_px": obj.centroid_px,
                    "velocity_px_per_frame": vel,
                    "approaching_camera": approaching,
                    "confidence": round(obj.confidence, 4),
                    "masked_from_static_bg": in_flow,
                }
            )

            # Update per-track state for next frame
            self._prev_centroids[obj.track_id] = centroid  # type: ignore[assignment]
            self._prev_areas[obj.track_id] = area

        dynamic_px_count = int(np.count_nonzero(dynamic_mask))

        # Distance from frame centre to nearest tracked person centroid
        nearest_px: float | None = None
        frame_cx, frame_cy = w / 2.0, h / 2.0
        for obj in tracked_objects:
            cx, cy = obj.centroid_px
            d = float(np.hypot(cx - frame_cx, cy - frame_cy))
            if nearest_px is None or d < nearest_px:
                nearest_px = d

        scene_state: dict[str, Any] = {
            "frame_idx": frame_idx,
            "timestamp_s": timestamp_s,
            "objects": objects_out,
            "unclassified_motion_regions": [
                {
                    "bbox_px": r.bbox_px,
                    "area_px2": r.area_px2,
                    "mean_flow_magnitude": r.mean_flow_magnitude,
                    "above_stop_threshold": r.above_stop_threshold,
                }
                for r in unclassified
            ],
            "flow_computed_this_frame": flow_computed_this_frame,
            "dynamic_pixel_count": dynamic_px_count,
            "nearest_person_px": (
                round(nearest_px, 2) if nearest_px is not None else None
            ),
        }

        # ---- Step 7: Visualization --------------------------------------
        annotated = self._draw_annotations(
            frame,
            tracked_objects,
            velocities,
            unclassified,
            current_flow_mag,
            frame_idx,
            fps,
        )

        # ---- Update per-frame state for next call -----------------------
        self._prev_frame = frame.copy()

        return FrameResult(
            annotated_frame=annotated,
            scene_state=scene_state,
            dynamic_mask=dynamic_mask,
        )
