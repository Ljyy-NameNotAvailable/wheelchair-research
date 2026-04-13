"""
mask_generator.py — Layer 1: Dynamic Mask Generation.

Implements the two mask sources and their union:
  1. Detection mask — zero extra compute, reuses YOLO bounding boxes.
  2. Flow mask — from RAFT-S at 5 FPS with ego-motion compensation.
  3. Union: dynamic_mask = detection_mask | flow_mask

Between RAFT-S calls the flow mask is propagated forward via ByteTrack's
Kalman-predicted bounding boxes (NGD-SLAM-style propagation). This means
the SLAM thread is never blocked waiting for GPU inference.

Spec ref: dynamic-components-design.md, Section 2 (all subsections).
Thresholds from spec:
  θ      = 2.0 px/frame — pixel is dynamic, exclude from SLAM
  θ_stop = 8.0 px/frame — unclassified fast-moving object, trigger deceleration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data containers for tracks (minimal representation; full track data comes
# from the ROS2 node via Ultralytics model.track() results)
# ---------------------------------------------------------------------------

# YOLO class names that are treated as unconditionally dynamic
DYNAMIC_CLASSES: frozenset[str] = frozenset({"person", "wheelchair", "generic_obstacle"})


@dataclass
class TrackBox:
    """
    Lightweight representation of one ByteTrack track, sufficient for mask
    generation and Kalman-based propagation.

    Populated from Ultralytics model.track() output in the ROS2 node.
    """
    track_id: int
    class_name: str
    bbox_xyxy: tuple[int, int, int, int]    # pixel coords (x1, y1, x2, y2)
    confidence: float
    # Kalman-predicted position for the NEXT frame (propagation between RAFT calls)
    predicted_bbox_xyxy: Optional[tuple[int, int, int, int]] = None
    # 2-D velocity from ByteTrack Kalman filter, in pixels/frame
    velocity_px: tuple[float, float] = (0.0, 0.0)


# ---------------------------------------------------------------------------
# Mask generation core class
# ---------------------------------------------------------------------------

class DynamicMaskGenerator:
    """
    Produces a unified boolean mask marking all dynamic pixels in a frame.

    Usage (called once per camera frame at 30 FPS):
      generator = DynamicMaskGenerator(height=480, width=640)
      mask = generator.update(tracks, flow_result_or_none)

    The generator maintains state between RAFT-S calls so that the flow mask
    can be propagated forward using ByteTrack predictions at zero GPU cost.

    Spec ref: Section 2 — all subsections; Section 3 (SLAM integration context).
    """

    def __init__(
        self,
        height: int = 480,
        width: int = 640,
        flow_threshold_px: float = 2.0,
        stop_threshold_px: float = 8.0,
        raft_fps: int = 5,
        yolo_fps: int = 30,
    ) -> None:
        """
        Args:
            height:            Frame height in pixels.
            width:             Frame width in pixels.
            flow_threshold_px: θ — flow magnitude above which a pixel is dynamic.
            stop_threshold_px: θ_stop — flow magnitude triggering deceleration.
            raft_fps:          RAFT-S inference rate. Used only for logging.
            yolo_fps:          YOLO detection rate. Used only for logging.
        """
        self.height = height
        self.width = width
        self.flow_threshold_px = flow_threshold_px
        self.stop_threshold_px = stop_threshold_px

        # Last RAFT-S flow mask (refreshed at 5 FPS, propagated between calls)
        self._cached_flow_mask: np.ndarray = np.zeros(
            (height, width), dtype=bool
        )
        self._cached_stop_mask: np.ndarray = np.zeros(
            (height, width), dtype=bool
        )
        # The track boxes used to produce the cached flow mask (for propagation)
        self._tracks_at_last_flow_update: list[TrackBox] = []

        self._frame_count: int = 0
        logger.info(
            "DynamicMaskGenerator initialised: %dx%d, θ=%.1f px, θ_stop=%.1f px, "
            "RAFT-S %d FPS, YOLO %d FPS",
            width, height, flow_threshold_px, stop_threshold_px, raft_fps, yolo_fps,
        )

    # ------------------------------------------------------------------
    # Detection mask (Source 1)
    # ------------------------------------------------------------------

    def build_detection_mask(self, tracks: list[TrackBox]) -> np.ndarray:
        """
        Build a boolean mask from YOLO bounding boxes for dynamic-class objects.

        Any pixel inside a bounding box of a dynamic class is set to True.
        This costs ~0 ms as it purely reuses already-computed YOLO output.

        Spec ref: Section 2a — "any pixel inside a detected bounding box for
        a dynamic class (person, wheelchair, generic_obstacle) is marked dynamic."

        Args:
            tracks: List of active ByteTrack tracks for this frame.

        Returns:
            detection_mask: bool array (H, W).
        """
        mask = np.zeros((self.height, self.width), dtype=bool)
        for track in tracks:
            if track.class_name not in DYNAMIC_CLASSES:
                continue
            x1, y1, x2, y2 = track.bbox_xyxy
            # Clamp to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.width, x2)
            y2 = min(self.height, y2)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True
        return mask

    # ------------------------------------------------------------------
    # Flow mask propagation (between RAFT-S calls)
    # ------------------------------------------------------------------

    def _propagate_flow_mask(
        self,
        cached_mask: np.ndarray,
        old_tracks: list[TrackBox],
        new_tracks: list[TrackBox],
    ) -> np.ndarray:
        """
        Propagate the previous flow mask to the current frame using the
        displacement of matched ByteTrack bounding boxes.

        For each track that exists in both old_tracks and new_tracks, shift
        the masked region by the bounding-box centre displacement. For any
        track that only appears in new_tracks (new detection), add its bbox
        to the propagated mask unconditionally (conservative).

        This is the NGD-SLAM-style propagation described in:
          Zhang et al. (2024) — NGD-SLAM; adapted here per spec Section 3a:
          "propagate the mask using lightweight tracking rather than waiting
          for the next GPU inference."

        Spec ref: Section 2b — "between RAFT-S calls, propagate the previous
        flow mask forward using ByteTrack's Kalman-predicted bounding boxes."

        Args:
            cached_mask: Previous flow mask (H, W) bool.
            old_tracks:  Tracks at the time the cached mask was computed.
            new_tracks:  Tracks in the current frame.

        Returns:
            Propagated mask (H, W) bool.
        """
        propagated = np.zeros((self.height, self.width), dtype=bool)

        old_by_id: dict[int, TrackBox] = {t.track_id: t for t in old_tracks}
        new_by_id: dict[int, TrackBox] = {t.track_id: t for t in new_tracks}

        for tid, new_track in new_by_id.items():
            x1n, y1n, x2n, y2n = new_track.bbox_xyxy
            x1n = max(0, x1n); y1n = max(0, y1n)
            x2n = min(self.width, x2n); y2n = min(self.height, y2n)
            if x2n <= x1n or y2n <= y1n:
                continue

            if tid in old_by_id:
                old_track = old_by_id[tid]
                x1o, y1o, x2o, y2o = old_track.bbox_xyxy

                # Only propagate if the old bbox was actually masked
                old_cx = (x1o + x2o) // 2
                old_cy = (y1o + y2o) // 2
                old_cx = max(0, min(self.width - 1, old_cx))
                old_cy = max(0, min(self.height - 1, old_cy))

                if not cached_mask[old_cy, old_cx]:
                    # This track's bbox was not masked last frame; skip.
                    continue

            # The track's region contributes to the propagated mask
            propagated[y1n:y2n, x1n:x2n] = True

        return propagated

    # ------------------------------------------------------------------
    # Main update entry point
    # ------------------------------------------------------------------

    def update(
        self,
        tracks: list[TrackBox],
        flow_dynamic_mask: Optional[np.ndarray],
        flow_stop_mask: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the unified dynamic mask for one camera frame.

        Called at 30 FPS. flow_dynamic_mask is provided only on the 1-in-6
        frames when RAFT-S fires; on other frames both flow masks are None and
        the cached flow mask is propagated forward using track positions.

        Spec ref: Section 2c — "dynamic_mask = detection_mask | flow_mask".

        Args:
            tracks:            Active ByteTrack tracks for this frame.
            flow_dynamic_mask: (H, W) bool mask from RAFT-S, or None.
            flow_stop_mask:    (H, W) bool stop mask from RAFT-S, or None.

        Returns:
            unified_mask:     (H, W) bool — all dynamic pixels.
            detection_mask:   (H, W) bool — YOLO-only dynamic pixels.
            effective_flow:   (H, W) bool — flow or propagated flow mask used.
        """
        self._frame_count += 1

        # --- Source 1: detection mask (always available, 0 extra cost) ---
        detection_mask = self.build_detection_mask(tracks)

        # --- Source 2: flow mask (fresh or propagated) ---
        if flow_dynamic_mask is not None and flow_stop_mask is not None:
            # RAFT-S fired this frame — update the cache
            self._cached_flow_mask = flow_dynamic_mask.copy()
            self._cached_stop_mask = flow_stop_mask.copy()
            self._tracks_at_last_flow_update = list(tracks)
            effective_flow = flow_dynamic_mask
            logger.debug(
                "[mask_generator] Frame %d: fresh flow mask, "
                "%.1f%% pixels dynamic via flow.",
                self._frame_count,
                100.0 * float(flow_dynamic_mask.sum()) / flow_dynamic_mask.size,
            )
        else:
            # Between RAFT-S calls: propagate cached mask via track positions
            propagated = self._propagate_flow_mask(
                self._cached_flow_mask,
                self._tracks_at_last_flow_update,
                tracks,
            )
            effective_flow = propagated
            logger.debug(
                "[mask_generator] Frame %d: propagated flow mask.",
                self._frame_count,
            )

        # --- Union ---
        unified_mask = detection_mask | effective_flow

        return unified_mask, detection_mask, effective_flow

    def get_cached_stop_mask(self) -> np.ndarray:
        """
        Return the most recent stop-threshold mask (θ_stop = 8.0 px/frame).

        Updated only when RAFT-S fires (5 FPS). Between calls, the last known
        stop mask is used — conservative for safety purposes.

        Spec ref: Section 2 — θ_stop triggers immediate deceleration.

        Returns:
            stop_mask: (H, W) bool.
        """
        return self._cached_stop_mask

    def count_masked_pixels(self, mask: np.ndarray) -> int:
        """
        Return the number of True pixels in a mask.

        Used to populate slam_dynamic_pixels_excluded in the scene state.

        Spec ref: Section 5 — "slam_dynamic_pixels_excluded: diagnostic counter."

        Args:
            mask: Any boolean (H, W) array.

        Returns:
            Integer pixel count.
        """
        return int(mask.sum())
