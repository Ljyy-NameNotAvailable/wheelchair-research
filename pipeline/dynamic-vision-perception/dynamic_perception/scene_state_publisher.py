"""
scene_state_publisher.py — Layer 3: Scene State Extension.

Builds the extended scene state JSON and publishes it on the ROS2 topic
/wheelchair/scene_state as std_msgs/String (serialised JSON).

The schema extends the base pipeline scene state with four new fields:
  - masked_from_slam       (per-object bool)
  - unclassified_motion_regions (list from optical flow)
  - slam_dynamic_pixels_excluded (int, diagnostic counter)
  - above_stop_threshold   (per unclassified region bool)

The trajectory_stub.predict_trajectory() function computes velocity_mps and
time_to_collision_s for each track. See trajectory_stub.py for the Social LSTM
upgrade path.

Spec ref: dynamic-components-design.md, Sections 4–5, scene state schema.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Optional

import numpy as np

from .mask_generator import TrackBox, DYNAMIC_CLASSES
from .trajectory_stub import (
    TrackHistoryEntry,
    predict_trajectory,
    compute_time_to_collision,
)

logger = logging.getLogger(__name__)

# Wheelchair bounding radius used for TTC computation (metres)
WHEELCHAIR_RADIUS_M: float = 0.4
OBJECT_RADIUS_M: float = 0.3


# ---------------------------------------------------------------------------
# Per-object scene state builder
# ---------------------------------------------------------------------------

def build_object_entry(
    track: TrackBox,
    depth_image: np.ndarray,
    slam_pose: dict,
    unified_mask: np.ndarray,
    track_history: list[TrackHistoryEntry],
    camera_fx: float = 614.0,
    camera_fy: float = 614.0,
    camera_cx: float = 320.0,
    camera_cy: float = 240.0,
) -> dict:
    """
    Build the scene state entry for one tracked object.

    Computes:
      - 3-D position in the camera frame (from depth at bbox centroid)
      - velocity_mps from predict_trajectory() (Kalman linear for now)
      - approaching: True if the object is moving toward the wheelchair
      - time_to_collision_s using compute_time_to_collision()
      - masked_from_slam: True if any pixel in the bbox was in unified_mask

    Spec ref: Section 5 — full per-object schema.

    Args:
        track:        TrackBox for this object.
        depth_image:  (H, W) float32 depth in metres from RealSense.
        slam_pose:    Current wheelchair pose {"x", "y", "theta_deg"}.
        unified_mask: (H, W) bool mask from DynamicMaskGenerator.
        track_history: List of TrackHistoryEntry for this track (for trajectory).
        camera_fx/fy/cx/cy: Intrinsics for 3-D back-projection.

    Returns:
        dict matching the per-object schema in the spec.
    """
    x1, y1, x2, y2 = track.bbox_xyxy
    cx_px = (x1 + x2) // 2
    cy_px = (y1 + y2) // 2

    # Clamp centroid to valid range
    h, w = depth_image.shape[:2]
    cx_px = max(0, min(w - 1, cx_px))
    cy_px = max(0, min(h - 1, cy_px))

    # Distance from depth image at bbox centroid
    distance_m = float(depth_image[cy_px, cx_px])
    if distance_m <= 0.0 or not math.isfinite(distance_m):
        # Fall back to median of valid pixels in the bbox
        roi = depth_image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        valid = roi[roi > 0.0]
        distance_m = float(np.median(valid)) if len(valid) > 0 else 0.0

    # 3-D position via pinhole back-projection (camera frame, z-forward)
    if distance_m > 0.0:
        pos_x = (cx_px - camera_cx) * distance_m / camera_fx
        pos_y = (cy_px - camera_cy) * distance_m / camera_fy
        pos_z = distance_m
    else:
        pos_x, pos_y, pos_z = 0.0, 0.0, 0.0

    # Trajectory prediction (Kalman linear; Social LSTM upgrade: see trajectory_stub.py)
    trajectory = predict_trajectory(
        track_history if track_history else [
            TrackHistoryEntry(x_m=pos_x, y_m=pos_z, timestamp=time.time())
        ],
        horizon_s=3.0,
        step_s=0.5,
    )
    vx, vy = trajectory.velocity_mps

    # "approaching" = object moving toward wheelchair (negative z-axis component)
    # SPEC_AMBIGUITY: The spec does not define the approaching criterion precisely.
    # Conservative interpretation: approaching = True if the track's velocity has a
    # negative z component (closing distance toward camera / wheelchair).
    approaching = vx < 0.0 or vy < 0.0

    # Time to collision
    wheelchair_pos = (float(slam_pose.get("x", 0.0)), float(slam_pose.get("y", 0.0)))
    ttc = compute_time_to_collision(
        trajectory,
        wheelchair_pos=wheelchair_pos,
        wheelchair_radius_m=WHEELCHAIR_RADIUS_M,
        object_radius_m=OBJECT_RADIUS_M,
    )

    # masked_from_slam: any pixel in the bbox was excluded from SLAM this frame
    x1c = max(0, x1); y1c = max(0, y1)
    x2c = min(w, x2); y2c = min(h, y2)
    if x2c > x1c and y2c > y1c:
        roi_mask = unified_mask[y1c:y2c, x1c:x2c]
        masked_from_slam = bool(roi_mask.any())
    else:
        masked_from_slam = False

    return {
        "id": track.track_id,
        "class": track.class_name,
        "position_m": {
            "x": round(pos_x, 3),
            "y": round(pos_y, 3),
            "z": round(pos_z, 3),
        },
        "distance_m": round(distance_m, 3),
        "velocity_mps": {
            "x": round(vx, 3),
            "y": round(vy, 3),
        },
        "approaching": approaching,
        "time_to_collision_s": round(ttc, 2) if ttc is not None else None,
        "confidence": round(track.confidence, 3),
        "depth_source": "realsense",
        "masked_from_slam": masked_from_slam,
    }


# ---------------------------------------------------------------------------
# Scene state assembler
# ---------------------------------------------------------------------------

class SceneStateBuilder:
    """
    Assembles the full extended scene state JSON for one pipeline frame.

    Called by the ROS2 node after mask generation and SLAM protection are
    complete. Maintains per-track history for trajectory prediction.

    Spec ref: Sections 4 and 5 — scene state schema and optical flow safety layer.
    """

    def __init__(
        self,
        max_history_len: int = 30,
        camera_intrinsics: Optional[dict] = None,
    ) -> None:
        """
        Args:
            max_history_len:    Max track history entries kept per track ID.
                                At 10 FPS, 30 entries = 3 seconds of history.
            camera_intrinsics:  Optional dict with keys fx, fy, cx, cy.
                                Defaults to RealSense D435i approximate values.
        """
        self._track_histories: dict[int, list[TrackHistoryEntry]] = {}
        self._max_history_len = max_history_len

        if camera_intrinsics is None:
            camera_intrinsics = {
                "fx": 614.0, "fy": 614.0, "cx": 320.0, "cy": 240.0
            }
        self._cam = camera_intrinsics

    def _update_track_history(
        self,
        track: TrackBox,
        depth_image: np.ndarray,
        timestamp: float,
    ) -> None:
        """
        Append the current observation to the track's position history.

        World position is approximated as the camera-frame position (x, z)
        since full SLAM-frame projection requires the transform at each step.
        This is sufficient for the Kalman linear extrapolation in the current
        stub; the Social LSTM upgrade would use proper world coordinates.
        """
        x1, y1, x2, y2 = track.bbox_xyxy
        cx_px = max(0, min(depth_image.shape[1] - 1, (x1 + x2) // 2))
        cy_px = max(0, min(depth_image.shape[0] - 1, (y1 + y2) // 2))
        d = float(depth_image[cy_px, cx_px])
        if d <= 0.0:
            return  # invalid depth, skip this observation

        pos_x = (cx_px - self._cam["cx"]) * d / self._cam["fx"]

        tid = track.track_id
        if tid not in self._track_histories:
            self._track_histories[tid] = []
        history = self._track_histories[tid]
        history.append(TrackHistoryEntry(x_m=pos_x, y_m=d, timestamp=timestamp))
        if len(history) > self._max_history_len:
            history.pop(0)

    def _prune_stale_tracks(self, active_ids: set[int]) -> None:
        """Remove history for tracks that are no longer active."""
        stale = [tid for tid in self._track_histories if tid not in active_ids]
        for tid in stale:
            del self._track_histories[tid]

    def build(
        self,
        tracks: list[TrackBox],
        depth_image: np.ndarray,
        unified_mask: np.ndarray,
        slam_pose: dict,
        unclassified_motion_regions: list[dict],
        slam_dynamic_pixels_excluded: int,
        motion_mask_active: bool,
        timestamp: Optional[float] = None,
    ) -> dict:
        """
        Build and return the full extended scene state dict.

        Spec ref: Section 5 — complete schema with all four new fields.

        Args:
            tracks:                        Active ByteTrack tracks.
            depth_image:                   (H, W) float32 depth in metres.
            unified_mask:                  (H, W) bool mask.
            slam_pose:                     {"x", "y", "theta_deg"} from SLAMProtector.
            unclassified_motion_regions:   From RAFTSFlowProcessor.find_unclassified_*.
            slam_dynamic_pixels_excluded:  From DynamicMaskGenerator.count_masked_pixels().
            motion_mask_active:            True if RAFT-S has produced at least one result.
            timestamp:                     POSIX timestamp. Defaults to time.time().

        Returns:
            scene_state: dict ready for json.dumps().
        """
        ts = timestamp if timestamp is not None else time.time()

        # Update track histories and prune stale ones
        active_ids: set[int] = {t.track_id for t in tracks}
        self._prune_stale_tracks(active_ids)
        for track in tracks:
            self._update_track_history(track, depth_image, ts)

        # Build per-object entries
        objects: list[dict] = []
        for track in tracks:
            history = self._track_histories.get(track.track_id, [])
            entry = build_object_entry(
                track=track,
                depth_image=depth_image,
                slam_pose=slam_pose,
                unified_mask=unified_mask,
                track_history=history,
                camera_fx=self._cam["fx"],
                camera_fy=self._cam["fy"],
                camera_cx=self._cam["cx"],
                camera_cy=self._cam["cy"],
            )
            objects.append(entry)

        # nearest obstacle
        distances = [obj["distance_m"] for obj in objects if obj["distance_m"] > 0.0]
        nearest_obstacle_m = round(min(distances), 3) if distances else None

        scene_state = {
            "timestamp": round(ts, 3),
            "objects": objects,
            "motion_mask_active": motion_mask_active,
            "unclassified_motion_regions": unclassified_motion_regions,
            "slam_dynamic_pixels_excluded": slam_dynamic_pixels_excluded,
            "nearest_obstacle_m": nearest_obstacle_m,
            "slam_pose": {
                "x": round(float(slam_pose.get("x", 0.0)), 3),
                "y": round(float(slam_pose.get("y", 0.0)), 3),
                "theta_deg": round(float(slam_pose.get("theta_deg", 0.0)), 2),
            },
        }
        return scene_state

    def to_json(self, scene_state: dict) -> str:
        """
        Serialise a scene state dict to a JSON string.

        Args:
            scene_state: Output of build().

        Returns:
            JSON string suitable for publishing as std_msgs/String.
        """
        return json.dumps(scene_state, separators=(",", ":"))
