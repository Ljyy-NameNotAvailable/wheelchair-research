"""
trajectory_stub.py — Kalman linear extrapolation + Social LSTM integration stub.

Spec ref: dynamic-components-design.md, Section 9 — "Integration Point for
Trajectory Prediction."

Current implementation:
  predict_trajectory() returns ByteTrack's Kalman linear extrapolation — a
  constant-velocity linear prediction over a short horizon.

Future implementation:
  Replace or augment the return value of predict_trajectory() with a
  `predicted_trajectory` array from a Social LSTM node (Alahi et al., 2016)
  trained on JRDB indoor pedestrian data. See the SOCIAL LSTM PLUG-IN POINT
  comment block inside predict_trajectory() for the exact replacement location.

Spec note on the upgrade path (Section 9):
  "Replace or augment the velocity_mps field with a predicted_trajectory array:
   a sequence of (x, y, t) waypoints over a 2–3 second horizon."
  "The time_to_collision_s field becomes based on the predicted path rather
   than linear extrapolation."
  "A Social LSTM node … subscribing to the per-track position history published
   by ByteTrack and publishing the predicted trajectory for each active track.
   This node is entirely decoupled from Layers 1–3 and can be toggled off
   without affecting SLAM protection or optical flow masking."
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Track history entry
# ---------------------------------------------------------------------------

@dataclass
class TrackHistoryEntry:
    """
    One timestamped observation of a tracked object's world position.

    x_m, y_m: 2-D world position in metres (from SLAM coordinate frame).
    timestamp: POSIX timestamp in seconds.
    """
    x_m: float
    y_m: float
    timestamp: float


@dataclass
class PredictedTrajectory:
    """
    Output of predict_trajectory() — a sequence of future waypoints.

    waypoints: list of (x_m, y_m, t_s) tuples. t_s is the predicted time
               (POSIX seconds) at which the object will be at (x_m, y_m).
    velocity_mps: instantaneous velocity vector (vx, vy) in m/s.
    source: "kalman_linear" (current) or "social_lstm" (future).
    """
    waypoints: list[tuple[float, float, float]]  # (x_m, y_m, t_s)
    velocity_mps: tuple[float, float]
    source: str = "kalman_linear"


# ---------------------------------------------------------------------------
# Trajectory prediction function
# ---------------------------------------------------------------------------

def predict_trajectory(
    track_history: list[TrackHistoryEntry],
    horizon_s: float = 3.0,
    step_s: float = 0.5,
    social_lstm_client=None,
) -> PredictedTrajectory:
    """
    Predict the future trajectory of a tracked object over a time horizon.

    Current implementation: Kalman constant-velocity linear extrapolation
    using the last two observations from ByteTrack's position history.

    -----------------------------------------------------------------------
    SOCIAL LSTM PLUG-IN POINT:

    To upgrade this function to Social LSTM (Alahi et al., 2016):

    1. Launch a separate ROS2 node (e.g. social_lstm_node) that:
       - Subscribes to /wheelchair/track_histories (per-track position lists)
       - Runs the Social LSTM model (trained on JRDB indoor pedestrian data)
       - Publishes /wheelchair/predicted_trajectories as a custom message

    2. In this function, replace the linear extrapolation block (marked below
       with #--- LINEAR EXTRAPOLATION ---) with a call to the Social LSTM
       ROS2 service or an async result from the social_lstm_client arg:

         if social_lstm_client is not None:
             return social_lstm_client.predict(track_history, horizon_s, step_s)

    3. The returned PredictedTrajectory.source should be set to "social_lstm".

    4. The time_to_collision_s computation in scene_state_publisher.py uses
       the waypoints list — it will automatically benefit from Social LSTM
       without any further changes to Layers 1–3.

    Open question (from the Dynamic Environment Understanding review, 2026-04-10):
      The latency of Social LSTM on Jetson Orin Nano running concurrently with
      YOLOv8n + RAFT-S has not been characterised. Empirical profiling in the
      target environment is the primary required contribution before deploying
      Social LSTM in the fast loop.
    -----------------------------------------------------------------------

    Args:
        track_history:       Ordered list of TrackHistoryEntry, most recent last.
                             At least 2 entries are needed for velocity estimation.
        horizon_s:           Prediction horizon in seconds. Default 3.0 s.
        step_s:              Waypoint spacing in seconds. Default 0.5 s.
        social_lstm_client:  Optional Social LSTM client. If not None and the
                             Social LSTM node is available, it will be used instead
                             of linear extrapolation. Currently always None.

    Returns:
        PredictedTrajectory with linear-extrapolated waypoints.

    Raises:
        ValueError: If track_history is empty.
    """
    if not track_history:
        raise ValueError(
            "predict_trajectory requires at least one TrackHistoryEntry. "
            "Got empty history."
        )

    # --- SOCIAL LSTM PLUG-IN (future): replace block below when client is available ---
    if social_lstm_client is not None:
        # Future: return social_lstm_client.predict(track_history, horizon_s, step_s)
        logger.warning(
            "[trajectory_stub] social_lstm_client provided but Social LSTM "
            "integration is not yet implemented. Falling back to linear extrapolation."
        )

    # --- LINEAR EXTRAPOLATION (current implementation) ---
    # Estimate velocity from the most recent pair of observations.
    if len(track_history) >= 2:
        p_curr = track_history[-1]
        p_prev = track_history[-2]
        dt = p_curr.timestamp - p_prev.timestamp
        if dt > 1e-6:
            vx = (p_curr.x_m - p_prev.x_m) / dt
            vy = (p_curr.y_m - p_prev.y_m) / dt
        else:
            vx, vy = 0.0, 0.0
    else:
        # Only one observation — cannot estimate velocity.
        p_curr = track_history[-1]
        vx, vy = 0.0, 0.0

    # Generate waypoints at fixed time steps over the horizon
    now = p_curr.timestamp
    n_steps = max(1, int(horizon_s / step_s))
    waypoints: list[tuple[float, float, float]] = []
    for i in range(1, n_steps + 1):
        t_future = now + i * step_s
        x_future = p_curr.x_m + vx * (i * step_s)
        y_future = p_curr.y_m + vy * (i * step_s)
        waypoints.append((round(x_future, 4), round(y_future, 4), round(t_future, 3)))

    return PredictedTrajectory(
        waypoints=waypoints,
        velocity_mps=(round(vx, 4), round(vy, 4)),
        source="kalman_linear",
    )


# ---------------------------------------------------------------------------
# Time-to-collision helper
# ---------------------------------------------------------------------------

def compute_time_to_collision(
    trajectory: PredictedTrajectory,
    wheelchair_pos: tuple[float, float],
    wheelchair_radius_m: float = 0.4,
    object_radius_m: float = 0.3,
) -> Optional[float]:
    """
    Estimate time-to-collision from a predicted trajectory.

    Checks each waypoint of the trajectory to find the earliest time at which
    the predicted object position is within collision distance of the wheelchair.

    With Social LSTM (future): this function works unchanged because it operates
    on the waypoints list — Social LSTM produces the same schema.

    Spec ref: Scene state field "time_to_collision_s" — currently linear
    extrapolation; will become trajectory-based when Social LSTM is active.

    Args:
        trajectory:           Output of predict_trajectory().
        wheelchair_pos:       (x_m, y_m) of the wheelchair in world frame.
        wheelchair_radius_m:  Collision radius of the wheelchair. Default 0.4 m.
        object_radius_m:      Collision radius of the tracked object. Default 0.3 m.

    Returns:
        Estimated time to collision in seconds from now, or None if no collision
        is predicted within the trajectory horizon.
    """
    collision_dist = wheelchair_radius_m + object_radius_m
    wx, wy = wheelchair_pos

    for x_m, y_m, t_s in trajectory.waypoints:
        dist = math.hypot(x_m - wx, y_m - wy)
        if dist < collision_dist:
            # Approximate: return time from now to this waypoint
            # (the first waypoint is step_s seconds from now)
            now = trajectory.waypoints[0][2] - (
                trajectory.waypoints[1][2] - trajectory.waypoints[0][2]
                if len(trajectory.waypoints) > 1 else 0.5
            )
            return max(0.0, round(t_s - now, 2))

    return None
