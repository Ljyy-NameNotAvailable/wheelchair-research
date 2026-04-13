# Dynamic Components Design — Vision-Only Wheelchair Perception Pipeline

**Date:** 2026-04-12
**Scope:** Implementation design for the dynamic perception sub-system (Stages 3–4 extension + optical flow layer)
**Depends on:** `pipeline-proposal.md` (vision-only pipeline, same directory)
**Literature basis:** Dynamic Environment Understanding review (2026-04-10), Vision-Only Pipeline review (2026-04-12)

---

## 1. The Core Problem

The vision-only pipeline has a structural tension between its two primary outputs:

- **Static map** (from ORB-SLAM3 / RTAB-Map): requires a stable, feature-rich representation of walls, doorways, and fixed furniture
- **Dynamic objects** (from YOLO + ByteTrack): the very people moving through those corridors

If a person walks through the camera frame while ORB-SLAM3 extracts ORB features, their body becomes a source of spurious keypoints that corrupt the static map. On a subsequent pass through the same corridor, the wheelchair may try to localize against a phantom "wall" where a person was standing.

The dynamic component sub-system's job is to **maintain a clean boundary between static and dynamic** throughout the pipeline. There are three implementation layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 1 — Dynamic Mask Generation                                       │
│  YOLO boxes + RAFT-S optical flow → unified dynamic mask                 │
└─────────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 2 — SLAM Protection                                               │
│  Mask → exclude dynamic pixels from ORB-SLAM3 / RTAB-Map feature set    │
└─────────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 3 — Scene State Extension                                         │
│  Motion mask output + flow magnitude → enrich scene state for slow loop  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer 1 — Dynamic Mask Generation

### 2a. Detection-Based Mask (zero extra compute)

YOLO already runs at every fast-loop frame producing bounding boxes. These boxes can directly define the dynamic mask: any pixel inside a detected `person`, `wheelchair`, or other classified dynamic object bounding box is excluded from SLAM feature extraction.

**Cost:** zero additional compute — this reuses existing YOLO output.
**Limitation:** YOLO operates on RGB frames with a detection confidence threshold. A person partially occluded by a door frame may produce a low-confidence box that gets discarded, leaving that region unmasked. Fast-moving unclassified objects (a hospital cart moving quickly) will be entirely missed.

This is the baseline. The mask from this source alone is sufficient in sparse environments.

### 2b. Optical Flow Mask (catches what YOLO misses)

RAFT-S (Teed & Deng, 2020) runs at 5–10 FPS on the Orin Nano GPU alongside YOLO, producing a dense optical flow field. Any pixel where flow magnitude exceeds a threshold θ is flagged as dynamic, regardless of classification.

```
flow_mask = (flow_magnitude > θ)      # θ ≈ 2.0 px/frame empirically
```

This catches:
- Fast-moving objects between YOLO frames (cart in a corridor)
- Partially occluded people where YOLO confidence is below threshold
- Any object class YOLO was not trained on

**Cost:** ~20 ms GPU per RAFT-S call at 5 FPS → amortized ~3 ms per YOLO frame (1 flow call per 6 detection frames).
**Limitation:** Camera motion during the wheelchair's own movement also produces optical flow. Must compensate using ego-motion from the SLAM pose or IMU (RealSense D435i has an integrated IMU). Without ego-motion compensation, any wheelchair turn will produce spurious flow across the entire frame.

**Ego-motion compensation:** subtract the expected background flow field derived from the wheelchair's current angular velocity (from IMU) and linear velocity estimate (from SLAM). Pixels where residual flow > θ after subtraction are dynamic.

### 2c. Unified Mask

```
dynamic_mask = detection_mask | flow_mask
```

The union of both masks. Pixels in either source are excluded from SLAM feature extraction.

For the SLAM input, this mask is applied before ORB feature extraction (ORB-SLAM3) or before the RTAB-Map visual odometry step. Neither SLAM library requires modification to accept this mask — both allow a per-frame exclusion mask at their input interface.

---

## 3. Layer 2 — SLAM Integration

### 3a. ORB-SLAM3 (recommended for monocular / stereo / RGB-D)

ORB-SLAM3 (Campos et al., 2021) exposes the mask input at the `TrackRGBD()` / `TrackStereo()` call signature. Pass the dynamic mask as the optional exclusion mask argument:

```python
# ROS2 node: feed mask at each tracking call
slam.TrackRGBD(im_rgb, im_depth, timestamp, mask=dynamic_mask)
```

ORB feature extraction skips masked pixels entirely. The tracking thread runs at 30–40 FPS unblocked because the mask computation (YOLO boxes: ~0 ms, flow: deferred to 5 FPS) does not block the tracking call.

This is structurally the same approach validated by Bescos et al. (2018) in DynaSLAM, which demonstrated that masking dynamic regions before ORB-SLAM2 feature extraction substantially outperforms unmodified visual SLAM in dynamic environments. The key architectural difference here: DynaSLAM uses deep learning for the mask generation and blocks the tracking thread waiting for GPU inference. The design above avoids this by:
1. Using YOLO's existing output (already running in the fast loop) for the primary mask
2. Running RAFT-S at a lower rate than the tracking thread, so the flow mask is stale by at most one flow interval (200 ms at 5 FPS)

**Stale mask behavior:** between RAFT-S calls, the flow mask from the previous call is propagated forward by ByteTrack's predicted bounding boxes. This is the core insight from NGD-SLAM (Zhang et al., 2024): propagate the mask using lightweight tracking rather than waiting for the next GPU inference, achieving GPU-free SLAM performance at 60 FPS on CPU. The same propagation logic applies here — ByteTrack already predicts the next position of each tracked object via Kalman filter, so the mask can be updated each frame using the predicted boxes without waiting for YOLO or RAFT-S.

### 3b. RTAB-Map (preferred when Nav2 occupancy grid output is required)

RTAB-Map (Labbe & Michaud, 2019) has a built-in dynamic object detection parameter (`Mem/NotLinkedNodesKept=false`) and can accept an external segmentation mask via the `RGBD/ProximityPathMaxNeighbors` pipeline. The simpler integration path:

1. Publish the dynamic mask as a separate ROS2 topic
2. Use the RTAB-Map ROS2 wrapper's mask input parameter to subscribe to it
3. RTAB-Map excludes masked regions from its loop closure detection, preventing dynamic objects from creating false loop closures

RTAB-Map's memory management (limiting the active nodes to maintain real-time performance as the map grows) interacts correctly with this masking — masked frames that produce fewer landmarks do not degrade loop closure quality because RTAB-Map selectively retains high-landmark frames.

---

## 4. Layer 3 — Optical Flow as Independent Safety Layer

Beyond its role in the dynamic mask, RAFT-S provides a standalone output: a **motion mask** that represents any moving region in the frame, with a per-pixel flow magnitude.

This is published as a separate field in the scene state and used by the fast loop independently of object classification:

**If flow magnitude exceeds a second, higher threshold θ_stop:**
- Trigger immediate deceleration regardless of YOLO classification
- A fast-moving unclassified object is treated as an obstacle by default
- This is the "if in doubt, slow down" rule implemented at the sensor level

**If flow magnitude exceeds θ but is below θ_stop:**
- Log the motion region as an unclassified dynamic object
- Publish bounding box + magnitude to the scene state
- ByteTrack receives this as a low-confidence detection and attempts to assign an ID

This design mirrors the approach validated by Ou et al. (2022) in their indoor assistive navigation system: panoptic segmentation provides semantic labels for known object classes, while geometric cues handle objects outside the segmentation vocabulary. Here, RAFT-S serves the geometric cue role — motion is motion, regardless of what the moving object is.

---

## 5. Scene State Extensions

The existing scene state schema (from `pipeline-proposal.md`) is extended with four new fields:

```json
{
  "timestamp": 1744400000.456,
  "objects": [
    {
      "id": 3,
      "class": "person",
      "position_m": {"x": 1.1, "y": 0.2, "z": 0.0},
      "distance_m": 1.12,
      "velocity_mps": {"x": -0.4, "y": 0.1},
      "approaching": true,
      "time_to_collision_s": 2.8,
      "confidence": 0.91,
      "depth_source": "realsense",
      "masked_from_slam": true
    }
  ],
  "motion_mask_active": true,
  "unclassified_motion_regions": [
    {
      "bbox": [320, 100, 420, 300],
      "flow_magnitude_px": 8.3,
      "area_px2": 30000,
      "above_stop_threshold": false
    }
  ],
  "slam_dynamic_pixels_excluded": 14200,
  "nearest_obstacle_m": 1.12,
  "slam_pose": {"x": 3.2, "y": 1.5, "theta_deg": 12.4}
}
```

New fields:
- `masked_from_slam` (per object): whether this track's region was excluded from SLAM this frame
- `unclassified_motion_regions`: flow-detected motion not matched to any YOLO track
- `slam_dynamic_pixels_excluded`: diagnostic counter — if this is high, the map is under stress
- `above_stop_threshold` (per unclassified region): whether this region triggered the immediate deceleration rule

---

## 6. Compute Budget

All additions on top of the existing vision-only pipeline:

| Component | Frequency | GPU cost | CPU cost |
|-----------|-----------|----------|----------|
| Detection mask (from YOLO boxes) | 30 FPS | ~0 ms (reuse) | ~0.5 ms |
| RAFT-S optical flow | 5 FPS | ~20 ms | — |
| Flow mask + union | 5 FPS | — | ~1 ms |
| ByteTrack mask propagation between flow calls | 30 FPS | — | ~1 ms extra |
| ORB-SLAM3 (with mask, fewer features) | 10 FPS | — | ≤30 ms (faster than unmasked) |

**Net GPU impact:** RAFT-S at 5 FPS adds ~20 ms every 200 ms. Between RAFT-S calls, no additional GPU load. This can be scheduled in the gap between YOLO inferences without contention if YOLO runs at ~12 ms per frame (30 FPS) and RAFT-S fires once every 6 frames.

**On Jetson Orin Nano with CUDA streams:** YOLO and RAFT-S can share the GPU if placed on separate CUDA streams with priority. YOLO gets the high-priority stream; RAFT-S runs in a lower-priority stream and yields when YOLO needs the bus.

---

## 7. Implementation Order

This sub-system can be added after the base vision-only pipeline is running:

1. **Step 1**: Add detection-based mask → feed YOLO boxes directly to ORB-SLAM3 mask input. Verify the SLAM map stays clean when a person walks through the frame.
2. **Step 2**: Add RAFT-S at 5 FPS. Log the raw flow field and tune θ empirically in the target corridor environment.
3. **Step 3**: Implement ego-motion compensation using IMU angular velocity.
4. **Step 4**: Union both masks; compare map quality (tracking loss rate, loop closure accuracy) against Step 1 baseline.
5. **Step 5**: Add unclassified motion region publishing to scene state. Test the stop-threshold rule with a fast-moving cart.

---

## 8. Open Empirical Questions

These require measurements in the target environment; existing literature does not resolve them for the wheelchair context:

1. **What θ value correctly separates ego-motion from dynamic object motion** in wheelchair-height corridor video? The IMU compensation will never be perfect (floor vibration, wheel slip). The threshold must be tuned empirically.

2. **How many SLAM features are lost per frame when the dynamic mask is applied** in a crowded corridor? If a full corridor with 3–4 people results in masking 30–40% of the frame, ORB-SLAM3 may lose tracking. This is the featureless-wall problem compounded by the masking problem.

3. **Does the stale flow mask (propagated via ByteTrack prediction) introduce more SLAM corruption than it prevents** in high-speed turn scenarios? This is the core robustness question for the NGD-SLAM-style propagation approach.

---

## 9. Integration Point for Trajectory Prediction

The dynamic component sub-system is designed so trajectory prediction can be added without modifying any of Layers 1–3:

**Where it plugs in:** Between ByteTrack's track output and the scene state publisher.

**Current state:** `velocity_mps` in the scene state is computed from ByteTrack's Kalman filter — a constant-velocity linear extrapolation of the last N position observations per track.

**Trajectory prediction upgrade path:**
- Replace or augment the `velocity_mps` field with a `predicted_trajectory` array: a sequence of (x, y, t) waypoints over a 2–3 second horizon
- The `time_to_collision_s` field becomes based on the predicted path rather than linear extrapolation
- The LLM slow loop can query "will this person be in the doorway in 2 seconds" rather than "is this person approaching"

**Minimum addition required:** A Social LSTM node (Alahi et al., 2016) or simpler MLP trained on JRDB indoor pedestrian data, subscribing to the per-track position history published by ByteTrack and publishing the predicted trajectory for each active track. This node is entirely decoupled from Layers 1–3 and can be toggled off without affecting SLAM protection or optical flow masking.

The latency characterization of Social LSTM on Jetson Orin Nano concurrent with the full vision stack remains an open question from the Dynamic Environment Understanding review (2026-04-10) and would be the primary empirical contribution of a trajectory prediction research phase.
