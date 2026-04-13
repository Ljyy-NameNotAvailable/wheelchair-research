# Vision-Only Indoor Wheelchair Perception Pipeline — Restricted Proposal

**Date:** 2026-04-12
**Constraint:** Camera sensors only — no LiDAR
**Hardware target:** NVIDIA Jetson Orin Nano (~20 TOPS INT8)
**Context:** Non-LLM perception stack feeding a fast safety loop and a structured scene interface for LLM teammates

---

## Sensor Options (choose one)

The pipeline works at three levels depending on what camera hardware is available. Each level adds depth capability:

| Level | Hardware | Depth source | Cost estimate |
|-------|----------|-------------|---------------|
| **A — RGB-D** | Intel RealSense D435i | Hardware structured light | ~$200 |
| **B — Stereo** | OAK-D / ZED Mini | Stereo matching (on-device) | ~$150–300 |
| **C — Monocular** | Any USB/CSI camera | Monocular depth estimation (neural net) | <$50 |

**Recommendation: Level A (RealSense D435i)** if budget allows. It gives the most reliable depth with minimal compute cost. Level C is the most constrained but works if only a basic camera is available.

---

## Two-Loop Architecture (same principle, vision-only)

```
┌──────────────────────────────────────────────────────────────────┐
│  FAST LOOP  (~20–50 ms)                                          │
│  Camera → YOLO detection → Depth → ByteTrack → Obstacle output   │
└──────────────────────────────────────────────────────────────────┘
                      ↓  publishes scene state
┌──────────────────────────────────────────────────────────────────┐
│  SLOW LOOP  (~500 ms – 2 s)                                      │
│  Scene snapshot → LLM teammates → Navigation intent              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Full Vision-Only Pipeline — Stage by Stage

### Stage 1 — Object Detection

Same as before: **YOLOv8n or YOLOv10n** on the RGB stream.

- Input: 640×480 RGB @ 30 FPS
- Target: <15 ms inference with TensorRT INT8
- Classes: `person`, `door`, `wheelchair`, `generic_obstacle`

No change from the original pipeline. YOLO does not need LiDAR.

---

### Stage 2 — Depth Estimation (replaces LiDAR for 3D position)

This is the key difference from the LiDAR pipeline. Choose based on sensor:

**Level A — RealSense D435i:**
- Use the hardware depth stream directly (same as original pipeline)
- Depth at bounding box centroid → 3D position
- Cost: ~0 extra compute
- Reliable up to ~4 m indoors

**Level B — Stereo camera (OAK-D / ZED):**
- On-device stereo matching (handled by the camera's own processor)
- Output is a depth map, used same way as RealSense
- Good up to ~5 m; less accurate than RealSense at close range

**Level C — Monocular camera:**
- Run **Depth Anything V2** (small variant) to estimate a dense depth map
- On Jetson Orin Nano: ~15–25 FPS for the small model
- Important limitation: monocular depth is *relative* (not metric) unless calibrated
- To get metric depth from monocular: place known-size objects in the scene as calibration anchors, or use the wheelchair's known floor plane height

**Optical flow as a complement (all levels):**
- Run a lightweight optical flow model (e.g. LiteFlowNet or RAFT-small) at 5–10 FPS
- Use flow magnitude to detect *motion* in the scene independent of classification
- Catches moving objects YOLO may miss or misclassify (e.g., a fast-moving cart)
- Flow output feeds into the scene state as a motion mask

---

### Stage 3 — Multi-Object Tracking

Same as original: **ByteTrack** (default) or **OC-SORT** (for occlusion-heavy corridors).

ByteTrack is integrated directly into Ultralytics — call `model.track()`.

---

### Stage 4 — Visual SLAM (replaces 2D LiDAR SLAM)

This is the largest change. Without LiDAR, the static map must come from vision.

**Recommended: ORB-SLAM3** (monocular, stereo, or RGB-D mode)
- Works with all three sensor levels
- Builds a sparse map of the environment in real time
- Robust to indoor environments
- Well-maintained, ROS2 wrapper available
- CPU-only operation possible, leaving GPU for YOLO + depth

**Alternative: RTAB-Map** (RGB-D or stereo)
- Produces a 2D occupancy grid (same format as LiDAR SLAM output)
- Easier to integrate with Nav2 costmaps
- Higher memory usage but better map quality for navigation

**Alternative: OpenVINS** (if camera + IMU available, e.g. RealSense D435i has IMU)
- Visual-inertial odometry — very accurate localization
- Does not produce a full map but gives precise pose for costmap updates

**Recommended combination:**
- **ORB-SLAM3 or RTAB-Map** for mapping (run at reduced frequency, ~5–10 FPS)
- **Dynamic object removal**: filter out YOLO-detected dynamic objects from the SLAM point cloud to prevent them from corrupting the static map

---

### Stage 5 — Scene State Publisher

Same structured message as before, minus the LiDAR-specific fields:

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
      "depth_source": "realsense"
    }
  ],
  "motion_mask_active": true,
  "nearest_obstacle_m": 1.12,
  "slam_pose": {"x": 3.2, "y": 1.5, "theta_deg": 12.4}
}
```

---

### Stage 6 — Nav2 Costmap Integration

Same as original pipeline — publish dynamic obstacles as a costmap inflation layer.
The occupancy grid now comes from visual SLAM instead of LiDAR SLAM.

---

## Honest Limitations vs. LiDAR Pipeline

| Aspect | LiDAR pipeline | Vision-only pipeline |
|--------|---------------|---------------------|
| Floor-level obstacle detection | Excellent (LiDAR sees everything at floor) | Weaker — camera may miss low objects |
| Depth accuracy | ±2–5 cm (LiDAR) | ±5–20 cm (RGB-D), worse monocular |
| Works in darkness | Yes (LiDAR) | No (needs illumination) |
| Compute budget | LiDAR SLAM is CPU, frees GPU | Visual SLAM uses more CPU |
| Cost | +$100–300 for LiDAR | Potentially zero (camera only) |
| Mapping robustness | High | Medium (featureless walls are hard for visual SLAM) |

The most important limitation: **camera-only pipelines struggle with featureless environments** (blank white hospital corridors). ORB-SLAM3 needs visual features (edges, corners, textures) to localize. Add fiducial markers (ArUco tags on walls) as a mitigation if this becomes an issue.

---

## Compute Budget on Jetson Orin Nano

Rough estimates for concurrent operation (TensorRT INT8 where applicable):

| Component | Estimated GPU/CPU time |
|-----------|----------------------|
| YOLOv8n (TensorRT INT8) | ~12 ms GPU |
| ByteTrack | ~3 ms CPU |
| RealSense depth lookup | ~1 ms CPU |
| Depth Anything V2-small (Level C only) | ~40 ms GPU |
| ORB-SLAM3 (RGB-D mode, 10 FPS) | ~30 ms CPU |
| Optical flow RAFT-small (5 FPS) | ~20 ms GPU |
| Scene state assembly | ~2 ms CPU |

**Level A (RealSense):** YOLO + ByteTrack + depth + SLAM fits comfortably within 50 ms.
**Level C (monocular):** Depth Anything competes with YOLO for GPU — need to stagger or use separate CUDA streams.

---

## Research Questions This Opens

1. **Does Depth Anything V2-small provide sufficient metric depth accuracy for wheelchair obstacle avoidance at indoor ranges (0.5–4 m)?**
   → Direct empirical test, novel for wheelchair context

2. **How much does optical flow complement YOLO for detecting fast-moving objects that YOLO misses between frames?**
   → Ablation: YOLO alone vs. YOLO + flow, measured by recall on moving objects

3. **Does ORB-SLAM3 maintain stable localization in typical indoor clinical environments (corridors, hospital rooms)?**
   → Robustness study, directly relevant to deployment

4. **What is the end-to-end latency of the full vision-only stack on Jetson Orin Nano?**
   → Same benchmark gap as LiDAR pipeline — no published number exists

---

## Suggested Implementation Order

1. **Week 1**: YOLOv8n + ByteTrack on RGB stream — get detection + tracking working
2. **Week 1–2**: Add RealSense D435i depth → get 3D positions per detection
3. **Week 2–3**: Add ORB-SLAM3 or RTAB-Map for static map
4. **Week 3**: Add dynamic object removal from SLAM map
5. **Week 4**: Build scene state publisher, define ROS2 interface with LLM teammates
6. **Week 4–5**: Integrate with Nav2, run end-to-end corridor tests
7. **Week 5+**: Add optical flow if needed; benchmark latency; collect JRDB-style data
