# Indoor Wheelchair Perception Pipeline — Research Proposal

**Date:** 2026-04-11
**Context:** Non-LLM perception stack for shared autonomy power wheelchair. LLM/segmentation work handled by teammates separately.
**Hardware target:** NVIDIA Jetson Orin Nano (~20 TOPS INT8)
**Sensor assumption:** Intel RealSense D435i (RGB-D) + 2D LiDAR (e.g. RPLidar S2)

---

## The Two-Loop Architecture

The pipeline must never let intelligence block safety. Run two independent loops:

```
┌─────────────────────────────────────────────────────────────────┐
│  FAST LOOP  (~20–50 ms, runs continuously)                      │
│  Sensors → Detection → Tracking → Immediate obstacle avoidance  │
└─────────────────────────────────────────────────────────────────┘
                        ↓  publishes scene state
┌─────────────────────────────────────────────────────────────────┐
│  SLOW LOOP  (~500 ms – 2 s, runs periodically)                  │
│  Scene state snapshot → LLM (teammates) → Navigation intent     │
└─────────────────────────────────────────────────────────────────┘
```

If the slow loop is busy or fails, the fast loop keeps the wheelchair safe independently.

---

## Full Pipeline — Stage by Stage

### Stage 1 — Sensing

| Sensor | Role | Justification |
|--------|------|---------------|
| Intel RealSense D435i | RGB color stream → YOLO input; Depth stream → 3D positions | Best-documented RGB-D for indoor robots; integrated IMU; ROS2 driver mature |
| 2D LiDAR (RPLidar S2 or Hokuyo URG) | Floor-level obstacle detection; SLAM input | Reliable at wheelchair height; fast (15 Hz); covers blind spots below camera FOV |

**Why both?** RGB-D misses low objects (bags on floor, chair legs). LiDAR misses vertical objects that don't extend to floor (a person standing in a doorway, glass walls). Together they cover each other's blind spots.

---

### Stage 2 — Static Map (Background)

**Tool:** SLAM Toolbox (ROS2) using 2D LiDAR
**Output:** Occupancy grid map of the room/corridor
**Update strategy:** Treat map as mostly static; remove dynamic object traces using a dynamic object filter (e.g. DynamicFilter, IROS 2022)

The map gives the wheelchair a stable understanding of walls, doorways, and fixed furniture. Dynamic objects detected in later stages are layered on top of this static map.

---

### Stage 3 — Dynamic Object Detection (Fast Loop Core)

**Model:** YOLOv8n or YOLOv10n
**Input:** RGB stream from RealSense (640×480 @ 30 FPS)
**Target latency:** <15 ms inference on Jetson Orin Nano (TensorRT FP16)
**Classes to detect (minimum viable):**
- `person` — highest priority, most common dynamic object indoors
- `door` — open/closing state matters for navigation
- `wheelchair` — detect other wheelchair users
- `generic_obstacle` — bags, carts, anything unclassified but moving

**Key decisions to validate empirically:**
- YOLOv8n vs YOLOv10n: YOLOv10 removes the NMS post-processing step, potentially faster end-to-end
- FP16 vs INT8 quantization: INT8 is faster but may reduce accuracy on small/occluded people
- Input resolution: 640×480 vs 416×416 — tradeoff between accuracy and speed

---

### Stage 4 — 3D Position from Depth

**Method:** For each YOLO bounding box, sample the RealSense depth map at the box centroid (or median of center region to reject outliers).
**Output:** (x, y, z) in camera frame → transform to wheelchair frame via calibrated extrinsics
**Cost:** Near-zero — simple pixel lookup, no additional model needed
**Alternative if RealSense depth is noisy:** Depth Anything V2 (monocular depth estimation) as fallback; runs ~15 FPS on Jetson

This gives every detected object a 3D position without needing a full 3D detector.

---

### Stage 5 — Multi-Object Tracking

**Recommended tracker:** ByteTrack
**Why ByteTrack:**
- Integrates natively with Ultralytics YOLOv8 (one line of code)
- Tracks even low-confidence detections (important for partial occlusion in corridors)
- ~2–5 ms overhead per frame
- Best speed-accuracy tradeoff among current trackers (ECCV 2022)

**Output per tracked object:**
- Persistent ID (same person keeps same ID across frames)
- 3D position history (last N frames)
- Estimated velocity vector (dx/dt, dy/dt in wheelchair frame)
- Time since last seen (for re-identification after occlusion)

**Fallback for occlusion-heavy scenes:** OC-SORT — better re-identification after long occlusion (e.g. person walks behind a column), at ~10% speed cost.

---

### Stage 6 — 2D LiDAR Person Detection (Complementary)

**Model:** DR-SPAAM (IROS 2020)
**Input:** 2D LiDAR scan
**Output:** (x, y) positions of detected legs/people in floor plane
**Why:** Catches people that the RGB-D camera misses (low light, looking away from camera direction, below camera FOV)

**Fusion with Stage 5:** Associate LiDAR detections with RGB-D tracked objects by spatial proximity in the floor plane. If a LiDAR detection has no RGB-D match within 0.4m, treat it as an unclassified obstacle.

---

### Stage 7 — Scene State Publisher

All per-object information is assembled into a structured scene message published to a ROS2 topic at ~10–20 Hz:

```json
{
  "timestamp": 1744300000.123,
  "objects": [
    {
      "id": 7,
      "class": "person",
      "position_m": {"x": 1.2, "y": 0.3, "z": 0.0},
      "distance_m": 1.24,
      "velocity_mps": {"x": -0.3, "y": 0.0},
      "approaching": true,
      "time_to_collision_s": 4.1,
      "confidence": 0.87
    }
  ],
  "static_map_clearance_m": 0.6,
  "nearest_obstacle_m": 1.24
}
```

**Consumers:**
- **Fast loop**: obstacle avoidance costmap (direct, <50 ms)
- **Slow loop**: scene snapshot sent to LLM teammates every ~1 s

---

### Stage 8 — Costmap Integration (Fast Loop Output)

**Tool:** Nav2 (ROS2 Navigation Stack)
**Method:** Publish dynamic obstacles as an inflation layer on the occupancy costmap
**Effect:** The wheelchair's path planner automatically routes around detected people and objects
**This is the direct, non-LLM navigation output** — it works even when the LLM is unavailable

---

## Interface to LLM Teammates

The slow loop passes the scene state (Stage 7) plus current user joystick intent to the LLM module. Your teammates do not need to process video — they receive a clean structured description. This decouples your work cleanly.

```
Your pipeline  →  ROS2 topic /wheelchair/scene_state  →  LLM module (teammates)
                                                       →  Nav2 costmap (your fast loop)
```

---

## Research Questions This Pipeline Opens

1. **What is the optimal YOLO variant + quantization for <30ms end-to-end on Jetson Orin Nano for indoor person detection?**
   → Directly publishable benchmark study (gap confirmed in literature review)

2. **Does adding 2D LiDAR person detection (DR-SPAAM) meaningfully improve safety recall over RGB-D YOLO alone in indoor wheelchair scenarios?**
   → Ablation study on a real wheelchair testbed

3. **What is the minimum scene representation needed by the LLM for meaningful navigation assistance?**
   → Interface design study (joint work with LLM teammates)

4. **How does ByteTrack vs OC-SORT affect tracking stability in narrow indoor corridors with frequent occlusion?**
   → Tracking benchmark using JRDB or a custom indoor dataset

---

## Suggested Implementation Order

1. **Week 1–2**: Set up RealSense + YOLOv8n on Jetson, measure baseline latency
2. **Week 2–3**: Add ByteTrack, validate person tracking in your lab space
3. **Week 3–4**: Add 2D LiDAR + DR-SPAAM, implement simple fusion
4. **Week 4–5**: Build scene state publisher, define ROS2 interface with LLM teammates
5. **Week 5–6**: Integrate with Nav2 costmap, run end-to-end tests
6. **Week 6+**: Systematic benchmarking, data collection for research questions above

---

## Key Papers to Implement From

| Paper | What to use from it |
|-------|---------------------|
| YOLOv8 (Ultralytics, 2023) | Base detector — use official Ultralytics repo |
| YOLOv10 (Wang et al., NeurIPS 2024) | Compare vs YOLOv8n on Jetson |
| ByteTrack (Zhang et al., ECCV 2022) | Built into Ultralytics — `model.track()` |
| DR-SPAAM (Jia et al., IROS 2020) | LiDAR person detection — open source ROS package |
| Depth Anything V2 (Yang et al., NeurIPS 2024) | Depth fallback if RealSense noisy |
| ROS2 Nav2 survey (Macenski et al., 2023) | Nav2 costmap integration reference |
| JRDB dataset (Martin-Martin et al., TPAMI 2021) | Training/eval dataset for indoor person detection |
