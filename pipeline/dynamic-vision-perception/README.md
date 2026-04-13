# Dynamic Vision Perception — ROS2 Package

Implementation of the dynamic components sub-system for the vision-only indoor
wheelchair perception pipeline.

**Spec source:** `research/vision-only-perception-pipeline/dynamic-components-design.md`
**Hardware target:** NVIDIA Jetson Orin Nano (~20 TOPS INT8), Intel RealSense D435i
**ROS2 version:** Humble

---

## Architecture

Three layers on top of the base vision-only pipeline:

```
RGB stream (30 FPS)
      │
      ├── YOLO + ByteTrack (every frame)  ─────────► detection mask
      │                                                     │
      └── RAFT-S optical flow (every 6th frame, ~5 FPS)    │
              │                                             │
              ├── ego-motion compensation (IMU + SLAM)      │
              └── flow magnitude > θ ──────────────────► flow mask
                                                             │
                                              union (detection | flow)
                                                             │
                                                    unified_mask (H, W bool)
                                                      │            │
                                          ┌───────────┘    ┌───────────────┐
                                          │                │               │
                                   ORB-SLAM3           RTAB-Map      /wheelchair/
                                   TrackRGBD()         mask topic    dynamic_mask
                                   (mask= arg)
                                          │
                                   SLAM pose
                                          │
                              scene state builder (10 FPS)
                                          │
                              /wheelchair/scene_state (JSON)
```

---

## Requirements

**Python packages** (install via requirements.txt):
```bash
pip install -r requirements.txt
```

**ROS2 packages** (installed with ROS2 Humble):
- `rclpy`, `sensor_msgs`, `std_msgs`, `geometry_msgs`, `cv_bridge`

**Jetson Orin Nano note:** Install PyTorch and torchvision from the Jetson PyPI
index before running `pip install -r requirements.txt`:
```bash
pip install torch torchvision \
  --index-url https://developer.download.nvidia.com/compute/redist/jp/v60
```

---

## Build

```bash
# From your ROS2 workspace root:
cd ~/ros2_ws
# Copy or symlink this package:
cp -r <repo>/pipeline/dynamic-vision-perception src/dynamic_perception

colcon build --packages-select dynamic_perception
source install/setup.bash
```

---

## Usage

### Running with mock ORB-SLAM3 (no external SLAM process required)

```bash
ros2 launch dynamic_perception dynamic_perception.launch.py slam_mock:=true
```

### Running against real ORB-SLAM3

1. Start ORB-SLAM3 ROS2 wrapper (publishes `/orb_slam3/pose`).
2. Set `slam_mock_mode: false` in `config.yaml` and provide `orb_slam3_vocab_path`
   and `orb_slam3_config_path`.
3. Install the ORB-SLAM3 Python bindings (not included here).
4. Run:
```bash
ros2 launch dynamic_perception dynamic_perception.launch.py slam_mock:=false
```

### Running against real RealSense D435i

```bash
# Start the RealSense ROS2 driver first:
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true enable_depth:=true align_depth.enable:=true enable_gyro:=true

# Then launch this node:
ros2 launch dynamic_perception dynamic_perception.launch.py
```

### Running standalone (no ROS2)

The core logic classes have no hard ROS2 dependency and can be tested independently:

```python
import numpy as np
import torch
from dynamic_perception.mask_generator import DynamicMaskGenerator, TrackBox
from dynamic_perception.optical_flow import RAFTSFlowProcessor, EgoMotionState
from dynamic_perception.slam_protector import ORBSLAM3Interface, SLAMProtector
from dynamic_perception.scene_state_publisher import SceneStateBuilder
from dynamic_perception.trajectory_stub import predict_trajectory, TrackHistoryEntry

# Create dummy frame
frame = np.zeros((480, 640, 3), dtype=np.uint8)
depth = np.ones((480, 640), dtype=np.float32) * 1.5  # 1.5 m flat

# Layer 1: mask generation
gen = DynamicMaskGenerator()
tracks = [TrackBox(1, "person", (100, 100, 200, 300), 0.9)]
unified, det, flow = gen.update(tracks, None, None)

# Layer 2: SLAM (mock)
slam = ORBSLAM3Interface(mock_mode=True)
protector = SLAMProtector(slam)
pose = protector.protect_and_track(frame, depth, 0.0, unified)

# Layer 3: scene state
builder = SceneStateBuilder()
state = builder.build(tracks, depth, unified, pose, [], 0, False)
import json; print(json.dumps(state, indent=2))
```

---

## Configuration

Edit `config.yaml` to change parameters. Key parameters:

| Parameter | Default | Spec reference |
|-----------|---------|----------------|
| `flow_threshold_px` | 2.0 | θ — dynamic pixel threshold (Section 2b) |
| `stop_threshold_px` | 8.0 | θ_stop — deceleration trigger (Section 4) |
| `raft_iters` | 12 | RAFT-S refinement iterations (fewer = faster) |
| `slam_mock_mode` | true | Set false when ORB-SLAM3 bindings available |
| `yolo_conf_threshold` | 0.25 | YOLO detection confidence |
| `track_history_max_len` | 30 | Entries per track (= 3 s at 10 FPS) |

All parameters from the spec configuration table are present in `config.yaml`.

---

## Inputs

The node expects these topics to be already publishing before startup:

| Topic | Type | Source | Required |
|-------|------|--------|----------|
| `/camera/color/image_raw` | `sensor_msgs/Image` (bgr8, 640x480 @ 30 FPS) | RealSense driver | Yes |
| `/camera/depth/image_raw` | `sensor_msgs/Image` (32FC1, metres) | RealSense driver | Yes |
| `/camera/imu` | `sensor_msgs/Imu` | RealSense driver | Optional (ego-motion) |
| `/orb_slam3/pose` | `geometry_msgs/PoseStamped` | ORB-SLAM3 node | Optional (slam_mock=false) |

---

## Outputs

| Topic | Type | Rate | Contents |
|-------|------|------|----------|
| `/wheelchair/scene_state` | `std_msgs/String` (JSON) | 10 FPS | Full extended scene state (spec Section 5 schema) |
| `/wheelchair/dynamic_mask` | `sensor_msgs/Image` (mono8) | 30 FPS | 255 = dynamic pixel; feed to RTAB-Map mask input |
| `/wheelchair/decel_alert` | `std_msgs/String` (JSON) | On event | Published when θ_stop exceeded (spec Section 4) |

### Scene state JSON schema

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

---

## Module Reference

| Module | Layer | Description |
|--------|-------|-------------|
| `dynamic_perception/optical_flow.py` | 1 | RAFT-S wrapper, ego-motion compensation, unclassified region detection |
| `dynamic_perception/mask_generator.py` | 1 | Detection mask + flow mask union, inter-frame propagation |
| `dynamic_perception/slam_protector.py` | 2 | ORB-SLAM3 interface (mock + real stub), SLAMProtector orchestrator |
| `dynamic_perception/scene_state_publisher.py` | 3 | Per-object entry builder, SceneStateBuilder, JSON serialiser |
| `dynamic_perception/trajectory_stub.py` | — | Kalman linear extrapolation; Social LSTM upgrade point |
| `nodes/dynamic_perception_node.py` | All | ROS2 node wiring all layers; compute scheduling |
| `launch/dynamic_perception.launch.py` | — | ROS2 launch file |

---

## Notes for tester

### What is mocked
- **ORB-SLAM3** (`slam_mock_mode: true`): the `ORBSLAM3Interface` returns a
  slowly drifting synthetic pose. The mask is still computed and passed to the
  mock call so the mask pipeline is fully testable without ORB-SLAM3 installed.
- **RAFT-S weights**: torchvision pretrained weights are downloaded on first run
  (~50 MB). On Jetson without internet, set `raft_weights_path` to a local file.

### Known edge cases
1. **First RAFT-S call**: returns a zero flow mask (no previous frame). The
   detection mask alone covers this frame. This is expected behavior.
2. **No depth at bbox centroid**: falls back to median of valid pixels in the
   bounding box ROI. If the entire ROI has invalid depth (e.g., the object is
   too close or too far for the RealSense), `distance_m` will be 0.0.
3. **Track ID gaps**: if ByteTrack loses and re-acquires a track, the old
   history is pruned and a new entry is created. The first trajectory prediction
   for the re-acquired track will have only one observation and will return
   zero velocity.
4. **Masked pixel count approaching 30–40% of frame**: spec Section 8 notes
   this may cause ORB-SLAM3 to lose tracking in crowded corridors. The
   `slam_dynamic_pixels_excluded` field in the scene state monitors this.
   A value consistently above ~90,000 px (640×480×30%) warrants investigation.
5. **`stop_threshold_px` trigger**: when any unclassified motion region exceeds
   θ_stop = 8.0 px/frame, a message is published to `/wheelchair/decel_alert`.
   Actual deceleration must be handled by the Nav2 integration layer (out of scope).

### Suggested test inputs
- A static scene (no motion): verify `motion_mask_active` stays false until
  RAFT-S fires and that `slam_dynamic_pixels_excluded` is near zero.
- A person walking across the frame: verify `masked_from_slam: true` in the
  object entry and that the unified mask covers the bounding box.
- A fast-moving cart (not a YOLO class): verify it appears in
  `unclassified_motion_regions` and triggers `above_stop_threshold: true`
  when it moves faster than 8 px/frame.
- Wheelchair turning in place: verify ego-motion compensation prevents the
  entire frame from being flagged as dynamic (requires IMU topic to be live).

### Social LSTM upgrade path
See `dynamic_perception/trajectory_stub.py` — the `predict_trajectory()`
function contains a clearly marked `SOCIAL LSTM PLUG-IN POINT` comment block
with step-by-step instructions for wiring in a Social LSTM ROS2 node.
No changes to Layers 1–3 are required for that upgrade.
