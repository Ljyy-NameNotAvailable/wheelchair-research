"""
dynamic_perception_node.py — ROS2 node wiring all three pipeline layers.

Subscribes to:
  /camera/color/image_raw       sensor_msgs/Image  (RGB, 640×480 @ 30 FPS)
  /camera/depth/image_raw       sensor_msgs/Image  (depth, aligned, float32 m)
  /camera/imu                   sensor_msgs/Imu    (angular velocity for ego-motion)
  /orb_slam3/pose               geometry_msgs/PoseStamped (SLAM pose)

Publishes to:
  /wheelchair/scene_state       std_msgs/String    (JSON, extended schema)
  /wheelchair/dynamic_mask      sensor_msgs/Image  (mono8, 255=dynamic)

YOLO + ByteTrack: driven via ultralytics model.track() on each RGB frame.
RAFT-S:           fires every RAFT_FRAME_INTERVAL RGB frames (~5 FPS at 30 FPS input).
ORB-SLAM3 mask:   delivered via SLAMProtector.protect_and_track() at ~10 FPS.
Scene state:      published at SCENE_STATE_FPS Hz (configurable, default 10).

Spec ref: dynamic-components-design.md — all sections; compute scheduling Section 6.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

# Add parent package to path when running as a ROS2 node
_NODE_DIR = Path(__file__).resolve().parent
_PKG_DIR = _NODE_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from dynamic_perception.mask_generator import DynamicMaskGenerator, TrackBox
from dynamic_perception.optical_flow import RAFTSFlowProcessor, EgoMotionState
from dynamic_perception.slam_protector import ORBSLAM3Interface, SLAMProtector
from dynamic_perception.scene_state_publisher import SceneStateBuilder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scheduling constants (spec Section 6)
# ---------------------------------------------------------------------------
YOLO_FPS: int = 30
RAFT_FPS: int = 5
SLAM_FPS: int = 10
RAFT_FRAME_INTERVAL: int = YOLO_FPS // RAFT_FPS   # = 6
SLAM_FRAME_INTERVAL: int = YOLO_FPS // SLAM_FPS   # = 3

DYNAMIC_CLASSES_YOLO: list[str] = ["person", "wheelchair", "generic_obstacle"]


# ---------------------------------------------------------------------------
# Helper: parse Ultralytics track results into TrackBox list
# ---------------------------------------------------------------------------

def parse_ultralytics_tracks(result) -> list[TrackBox]:
    """
    Convert one Ultralytics Results object (from model.track()) into a list
    of TrackBox objects used by the core logic classes.

    Args:
        result: ultralytics.engine.results.Results from model.track().

    Returns:
        List of TrackBox, one per detection with a valid track ID.
    """
    tracks: list[TrackBox] = []
    if result.boxes is None:
        return tracks

    boxes = result.boxes
    for i in range(len(boxes)):
        # Track ID is None for unconfirmed tracks; skip those.
        track_id_tensor = boxes.id
        if track_id_tensor is None:
            continue
        tid = int(track_id_tensor[i].item())
        cls_idx = int(boxes.cls[i].item())
        class_name = result.names.get(cls_idx, "unknown")
        conf = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

        # ByteTrack does not expose pixel-space velocity directly via Ultralytics.
        # SPEC_AMBIGUITY: The spec calls for "ByteTrack's Kalman-predicted bounding
        # boxes" for mask propagation. Ultralytics does not expose the internal Kalman
        # state as public API. Conservative interpretation: use the current bbox as
        # the "predicted" bbox (effectively: no Kalman extrapolation, the bbox position
        # is treated as already Kalman-corrected). A future refactor could subclass
        # the ByteTracker to expose predicted_bbox before the correction step.
        track = TrackBox(
            track_id=tid,
            class_name=class_name,
            bbox_xyxy=(x1, y1, x2, y2),
            confidence=conf,
            predicted_bbox_xyxy=(x1, y1, x2, y2),
            velocity_px=(0.0, 0.0),
        )
        tracks.append(track)

    return tracks


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class DynamicPerceptionNode(Node):
    """
    Main ROS2 node for the dynamic perception sub-system.

    Wires together all three layers:
      Layer 1 — DynamicMaskGenerator + RAFTSFlowProcessor
      Layer 2 — SLAMProtector (with ORBSLAM3Interface)
      Layer 3 — SceneStateBuilder + publisher

    Frame counter drives the scheduling:
      - Every frame: YOLO + ByteTrack, detection mask, mask propagation
      - Every RAFT_FRAME_INTERVAL frames: RAFT-S inference
      - Every SLAM_FRAME_INTERVAL frames: SLAM tracking call + scene state publish
    """

    def __init__(self, config: dict) -> None:
        super().__init__("dynamic_perception_node")
        self._config = config
        self._bridge = CvBridge()
        self._frame_count: int = 0

        # --- Device ---
        device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device_str)
        self.get_logger().info(f"[node] Using device: {self._device}")

        # --- YOLO model ---
        try:
            from ultralytics import YOLO
            model_path = config.get("yolo_model_path", "yolov8n.pt")
            self._yolo = YOLO(model_path)
            self.get_logger().info(f"[node] YOLO loaded: {model_path}")
        except ImportError as e:
            raise ImportError(
                "ultralytics package is required. Install: pip install ultralytics"
            ) from e

        # --- Layer 1: optical flow + mask generator ---
        height = config.get("frame_height", 480)
        width = config.get("frame_width", 640)
        flow_threshold = config.get("flow_threshold_px", 2.0)
        stop_threshold = config.get("stop_threshold_px", 8.0)

        self._flow_processor = RAFTSFlowProcessor(
            device=self._device,
            height=height,
            width=width,
            flow_threshold_px=flow_threshold,
            stop_threshold_px=stop_threshold,
            weights_path=config.get("raft_weights_path", None),
            raft_iters=config.get("raft_iters", 12),
        )
        self._mask_gen = DynamicMaskGenerator(
            height=height,
            width=width,
            flow_threshold_px=flow_threshold,
            stop_threshold_px=stop_threshold,
        )

        # --- Layer 2: SLAM protector ---
        mock_slam = config.get("slam_mock_mode", True)
        slam_iface = ORBSLAM3Interface(mock_mode=mock_slam)
        self._slam_protector = SLAMProtector(slam_iface)

        # --- Layer 3: scene state builder ---
        cam_intr = config.get("camera_intrinsics", None)
        self._scene_builder = SceneStateBuilder(camera_intrinsics=cam_intr)

        # --- Ego-motion state (updated by IMU + SLAM callbacks) ---
        self._ego = EgoMotionState(
            camera_fx=cam_intr["fx"] if cam_intr else 614.0,
            camera_fy=cam_intr["fy"] if cam_intr else 614.0,
            camera_cx=cam_intr["cx"] if cam_intr else 320.0,
            camera_cy=cam_intr["cy"] if cam_intr else 240.0,
        )

        # --- State buffers ---
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_slam_pose: dict = {"x": 0.0, "y": 0.0, "theta_deg": 0.0}
        self._motion_mask_active: bool = False
        self._unclassified_regions: list[dict] = []

        # --- ROS2 subscribers ---
        self._rgb_sub = self.create_subscription(
            Image, "/camera/color/image_raw", self._rgb_callback, 10
        )
        self._depth_sub = self.create_subscription(
            Image, "/camera/depth/image_raw", self._depth_callback, 10
        )
        self._imu_sub = self.create_subscription(
            Imu, "/camera/imu", self._imu_callback, 10
        )
        self._slam_pose_sub = self.create_subscription(
            PoseStamped, "/orb_slam3/pose", self._slam_pose_callback, 10
        )

        # --- ROS2 publishers ---
        self._scene_state_pub = self.create_publisher(String, "/wheelchair/scene_state", 10)
        self._mask_pub = self.create_publisher(Image, "/wheelchair/dynamic_mask", 10)

        self.get_logger().info("[node] DynamicPerceptionNode initialised.")

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def _depth_callback(self, msg: Image) -> None:
        """Cache the latest depth image (float32, metres)."""
        try:
            self._latest_depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().error(f"[node] Depth decode error: {e}")

    def _imu_callback(self, msg: Imu) -> None:
        """
        Update ego-motion state from IMU.

        Spec ref: Section 2b — "ego-motion compensation: subtract expected
        background flow derived from IMU angular velocity."
        """
        self._ego.angular_velocity_rps = msg.angular_velocity.z

    def _slam_pose_callback(self, msg: PoseStamped) -> None:
        """
        Update SLAM pose from /orb_slam3/pose.

        In mock mode, this callback will not fire (the topic is not published).
        The SLAM protector's mock interface provides a drifting test pose instead.

        Spec ref: Section 3 — ORB-SLAM3 is assumed to be already running and
        publishing /orb_slam3/pose.
        """
        import math as _math
        pos = msg.pose.position
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        theta_deg = _math.degrees(2.0 * _math.atan2(qz, qw))
        self._latest_slam_pose = {
            "x": round(pos.x, 4),
            "y": round(pos.y, 4),
            "theta_deg": round(theta_deg, 2),
        }
        # Update SLAM linear velocity for ego-motion compensation
        # SPEC_AMBIGUITY: The spec mentions "SLAM linear velocity estimate" but
        # PoseStamped does not include velocity. Conservative: leave linear velocity
        # at 0.0 until a separate odometry/TwistStamped topic is wired. IMU
        # angular velocity already provides the dominant ego-motion term.
        self._ego.linear_velocity_mps = (0.0, 0.0)

    def _rgb_callback(self, msg: Image) -> None:
        """
        Main pipeline callback — fires on every RGB frame at 30 FPS.

        Drives all three layers according to the compute schedule in spec Section 6:
          - Every frame:                  YOLO + ByteTrack, detection mask
          - Every RAFT_FRAME_INTERVAL:    RAFT-S inference
          - Every SLAM_FRAME_INTERVAL:    SLAM tracking call + scene state publish
        """
        if self._latest_depth is None:
            self.get_logger().warn("[node] Waiting for first depth frame…")
            return

        try:
            frame_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"[node] RGB decode error: {e}")
            return

        self._frame_count += 1
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # ----------------------------------------------------------------
        # Stage 1: YOLO + ByteTrack (every frame, 30 FPS)
        # ----------------------------------------------------------------
        try:
            yolo_results = self._yolo.track(
                frame_bgr,
                persist=True,
                verbose=False,
                conf=self._config.get("yolo_conf_threshold", 0.25),
                iou=self._config.get("yolo_iou_threshold", 0.45),
            )
        except Exception as e:
            self.get_logger().error(f"[node] YOLO tracking error: {e}")
            return

        tracks: list[TrackBox] = []
        if yolo_results:
            tracks = parse_ultralytics_tracks(yolo_results[0])

        # ----------------------------------------------------------------
        # Stage 2: RAFT-S (every RAFT_FRAME_INTERVAL frames, ~5 FPS)
        # ----------------------------------------------------------------
        flow_dynamic_mask: Optional[np.ndarray] = None
        flow_stop_mask: Optional[np.ndarray] = None

        if self._frame_count % RAFT_FRAME_INTERVAL == 0:
            try:
                flow_result = self._flow_processor.update(frame_bgr, ego=self._ego)
                flow_dynamic_mask, flow_stop_mask = (
                    self._flow_processor.compute_flow_mask(flow_result)
                )
                # Build detection mask for unclassified region subtraction
                det_mask_for_flow = self._mask_gen.build_detection_mask(tracks)
                self._unclassified_regions = (
                    self._flow_processor.find_unclassified_motion_regions(
                        flow_mask=flow_dynamic_mask,
                        stop_mask=flow_stop_mask,
                        yolo_detection_mask=det_mask_for_flow,
                        flow_magnitude=flow_result.flow_magnitude,
                        min_area_px2=self._config.get("min_unclassified_area_px2", 500),
                    )
                )
                self._motion_mask_active = True

                # Deceleration trigger (spec Section 4)
                above_stop_regions = [
                    r for r in self._unclassified_regions if r["above_stop_threshold"]
                ]
                if above_stop_regions:
                    self.get_logger().warn(
                        f"[node] STOP THRESHOLD EXCEEDED: "
                        f"{len(above_stop_regions)} unclassified fast-moving region(s). "
                        f"Triggering deceleration signal."
                    )
                    self._publish_deceleration_signal(above_stop_regions)

            except Exception as e:
                self.get_logger().error(f"[node] RAFT-S error: {e}")

        # ----------------------------------------------------------------
        # Stage 3: Unified mask generation (every frame, 30 FPS)
        # ----------------------------------------------------------------
        unified_mask, detection_mask, effective_flow = self._mask_gen.update(
            tracks=tracks,
            flow_dynamic_mask=flow_dynamic_mask,
            flow_stop_mask=flow_stop_mask,
        )
        slam_pixels_excluded = self._mask_gen.count_masked_pixels(unified_mask)

        # Publish mask topic (for RTAB-Map or debugging)
        self._publish_mask(unified_mask, msg.header)

        # ----------------------------------------------------------------
        # Stage 4: SLAM protection (every SLAM_FRAME_INTERVAL frames, 10 FPS)
        # ----------------------------------------------------------------
        if self._frame_count % SLAM_FRAME_INTERVAL == 0:
            try:
                slam_pose = self._slam_protector.protect_and_track(
                    rgb=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                    depth=self._latest_depth,
                    timestamp=timestamp,
                    unified_mask=unified_mask,
                )
                self._latest_slam_pose = slam_pose
            except Exception as e:
                self.get_logger().error(f"[node] SLAM protector error: {e}")

            # ----------------------------------------------------------------
            # Stage 5: Scene state publish (piggybacks on SLAM cadence, 10 FPS)
            # ----------------------------------------------------------------
            try:
                scene_state = self._scene_builder.build(
                    tracks=tracks,
                    depth_image=self._latest_depth,
                    unified_mask=unified_mask,
                    slam_pose=self._latest_slam_pose,
                    unclassified_motion_regions=self._unclassified_regions,
                    slam_dynamic_pixels_excluded=slam_pixels_excluded,
                    motion_mask_active=self._motion_mask_active,
                    timestamp=timestamp,
                )
                json_str = self._scene_builder.to_json(scene_state)
                out_msg = String()
                out_msg.data = json_str
                self._scene_state_pub.publish(out_msg)
                self.get_logger().debug(
                    f"[node] Scene state published: {len(tracks)} tracks, "
                    f"{slam_pixels_excluded} masked pixels."
                )
            except Exception as e:
                self.get_logger().error(f"[node] Scene state build error: {e}")

    def _publish_mask(self, mask: np.ndarray, header) -> None:
        """
        Publish the unified dynamic mask as a mono8 Image message.

        255 = dynamic pixel (exclude from SLAM / RTAB-Map).
        0   = static pixel.

        Spec ref: Section 3b — RTAB-Map subscribes to this topic as its mask input.
        """
        mask_u8 = (mask.astype(np.uint8) * 255)
        mask_msg = self._bridge.cv2_to_imgmsg(mask_u8, encoding="mono8")
        mask_msg.header = header
        self._mask_pub.publish(mask_msg)

    def _publish_deceleration_signal(self, regions: list[dict]) -> None:
        """
        Publish a deceleration signal when θ_stop is exceeded.

        Spec ref: Section 4 — "trigger immediate deceleration regardless of
        YOLO classification."

        SPEC_AMBIGUITY: The spec does not define the exact ROS2 message type or
        topic for the deceleration signal. Conservative implementation: publish a
        JSON warning on /wheelchair/decel_alert as std_msgs/String. A Nav2
        integration point would replace this with an appropriate velocity command.
        Actual deceleration is Nav2's responsibility (out of scope per spec).
        """
        try:
            from std_msgs.msg import String as StringMsg
            if not hasattr(self, "_decel_pub"):
                self._decel_pub = self.create_publisher(
                    StringMsg, "/wheelchair/decel_alert", 10
                )
            payload = json.dumps({
                "alert": "stop_threshold_exceeded",
                "regions": regions,
                "timestamp": time.time(),
            })
            msg = StringMsg()
            msg.data = payload
            self._decel_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"[node] Decel signal error: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    """
    ROS2 node entry point.

    Loads config.yaml from the package root, then spins the node.
    """
    config_path = _PKG_DIR / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. "
            "Run from the package root or set the correct path."
        )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    rclpy.init(args=args)
    node = DynamicPerceptionNode(config=config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
