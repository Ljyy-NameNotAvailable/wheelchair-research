"""
slam_protector.py — Layer 2: SLAM Protection.

Feeds the unified dynamic mask to the ORB-SLAM3 interface at each tracking
call, preventing dynamic-object pixels from corrupting the static map.

ORB-SLAM3 interface assumption:
  ORB-SLAM3 is already running as a separate process and is assumed to either:
    (a) expose a Python binding via slam.TrackRGBD(rgb, depth, t, mask=mask),
    (b) accept a mask topic via a ROS2 service or shared-memory interface.

  DO NOT implement ORB-SLAM3 itself. This module mocks its interface with
  a clearly marked stub so the rest of the pipeline can be tested independently.

Between RAFT-S calls (at 30 FPS), the mask is propagated from mask_generator.py
via ByteTrack predictions — the SLAM thread therefore never waits for GPU
inference to complete.

Spec ref: dynamic-components-design.md, Section 3a (ORB-SLAM3 integration),
          Section 3b (RTAB-Map integration stub).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORB-SLAM3 interface stub
# ---------------------------------------------------------------------------

class ORBSLAM3Interface:
    """
    Mock interface to an external ORB-SLAM3 process.

    In production, replace the body of track_rgbd() with the actual call to
    the ORB-SLAM3 Python bindings (ORB_SLAM3.System.TrackRGBD) or a ROS2
    service call to the ORB-SLAM3 ROS2 wrapper.

    Spec ref: Section 3a:
      "ORB-SLAM3 exposes the mask input at the TrackRGBD() call signature.
       Pass the dynamic mask as the optional exclusion mask argument."

    Assumed ROS2 topics published by ORB-SLAM3 (already running externally):
      /orb_slam3/pose    — geometry_msgs/PoseStamped
      /orb_slam3/map     — sensor_msgs/PointCloud2 (optional)
    """

    def __init__(self, mock_mode: bool = True) -> None:
        """
        Args:
            mock_mode: If True, all calls are no-ops that return dummy pose.
                       Set to False when real ORB-SLAM3 bindings are available.
        """
        self.mock_mode = mock_mode
        self._last_pose: dict = {"x": 0.0, "y": 0.0, "theta_deg": 0.0}
        self._tracking_state: str = "OK"  # "OK", "LOST", "RECENTLY_LOST"
        self._call_count: int = 0

        if mock_mode:
            logger.warning(
                "[slam_protector] ORBSLAM3Interface running in MOCK MODE. "
                "TrackRGBD() calls will be no-ops. Replace with real bindings "
                "when ORB-SLAM3 Python package is available."
            )
        else:
            # REAL IMPLEMENTATION HOOK:
            # import ORB_SLAM3
            # self._slam = ORB_SLAM3.System(
            #     vocab_path="/path/to/ORBvoc.txt",
            #     config_path="/path/to/RealSense_D435i.yaml",
            #     sensor=ORB_SLAM3.Sensor.RGBD,
            # )
            raise NotImplementedError(
                "Real ORB-SLAM3 bindings not yet wired. Set mock_mode=True "
                "to run without ORB-SLAM3, or install the ORB-SLAM3 Python "
                "package and uncomment the binding code in slam_protector.py."
            )

    def track_rgbd(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        timestamp: float,
        dynamic_mask: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Send one RGB-D frame to ORB-SLAM3 with the dynamic exclusion mask.

        Spec ref: Section 3a —
          "slam.TrackRGBD(im_rgb, im_depth, timestamp, mask=dynamic_mask)"
          "ORB feature extraction skips masked pixels entirely."

        Args:
            rgb:          (H, W, 3) uint8 RGB image.
            depth:        (H, W) float32 depth in metres.
            timestamp:    POSIX timestamp (float seconds).
            dynamic_mask: Optional (H, W) bool or uint8 mask. True = exclude.
                          Passed directly to ORB-SLAM3's mask argument.

        Returns:
            Pose dict: {"x": float, "y": float, "theta_deg": float}
                       representing the wheelchair's estimated 2-D pose.
            On tracking loss returns the last good pose.
        """
        self._call_count += 1

        if self.mock_mode:
            # MOCK: return a slowly drifting pose so the scene state has
            # something non-trivial to publish during development/testing.
            t = time.monotonic()
            mock_pose = {
                "x": round(3.0 + 0.1 * np.sin(t * 0.2), 3),
                "y": round(1.5 + 0.1 * np.cos(t * 0.2), 3),
                "theta_deg": round(10.0 + 2.0 * np.sin(t * 0.1), 2),
            }
            self._last_pose = mock_pose

            if dynamic_mask is not None:
                masked_px = int(dynamic_mask.sum()) if dynamic_mask.dtype == bool \
                    else int((dynamic_mask > 0).sum())
                logger.debug(
                    "[slam_protector] Mock TrackRGBD call #%d: "
                    "%d pixels excluded via dynamic mask.",
                    self._call_count, masked_px,
                )
            return mock_pose

        # REAL IMPLEMENTATION HOOK (unreachable in mock mode):
        # mask_u8 = (dynamic_mask.astype(np.uint8) * 255
        #            if dynamic_mask is not None else None)
        # pose_matrix = self._slam.TrackRGBD(rgb, depth, timestamp, mask=mask_u8)
        # self._last_pose = self._matrix_to_2d_pose(pose_matrix)
        # self._tracking_state = self._slam.GetTrackingState()
        # return self._last_pose
        raise NotImplementedError("Unreachable in mock_mode=True.")  # pragma: no cover

    def get_tracking_state(self) -> str:
        """
        Return the current ORB-SLAM3 tracking state.

        Returns:
            One of: "OK", "LOST", "RECENTLY_LOST" (mirrors ORB-SLAM3 enum names).
        """
        if self.mock_mode:
            return "OK"
        # REAL: return self._tracking_state
        raise NotImplementedError  # pragma: no cover

    def get_last_pose(self) -> dict:
        """
        Return the most recent successfully computed pose.

        Returns:
            {"x": float, "y": float, "theta_deg": float}
        """
        return dict(self._last_pose)

    @staticmethod
    def _matrix_to_2d_pose(T: "np.ndarray") -> dict:  # noqa: F821
        """
        Extract (x, y, yaw) from a 4×4 SE3 transform matrix returned by ORB-SLAM3.

        This stub is used only when real bindings are active (mock_mode=False).
        ORB-SLAM3 returns the camera-to-world transform as a 4×4 numpy array.
        """
        x = float(T[0, 3])
        y = float(T[2, 3])  # ORB-SLAM3 uses z-forward convention
        import math
        theta_rad = math.atan2(float(T[0, 2]), float(T[2, 2]))
        return {"x": x, "y": y, "theta_deg": math.degrees(theta_rad)}


# ---------------------------------------------------------------------------
# RTAB-Map interface stub
# ---------------------------------------------------------------------------

class RTABMapInterface:
    """
    Minimal stub for RTAB-Map integration.

    RTAB-Map is not the primary SLAM backend; this stub documents the
    alternative integration path described in spec Section 3b.

    In the RTAB-Map path:
      1. Publish dynamic_mask as sensor_msgs/Image to a ROS2 topic.
      2. Configure the RTAB-Map ROS2 node to subscribe to that topic as its
         mask input parameter (Mem/ExcludeMaskTopic or equivalent).
      3. RTAB-Map excludes masked regions from loop closure detection.

    No code changes to RTAB-Map are required — this class only publishes
    the mask topic; the actual SLAM node is assumed to be running externally.

    Spec ref: Section 3b.
    """

    def __init__(self) -> None:
        logger.info(
            "[slam_protector] RTABMapInterface stub created. "
            "Mask publishing to /wheelchair/dynamic_mask is handled by the "
            "ROS2 node (dynamic_perception_node.py). "
            "Configure rtabmap_ros to subscribe to that topic."
        )

    def get_integration_instructions(self) -> str:
        """Return human-readable RTAB-Map integration instructions."""
        return (
            "RTAB-Map integration:\n"
            "  1. In your rtabmap_ros launch file, add:\n"
            "       'Mem/ExcludeMaskTopic': '/wheelchair/dynamic_mask'\n"
            "  2. The dynamic_perception_node publishes the mask on that topic\n"
            "     as sensor_msgs/Image (mono8, 255=exclude).\n"
            "  3. No code changes to RTAB-Map are required.\n"
            "  Spec ref: dynamic-components-design.md Section 3b."
        )


# ---------------------------------------------------------------------------
# SLAMProtector — orchestrates mask delivery
# ---------------------------------------------------------------------------

class SLAMProtector:
    """
    Orchestrates delivery of the dynamic mask to the SLAM backend.

    Called once per SLAM frame (10 FPS per spec Section 6). Passes the
    unified mask from DynamicMaskGenerator to the ORB-SLAM3 interface
    and returns the current pose for use in the scene state publisher.

    Spec ref: Section 3 — "feed the unified mask to ORB-SLAM3 via its mask
    argument at each TrackRGBD() call."
    """

    def __init__(self, slam_interface: ORBSLAM3Interface) -> None:
        """
        Args:
            slam_interface: An ORBSLAM3Interface instance (real or mock).
        """
        self._slam = slam_interface
        self._call_count: int = 0

    def protect_and_track(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        timestamp: float,
        unified_mask: np.ndarray,
    ) -> dict:
        """
        Feed one RGB-D frame plus its dynamic mask to ORB-SLAM3.

        The mask ensures ORB feature extraction skips dynamic pixels, keeping
        the static map free of spurious keypoints from people or moving objects.

        Spec ref: Section 3a:
          "ORB feature extraction skips masked pixels entirely. The tracking
           thread runs at 30–40 FPS unblocked because the mask computation
           does not block the tracking call."

        Args:
            rgb:          (H, W, 3) uint8 RGB image.
            depth:        (H, W) float32 depth in metres.
            timestamp:    POSIX timestamp in seconds.
            unified_mask: (H, W) bool mask from DynamicMaskGenerator.update().

        Returns:
            pose: {"x": float, "y": float, "theta_deg": float}
        """
        self._call_count += 1
        pose = self._slam.track_rgbd(rgb, depth, timestamp, dynamic_mask=unified_mask)
        logger.debug(
            "[slam_protector] SLAM call #%d: pose=(%s), "
            "mask pixels=%d, tracking_state=%s",
            self._call_count,
            pose,
            int(unified_mask.sum()),
            self._slam.get_tracking_state(),
        )
        return pose

    def is_tracking_healthy(self) -> bool:
        """
        Return True if ORB-SLAM3 is currently tracking successfully.

        Returns:
            bool — True if state is "OK", False if "LOST" or "RECENTLY_LOST".
        """
        return self._slam.get_tracking_state() == "OK"
