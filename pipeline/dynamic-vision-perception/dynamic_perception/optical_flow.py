"""
optical_flow.py — RAFT-S wrapper with ego-motion compensation.

Spec reference: dynamic-components-design.md, Section 2b and Section 2 (ego-motion
compensation). Runs at 5 FPS on a lower-priority CUDA stream. Between RAFT-S
calls the previous flow mask is propagated via ByteTrack Kalman predictions
(handled in mask_generator.py).

Ego-motion compensation:
  Subtract the expected background flow derived from IMU angular velocity
  (RealSense D435i built-in IMU) and SLAM linear velocity estimate before
  applying the flow threshold θ.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EgoMotionState:
    """Ego-motion estimates used for background-flow subtraction."""
    angular_velocity_rps: float = 0.0     # radians per second (IMU yaw rate)
    linear_velocity_mps: tuple[float, float] = (0.0, 0.0)  # (vx, vy) from SLAM
    camera_fx: float = 614.0             # focal length x (RealSense D435i default)
    camera_fy: float = 614.0             # focal length y
    camera_cx: float = 320.0             # principal point x (640 px wide)
    camera_cy: float = 240.0             # principal point y (480 px tall)
    depth_m: float = 1.5                 # scene depth estimate used for flow scale


@dataclass
class FlowResult:
    """Output of one RAFT-S inference call."""
    flow_uv: np.ndarray                  # shape (H, W, 2), float32, px/frame
    flow_magnitude: np.ndarray           # shape (H, W), float32
    timestamp: float = field(default_factory=time.monotonic)
    frame_index: int = 0


# ---------------------------------------------------------------------------
# RAFT-S model loader
# ---------------------------------------------------------------------------

def load_raft_small(device: torch.device, weights_path: Optional[str] = None) -> torch.nn.Module:
    """
    Load RAFT-S from torchvision or from original RAFT repo weights.

    Spec ref: design doc — "use the torchvision optical flow module
    (torchvision.models.optical_flow.raft_small) or the original RAFT repo weights".

    Args:
        device: torch device to load the model onto.
        weights_path: Optional path to custom RAFT-S weights (.pth). If None,
                      torchvision pretrained weights are used.

    Returns:
        Loaded RAFT-S model in eval mode.
    """
    try:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        if weights_path is not None:
            model = raft_small(weights=None)
            state = torch.load(weights_path, map_location=device)
            # Handle both raw state dict and checkpoint dicts
            if "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=False)
            logger.info("RAFT-S loaded from custom weights: %s", weights_path)
        else:
            model = raft_small(weights=Raft_Small_Weights.DEFAULT)
            logger.info("RAFT-S loaded with torchvision pretrained weights.")
        model = model.to(device).eval()
        return model
    except ImportError as exc:
        raise ImportError(
            "torchvision >= 0.13 is required for raft_small. "
            "Install with: pip install torchvision"
        ) from exc


# ---------------------------------------------------------------------------
# Ego-motion background flow estimation
# ---------------------------------------------------------------------------

def estimate_background_flow(
    height: int,
    width: int,
    ego: EgoMotionState,
    dt: float = 1.0 / 30.0,
) -> np.ndarray:
    """
    Estimate the expected optical flow due to wheelchair ego-motion (rotation + translation).

    Uses the pinhole camera model. For a rotating camera with angular velocity ω (yaw)
    and translating with velocity (vx, vy), the expected flow at pixel (u, v) is:

        flow_u = -ω * (u - cx)   (yaw-induced horizontal flow)
        flow_v = -ω * (v - cy)   (yaw-induced vertical flow, small)
        + translation-induced term (f * vx / Z, f * vy / Z)

    Spec ref: Section 2b — "subtract expected background flow derived from IMU
    angular velocity and SLAM linear velocity estimate".

    Args:
        height: Frame height in pixels.
        width:  Frame width in pixels.
        ego:    Current ego-motion estimates.
        dt:     Time step between frames in seconds.

    Returns:
        background_flow: np.ndarray of shape (H, W, 2), float32.
                         Channel 0 = horizontal (u) component.
                         Channel 1 = vertical (v) component.
    """
    u_coords, v_coords = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    # Rotation-induced flow (yaw dominates for a ground vehicle)
    omega_dt = ego.angular_velocity_rps * dt
    flow_u_rot = -omega_dt * (u_coords - ego.camera_cx)
    flow_v_rot = -omega_dt * (v_coords - ego.camera_cy)

    # Translation-induced flow (forward/lateral velocity via thin-lens projection)
    # f * v_x / Z  and  f * v_y / Z  in pixels per frame
    vx, vy = ego.linear_velocity_mps
    if ego.depth_m > 0.01:
        flow_u_trans = ego.camera_fx * vx * dt / ego.depth_m
        flow_v_trans = ego.camera_fy * vy * dt / ego.depth_m
    else:
        flow_u_trans = 0.0
        flow_v_trans = 0.0

    background_flow = np.stack(
        [flow_u_rot + flow_u_trans, flow_v_rot + flow_v_trans],
        axis=-1,
    )
    return background_flow


# ---------------------------------------------------------------------------
# Core optical flow processor
# ---------------------------------------------------------------------------

class RAFTSFlowProcessor:
    """
    Wraps RAFT-S inference and produces a dynamic-motion mask.

    Lifecycle:
      - Instantiate once at node startup.
      - Call update() once every 6 YOLO frames (~5 FPS) with the latest frame pair.
      - Read .last_result for the FlowResult used by mask_generator.py.

    The model runs on a lower-priority CUDA stream (stream_priority=-1 in PyTorch
    sense, meaning the default stream is higher priority). On Jetson, YOLO holds
    the high-priority stream; RAFT-S yields automatically.

    Spec ref: Section 2b, Section 6 (compute scheduling).
    """

    def __init__(
        self,
        device: torch.device,
        height: int = 480,
        width: int = 640,
        flow_threshold_px: float = 2.0,
        stop_threshold_px: float = 8.0,
        weights_path: Optional[str] = None,
        raft_iters: int = 12,
    ) -> None:
        """
        Args:
            device:              torch device (cuda or cpu).
            height:              Frame height. Default 480 (RealSense D435i).
            width:               Frame width.  Default 640.
            flow_threshold_px:   θ — pixels/frame above which a pixel is dynamic.
                                 Spec default: 2.0 px/frame.
            stop_threshold_px:   θ_stop — above this, trigger deceleration.
                                 Spec default: 8.0 px/frame.
            weights_path:        Optional custom RAFT-S weights.
            raft_iters:          Number of RAFT recurrent refinement iterations.
                                 Lower = faster, less accurate. 12 is a good tradeoff.
        """
        self.device = device
        self.height = height
        self.width = width
        self.flow_threshold_px = flow_threshold_px
        self.stop_threshold_px = stop_threshold_px
        self.raft_iters = raft_iters

        self.model = load_raft_small(device, weights_path)

        # CUDA stream for RAFT-S (separate from YOLO's default stream)
        if device.type == "cuda":
            self._cuda_stream: Optional[torch.cuda.Stream] = torch.cuda.Stream(
                device=device, priority=0  # 0 = low priority on Jetson CUDA
            )
        else:
            self._cuda_stream = None

        self.last_result: Optional[FlowResult] = None
        self._prev_frame_tensor: Optional[torch.Tensor] = None
        self._frame_index: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Convert a BGR uint8 frame to a float32 RGB tensor normalised to [0, 1].
        RAFT-S expects input in [0, 255] float — torchvision normalises internally,
        so we provide uint8-range float here.

        Shape: (1, 3, H, W).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # torchvision RAFT expects values in [0, 255] as float32
        tensor = torch.from_numpy(frame_rgb).float()  # (H, W, 3)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        return tensor.to(self.device)

    @staticmethod
    def _flow_from_model_output(flow_output) -> np.ndarray:
        """
        Extract the final flow prediction from RAFT-S output.

        torchvision RAFT returns a list of tensors (one per iteration);
        the last element is the highest-quality estimate.
        Shape output: (H, W, 2) float32 numpy array.
        """
        if isinstance(flow_output, (list, tuple)):
            flow_tensor = flow_output[-1]  # final refinement step
        else:
            flow_tensor = flow_output
        # flow_tensor: (1, 2, H, W)
        flow_np = flow_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
        return flow_np.astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame_bgr: np.ndarray,
        ego: Optional[EgoMotionState] = None,
    ) -> FlowResult:
        """
        Run RAFT-S on the current frame relative to the previous frame.

        Applies ego-motion compensation if an EgoMotionState is provided.
        Stores the result in self.last_result and returns it.

        Spec ref: Section 2b — "run RAFT-S at 5 FPS … ego-motion compensation".

        Args:
            frame_bgr: Current BGR frame, uint8, shape (H, W, 3).
            ego:       Current ego-motion state for background subtraction.
                       If None, no compensation is applied.

        Returns:
            FlowResult with flow field and magnitude map.

        Raises:
            RuntimeError: If called before a previous frame exists (first call
                          stores frame and returns zero-flow result).
        """
        self._frame_index += 1
        current_tensor = self._preprocess(frame_bgr)

        if self._prev_frame_tensor is None:
            # First call — no previous frame available. Return zero flow.
            self._prev_frame_tensor = current_tensor
            zero_flow = np.zeros((self.height, self.width, 2), dtype=np.float32)
            zero_mag = np.zeros((self.height, self.width), dtype=np.float32)
            result = FlowResult(
                flow_uv=zero_flow,
                flow_magnitude=zero_mag,
                frame_index=self._frame_index,
            )
            self.last_result = result
            return result

        context = (
            torch.cuda.stream(self._cuda_stream)
            if self._cuda_stream is not None
            else _nullcontext()
        )

        with context:
            with torch.no_grad():
                flow_output = self.model(
                    self._prev_frame_tensor,
                    current_tensor,
                    num_flow_updates=self.raft_iters,
                )

        flow_uv = self._flow_from_model_output(flow_output)  # (H, W, 2)

        # Ego-motion compensation
        if ego is not None:
            background = estimate_background_flow(
                self.height, self.width, ego
            )
            flow_uv = flow_uv - background

        flow_magnitude = np.linalg.norm(flow_uv, axis=-1)  # (H, W)

        self._prev_frame_tensor = current_tensor
        result = FlowResult(
            flow_uv=flow_uv,
            flow_magnitude=flow_magnitude,
            frame_index=self._frame_index,
        )
        self.last_result = result
        return result

    def compute_flow_mask(
        self,
        flow_result: FlowResult,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Threshold the flow magnitude to produce dynamic and stop masks.

        Spec ref: Section 2b thresholds:
          θ = 2.0 px/frame  → dynamic_mask (exclude from SLAM)
          θ_stop = 8.0 px/frame → stop_mask (trigger deceleration)

        Args:
            flow_result: A FlowResult from update().

        Returns:
            dynamic_mask: bool array (H, W), True where flow > θ.
            stop_mask:    bool array (H, W), True where flow > θ_stop.
        """
        mag = flow_result.flow_magnitude
        dynamic_mask = mag > self.flow_threshold_px
        stop_mask = mag > self.stop_threshold_px
        return dynamic_mask.astype(bool), stop_mask.astype(bool)

    def find_unclassified_motion_regions(
        self,
        flow_mask: np.ndarray,
        stop_mask: np.ndarray,
        yolo_detection_mask: np.ndarray,
        flow_magnitude: np.ndarray,
        min_area_px2: int = 500,
    ) -> list[dict]:
        """
        Find connected components of flow-detected motion not covered by YOLO detections.

        Spec ref: Section 4 and scene state schema — "unclassified_motion_regions".
        Also spec: "if flow magnitude exceeds θ but is below θ_stop, log the
        motion region as unclassified dynamic object."

        Args:
            flow_mask:           Bool array (H, W), pixels above θ.
            stop_mask:           Bool array (H, W), pixels above θ_stop.
            yolo_detection_mask: Bool array (H, W), pixels inside YOLO bounding boxes.
            flow_magnitude:      Float array (H, W) of per-pixel flow magnitude.
            min_area_px2:        Discard regions smaller than this (noise filter).

        Returns:
            List of dicts matching the scene state schema:
              {bbox, flow_magnitude_px, area_px2, above_stop_threshold}
        """
        # Unclassified = flow-dynamic but NOT covered by any YOLO detection
        unclassified = flow_mask & ~yolo_detection_mask
        unclassified_u8 = unclassified.astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            unclassified_u8, connectivity=8
        )

        regions: list[dict] = []
        for label_id in range(1, num_labels):  # 0 is background
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_area_px2:
                continue

            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])

            component_pixels = labels == label_id
            mean_mag = float(np.mean(flow_magnitude[component_pixels]))
            above_stop = bool(np.any(stop_mask[component_pixels]))

            regions.append({
                "bbox": [x, y, x + w, y + h],
                "flow_magnitude_px": round(mean_mag, 2),
                "area_px2": area,
                "above_stop_threshold": above_stop,
            })

        return regions


# ---------------------------------------------------------------------------
# Context manager shim for non-CUDA paths
# ---------------------------------------------------------------------------

class _nullcontext:
    """Minimal context manager used when running on CPU (no CUDA stream)."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
