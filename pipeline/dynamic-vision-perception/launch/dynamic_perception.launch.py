"""
dynamic_perception.launch.py — ROS2 launch file for the dynamic perception node.

Usage:
  ros2 launch dynamic_perception dynamic_perception.launch.py

Optional launch arguments:
  config_path:    Path to config.yaml (default: <package_root>/config.yaml)
  slam_mock:      'true' to run without real ORB-SLAM3 (default: 'true')
  yolo_model:     Path to YOLOv8 weights file (default: 'yolov8n.pt')
  log_level:      Logging verbosity: 'info', 'debug', 'warn' (default: 'info')

Spec ref: dynamic-components-design.md — compute scheduling Section 6.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


_PKG_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = str(_PKG_DIR / "config.yaml")


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        # ----------------------------------------------------------------
        # Launch arguments
        # ----------------------------------------------------------------
        DeclareLaunchArgument(
            "config_path",
            default_value=_DEFAULT_CONFIG,
            description="Absolute path to config.yaml",
        ),
        DeclareLaunchArgument(
            "slam_mock",
            default_value="true",
            description="Run ORB-SLAM3 interface in mock mode (true/false)",
        ),
        DeclareLaunchArgument(
            "yolo_model",
            default_value="yolov8n.pt",
            description="Path to YOLOv8 .pt or TensorRT .engine weights",
        ),
        DeclareLaunchArgument(
            "log_level",
            default_value="info",
            description="ROS2 log level (debug, info, warn, error)",
        ),

        # ----------------------------------------------------------------
        # Dynamic perception node
        # ----------------------------------------------------------------
        Node(
            package="dynamic_perception",
            executable="dynamic_perception_node",
            name="dynamic_perception_node",
            output="screen",
            parameters=[
                {
                    # These parameters are read inside the node via config.yaml;
                    # ROS2 parameters here override only the slam_mock flag for
                    # convenience at launch time without editing config.yaml.
                    "slam_mock_mode": LaunchConfiguration("slam_mock"),
                    "yolo_model_path": LaunchConfiguration("yolo_model"),
                }
            ],
            arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
            # Remappings allow adapting to different camera driver topic names
            remappings=[
                ("/camera/color/image_raw", "/camera/color/image_raw"),
                ("/camera/depth/image_raw", "/camera/depth/image_raw"),
                ("/camera/imu",             "/camera/imu"),
                ("/orb_slam3/pose",          "/orb_slam3/pose"),
                ("/wheelchair/scene_state",  "/wheelchair/scene_state"),
                ("/wheelchair/dynamic_mask", "/wheelchair/dynamic_mask"),
            ],
        ),
    ])
