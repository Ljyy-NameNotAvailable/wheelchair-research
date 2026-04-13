"""
dynamic_perception — Dynamic components sub-system for wheelchair vision pipeline.

Implements the 3-layer dynamic masking architecture described in:
  research/vision-only-perception-pipeline/dynamic-components-design.md

Layers:
  1. Dynamic Mask Generation  (mask_generator.py + optical_flow.py)
  2. SLAM Protection           (slam_protector.py)
  3. Scene State Extension     (scene_state_publisher.py)

Trajectory stub:               trajectory_stub.py
ROS2 node wiring:              ../nodes/dynamic_perception_node.py
"""
