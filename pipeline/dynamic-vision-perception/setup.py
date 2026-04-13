"""
setup.py — ROS2 ament_python build for the dynamic_perception package.

Build with:
  cd <workspace_root>
  colcon build --packages-select dynamic_perception
  source install/setup.bash
"""

from setuptools import find_packages, setup
from pathlib import Path

_PKG_ROOT = Path(__file__).parent

setup(
    name="dynamic_perception",
    version="0.1.0",
    packages=find_packages(exclude=["test", "nodes", "launch"]),
    data_files=[
        # ROS2 package index
        ("share/ament_index/resource_index/packages", ["resource/dynamic_perception"]),
        ("share/dynamic_perception", ["package.xml"]),
        # Launch files
        ("share/dynamic_perception/launch", ["launch/dynamic_perception.launch.py"]),
        # Default config
        ("share/dynamic_perception/config", ["config.yaml"]),
    ],
    install_requires=[
        "setuptools",
    ],
    zip_safe=True,
    maintainer="Wheelchair Research",
    maintainer_email="research@wheelchair-project.local",
    description=(
        "Dynamic components sub-system for the vision-only indoor wheelchair "
        "perception pipeline."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # ROS2 node entry point
            "dynamic_perception_node = nodes.dynamic_perception_node:main",
        ],
    },
)
