# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Always source the workspace environment before running any ROS2 commands:

```bash
source ~/ur5_ws/fast_setup.sh
```

This sets `ROS_DOMAIN_ID=42`, restricts communication to localhost, enables CycloneDDS middleware, and sources both ROS2 Humble and the install overlay.

## Build Commands

```bash
# Build all packages
colcon build

# Build a specific package
colcon build --packages-select ur5e_scanner

# Run tests
colcon test --packages-select ur5e_scanner
colcon test-result --verbose

# Run a single test file
python -m pytest src/ur5e_scanner/test/test_foo.py
```

After rebuilding, always re-source the install overlay: `source install/setup.bash`

## Architecture Overview

### Control Approaches

The workspace implements three increasingly sophisticated approaches to UR5e control:

1. **Direct joint trajectory** — Publish `trajectory_msgs/JointTrajectory` directly to `/scaled_joint_trajectory_controller/joint_trajectory`. Used in `sanity.py` for quick tests.

2. **MoveIt2 action/service clients** — Use `/move_action` (MoveGroup action) or `/compute_cartesian_path` service. Implemented in `src/ur5e_scanning/`. Subscribe to `/joint_states` to seed the start state before planning.

3. **MoveItPy API** — High-level Python API used in `src/ur5_circle_path/`. Cleaner interface but requires MoveIt2 to be running.

4. **Vision + LLM servoing** — Flask server + YOLO (`best.pt`) + LLM (OpenRouter). Visual offsets are computed and mapped to Cartesian/joint corrections. See `integrated_ai_robot.py` (Moonshot/Kimi) and `newaiscanner.py` (Gemini).

### Key Packages

| Package | Type | Purpose |
|---|---|---|
| `Universal_Robots_ROS2_Driver` | External (upstream) | Hardware driver, controllers, MoveIt2 config |
| `ur5e_scanner` | ament_python | Scanning paths via direct trajectories and MoveIt2 |
| `ur5e_scanning` | ament_python | Advanced scanning using MoveIt2 services/actions |
| `ur5_circle_path` | script-only (no package.xml) | Circular path demo using MoveItPy |

### ROS2 Communication

**Subscriptions:** `/joint_states` (sensor_msgs/JointState) — current robot state

**Publications:** `/scaled_joint_trajectory_controller/joint_trajectory` (trajectory_msgs/JointTrajectory)

**MoveIt2 interfaces:**
- Action: `/move_action` (moveit_msgs/MoveGroup)
- Service: `/compute_cartesian_path` (moveit_msgs/GetCartesianPath)
- Service: `/get_planning_scene` (moveit_msgs/GetPlanningScene)
- Topic: `/display_planned_path` (moveit_msgs/DisplayTrajectory)

### Standalone Scripts (root of workspace)

These live outside the package structure and are run directly with `python`:

- `sanity.py` — Minimal test: sends a +0.2 rad shoulder_pan move
- `integrated_ai_robot.py` — Flask web server with YOLO + Moonshot LLM visual servoing
- `newaiscanner.py` — Same concept but uses Google Gemini 2.5 Flash

### YOLO Model

`best.pt` (22 MB) at the workspace root is the pre-trained object detection model used by both AI scanner scripts. `CONFIDENCE_THRESHOLD = 0.85`.

## Launching the Robot

```bash
# Hardware bringup (UR5e)
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=<IP> use_fake_hardware:=false

# MoveIt2 (separate terminal)
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e

# Run a scanning node
ros2 run ur5e_scanning scanning_node
```

## DDS Configuration

`cyclonedds.xml` at the workspace root configures CycloneDDS with `MaxMessageSize=10MB`. The `fast_setup.sh` sets `CYCLONEDDS_URI` to point to this file.
