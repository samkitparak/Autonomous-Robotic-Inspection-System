# UR5e Scanning Workspace

ROS2 Humble workspace for autonomous inspection and visual servoing with a Universal Robots UR5e arm and Intel RealSense D455 camera.

## Dependencies

- ROS2 Humble
- [Universal_Robots_ROS2_Driver](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver) (humble branch) — clone into `src/`
- MoveIt2 for ROS2 Humble
- Python: `ultralytics`, `flask`, `openai`, `cv2`, `numpy`

## Setup

```bash
# Clone this repo
git clone <repo-url> ur5_ws
cd ur5_ws

# Clone the upstream driver (not included in this repo)
git clone -b humble https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver.git src/Universal_Robots_ROS2_Driver

# Build
colcon build

# Source the workspace (sets ROS_DOMAIN_ID=42, CycloneDDS, localhost-only)
source fast_setup.sh
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your API key before running the AI scanner scripts:

```bash
cp .env.example .env
# edit .env with your OPENROUTER_API_KEY
export $(cat .env | xargs)
```

## Packages

| Package | Purpose |
|---|---|
| `ur5e_scanner` | Direct trajectory scanning and MoveIt2-based scanning (IK + planning) |
| `ur5e_scanning` | Advanced scanning via MoveIt2 action/service clients; circular and helical paths |
| `ur5e_vision` | YOLO object detection on RealSense depth+color → 3D pose → LLM-guided servoing |
| `ur5_circle_path` | Circular path demo using the MoveItPy high-level API |

## Standalone Scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `sanity.py` | Minimal test: moves shoulder_pan_joint by +0.2 rad |
| `integrated_ai_robot.py` | Flask server + YOLO + Moonshot/Kimi LLM visual servoing |
| `newaiscanner.py` | Flask server + YOLO + Google Gemini 2.5 Flash visual servoing |

Run with: `python scripts/<script>.py`

## Launching the Robot

```bash
# Terminal 1 — Hardware bringup
source ~/ur5_ws/fast_setup.sh
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=<IP> use_fake_hardware:=false

# Terminal 2 — MoveIt2
source ~/ur5_ws/fast_setup.sh
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e

# Terminal 3 — Scanning node
source ~/ur5_ws/fast_setup.sh
ros2 run ur5e_scanning scanning_node

# Or vision pipeline
source ~/ur5_ws/fast_setup.sh
ros2 launch ur5e_vision camera_transform.launch.py
ros2 run ur5e_vision object_localizer
ros2 run ur5e_vision llm_planner
```

## YOLO Model

`best.pt` (22 MB) is the pre-trained object detection model used by the AI scanner scripts.
`CONFIDENCE_THRESHOLD = 0.85`.

## ROS2 Interfaces

**Topics:**
- `/joint_states` (sensor_msgs/JointState) — robot state
- `/scaled_joint_trajectory_controller/joint_trajectory` (trajectory_msgs/JointTrajectory) — direct control

**MoveIt2:**
- Action: `/move_action` (moveit_msgs/MoveGroup)
- Service: `/compute_cartesian_path` (moveit_msgs/GetCartesianPath)

## DDS Configuration

`cyclonedds.xml` sets `MaxMessageSize=10MB` to handle uncompressed camera streams.
`fast_setup.sh` sets `CYCLONEDDS_URI` to point to this file.
