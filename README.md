# UR5e Scanning Workspace

ROS2 Humble workspace for autonomous inspection and visual servoing with a Universal Robots UR5e arm and Intel RealSense D455 camera.

## Dependencies

- ROS2 Humble
- [Universal_Robots_ROS2_Driver](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver) (humble branch) — clone into `src/`
- MoveIt2 for ROS2 Humble
- Python: `ultralytics`, `flask`, `requests`, `cv2`, `numpy`

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

## LLM Server

Label reading is handled by a self-hosted LLM server running on a separate GPU PC.

| Setting | Value |
|---|---|
| Host | `172.22.132.20:8001` |
| Model | `llama3.2-vision:latest` (via Ollama) |
| Endpoints | `GET /health`, `POST /read-label`, `POST /decide` |

Test connectivity before running the vision pipeline:

```bash
python scripts/test_ollama.py
```

## Packages

| Package | Purpose |
|---|---|
| `ur5e_scanner` | Direct trajectory scanning and MoveIt2-based scanning (IK + planning) |
| `ur5e_scanning` | Advanced scanning via MoveIt2 action/service clients; circular and helical paths |
| `ur5e_vision` | YOLO object detection → 3D pose → orbit inspection → LLM label reading via `inspection_dashboard` |
| `ur5_circle_path` | Circular path demo using the MoveItPy high-level API |

## Standalone Scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `sanity.py` | Minimal test: moves shoulder_pan_joint by +0.2 rad |
| `integrated_ai_robot.py` | Flask server + YOLO + Moonshot/Kimi LLM visual servoing |
| `newaiscanner.py` | Flask server + YOLO + Google Gemini 2.5 Flash visual servoing |
| `test_ollama.py` | Connectivity test for the LLM server (`/health` + `/read-label`) |

Run with: `python scripts/<script>.py`

## Launching the Robot

```bash
# Terminal 1 — Hardware bringup
source ~/ur5_ws/fast_setup.sh
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=<IP> use_fake_hardware:=false

# Terminal 2 — MoveIt2
source ~/ur5_ws/fast_setup.sh
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e

# Terminal 3 — Camera + object localizer
source ~/ur5_ws/fast_setup.sh
ros2 launch ur5e_vision camera_transform.launch.py
ros2 run ur5e_vision object_localizer

# Terminal 4 — Inspection dashboard (web UI on port 5000)
source ~/ur5_ws/fast_setup.sh
ros2 run ur5e_vision inspection_dashboard
```

Open `http://<robot-ip>:5000` in a browser to access the inspection dashboard.

To override the LLM server address at launch:

```bash
ros2 run ur5e_vision inspection_dashboard --ros-args -p llm_server:=http://172.22.132.20:8001
```

## YOLO Model

`best.pt` (22 MB) is the pre-trained object detection model. `CONFIDENCE_THRESHOLD = 0.85`.

## ROS2 Interfaces

**Topics:**
- `/joint_states` (sensor_msgs/JointState) — robot state
- `/scaled_joint_trajectory_controller/joint_trajectory` (trajectory_msgs/JointTrajectory) — direct control
- `/detected_object/image` (sensor_msgs/Image) — YOLO-annotated camera frame
- `/detected_object/pose` (geometry_msgs/PoseStamped) — 3D object position in base frame

**MoveIt2:**
- Action: `/move_action` (moveit_msgs/MoveGroup)
- Service: `/compute_cartesian_path` (moveit_msgs/GetCartesianPath)

## DDS Configuration

`cyclonedds.xml` sets `MaxMessageSize=10MB` to handle uncompressed camera streams.
`fast_setup.sh` sets `CYCLONEDDS_URI` to point to this file.
