# Autonomous Robotic Inspection System

**Fully automated quality inspection for circular economy remanufacturing — zero human involvement after setup.**

Built as a Student Assistant at the [wbk Institute for Production Science](https://www.wbk.kit.edu/), Karlsruhe Institute of Technology (KIT), under the DFG-funded research programme [SFB 1574 B01](https://www.sfb1574.kit.edu/).

---

## The Problem

In circular economy remanufacturing, used industrial parts — power tools, motors, components — are recovered, refurbished, and returned to service instead of being scrapped. The bottleneck is inspection: a human has to physically examine every part, read its label, and decide if it passes. It's slow, inconsistent, and impossible to scale.

**The goal: eliminate that human from the loop entirely.**

---

## What It Does

On a single launch command, the system handles the full inspection autonomously:

1. **Workspace calibration** — detects an ArUco marker and solves for its 6-DOF pose in the robot's frame using solvePnP. No pre-programmed positions.
2. **Part detection** — YOLOv8 finds the target object in the live camera feed (confidence ≥ 0.85).
3. **3D localisation** — the bounding box centroid is deprojected through the aligned depth stream into a metric XYZ coordinate in the robot's base frame.
4. **Orbit generation** — 20 viewpoints are computed around the detected object pose across 3 concentric rings (r = 0.20 / 0.28 / 0.35m). Each viewpoint is analytically aimed at the object centre using Shepperd's method for quaternion computation. The path adapts to wherever the object actually is.
5. **Autonomous capture** — the UR5e moves to each viewpoint via MoveIt2, captures a frame at each position.
6. **VLM analysis** — the best frame is sent to a self-hosted vision-language model (Ollama + FastAPI) which reads the label text, extracts part identifiers, and flags visual anomalies.
7. **Pass/fail report** — results appear on a live web dashboard (Flask, port 5000).

---

## System Architecture

```
Intel RealSense D455 (RGB-D, 640×480 @ 30fps)
        │
        ▼
┌─────────────────────┐     ┌──────────────────────┐
│  workspace_calibrator│     │   object_localizer   │
│  ArUco → solvePnP   │     │  YOLO + depth deproj │
│  → /workspace_anchor│     │  → /object_pose      │
└─────────────────────┘     └──────────┬───────────┘
                                        │
                            ┌───────────▼───────────┐
                            │  inspection_dashboard  │
                            │  Orbit generation      │
                            │  State machine         │
                            │  Flask web UI          │
                            └───────────┬───────────┘
                                        │
                            ┌───────────▼───────────┐
                            │     robot_mover        │
                            │  MoveIt2 IK → plan     │
                            │  → joint trajectory    │
                            └───────────┬───────────┘
                                        │
                                   UR5e arm
                                        │
                            ┌───────────▼───────────┐
                            │     VLM Server         │
                            │  Ollama + FastAPI      │
                            │  llama3.2-vision       │
                            │  /read-label endpoint  │
                            └───────────────────────┘
```

**ROS 2 Humble nodes (all custom-built):**

| Node | Role |
|---|---|
| `realsense_publisher` | Owns D455 via pyrealsense2 SDK — publishes color + aligned depth + CameraInfo |
| `camera_tf_publisher` | Static TF: tool0 → camera_link (35mm offset, centered, straight-down) |
| `workspace_calibrator` | ArUco detection → solvePnP → TF2 → `/workspace_anchor` (latches at 5 stable frames) |
| `object_localizer` | YOLO + depth deprojection → TF2 → `/object_pose` |
| `ur_robot_driver` | Hardware interface: ROS 2 controllers ↔ UR reverse-interface protocol |
| `ur_moveit_config` | MoveIt2 move group, planning scene, IK server |
| `inspection_dashboard` | Flask orchestrator: state machine + live YOLO feed + VLM call + web report |

---

## Stack

- **Robot:** Universal Robots UR5e
- **Camera:** Intel RealSense D455 (RGB-D)
- **Framework:** ROS 2 Humble, MoveIt2, TF2
- **Detection:** YOLOv8 (fine-tuned, 22MB, confidence ≥ 0.85)
- **VLM:** llama3.2-vision via Ollama + FastAPI (self-hosted — evaluated Gemini 2.5 Flash and OpenRouter before switching to avoid API costs and latency variance)
- **Language:** Python
- **Web UI:** Flask

---

## Hard Problems Solved

**OpenCV 4.12 API removal**
The entire ArUco detection API (`detectMarkers`, `estimatePoseSingleMarkers`) was removed in OpenCV 4.12 with no deprecation warning. Required a full rewrite of the workspace calibrator using the new `ArucoDetector` interface with manual `rvec`/`tvec` handling via `solvePnP`.

**MoveIt2 trajectory timestamp corruption**
MoveIt's execute layer was corrupting trajectory timestamps, causing jerky and unreliable motion. Diagnosed the issue and bypassed the execute layer entirely — publishing directly to `/scaled_joint_trajectory_controller/joint_trajectory`.

**CycloneDDS UDP buffer overflow**
Publishing 30fps RGB-D frames over DDS caused `ENOBUFS` errors at the CycloneDDS C layer that bypass all Python-level error handling. Partially mitigated via `sysctl` socket buffer tuning and a custom `cyclonedds.xml` config (`MaxMessageSize=10MB`).

**Viewpoint geometry rewrite**
After switching from an offset/tilted camera mount to a centered straight-down configuration, the entire orbit geometry had to be rederived — new radii, new height formula, new analytic quaternion solver using Shepperd's method to aim each viewpoint at the object centre.

---

## What's Built vs Off-the-Shelf

**Built from scratch:**
- All 7 ROS 2 nodes
- Tiered orbit viewpoint generator with analytic quaternion solver
- Depth deprojection pipeline (YOLO centroid → 3D pose)
- VLM FastAPI server (`/read-label` endpoint, Ollama wrapper)
- Flask inspection dashboard + state machine + SSE
- Custom camera mount (parametric CAD, 3D printed, ISO 9409-1-50-4-M6 flange)
- CycloneDDS configuration for image streaming

**Off-the-shelf:**
- `Universal_Robots_ROS2_Driver` (UR's open-source driver)
- `ur_moveit_config` (ships with the driver)
- YOLOv8 (pre-trained weights, fine-tuned on target parts)
- Ollama inference runtime

---

## Setup

```bash
# Clone this repo
git clone https://github.com/samkitparak/Autonomous-Robotic-Inspection-System ur5_ws
cd ur5_ws

# Clone the upstream UR driver
git clone -b humble https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver.git src/Universal_Robots_ROS2_Driver

# Build
colcon build

# Source (sets ROS_DOMAIN_ID=42, CycloneDDS, localhost-only)
source fast_setup.sh
```

Copy `.env.example` to `.env` and configure your robot IP and LLM server address.

**Launch:**
```bash
# Terminal 1 — Hardware
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=<IP>

# Terminal 2 — MoveIt2
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e

# Terminal 3 — Camera + localiser
ros2 launch ur5e_vision camera_transform.launch.py
ros2 run ur5e_vision object_localizer

# Terminal 4 — Inspection dashboard
ros2 run ur5e_vision inspection_dashboard
# → Open http://<robot-ip>:5000
```

Test LLM server connectivity:
```bash
python scripts/test_ollama.py
```

---

## Status

| Component | Status |
|---|---|
| Robot connection + controllers | ✅ Confirmed |
| Camera stream + depth alignment | ✅ Confirmed |
| ArUco workspace calibration | ✅ Confirmed |
| YOLO detection + 3D localisation | ✅ Confirmed |
| MoveIt2 trajectory execution | ✅ Confirmed |
| Full orbit + capture sequence | 🔄 Integration testing |
| VLM label read from live frame | 🔄 Integration testing |
| Defect detection (Phase B) | ⬜ Planned |

---

## Research Context

This project is part of **SFB 1574 B01** — a DFG-funded collaborative research centre at KIT investigating automated processes for circular economy remanufacturing. The inspection step is the current bottleneck in remanufacturing lines; this system is designed to replace or augment the human inspector entirely.

---

*Samkit Parak · wbk Institute for Production Science, KIT · paraksamkit@gmail.com*
