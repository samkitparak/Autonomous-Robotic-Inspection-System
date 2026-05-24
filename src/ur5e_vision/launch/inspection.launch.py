"""
inspection.launch.py
--------------------
One-command launch for the full UR5e autonomous inspection system.

Usage:
  ros2 launch ur5e_vision inspection.launch.py robot_ip:=<IP>

Arguments:
  robot_ip        : IP address of the UR5e controller (required)
  llm_server      : Base URL of the LLM inference server (default: http://172.22.132.20:8001)
  move_speed      : MoveIt2 velocity scaling factor 0-1 (default: 0.15)
  port            : Dashboard web server port (default: 5000)
  use_fake_hardware : Set true to run without physical robot (default: false)

Nodes started:
  1. ur_robot_driver        — hardware bringup + controllers
  2. ur_moveit_config       — MoveIt2 move group + RViz
  3. realsense_publisher    — RealSense D455 color + depth
  4. camera_transform       — static TF: tool0 → camera_link
  5. workspace_calibrator   — ArUco marker detection → /workspace_anchor
  6. object_localizer       — YOLO detection + 3D pose via depth
  7. inspection_dashboard   — Flask web UI + structured inspection loop
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ------------------------------------------------------------------ args
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        description='IP address of the UR5e robot controller (e.g. 192.168.1.100)')

    llm_server_arg = DeclareLaunchArgument(
        'llm_server',
        default_value='http://172.22.132.20:8001',
        description='Base URL of the LLM inference server')

    move_speed_arg = DeclareLaunchArgument(
        'move_speed',
        default_value='0.15',
        description='MoveIt2 velocity scaling factor (0.0 – 1.0)')

    port_arg = DeclareLaunchArgument(
        'port',
        default_value='5000',
        description='Port for the Flask web dashboard')

    use_fake_arg = DeclareLaunchArgument(
        'use_fake_hardware',
        default_value='false',
        description='Set true to run without physical robot')

    table_z_arg = DeclareLaunchArgument(
        'table_z',
        default_value='0.1',
        description='Z-height of work surface top face in base_link frame (metres). '
                    'Measure with ruler from robot base to table surface.')

    # --------------------------------------------------------- 1. ur_control
    ur_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ur_robot_driver'),
                'launch', 'ur_control.launch.py',
            ])
        ]),
        launch_arguments={
            'ur_type':            'ur5e',
            'robot_ip':           LaunchConfiguration('robot_ip'),
            'use_fake_hardware':  LaunchConfiguration('use_fake_hardware'),
            'launch_rviz':        'false',
        }.items(),
    )

    # --------------------------------------------------------- 2. ur_moveit
    # Delay so controllers are active before MoveIt2 tries to connect
    ur_moveit = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    PathJoinSubstitution([
                        FindPackageShare('ur_moveit_config'),
                        'launch', 'ur_moveit.launch.py',
                    ])
                ]),
                launch_arguments={
                    'ur_type':   'ur5e',
                    'launch_rviz': 'true',
                }.items(),
            )
        ],
    )

    # ------------------------------------------------------ 3. realsense
    # Delay until after driver is ready
    realsense = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='ur5e_vision',
                executable='realsense_publisher',
                name='realsense_publisher',
                output='log',
            )
        ],
    )

    # -------------------------------------------------- 4. camera_transform
    camera_transform = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    PathJoinSubstitution([
                        FindPackageShare('ur5e_vision'),
                        'launch', 'camera_transform.launch.py',
                    ])
                ]),
            )
        ],
    )

    # ------------------------------------------- 5. workspace_calibrator
    workspace_calibrator = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='ur5e_vision',
                executable='workspace_calibrator',
                name='workspace_calibrator',
                output='screen',
                parameters=[{
                    'marker_length': 0.15,
                    'marker_id':     0,
                    'base_frame':    'base_link',
                    'stable_frames': 5,
                }],
            )
        ],
    )

    # -------------------------------------------------- 7. object_localizer
    vision_params = PathJoinSubstitution([
        FindPackageShare('ur5e_vision'),
        'config', 'vision_params.yaml',
    ])

    object_localizer = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='ur5e_vision',
                executable='object_localizer',
                name='object_localizer',
                output='log',
                parameters=[vision_params],
            )
        ],
    )

    # ----------------------------------------------- 8. inspection_dashboard
    # Start last — needs MoveIt2 and camera topics to be up
    inspection_dashboard = TimerAction(
        period=10.0,
        actions=[
            Node(
                package='ur5e_vision',
                executable='inspection_dashboard',
                name='inspection_dashboard',
                output='screen',
                parameters=[{
                    'llm_server': LaunchConfiguration('llm_server'),
                    'move_speed': LaunchConfiguration('move_speed'),
                    'port':       LaunchConfiguration('port'),
                    'table_z':    LaunchConfiguration('table_z'),
                }],
            )
        ],
    )

    return LaunchDescription([
        robot_ip_arg,
        llm_server_arg,
        move_speed_arg,
        port_arg,
        use_fake_arg,
        table_z_arg,
        ur_control,
        ur_moveit,
        realsense,
        camera_transform,
        workspace_calibrator,
        object_localizer,
        inspection_dashboard,
    ])
