"""
camera_transform.launch.py
--------------------------
Publishes the static TF2 transform:  tool0 → camera_link

Camera geometry (centered mount):
  - Camera is directly below tool0 — no lateral offset, no tilt
  - Camera optical axis = tool0 -Z axis (straight down from the flange)
  - Z offset ≈ -33 mm (flange thickness 8 mm + tab height 25 mm)
    Measure precisely from the printed part and update camera_z_offset.

To verify visually after launching:
  ros2 run rviz2 rviz2
  Add display → TF, set Fixed Frame to 'base_link'.
  Confirm camera_link is directly below tool0 with axes aligned.

Arguments:
  parent_frame    : parent TF frame (default: tool0)
  camera_frame    : child TF frame — must match RealSense driver (default: camera_link)
  camera_z_offset : distance from tool0 face to camera lens, negative (default: -0.033)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Distance from tool0 face to camera lens center (metres, negative = below flange).
    # Measured: flange 8 mm + tab 25 mm + lens recess ≈ 35 mm total → -0.035
    CAMERA_Z_OFFSET = -0.035

    return LaunchDescription([
        DeclareLaunchArgument(
            'parent_frame', default_value='tool0',
            description='Robot flange frame to which the camera is attached'),
        DeclareLaunchArgument(
            'camera_frame', default_value='camera_link',
            description='Camera base frame published by the RealSense driver. '
                        'Check with: ros2 run tf2_tools view_frames'),
        DeclareLaunchArgument(
            'camera_z_offset', default_value=str(CAMERA_Z_OFFSET),
            description='Distance from tool0 face to camera lens (metres, negative).'),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_tf_publisher',
            output='screen',
            arguments=[
                '--frame-id',       LaunchConfiguration('parent_frame'),
                '--child-frame-id', LaunchConfiguration('camera_frame'),
                '--x',     '0.0',
                '--y',     '0.0',
                '--z',     LaunchConfiguration('camera_z_offset'),
                '--roll',  '0.0',
                '--pitch', '0.0',
                '--yaw',   '0.0',
            ],
        ),
    ])
