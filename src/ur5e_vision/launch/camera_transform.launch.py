"""
camera_transform.launch.py
--------------------------
Publishes the static TF2 transform:  tool0 → camera_link

Camera geometry (verify against CAD):
  - Translation : +17 cm along tool0 X axis  (camera is to the +X side of the flange)
                  If your camera is on the -X side, set camera_offset_x to -0.17
  - Rotation    : pitch = -60°
                  This rotates the camera's optical axis (camera Z) from tool0 +Z
                  (straight down when arm is in work pose) toward tool0 -X (center).
                  Result: optical axis = [-sin60°, 0, cos60°] = [-0.866, 0, 0.5] in tool0

Geometry check (arm 10 cm above table, TCP at origin):
  Camera at [0.17, 0, 0] m in tool0, looking along [-0.866, 0, 0.5].
  Ray hits table (tool0 Z = 0.1 m) at X ≈ 0 m — i.e., roughly below the TCP. ✓

To verify visually after launching:
  ros2 run rviz2 rviz2
  Add display → TF, set Fixed Frame to 'base_link'.
  Confirm camera_link is 17 cm in +X from tool0, with Z axis angled toward center.

Arguments:
  parent_frame       : parent TF frame (default: tool0)
  camera_frame       : child TF frame — must match RealSense driver output (default: camera_link)
  camera_offset_x    : lateral offset in metres (default: 0.17)
  camera_pitch_rad   : pitch in radians (default: -1.0472 = -60°)
"""

import math
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ----- physical constants -----
    OFFSET_X   =  0.17               # metres  — camera is +X from tool0
    OFFSET_Y   =  0.0
    OFFSET_Z   =  0.0
    ROLL       =  0.0
    PITCH      = -math.pi / 3        # -60° : rotates camera Z toward -X (center)
    YAW        =  0.0

    return LaunchDescription([
        DeclareLaunchArgument(
            'parent_frame', default_value='tool0',
            description='Robot flange frame to which the camera is attached'),
        DeclareLaunchArgument(
            'camera_frame', default_value='camera_link',
            description='Camera base frame published by the RealSense driver. '
                        'Check with: ros2 run tf2_tools view_frames'),
        DeclareLaunchArgument(
            'camera_offset_x', default_value=str(OFFSET_X),
            description='Camera offset along tool0 X axis (metres). '
                        'Use negative value if camera is on the -X side.'),
        DeclareLaunchArgument(
            'camera_pitch_rad', default_value=str(PITCH),
            description='Pitch of camera frame relative to tool0 (radians). '
                        '-pi/3 (-60°) tilts optical axis from +Z toward -X.'),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_tf_publisher',
            output='screen',
            arguments=[
                '--frame-id',       LaunchConfiguration('parent_frame'),
                '--child-frame-id', LaunchConfiguration('camera_frame'),
                '--x',     LaunchConfiguration('camera_offset_x'),
                '--y',     str(OFFSET_Y),
                '--z',     str(OFFSET_Z),
                '--roll',  str(ROLL),
                '--pitch', LaunchConfiguration('camera_pitch_rad'),
                '--yaw',   str(YAW),
            ],
        ),
    ])
