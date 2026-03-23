#!/usr/bin/env python3
import rclpy
import math
import numpy as np

from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit.planning import MoveItPy
from moveit.planning import PlanRequestParameters
from moveit.core.robot_state import RobotState
from moveit.planning.task_composer import TaskComposerClient
from tf_transformations import quaternion_from_matrix

class UR5CirclePath(Node):
    def __init__(self):
        super().__init__('ur5_circle_path')

        # Initialize MoveItPy (new in ROS2)
        self.moveit = MoveItPy(node_name="moveit_py")
        self.arm = self.moveit.get_planning_component("ur_manipulator")

    def circle_scan(self, center, radius=0.25, z_offset=0.15, num_points=36):
        """
        Generate and execute a circular path around a part, with the tool Z-axis pointing at the part center.
        center: (x, y, z) of part center
        radius: orbit radius
        z_offset: height offset from part center
        num_points: number of waypoints around the circle
        """
        waypoints = []

        # Current pose
        current_state = self.moveit.get_current_state()
        current_pose = self.arm.get_current_state().end_effector_state
        waypoints.append(current_pose)

        for i in range(num_points + 1):
            angle = 2.0 * math.pi * (i / num_points)

            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            z = center[2] + z_offset

            # Vector from EE → center
            direction = np.array([center[0] - x, center[1] - y, center[2] - z])
            direction = direction / np.linalg.norm(direction)

            # "Up" vector (world Z)
            up = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(direction, up)) > 0.95:
                up = np.array([0.0, 1.0, 0.0])

            # Build rotation matrix so Z-axis points to target
            z_axis = direction
            y_axis = np.cross(z_axis, up)
            y_axis /= np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)

            R = np.array([[x_axis[0], y_axis[0], z_axis[0], 0],
                          [x_axis[1], y_axis[1], z_axis[1], 0],
                          [x_axis[2], y_axis[2], z_axis[2], 0],
                          [0, 0, 0, 1]])

            quat = quaternion_from_matrix(R)

            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = float(z)
            pose.orientation.x = float(quat[0])
            pose.orientation.y = float(quat[1])
            pose.orientation.z = float(quat[2])
            pose.orientation.w = float(quat[3])

            waypoints.append(pose)

        # Plan Cartesian path
        params = PlanRequestParameters()
        params.max_acceleration_scaling_factor = 0.2
        params.max_velocity_scaling_factor = 0.2

        (plan, fraction) = self.arm.compute_cartesian_path(
            waypoints=waypoints,
            max_step=0.01,
            jump_threshold=0.0
        )

        if fraction < 0.9:
            self.get_logger().warn(f"Only {fraction*100:.1f}% of path planned!")
        else:
            self.get_logger().info("Path planned successfully.")

        # Execute trajectory
        if plan:
            self.arm.execute(plan)


def main(args=None):
    rclpy.init(args=args)
    node = UR5CirclePath()

    # Example: part center
    center = (0.5, 0.0, 0.1)
    node.circle_scan(center=center, radius=0.25, z_offset=0.15, num_points=36)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
