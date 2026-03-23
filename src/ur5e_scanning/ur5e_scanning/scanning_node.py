#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    PositionIKRequest, 
    RobotState, 
    Constraints, 
    OrientationConstraint,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
    RobotTrajectory,
    DisplayTrajectory
)
from moveit_msgs.srv import GetCartesianPath, GetPlanningScene
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import numpy as np
from math import pi, cos, sin, sqrt, atan2
import time
import sys

def quaternion_from_euler(roll, pitch, yaw):
    """Convert Euler angles to quaternion"""
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    
    return q

class UR5eScanningNode(Node):
    def __init__(self):
        super().__init__('ur5e_scanning_node')
        
        self.get_logger().info('Initializing UR5e Scanning Node...')
        
        # Parameters for scanning
        self.declare_parameters(
            namespace='',
            parameters=[
                ('part_center', [0.4, 0.0, 0.2]),  # [x, y, z] in meters
                ('scan_radius', 0.3),               # Distance from part
                ('num_scan_points', 8),             # Points per circle
                ('scan_heights', [0.1, 0.2, 0.3]),  # Multiple scan levels
                ('scan_type', 'circular'),          # 'circular' or 'helical'
                ('robot_speed', 0.1),               # Speed factor (0-1)
            ]
        )
        
        # Get parameters
        self.part_center = self.get_parameter('part_center').value
        self.scan_radius = self.get_parameter('scan_radius').value
        self.num_scan_points = self.get_parameter('num_scan_points').value
        self.scan_heights = self.get_parameter('scan_heights').value
        self.scan_type = self.get_parameter('scan_type').value
        self.robot_speed = self.get_parameter('robot_speed').value
        
        # Action client for MoveGroup
        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            '/move_action'
        )
        
        # Service client for Cartesian path
        self.cartesian_path_client = self.create_client(
            GetCartesianPath,
            '/compute_cartesian_path'
        )
        
        # Service client for planning scene
        self.planning_scene_client = self.create_client(
            GetPlanningScene,
            '/get_planning_scene'
        )
        
        # Publisher for trajectory visualization
        self.display_pub = self.create_publisher(
            DisplayTrajectory,
            '/display_planned_path',
            10
        )
        
        # Subscriber for joint states
        self.joint_states = None
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Wait for services
        self.get_logger().info('Waiting for MoveIt services...')
        self.move_group_client.wait_for_server()
        self.cartesian_path_client.wait_for_service()
        
        self.get_logger().info('UR5e Scanning Node initialized!')
        
    def joint_state_callback(self, msg):
        """Store current joint states"""
        self.joint_states = msg
    
    def calculate_camera_orientation(self, camera_pos, target_pos):
        """Calculate orientation to point camera at target"""
        # Vector from camera to target
        dx = target_pos[0] - camera_pos[0]
        dy = target_pos[1] - camera_pos[1]
        dz = target_pos[2] - camera_pos[2]
        
        # Calculate yaw (rotation around z-axis)
        yaw = atan2(dy, dx)
        
        # Calculate pitch (rotation around y-axis)
        horizontal_dist = sqrt(dx**2 + dy**2)
        pitch = atan2(-dz, horizontal_dist)
        
        # Roll is typically 0 for scanning
        roll = 0
        
        return roll, pitch, yaw
    
    def create_pose_stamped(self, x, y, z, roll, pitch, yaw):
        """Create a PoseStamped message"""
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base_link"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = z
        
        pose_stamped.pose.orientation = quaternion_from_euler(roll, pitch, yaw)
        
        return pose_stamped
    
    def generate_circular_waypoints(self, height):
        """Generate circular scanning waypoints at given height"""
        waypoints = []
        
        for i in range(self.num_scan_points):
            angle = 2 * pi * i / self.num_scan_points
            
            # Calculate position
            x = self.part_center[0] + self.scan_radius * cos(angle)
            y = self.part_center[1] + self.scan_radius * sin(angle)
            z = height
            
            # Calculate orientation to look at part
            camera_pos = [x, y, z]
            target_pos = [self.part_center[0], self.part_center[1], height]
            roll, pitch, yaw = self.calculate_camera_orientation(camera_pos, target_pos)
            
            # Create pose
            pose_stamped = self.create_pose_stamped(x, y, z, roll, pitch, yaw)
            waypoints.append(pose_stamped)
        
        # Close the loop
        waypoints.append(waypoints[0])
        
        return waypoints
    
    def generate_helical_waypoints(self):
        """Generate helical (spiral) scanning waypoints"""
        waypoints = []
        
        points_per_level = self.num_scan_points
        total_points = points_per_level * len(self.scan_heights)
        
        for i in range(total_points):
            # Continuous angle for spiral
            angle = 2 * pi * i / points_per_level
            
            # Interpolate height
            height_progress = i / (total_points - 1) if total_points > 1 else 0
            height = self.scan_heights[0] + height_progress * (self.scan_heights[-1] - self.scan_heights[0])
            
            # Calculate position
            x = self.part_center[0] + self.scan_radius * cos(angle)
            y = self.part_center[1] + self.scan_radius * sin(angle)
            z = height
            
            # Calculate orientation
            camera_pos = [x, y, z]
            target_pos = [self.part_center[0], self.part_center[1], height]
            roll, pitch, yaw = self.calculate_camera_orientation(camera_pos, target_pos)
            
            # Create pose
            pose_stamped = self.create_pose_stamped(x, y, z, roll, pitch, yaw)
            waypoints.append(pose_stamped)
        
        return waypoints
    
    async def compute_cartesian_path(self, waypoints):
        """Compute a Cartesian path through waypoints"""
        if not self.joint_states:
            self.get_logger().error("No joint states available")
            return None
            
        request = GetCartesianPath.Request()
        request.header.frame_id = "base_link"
        request.header.stamp = self.get_clock().now().to_msg()
        
        # Start state
        request.start_state.joint_state = self.joint_states
        request.start_state.is_diff = False
        
        # Group name
        request.group_name = "ur_manipulator"
        
        # Waypoints
        request.waypoints = waypoints
        
        # Parameters
        request.max_step = 0.01  # 1cm resolution
        request.jump_threshold = 0.0  # Disable jump threshold
        request.avoid_collisions = True
        
        # Call service
        future = self.cartesian_path_client.call_async(request)
        response = await future
        
        if response.fraction > 0.9:  # 90% of path computed
            self.get_logger().info(f"Computed {response.fraction*100:.1f}% of Cartesian path")
            return response.solution
        else:
            self.get_logger().warn(f"Could only compute {response.fraction*100:.1f}% of path")
            return None
    
    async def execute_trajectory(self, trajectory):
        """Execute a planned trajectory"""
        if not trajectory:
            return False
            
        # Visualize trajectory
        display_msg = DisplayTrajectory()
        display_msg.trajectory.append(trajectory)
        self.display_pub.publish(display_msg)
        
        # Create MoveGroup goal
        goal = MoveGroup.Goal()
        goal.request.group_name = "ur_manipulator"
        goal.request.num_planning_attempts = 1
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = self.robot_speed
        goal.request.max_acceleration_scaling_factor = self.robot_speed * 0.5
        
        # Since we already have a trajectory, we'll execute it directly
        # This is a simplified approach - in practice you might need ExecuteTrajectory action
        
        self.get_logger().info("Executing trajectory...")
        # Here you would send the trajectory to the robot controller
        # For now, we'll just simulate the execution
        await self.simulate_trajectory_execution(trajectory)
        
        return True
    
    async def simulate_trajectory_execution(self, trajectory):
        """Simulate trajectory execution with delays"""
        if trajectory.joint_trajectory.points:
            num_points = len(trajectory.joint_trajectory.points)
            self.get_logger().info(f"Executing {num_points} trajectory points...")
            
            for i, point in enumerate(trajectory.joint_trajectory.points):
                self.get_logger().info(f"Point {i+1}/{num_points}")
                # In real implementation, send this to robot controller
                import time
            	time.sleep(0.1)  # Simulate execution time
    	
    async def sleep_async(self, duration):
        """Async sleep function"""
        import asyncio
        await asyncio.sleep(duration)
    
    async def move_to_named_target(self, target_name):
        """Move to a named target like 'home'"""
        goal = MoveGroup.Goal()
        goal.request.group_name = "ur_manipulator"
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        
        # Set named target
        goal.request.goal_constraints = []
        # In practice, you'd set joint constraints for the named position
        
        self.get_logger().info(f"Moving to {target_name} position...")
        
        # Send goal
        future = self.move_group_client.send_goal_async(goal)
        goal_handle = await future
        
        if not goal_handle.accepted:
            self.get_logger().error(f"Failed to move to {target_name}")
            return False
            
        result_future = goal_handle.get_result_async()
        result = await result_future
        
        return result.result.error_code.val == 1  # SUCCESS
    
    async def run_scanning_routine(self):
        """Main scanning routine"""
        self.get_logger().info("=" * 50)
        self.get_logger().info("Starting UR5e Scanning Routine")
        self.get_logger().info(f"Scan type: {self.scan_type}")
        self.get_logger().info(f"Part center: {self.part_center}")
        self.get_logger().info(f"Scan radius: {self.scan_radius}m")
        self.get_logger().info("=" * 50)
        
        # Wait for joint states
        while not self.joint_states:
            self.get_logger().info("Waiting for joint states...")
            import time
            time.sleep(1.0)
        
        # Move to home position
        await self.move_to_named_target("home")
        
        if self.scan_type == "circular":
            # Perform circular scans at each height
            for height in self.scan_heights:
                self.get_logger().info(f"\nScanning at height {height}m")
                waypoints = self.generate_circular_waypoints(height)
                
                # Compute Cartesian path
                trajectory = await self.compute_cartesian_path(waypoints)
                
                if trajectory:
                    await self.execute_trajectory(trajectory)
                    self.get_logger().info(f"Completed scan at height {height}m")
                else:
                    self.get_logger().warn(f"Failed to plan path at height {height}m")
                
                import time
            	time.sleep(1.0)
        
        elif self.scan_type == "helical":
            # Perform helical scan
            self.get_logger().info("\nExecuting helical scan")
            waypoints = self.generate_helical_waypoints()
            
            # Compute Cartesian path
            trajectory = await self.compute_cartesian_path(waypoints)
            
            if trajectory:
                await self.execute_trajectory(trajectory)
                self.get_logger().info("Completed helical scan")
            else:
                self.get_logger().warn("Failed to plan helical path")
        
        # Return to home
        await self.move_to_named_target("home")
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("Scanning routine complete!")
        self.get_logger().info("=" * 50)

def main(args=None):
    rclpy.init(args=args)
    
    node = UR5eScanningNode()
    
    # Create async executor
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    
    # Run the scanning routine
    async def run():
        await node.run_scanning_routine()
        rclpy.shutdown()
    
    # Execute
    try:
        executor.create_task(run())
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
