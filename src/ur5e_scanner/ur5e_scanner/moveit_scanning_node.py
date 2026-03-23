#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.msg import DisplayTrajectory, MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
import math
import time

class UR5eScanningWithMoveIt(Node):
    def __init__(self):
        super().__init__('ur5e_scanning_moveit')
        
        # MoveIt Action Client
        self.move_group_client = ActionClient(self, MoveGroup, '/move_action')
        
        # Publishers for visualization
        self.display_trajectory_pub = self.create_publisher(
            DisplayTrajectory, 
            '/move_group/display_planned_path', 
            1
        )
        
        # Basic scanning parameters - SAFE VALUES FOR TESTING
        self.scan_center_x = 0.3  # 30cm forward from base
        self.scan_center_y = 0.0  # centered
        self.scan_center_z = 0.3  # 30cm high - SAFE HEIGHT
        self.scan_radius = 0.1    # Small 10cm radius for testing
        self.scan_height = 0.05   # 5cm above part
        self.num_points = 6       # 6 points for testing (60° apart)
        
        self.planning_group = "ur_manipulator"
        
        self.get_logger().info('UR5e MoveIt Scanning Node initialized')
        self.get_logger().info('Waiting for MoveIt action server...')
        
        # Wait for MoveIt action server
        self.move_group_client.wait_for_server(timeout_sec=10.0)
        self.get_logger().info('MoveIt action server connected!')
        
    def create_move_group_goal(self, target_pose):
        """Create a MoveGroup action goal"""
        goal = MoveGroup.Goal()
        goal.request.group_name = self.planning_group
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.1  # Slow for safety
        goal.request.max_acceleration_scaling_factor = 0.1
        
        # Set target pose
        constraint = goal.request.goal_constraints.add()
        constraint.position_constraints.add()
        constraint.position_constraints[0].header.frame_id = "base_link"
        constraint.position_constraints[0].link_name = "tool0"
        constraint.position_constraints[0].target_point_offset.x = target_pose.position.x
        constraint.position_constraints[0].target_point_offset.y = target_pose.position.y  
        constraint.position_constraints[0].target_point_offset.z = target_pose.position.z
        constraint.position_constraints[0].weight = 1.0
        
        # Add orientation constraint
        constraint.orientation_constraints.add()
        constraint.orientation_constraints[0].header.frame_id = "base_link"
        constraint.orientation_constraints[0].link_name = "tool0"
        constraint.orientation_constraints[0].orientation = target_pose.orientation
        constraint.orientation_constraints[0].weight = 1.0
        
        return goal
        
    def generate_circular_path(self):
        """Generate circular scanning path around the part"""
        waypoints = []
        
        for i in range(self.num_points):
            angle = 2 * math.pi * i / self.num_points
            
            # Calculate position in circle
            x = self.scan_center_x + self.scan_radius * math.cos(angle)
            y = self.scan_center_y + self.scan_radius * math.sin(angle)
            z = self.scan_center_z + self.scan_height
            
            # Create pose
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            
            # Orientation - point tool down towards part
            # This is a simple downward orientation
            pose.orientation.x = 1.0  # Point down
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 0.0
            
            waypoints.append(pose)
            
        return waypoints
    
    def move_to_pose(self, target_pose):
        """Move robot to target pose using MoveIt"""
        self.get_logger().info(f'Planning to position: x={target_pose.position.x:.3f}, '
                             f'y={target_pose.position.y:.3f}, z={target_pose.position.z:.3f}')
        
        # Create goal
        goal = self.create_move_group_goal(target_pose)
        
        # Send goal
        future = self.move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected!')
            return False
            
        self.get_logger().info('Goal accepted, executing...')
        
        # Wait for result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        result = result_future.result()
        if result.result.error_code.val == MoveItErrorCodes.SUCCESS:
            self.get_logger().info('Movement completed successfully!')
            return True
        else:
            self.get_logger().error(f'Movement failed with error code: {result.result.error_code.val}')
            return False
    
    def execute_scanning_path(self):
        """Execute the scanning path using MoveIt"""
        waypoints = self.generate_circular_path()
        
        self.get_logger().info(f'Generated {len(waypoints)} waypoints for scanning')
        self.get_logger().info('Starting scanning sequence...')
        
        success_count = 0
        for i, waypoint in enumerate(waypoints):
            self.get_logger().info(f'Moving to waypoint {i+1}/{len(waypoints)}')
            
            if self.move_to_pose(waypoint):
                success_count += 1
                self.get_logger().info(f'Waypoint {i+1} completed. Pausing for scan...')
                time.sleep(2.0)  # Pause for "scanning"
            else:
                self.get_logger().error(f'Failed to reach waypoint {i+1}')
                break
                
        self.get_logger().info(f'Scanning path completed! {success_count}/{len(waypoints)} waypoints successful')

def main(args=None):
    rclpy.init(args=args)
    
    scanner = UR5eScanningWithMoveIt()
    
    try:
        # Wait for everything to be ready
        time.sleep(3.0)
        
        # Execute the scanning path
        scanner.execute_scanning_path()
        
        # Keep node alive to see final position
        self.get_logger().info('Scanning complete. Press Ctrl+C to exit.')
        rclpy.spin(scanner)
        
    except KeyboardInterrupt:
        scanner.get_logger().info('Scanning interrupted by user')
    
    finally:
        scanner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
