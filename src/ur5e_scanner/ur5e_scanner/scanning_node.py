#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.srv import GetPositionIK
import math
import time

class UR5eScanningPath(Node):
    def __init__(self):
        super().__init__('ur5e_scanning_path')
        
        # Publisher for target poses
        self.pose_publisher = self.create_publisher(PoseStamped, '/move_group/goal', 10)
        
        # Basic scanning parameters - adjust these for your part
        self.scan_center_x = 0.3  # meters from robot base
        self.scan_center_y = 0.0  # meters from robot base
        self.scan_center_z = 0.2  # height above table
        self.scan_radius = 0.15   # radius of circular path
        self.scan_height = 0.1    # how high above part to scan
        self.num_points = 8       # number of scanning positions
        
        self.get_logger().info('UR5e Scanning Path Node initialized')
        
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
            pose = PoseStamped()
            pose.header.frame_id = "base_link"
            pose.header.stamp = self.get_clock().now().to_msg()
            
            # Position
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            
            # Orientation - point camera down and towards center
            # Simple approach: keep same orientation for all points
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.707  # Point down (45 degrees)
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 0.707
            
            waypoints.append(pose)
            
        return waypoints
    
    def execute_scanning_path(self):
        """Execute the scanning path"""
        waypoints = self.generate_circular_path()
        
        self.get_logger().info(f'Generated {len(waypoints)} waypoints for scanning')
        
        for i, waypoint in enumerate(waypoints):
            self.get_logger().info(f'Moving to waypoint {i+1}/{len(waypoints)}')
            self.get_logger().info(f'Position: x={waypoint.pose.position.x:.3f}, '
                                 f'y={waypoint.pose.position.y:.3f}, '
                                 f'z={waypoint.pose.position.z:.3f}')
            
            # Publish the target pose
            self.pose_publisher.publish(waypoint)
            
            # Wait for movement to complete (adjust timing as needed)
            time.sleep(3.0)
            
        self.get_logger().info('Scanning path completed!')

def main(args=None):
    rclpy.init(args=args)
    
    scanner = UR5eScanningPath()
    
    try:
        # Wait a moment for publishers to establish
        time.sleep(2.0)
        
        # Execute the scanning path
        scanner.execute_scanning_path()
        
        # Keep node alive
        rclpy.spin(scanner)
        
    except KeyboardInterrupt:
        scanner.get_logger().info('Scanning interrupted by user')
    
    finally:
        scanner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
