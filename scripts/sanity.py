import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import sys
import time
import threading

class DirectMover(Node):
    def __init__(self):
        super().__init__('ur5_direct_mover')
        
        # 1. Publisher to the Driver (Bypassing MoveIt)
        self.publisher_ = self.create_publisher(
            JointTrajectory, 
            '/scaled_joint_trajectory_controller/joint_trajectory', 
            10
        )
        
        # 2. Subscriber to read start position
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        
        self.current_joints = {}
        self.joint_names = []
        self.received_first_state = False

        self.get_logger().info("Waiting for robot joint states...")

    def joint_state_callback(self, msg):
        self.joint_names = msg.name
        for i, name in enumerate(msg.name):
            self.current_joints[name] = msg.position[i]
        self.received_first_state = True

    def send_move_command(self):
        # Wait for data
        timeout = 0
        while not self.received_first_state:
            time.sleep(0.1)
            timeout += 1
            if timeout > 50:
                self.get_logger().error("❌ Timeout: No joint states received from robot.")
                return

        self.get_logger().info("✅ Robot State Received. Preparing Trajectory.")
        
        # UR5 Standard Joint Names (Order matters!)
        ordered_names = [
            "shoulder_pan_joint", 
            "shoulder_lift_joint", 
            "elbow_joint", 
            "wrist_1_joint", 
            "wrist_2_joint", 
            "wrist_3_joint"
        ]

        # Get current position ordered correctly
        start_positions = []
        for name in ordered_names:
            if name not in self.current_joints:
                self.get_logger().error(f"❌ Missing joint {name} in state msg!")
                return
            start_positions.append(self.current_joints[name])

        # Create Target: Move Base (Shoulder Pan) by +0.2 radians (~11 deg)
        target_positions = list(start_positions) # Copy
        target_positions[0] += 0.2 

        # --- BUILD MESSAGE ---
        msg = JointTrajectory()
        msg.header.frame_id = "base_link"
        msg.joint_names = ordered_names
        
        point = JointTrajectoryPoint()
        point.positions = target_positions
        point.time_from_start.sec = 4  # Take 4 seconds to move (Slow & Safe)
        
        msg.points.append(point)

        self.get_logger().info(f"🚀 Publishing Trajectory to Driver...")
        self.get_logger().info(f"   Moving Base Joint: {start_positions[0]:.2f} -> {target_positions[0]:.2f}")
        
        # Publish multiple times to ensure connection
        for _ in range(5):
            self.publisher_.publish(msg)
            time.sleep(0.1)
            
        self.get_logger().info("✅ Command Published. Watch the robot!")

def main():
    rclpy.init()
    node = DirectMover()
    
    # Spin in background to handle callbacks
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    try:
        node.send_move_command()
        time.sleep(4) # Wait for move to finish
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


