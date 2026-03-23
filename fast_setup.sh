source /opt/ros/humble/setup.bash
source ~/ur5_ws/install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file:///home/crc-b01/ur5_ws/cyclonedds.xml"
export LIBGL_ALWAYS_SOFTWARE=1
