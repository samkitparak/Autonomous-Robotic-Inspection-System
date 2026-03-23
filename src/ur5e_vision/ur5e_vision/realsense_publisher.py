"""
realsense_publisher.py
----------------------
Lightweight RealSense → ROS2 bridge using pyrealsense2 SDK directly.
Replaces the realsense2_camera ROS2 package (not installed on this machine).

Published topics (matching what object_localizer expects):
  /camera/camera/color/image_raw                   (sensor_msgs/Image, bgr8)
  /camera/camera/aligned_depth_to_color/image_raw  (sensor_msgs/Image, 16UC1, mm)
  /camera/camera/color/camera_info                 (sensor_msgs/CameraInfo)

The frame_id on all messages is 'camera_link' — this is the frame that
camera_transform.launch.py connects to tool0.

Parameters:
  fps            : capture frame rate (default 30)
  color_width    : colour resolution width  (default 640)
  color_height   : colour resolution height (default 480)
  device_serial  : optional RealSense serial number (default: first found)
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo

import numpy as np
import cv2
import threading

try:
    import pyrealsense2 as rs
    RS_AVAILABLE = True
except ImportError:
    RS_AVAILABLE = False


CAMERA_FRAME = 'camera_link'


class RealSensePublisher(Node):

    def __init__(self):
        super().__init__('realsense_publisher')

        self.declare_parameter('fps',           30)
        self.declare_parameter('color_width',   640)
        self.declare_parameter('color_height',  480)
        self.declare_parameter('device_serial', '')

        fps           = self.get_parameter('fps').value
        width         = self.get_parameter('color_width').value
        height        = self.get_parameter('color_height').value
        serial        = self.get_parameter('device_serial').value

        if not RS_AVAILABLE:
            self.get_logger().fatal(
                'pyrealsense2 not available. Install with: pip install pyrealsense2')
            return

        # ---- RealSense pipeline ----
        self._pipe    = rs.pipeline()
        cfg           = rs.config()

        if serial:
            cfg.enable_device(serial)

        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8,  fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16,   fps)

        # align depth to colour frame
        self._align = rs.align(rs.stream.color)

        profile = self._pipe.start(cfg)

        # Extract intrinsics and depth scale
        color_profile = profile.get_stream(rs.stream.color) \
                               .as_video_stream_profile()
        intr          = color_profile.get_intrinsics()

        depth_sensor       = profile.get_device().first_depth_sensor()
        self._depth_scale  = depth_sensor.get_depth_scale()   # metres per count

        self.get_logger().info(
            f'RealSense started: {width}x{height} @ {fps} fps  '
            f'depth_scale={self._depth_scale:.6f} m/count')
        self.get_logger().info(
            f'Intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} '
            f'cx={intr.ppx:.1f} cy={intr.ppy:.1f}')

        # ---- publishers ----
        self._color_pub  = self.create_publisher(
            Image, '/camera/camera/color/image_raw', 10)
        self._depth_pub  = self.create_publisher(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', 10)
        self._info_pub   = self.create_publisher(
            CameraInfo, '/camera/camera/color/camera_info', 10)

        # Pre-build CameraInfo (intrinsics don't change)
        self._camera_info = self._build_camera_info(intr, width, height)

        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        self.get_logger().info(
            'RealSensePublisher ready. Publishing to /camera/camera/...')

    # ------------------------------------------------------------------

    def _capture_loop(self):
        """Blocking capture loop — runs in its own thread so the timer isn't needed."""
        while self._running:
            try:
                frames = self._pipe.wait_for_frames(timeout_ms=1000)
            except Exception as e:
                self.get_logger().warn(f'Frame timeout: {e}', throttle_duration_sec=5)
                continue
            self._process_frames(frames)

    def _capture_cb(self):  # kept for API compatibility but no longer used
        pass

    def _process_frames(self, frames):

        aligned  = self._align.process(frames)
        color_f = aligned.get_color_frame()
        depth_f = aligned.get_depth_frame()

        if not color_f or not depth_f:
            return

        now = self.get_clock().now().to_msg()

        # Colour image — JPEG-encode before publishing to keep message well
        # under the UDP fragment limit (~921 KB raw → ~50 KB JPEG).
        color_arr = np.asanyarray(color_f.get_data())
        _, jpeg_buf = cv2.imencode(
            '.jpg', color_arr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpeg_arr = np.frombuffer(jpeg_buf, dtype=np.uint8)
        color_msg = self._arr_to_imgmsg(jpeg_arr, 'jpeg', now)
        self._color_pub.publish(color_msg)

        # Depth image — publish at half resolution to keep under UDP limit.
        # (640×480 16UC1 = 614 KB → 320×240 = 153 KB, manageable with frags)
        depth_arr = np.asanyarray(depth_f.get_data())
        depth_small = depth_arr[::2, ::2]          # simple 2× subsample
        depth_msg = self._arr_to_imgmsg(depth_small, '16UC1', now)
        self._depth_pub.publish(depth_msg)

        # CameraInfo
        self._camera_info.header.stamp = now
        self._info_pub.publish(self._camera_info)

    # ------------------------------------------------------------------

    @staticmethod
    def _arr_to_imgmsg(arr: np.ndarray, encoding: str, stamp) -> Image:
        msg                 = Image()
        msg.header.stamp    = stamp
        msg.header.frame_id = CAMERA_FRAME
        msg.encoding        = encoding
        msg.is_bigendian    = False

        if encoding == 'jpeg':
            # arr is already a 1-D byte buffer from imencode
            msg.height = 1
            msg.width  = arr.shape[0]
            msg.step   = arr.shape[0]
        elif encoding == '16UC1':
            msg.height = arr.shape[0]
            msg.width  = arr.shape[1]
            msg.step   = arr.shape[1] * 2
        else:
            msg.height = arr.shape[0]
            msg.width  = arr.shape[1]
            msg.step   = arr.shape[1] * (arr.nbytes // arr.size)

        msg.data = arr.tobytes()
        return msg

    @staticmethod
    def _build_camera_info(intr, width: int, height: int) -> CameraInfo:
        info                = CameraInfo()
        info.header.frame_id = CAMERA_FRAME
        info.width          = width
        info.height         = height

        # K matrix (row-major, 3x3)
        info.k = [
            intr.fx, 0.0,     intr.ppx,
            0.0,     intr.fy, intr.ppy,
            0.0,     0.0,     1.0,
        ]
        # Distortion (RealSense provides Brown-Conrady coefficients)
        info.distortion_model = 'plumb_bob'
        info.d = list(intr.coeffs)

        # Projection matrix P = [K | 0]
        info.p = [
            intr.fx, 0.0,     intr.ppx, 0.0,
            0.0,     intr.fy, intr.ppy, 0.0,
            0.0,     0.0,     1.0,      0.0,
        ]
        return info

    def destroy_node(self):
        self._running = False
        if RS_AVAILABLE:
            try:
                self._pipe.stop()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealSensePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
