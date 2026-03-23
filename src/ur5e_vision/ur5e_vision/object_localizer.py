"""
object_localizer.py
-------------------
Runs YOLO on the RealSense color stream, looks up depth at each detection
centre from the aligned depth image, back-projects to 3D in the camera
optical frame, then transforms to the robot base frame using TF2.

Requires camera_transform.launch.py (tool0 → camera_link) to be running
alongside the RealSense driver and the UR5e robot driver.

Topics subscribed:
  /camera/camera/color/image_raw                   (sensor_msgs/Image)
  /camera/camera/aligned_depth_to_color/image_raw  (sensor_msgs/Image)
  /camera/camera/color/camera_info                 (sensor_msgs/CameraInfo)

Topics published:
  /detected_object/pose    (geometry_msgs/PoseStamped)  — 3D pose in base frame
  /detected_object/image   (sensor_msgs/Image)          — annotated colour image

Parameters (all overridable via ROS params or launch):
  model_path            : path to YOLO weights  (default: best.pt)
  confidence_threshold  : YOLO confidence cutoff (default: 0.5)
  target_class          : only publish poses for this YOLO class name (default: '' = any class)
  base_frame            : target frame for pose output (default: base_link)
  depth_scale           : metres per depth unit   (default: 0.001  = RealSense mm)
  color_topic           : colour image topic
  depth_topic           : aligned depth topic
  camera_info_topic     : camera info topic
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped

import numpy as np
import cv2

from cv_bridge import CvBridge

import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PointStamped transform support

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ObjectLocalizer(Node):

    def __init__(self):
        super().__init__('object_localizer')

        # ---- parameters ----
        self.declare_parameter('model_path', 'best.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('target_class', '')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('color_topic',
                               '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic',
                               '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic',
                               '/camera/camera/color/camera_info')

        model_path        = self.get_parameter('model_path').value
        self.conf_thresh  = self.get_parameter('confidence_threshold').value
        self.target_class = self.get_parameter('target_class').value.strip()
        self.base_frame   = self.get_parameter('base_frame').value
        self.depth_scale  = self.get_parameter('depth_scale').value
        color_topic       = self.get_parameter('color_topic').value
        depth_topic       = self.get_parameter('depth_topic').value
        info_topic        = self.get_parameter('camera_info_topic').value

        # ---- TF2 ----
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- YOLO ----
        if YOLO_AVAILABLE:
            self.model = YOLO(model_path)
            self.get_logger().info(f'Loaded YOLO model: {model_path}')
            self.get_logger().info(
                f'Available classes: {list(self.model.names.values())}')
            if self.target_class:
                self.get_logger().info(
                    f'Target class filter: "{self.target_class}" '
                    f'(only this class will drive robot motion)')
            else:
                self.get_logger().info(
                    'No target_class filter — highest-confidence detection of any class used')
        else:
            self.model = None
            self.get_logger().warn(
                'ultralytics not installed — detections disabled. '
                'Install with: pip install ultralytics')

        # ---- state ----
        self.bridge       = CvBridge()
        self.camera_info  = None   # filled by _camera_info_cb
        self.depth_image  = None   # filled by _depth_cb
        self.camera_frame = None   # optical frame from CameraInfo header

        # ---- subscribers ----
        self.create_subscription(CameraInfo, info_topic,
                                 self._camera_info_cb, 10)
        self.create_subscription(Image, depth_topic,
                                 self._depth_cb, 10)
        self.create_subscription(Image, color_topic,
                                 self._color_cb, 10)

        # ---- publishers ----
        self.pose_pub  = self.create_publisher(PoseStamped,
                                               '/detected_object/pose', 10)
        self.image_pub = self.create_publisher(Image,
                                               '/detected_object/image', 10)

        self.get_logger().info(
            f'ObjectLocalizer ready. Waiting for camera data on {color_topic}')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _camera_info_cb(self, msg: CameraInfo):
        if self.camera_info is not None:
            return  # only need it once
        self.camera_info  = msg
        self.camera_frame = msg.header.frame_id
        # Intrinsics from K matrix:  [fx 0 cx; 0 fy cy; 0 0 1]
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.get_logger().info(
            f'Camera intrinsics received. Frame: {self.camera_frame} '
            f'fx={self.fx:.1f} fy={self.fy:.1f} cx={self.cx:.1f} cy={self.cy:.1f}')

    def _depth_cb(self, msg: Image):
        # 16-bit depth at half resolution (320x240) — published by realsense_publisher
        # to stay under DDS UDP limits. Intrinsics are scaled accordingly in
        # _pixel_to_base_pose via self._depth_scale_factor.
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='passthrough')
        self.depth_stamp = msg.header.stamp

    def _color_cb(self, msg: Image):
        if self.camera_info is None:
            self.get_logger().warn('Waiting for camera_info…', throttle_duration_sec=5)
            return
        if self.depth_image is None:
            self.get_logger().warn('Waiting for depth image…', throttle_duration_sec=5)
            return
        if self.model is None:
            return

        # Colour is JPEG-encoded (encoding field = 'jpeg') — decode manually
        if msg.encoding == 'jpeg':
            import numpy as np
            jpeg_arr   = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            color_image = cv2.imdecode(jpeg_arr, cv2.IMREAD_COLOR)
        else:
            color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results     = self.model(color_image, conf=self.conf_thresh, verbose=False)
        annotated   = color_image.copy()

        best_conf      = -1.0
        best_pose      = None

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf   = float(box.conf[0])
                label  = self.model.names[int(box.cls[0])]

                # Skip detections that don't match the target class filter
                if self.target_class and label != self.target_class:
                    colour = (128, 128, 128)  # grey = ignored class
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 1)
                    cv2.putText(annotated, f'[ignored] {label} {conf:.2f}',
                                (x1, max(y1 - 8, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1)
                    continue

                # --- 3D localisation ---
                u = (x1 + x2) // 2
                v = (y1 + y2) // 2
                pose_base = self._pixel_to_base_pose(u, v, msg.header.stamp)

                # Track highest-confidence detection with a valid 3D pose
                if pose_base and conf > best_conf:
                    best_conf = conf
                    best_pose = pose_base

                # --- annotation ---
                colour = (0, 255, 0) if pose_base else (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

                if pose_base:
                    p = pose_base.pose.position
                    tag = (f'{label} {conf:.2f} | '
                           f'x={p.x:.3f} y={p.y:.3f} z={p.z:.3f} m')
                else:
                    tag = f'{label} {conf:.2f} | no depth'

                cv2.putText(annotated, tag, (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
                self.get_logger().info(tag)

        # Publish only the best detection so the planner always gets the most
        # confident 3D pose (not whichever box happened to be last in the loop).
        if best_pose is not None:
            self.pose_pub.publish(best_pose)

        self.image_pub.publish(
            self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8'))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pixel_to_base_pose(self, u: int, v: int, stamp) -> PoseStamped | None:
        """
        Back-project pixel (u, v) to a 3D PoseStamped in self.base_frame.

        Steps:
          1. Sample median depth from a 11×11 window around (u, v).
          2. Convert depth units to metres using depth_scale.
          3. Apply pinhole model to get (X, Y, Z) in camera optical frame.
          4. TF2-transform the point to base_link (or configured base_frame).
        """
        # Depth image is 2× subsampled (320×240) relative to the colour image
        # (640×480) where YOLO runs.  Scale pixel coords and intrinsics down.
        ud = u // 2
        vd = v // 2
        fx_d = self.fx / 2.0
        fy_d = self.fy / 2.0
        cx_d = self.cx / 2.0
        cy_d = self.cy / 2.0

        h, w = self.depth_image.shape[:2]
        r = 5  # half-window size (pixels) in depth space
        roi = self.depth_image[
            max(0, vd - r): min(h, vd + r + 1),
            max(0, ud - r): min(w, ud + r + 1),
        ]
        valid = roi[roi > 0]
        if len(valid) == 0:
            self.get_logger().warn(
                f'No valid depth around pixel ({u}, {v})', throttle_duration_sec=2)
            return None

        depth_m = float(np.median(valid)) * self.depth_scale

        # Pinhole back-projection in camera optical frame
        # (optical frame: Z forward, X right, Y down — ROS REP-103)
        x_cam = (ud - cx_d) * depth_m / fx_d
        y_cam = (vd - cy_d) * depth_m / fy_d
        z_cam = depth_m

        # Build PointStamped in the camera optical frame.
        # Use time=0 (latest available TF) rather than the image stamp to avoid
        # "extrapolation into the future" errors when image timestamps are newer
        # than the last joint-state broadcast from the robot driver.
        point_cam = PointStamped()
        point_cam.header.stamp    = Time().to_msg()  # 0 → latest available
        point_cam.header.frame_id = self.camera_frame
        point_cam.point.x = x_cam
        point_cam.point.y = y_cam
        point_cam.point.z = z_cam

        # Transform to base frame
        try:
            point_base = self.tf_buffer.transform(
                point_cam, self.base_frame,
                timeout=Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(f'TF transform failed: {e}',
                                   throttle_duration_sec=2)
            return None

        pose = PoseStamped()
        pose.header             = point_base.header
        pose.pose.position      = point_base.point
        pose.pose.orientation.w = 1.0  # orientation undefined; set identity
        return pose


def main(args=None):
    rclpy.init(args=args)
    node = ObjectLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
