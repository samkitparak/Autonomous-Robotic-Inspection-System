"""
workspace_calibrator.py
-----------------------
Detects an ArUco marker (DICT_4X4_50, ID 0) in the RealSense color stream,
transforms its pose to the robot base frame via TF2, and latches it as the
workspace anchor once N stable detections agree.

Once latched, /workspace_anchor is republished on every frame so downstream
nodes that subscribe after startup don't miss it.

Topics subscribed:
  /camera/camera/color/image_raw       (sensor_msgs/Image)
  /camera/camera/color/camera_info     (sensor_msgs/CameraInfo)

Topics published:
  /workspace_anchor                    (geometry_msgs/PoseStamped) — latched pose in base frame
  /workspace_calibrator/image          (sensor_msgs/Image)         — annotated debug view

Parameters:
  marker_length      : physical marker side length in metres  (default: 0.15)
  marker_id          : ArUco marker ID to track               (default: 0)
  base_frame         : target TF frame for output             (default: base_link)
  stable_frames      : consecutive detections before latching  (default: 5)
  color_topic        : color image topic
  camera_info_topic  : camera info topic
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PoseStamped transform support


# OpenCV 4.7+ removed the old functional API; use ArucoDetector class instead.
def _make_aruco_detector(dictionary):
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


# Object-space corners for a square marker (for solvePnP pose estimation).
# OpenCV 4.7+ removed estimatePoseSingleMarkers — we use solvePnP directly.
def _marker_obj_pts(marker_length: float) -> np.ndarray:
    h = marker_length / 2.0
    return np.array([[-h,  h, 0], [h,  h, 0],
                     [ h, -h, 0], [-h, -h, 0]], dtype=np.float32)


class WorkspaceCalibrator(Node):

    def __init__(self):
        super().__init__('workspace_calibrator')

        # ---- parameters ----
        self.declare_parameter('marker_length',    0.15)
        self.declare_parameter('marker_id',        0)
        self.declare_parameter('base_frame',       'base_link')
        self.declare_parameter('stable_frames',    5)
        self.declare_parameter('color_topic',
                               '/camera/camera/color/image_raw')
        self.declare_parameter('camera_info_topic',
                               '/camera/camera/color/camera_info')

        self.marker_length  = float(self.get_parameter('marker_length').value)
        self.marker_id      = int(self.get_parameter('marker_id').value)
        self.base_frame     = self.get_parameter('base_frame').value
        self._stable_needed = int(self.get_parameter('stable_frames').value)
        color_topic         = self.get_parameter('color_topic').value
        info_topic          = self.get_parameter('camera_info_topic').value

        # ---- ArUco setup ----
        self._aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self._detector     = _make_aruco_detector(self._aruco_dict)
        self._obj_pts      = _marker_obj_pts(self.marker_length)

        # ---- camera state ----
        self.bridge         = CvBridge()
        self._camera_frame  = None
        self._camera_matrix = None
        self._dist_coeffs   = None

        # ---- detection state ----
        self._stable_buf    = []     # PoseStamped list, accumulates before latch
        self._anchor        = None   # latched PoseStamped (None until locked)
        self._anchor_locked = False

        # ---- TF2 ----
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- subscribers ----
        self.create_subscription(CameraInfo, info_topic, self._info_cb, 10)
        self.create_subscription(Image, color_topic, self._image_cb, 10)

        # ---- publishers ----
        self._anchor_pub = self.create_publisher(PoseStamped, '/workspace_anchor', 10)
        self._debug_pub  = self.create_publisher(Image, '/workspace_calibrator/image', 10)

        self.get_logger().info(
            f'WorkspaceCalibrator ready — DICT_4X4_50 ID={self.marker_id} '
            f'size={self.marker_length}m, need {self._stable_needed} stable frames')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _info_cb(self, msg: CameraInfo):
        if self._camera_frame is not None:
            return
        self._camera_frame  = msg.header.frame_id
        self._camera_matrix = np.array(
            [[msg.k[0], 0,        msg.k[2]],
             [0,        msg.k[4], msg.k[5]],
             [0,        0,        1       ]], dtype=np.float64)
        self._dist_coeffs = np.array(msg.d, dtype=np.float64)
        self.get_logger().info(f'Camera intrinsics received. Frame: {self._camera_frame}')

    def _image_cb(self, msg: Image):
        if self._camera_frame is None:
            return

        # Decode (realsense_publisher may send JPEG-encoded images)
        if msg.encoding == 'jpeg':
            jpeg_arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            frame    = cv2.imdecode(jpeg_arr, cv2.IMREAD_COLOR)
        else:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if frame is None:
            return

        annotated = frame.copy()
        found     = self._detect_and_update(annotated)

        self._draw_status(annotated, found)
        self._debug_pub.publish(self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8'))

        # Republish anchor every frame so late subscribers catch it
        if self._anchor_locked:
            self._anchor_pub.publish(self._anchor)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_and_update(self, annotated) -> bool:
        """
        Run ArUco detection. Returns True if the target marker was detected
        this frame and a valid base-frame pose was obtained.
        """
        gray = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None:
            if not self._anchor_locked:
                self._stable_buf.clear()
            return False

        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        flat_ids = ids.flatten().tolist()

        if self.marker_id not in flat_ids:
            if not self._anchor_locked:
                self._stable_buf.clear()
            return False

        idx = flat_ids.index(self.marker_id)
        corner_2d = corners[idx].reshape(4, 2)
        _, rvec, tvec = cv2.solvePnP(
            self._obj_pts, corner_2d,
            self._camera_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        rvec = rvec.flatten()
        tvec = tvec.flatten()

        cv2.drawFrameAxes(
            annotated, self._camera_matrix, self._dist_coeffs,
            rvec, tvec, self.marker_length * 0.5)

        pose_base = self._pose_to_base(tvec, rvec)
        if pose_base is None:
            return False

        if not self._anchor_locked:
            self._stable_buf.append(pose_base)
            if len(self._stable_buf) >= self._stable_needed:
                self._latch(pose_base)
        return True

    def _pose_to_base(self, tvec, rvec) -> PoseStamped | None:
        """
        Convert ArUco tvec/rvec (camera optical frame) to PoseStamped in base_frame.
        Uses the full TF chain: camera_optical → camera_link → tool0 → base_link.
        """
        R, _ = cv2.Rodrigues(rvec)
        qx, qy, qz, qw = self._rot_to_quat(R)

        pose_cam = PoseStamped()
        pose_cam.header.stamp       = Time().to_msg()   # 0 = latest available TF
        pose_cam.header.frame_id    = self._camera_frame
        pose_cam.pose.position.x    = float(tvec[0])
        pose_cam.pose.position.y    = float(tvec[1])
        pose_cam.pose.position.z    = float(tvec[2])
        pose_cam.pose.orientation.x = float(qx)
        pose_cam.pose.orientation.y = float(qy)
        pose_cam.pose.orientation.z = float(qz)
        pose_cam.pose.orientation.w = float(qw)

        try:
            return self.tf_buffer.transform(
                pose_cam, self.base_frame, timeout=Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(
                f'TF transform failed: {e}', throttle_duration_sec=2)
            return None

    def _latch(self, pose: PoseStamped):
        self._anchor        = pose
        self._anchor_locked = True
        p = pose.pose.position
        self.get_logger().info(
            f'Workspace anchor LATCHED: '
            f'({p.x:.4f}, {p.y:.4f}, {p.z:.4f}) m in {self.base_frame}')
        self._anchor_pub.publish(self._anchor)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rot_to_quat(R) -> tuple[float, float, float, float]:
        """Rotation matrix → (qx, qy, qz, qw). Shepperd's method."""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return float(x), float(y), float(z), float(w)

    def _draw_status(self, frame, found: bool):
        if self._anchor_locked:
            p = self._anchor.pose.position
            cv2.putText(frame,
                        f'ANCHOR LOCKED  ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) m',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif found:
            n = len(self._stable_buf)
            cv2.putText(frame,
                        f'Accumulating {n}/{self._stable_needed}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        else:
            cv2.putText(frame,
                        f'Looking for ArUco ID {self.marker_id}...',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def main(args=None):
    rclpy.init(args=args)
    node = WorkspaceCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
