"""
data_collector.py
-----------------
Autonomous data collection for VLM fine-tuning.

Moves the robot to random viewpoints around the detected object, captures a
raw camera image + joint state at each position, and saves everything to disk.

The dataset is the raw ingredient for fine-tuning Qwen2-VL (or similar) to
reliably read labels in YOUR specific workspace, lighting, and camera rig.

Usage:
  ros2 run ur5e_vision data_collector --ros-args \\
      -p num_viewpoints:=80 \\
      -p output_dir:=/home/crc-b01/robot_dataset

After collection:
  1. Open <output_dir>/label.txt and type the label text (once for all images).
  2. Run:  python3 build_dataset.py <output_dir>
  This creates training_data.jsonl ready for Unsloth/Qwen2-VL fine-tuning.

Parameters:
  num_viewpoints   : total images to collect (default 80)
  radius_min       : closest standoff from object in metres (default 0.15)
  radius_max       : furthest standoff in metres (default 0.40)
  height_jitter    : random ±offset from optimal camera height (default 0.07)
  move_speed       : MoveIt2 velocity scaling 0-1 (default 0.20)
  output_dir       : where to save images + metadata
"""

import json
import math
import os
import random
import threading
import time
from dataclasses import asdict
from datetime import datetime

import cv2
import numpy as np
import rclpy
import rclpy.executors
from rclpy.node import Node

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, JointState

from ur5e_vision.robot_mover import RobotMover
from ur5e_vision.viewpoint_generator import (
    CAMERA_OFFSET_M,
    Viewpoint,
    _TAN30,
)


class DataCollector(Node):

    def __init__(self):
        super().__init__('data_collector')

        # ---- parameters ----
        self.declare_parameter('num_viewpoints',  80)
        self.declare_parameter('radius_min',      0.15)
        self.declare_parameter('radius_max',      0.40)
        self.declare_parameter('height_jitter',   0.07)
        self.declare_parameter('move_speed',      0.20)
        self.declare_parameter('output_dir',
                               os.path.expanduser('~/robot_dataset'))

        self._n          = self.get_parameter('num_viewpoints').value
        self._r_min      = self.get_parameter('radius_min').value
        self._r_max      = self.get_parameter('radius_max').value
        self._h_jitter   = self.get_parameter('height_jitter').value
        speed            = self.get_parameter('move_speed').value
        base_dir         = self.get_parameter('output_dir').value

        # ---- output directory ----
        session = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._out  = os.path.join(base_dir, session)
        self._imgs = os.path.join(self._out, 'images')
        os.makedirs(self._imgs, exist_ok=True)

        # ---- state ----
        self._bridge       = CvBridge()
        self._latest_img   = None
        self._img_lock     = threading.Lock()
        self._obj_pose     = None
        self._obj_lock     = threading.Lock()
        self._metadata     = []

        # ---- robot mover ----
        self._mover = RobotMover(self, speed=speed)

        # ---- subscribers ----
        self.create_subscription(
            PoseStamped, '/detected_object/pose', self._pose_cb, 10)
        self.create_subscription(
            Image, '/camera/camera/color/image_raw', self._img_cb, 10)

        # ---- kick off collection in background ----
        t = threading.Thread(target=self._collect_loop, daemon=True)
        t.start()

        self.get_logger().info(
            f'DataCollector ready — waiting for object detection…\n'
            f'Output: {self._out}')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _pose_cb(self, msg: PoseStamped):
        with self._obj_lock:
            self._obj_pose = msg

    def _img_cb(self, msg: Image):
        # Decode JPEG-encoded raw image from realsense_publisher
        if msg.encoding == 'jpeg':
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self._img_lock:
            self._latest_img = img

    # ------------------------------------------------------------------
    # Collection logic
    # ------------------------------------------------------------------

    def _collect_loop(self):
        # 1. Wait for first object detection
        self.get_logger().info('Waiting for YOLO to detect the object…')
        while rclpy.ok():
            with self._obj_lock:
                pose = self._obj_pose
            if pose is not None:
                break
            time.sleep(0.2)

        obj_x = pose.pose.position.x
        obj_y = pose.pose.position.y
        obj_z = pose.pose.position.z
        self.get_logger().info(
            f'Object detected at ({obj_x:.3f}, {obj_y:.3f}, {obj_z:.3f}) m  '
            f'— generating {self._n} random viewpoints')

        # 2. Generate random viewpoints
        viewpoints = self._random_viewpoints(obj_x, obj_y, obj_z)
        random.shuffle(viewpoints)   # vary motion order → less repetitive data

        # 3. Visit each viewpoint
        success = 0
        for i, (vp, meta) in enumerate(viewpoints):
            self.get_logger().info(
                f'[{i+1}/{self._n}]  r={meta["radius_m"]:.2f} m  '
                f'θ={math.degrees(meta["angle_rad"]):.0f}°  '
                f'z={vp.z:.3f} m')

            ok, msg = self._mover.move_to_viewpoint(vp)
            if not ok:
                self.get_logger().warn(f'  Move failed: {msg} — skipping')
                continue

            # Brief settle — let image buffer refresh after motion stops
            time.sleep(0.4)

            # Grab image + joint state
            with self._img_lock:
                img = self._latest_img.copy() if self._latest_img is not None else None
            js = self._mover._joint_state

            if img is None:
                self.get_logger().warn('  No image available — skipping')
                continue

            # Save image
            fname = f'img_{success + 1:04d}.jpg'
            fpath = os.path.join(self._imgs, fname)
            cv2.imwrite(fpath, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Save metadata record
            record = {
                'image':     os.path.join('images', fname),
                'obj_pos':   {'x': obj_x, 'y': obj_y, 'z': obj_z},
                'ee_pos':    {'x': vp.x,  'y': vp.y,  'z': vp.z},
                **meta,
            }
            if js is not None:
                joints = dict(zip(js.name, [round(p, 5) for p in js.position]))
                record['joints'] = {k: joints[k] for k in self._mover.ARM_JOINTS
                                    if k in joints}
            self._metadata.append(record)
            success += 1

        # 4. Save metadata + write label placeholder
        meta_path  = os.path.join(self._out, 'metadata.jsonl')
        label_path = os.path.join(self._out, 'label.txt')
        build_path = os.path.join(self._out, 'build_dataset.py')

        with open(meta_path, 'w') as f:
            for r in self._metadata:
                f.write(json.dumps(r) + '\n')

        with open(label_path, 'w') as f:
            f.write('# Type the label text from the tool below (replace this line).\n')
            f.write('# Include ALL readable text: brand, model, watts, RPM, voltage, etc.\n')
            f.write('BRAND MODEL-NUMBER, 750W, 11000RPM, 220-240V\n')

        # Write the finalization script alongside the data
        self._write_build_script(build_path)

        self.get_logger().info(
            f'\n{"="*60}\n'
            f'Collection done: {success}/{self._n} images saved\n'
            f'Output: {self._out}\n\n'
            f'Next steps:\n'
            f'  1. Edit  {label_path}\n'
            f'     → Replace the placeholder with the ACTUAL label text\n\n'
            f'  2. Run   python3 {build_path}\n'
            f'     → Creates training_data.jsonl ready for Unsloth\n'
            f'{"="*60}')

    def _random_viewpoints(
        self, obj_x: float, obj_y: float, obj_z: float
    ) -> list[tuple[Viewpoint, dict]]:
        """
        Sample N random viewpoints in a hemisphere above the object.

        Each viewpoint uses the same height formula as generate_orbit so the
        camera always points at the object centre. A small random height jitter
        gives elevation variation without breaking the camera geometry too much.
        """
        results = []
        for _ in range(self._n):
            r     = random.uniform(self._r_min, self._r_max)
            theta = random.uniform(0.0, 2 * math.pi)

            # Optimal height: camera ray passes through object centre
            h = (r + CAMERA_OFFSET_M) * _TAN30
            # Jitter: sample from a normal distribution, clamped
            h += random.gauss(0.0, self._h_jitter / 2)
            h  = max(h, 0.10)   # never crash into the table

            ex = obj_x + r * math.cos(theta)
            ey = obj_y + r * math.sin(theta)
            ez = obj_z + h

            vp = Viewpoint(x=ex, y=ey, z=ez, yaw=theta)
            meta = {
                'radius_m':  round(r, 4),
                'angle_rad': round(theta, 4),
                'height_m':  round(h, 4),
            }
            results.append((vp, meta))
        return results

    @staticmethod
    def _write_build_script(path: str):
        """Write a self-contained script to turn the dataset into training JSONL."""
        script = '''\
#!/usr/bin/env python3
"""
build_dataset.py  — run once after data collection.

Usage:
  python3 build_dataset.py <session_dir>

Reads:
  <session_dir>/metadata.jsonl
  <session_dir>/label.txt

Writes:
  <session_dir>/training_data.jsonl   ← feed this to Unsloth

Format is Qwen2-VL / Unsloth chat format with a single vision+text user turn.
"""
import json
import os
import sys

def main():
    if len(sys.argv) < 2:
        session = os.path.dirname(os.path.abspath(__file__))
    else:
        session = sys.argv[1]

    label_path = os.path.join(session, 'label.txt')
    meta_path  = os.path.join(session, 'metadata.jsonl')
    out_path   = os.path.join(session, 'training_data.jsonl')

    # Read label text (skip comment lines)
    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    if not lines:
        print('ERROR: label.txt is empty. Fill in the label text first.')
        sys.exit(1)
    label_text = ' '.join(lines)
    print(f'Label text: {label_text!r}')

    # Read metadata
    records = []
    with open(meta_path) as f:
        for line in f:
            records.append(json.loads(line))

    # Build training JSONL
    written = 0
    with open(out_path, 'w') as f:
        for rec in records:
            img_path = os.path.join(session, rec['image'])
            if not os.path.exists(img_path):
                print(f'  Missing: {img_path} — skipping')
                continue

            entry = {
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image', 'image': img_path},
                            {'type': 'text',
                             'text': (
                                 'This is a photo of an industrial power tool taken '
                                 'by a wrist-mounted camera on a robot arm. '
                                 'What text can you read on the tool label? '
                                 'Include brand, model number, power rating, speed, '
                                 'voltage, and any other text.'
                             )},
                        ],
                    },
                    {
                        'role': 'assistant',
                        'content': label_text,
                    },
                ]
            }
            f.write(json.dumps(entry) + '\\n')
            written += 1

    print(f'Wrote {written} training examples → {out_path}')
    print()
    print('To fine-tune on the RTX PC:')
    print('  1. Copy the session dir to the RTX PC')
    print('  2. Run the Unsloth fine-tuning script with training_data.jsonl')

if __name__ == '__main__':
    main()
'''
        with open(path, 'w') as f:
            f.write(script)
        os.chmod(path, 0o755)


def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
