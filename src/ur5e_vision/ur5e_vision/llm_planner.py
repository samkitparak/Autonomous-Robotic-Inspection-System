"""
llm_planner.py
--------------
LLM-driven robot planning loop for label reading on an angle grinder.

Flow:
  1.  object_localizer publishes /detected_object/pose  (first detection triggers start)
  2.  LLM receives: current camera image + object 3D position + EE position
  3.  LLM responds with a tool call  →  robot executes the move
  4.  Robot settles, new image captured  →  repeat from step 2
  5.  LLM calls report_label()  →  loop ends, state returns to IDLE

LLM tools available:
  move_to_viewpoint(x, y, z)                          — absolute position in base frame
  adjust_view(dx, dy, dz)                             — relative offset from current position
  scan_orbit(radius, num_points, start_angle_deg,
             sweep_deg, height)                       — orbit around detected object
  report_label(text, confidence)                      — terminal: label has been read

Topics subscribed:
  /detected_object/pose   (geometry_msgs/PoseStamped)
  /detected_object/image  (sensor_msgs/Image)

Parameters (all overridable via ROS params or vision_params.yaml):
  api_key    : OpenRouter API key (or set OPENROUTER_API_KEY env var)
  model      : LLM model string (default: google/gemini-2.5-flash)
  base_url   : OpenRouter base URL
  max_steps  : safety limit on LLM iterations (default: 20)
  move_speed : MoveIt2 velocity scaling 0–1 (default: 0.15)
"""

import base64
import json
import os
import threading
import time
from enum import Enum, auto

import cv2
import rclpy
import rclpy.executors
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from openai import OpenAI
from sensor_msgs.msg import Image

from ur5e_vision.robot_mover import RobotMover
from ur5e_vision.viewpoint_generator import (
    generate_orbit,
    coverage_angles_for_label,
)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class State(Enum):
    IDLE     = auto()   # waiting for first YOLO detection
    PLANNING = auto()   # LLM is reasoning / we are waiting for a tool call
    MOVING   = auto()   # robot executing a motion
    DONE     = auto()   # label reported; ready to idle again


# ---------------------------------------------------------------------------
# LLM tool definitions  (OpenAI tool-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'move_to_viewpoint',
            'description': (
                'Move the camera to an absolute position (x, y, z) in the robot '
                'base frame.  The camera is offset 17 cm from the end-effector and '
                'angled 60° toward the center, so the image will shift accordingly. '
                'Use this to reposition around the object.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'x': {
                        'type': 'number',
                        'description': 'X metres in robot base frame. '
                                       'Positive = away from robot body.',
                    },
                    'y': {
                        'type': 'number',
                        'description': 'Y metres. Positive = robot left side.',
                    },
                    'z': {
                        'type': 'number',
                        'description': 'Height in metres above table. '
                                       'Minimum safe value is 0.10 m.',
                    },
                },
                'required': ['x', 'y', 'z'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'adjust_view',
            'description': (
                'Make a small relative movement from the current end-effector '
                'position.  Use for fine adjustments when already near the label.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'dx': {
                        'type': 'number',
                        'description': 'Offset in X (metres). Positive = forward.',
                    },
                    'dy': {
                        'type': 'number',
                        'description': 'Offset in Y (metres). Positive = left.',
                    },
                    'dz': {
                        'type': 'number',
                        'description': 'Offset in Z (metres). Positive = up.',
                    },
                },
                'required': ['dx', 'dy', 'dz'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'scan_orbit',
            'description': (
                'Execute a smooth orbit of N viewpoints around the detected object. '
                'The system automatically computes the correct EE height and orientation '
                'so the offset camera looks at the object from each angle. '
                'Use this to systematically search all faces of the object for the label. '
                'A fresh image is returned after the orbit completes.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'radius': {
                        'type': 'number',
                        'description': (
                            'Horizontal standoff from the object centre (metres). '
                            'Recommended: 0.20–0.35 m. Smaller = closer / larger in frame.'
                        ),
                    },
                    'num_points': {
                        'type': 'integer',
                        'description': 'Number of waypoints (4–12). More = thorough, slower.',
                    },
                    'start_angle_deg': {
                        'type': 'number',
                        'description': (
                            'Starting angle of the orbit in degrees from world +X axis. '
                            'Omit or set to -1 to start from the current EE position '
                            '(avoids a large repositioning move).'
                        ),
                    },
                    'sweep_deg': {
                        'type': 'number',
                        'description': (
                            'Angular span of the orbit in degrees. '
                            '360 = full circle, 180 = half circle. Default 360.'
                        ),
                    },
                    'height': {
                        'type': 'number',
                        'description': (
                            'EE height above the object in metres. '
                            'Omit to use the optimal value computed from the camera geometry '
                            '(≈ 0.21 m for radius=0.20, ≈ 0.24 m for radius=0.25).'
                        ),
                    },
                },
                'required': ['radius', 'num_points'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'report_label',
            'description': (
                'Report the text found on the angle grinder label. '
                'Only call this when the label text is clearly legible in the image.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'text': {
                        'type': 'string',
                        'description': 'The full text read from the label.',
                    },
                    'confidence': {
                        'type': 'number',
                        'description': 'Reading confidence from 0.0 to 1.0.',
                    },
                },
                'required': ['text', 'confidence'],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI controller for a UR5e robot arm with an Intel RealSense camera.

Goal: Read the text label on the angle grinder on the table.

Strategy (follow in order):
1. Call scan_orbit (radius=0.20, num_points=6) — the system will automatically orbit the
   object, capture an image at every angle, then send ALL images to you at once.
   You will then identify which image shows the label and report the text.

2. If the first orbit doesn't show legible text, call scan_orbit again with radius=0.12
   (closer) to get a tighter view.

3. Only use move_to_viewpoint or adjust_view if you need a very specific angle that
   the orbit didn't cover.

4. Call report_label with ALL text you can read (brand, model, watts, RPM, etc.).

Notes:
- Leave the height parameter unset — it is computed automatically.
- The scan_orbit tool handles everything: movement, image capture, and analysis.
- Safe Z minimum is enforced automatically."""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class LLMPlanner(Node):

    def __init__(self):
        super().__init__('llm_planner')

        # ---- parameters ----
        self.declare_parameter('api_key',   '')
        self.declare_parameter('model',     'google/gemini-2.5-flash')
        self.declare_parameter('base_url',  'https://openrouter.ai/api/v1')
        self.declare_parameter('max_steps', 20)
        self.declare_parameter('move_speed', 0.15)

        api_key  = (self.get_parameter('api_key').value or
                    os.environ.get('OPENROUTER_API_KEY', '') or
                    'no-key')   # ollama accepts any non-empty string
        self.model     = self.get_parameter('model').value
        base_url       = self.get_parameter('base_url').value
        self.max_steps = self.get_parameter('max_steps').value
        speed          = self.get_parameter('move_speed').value

        # ---- LLM client ----
        self.llm = OpenAI(api_key=api_key, base_url=base_url)

        # ---- state ----
        self._state       = State.IDLE
        self._state_lock  = threading.Lock()
        self._latest_img  = None          # cv2 BGR
        self._latest_pose = None          # PoseStamped in base frame
        self._img_event   = threading.Event()   # fires when a new image arrives
        self._messages: list = []

        # ---- ROS ----
        self.bridge  = CvBridge()
        cb_group = ReentrantCallbackGroup()

        self.create_subscription(
            Image, '/detected_object/image',
            self._image_cb, 10, callback_group=cb_group)
        self.create_subscription(
            PoseStamped, '/detected_object/pose',
            self._pose_cb, 10, callback_group=cb_group)

        # ---- motion ----
        self.mover = RobotMover(self, speed=speed)

        self.get_logger().info(
            f'LLMPlanner ready. Model: {self.model}. '
            f'Waiting for object detection on /detected_object/pose …')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _image_cb(self, msg: Image):
        self._latest_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self._img_event.set()

    def _pose_cb(self, msg: PoseStamped):
        with self._state_lock:
            self._latest_pose = msg
            if self._state == State.IDLE:
                p = msg.pose.position
                self.get_logger().info(
                    f'Object detected at '
                    f'x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}. '
                    f'Starting LLM planning loop.')
                self._state = State.PLANNING
                threading.Thread(
                    target=self._planning_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # Planning loop  (background thread)
    # ------------------------------------------------------------------

    def _planning_loop(self):
        self._messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        step = 0

        while step < self.max_steps:
            step += 1
            self.get_logger().info(f'--- LLM step {step}/{self.max_steps} ---')

            # Wait for a fresh camera frame (up to 5 s)
            self._img_event.clear()
            if not self._img_event.wait(timeout=5.0):
                self.get_logger().warn('Timed out waiting for camera image.')
                continue

            # Build user message with image + state
            self._messages.append(self._build_user_message())

            # Call LLM
            try:
                response = self.llm.chat.completions.create(
                    model=self.model,
                    messages=self._messages,
                    tools=TOOLS,
                    tool_choice='auto',
                )
            except Exception as e:
                self.get_logger().error(f'LLM API error: {e}')
                time.sleep(2.0)
                continue

            assistant_msg = response.choices[0].message
            # Store assistant message (serialise to plain dict)
            self._messages.append(
                assistant_msg.model_dump(exclude_unset=True, exclude_none=True))

            if not assistant_msg.tool_calls:
                # LLM gave text without a tool call — log and nudge it
                self.get_logger().info(f'LLM text (no tool): {assistant_msg.content}')
                self._messages.append({
                    'role': 'user',
                    'content': (
                        'Please respond with a tool call — either move the robot '
                        'to a better viewpoint or report the label if you can read it.'
                    ),
                })
                continue

            # Dispatch every tool call in the response
            terminal = False
            for tc in assistant_msg.tool_calls:
                result_text, is_done = self._dispatch(tc)
                self.get_logger().info(
                    f'  tool={tc.function.name}  result={result_text}')
                self._messages.append({
                    'role': 'tool',
                    'tool_call_id': tc.id,
                    'content': result_text,
                })
                if is_done:
                    terminal = True
                    break

            if terminal:
                break

            # Brief settle pause before next LLM step
            time.sleep(0.5)

        with self._state_lock:
            if self._state != State.DONE:
                self.get_logger().warn(
                    f'Max steps ({self.max_steps}) reached without reading label.')
            self._state = State.IDLE
            self.get_logger().info('Planning loop ended. Back to IDLE.')

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, tool_call) -> tuple[str, bool]:
        """Execute a tool call. Returns (result_text, is_terminal)."""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return f'Error parsing arguments: {e}', False

        if name == 'move_to_viewpoint':
            return self._exec_move(
                args.get('x', 0.0),
                args.get('y', 0.0),
                args.get('z', 0.15),
            ), False

        if name == 'adjust_view':
            return self._exec_adjust(
                args.get('dx', 0.0),
                args.get('dy', 0.0),
                args.get('dz', 0.0),
            ), False

        if name == 'scan_orbit':
            # Returns (text, is_terminal) — terminal when label is found in images
            return self._exec_scan_orbit(
                radius          = float(args.get('radius', 0.25)),
                num_points      = int(args.get('num_points', 8)),
                start_angle_deg = float(args.get('start_angle_deg', -1.0)),
                sweep_deg       = float(args.get('sweep_deg', 360.0)),
                height          = args.get('height'),   # None = auto
            )

        if name == 'report_label':
            return self._exec_report(
                args.get('text', ''),
                args.get('confidence', 0.0),
            ), True

        return f'Unknown tool: {name}', False

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _exec_move(self, x: float, y: float, z: float) -> str:
        # Safety clamps
        z = max(z, 0.10)

        # Cap maximum step distance to 0.25 m
        note = ''
        current = self.mover.get_current_ee_pose()
        if current:
            dx = x - current['x']
            dy = y - current['y']
            dz = z - current['z']
            dist = (dx**2 + dy**2 + dz**2) ** 0.5
            max_step = 0.25
            if dist > max_step:
                scale = max_step / dist
                x = current['x'] + dx * scale
                y = current['y'] + dy * scale
                z = current['z'] + dz * scale
                note = (f' (step clamped from {dist:.2f} m to {max_step} m; '
                        f'call again to continue toward goal)')

        with self._state_lock:
            self._state = State.MOVING

        success, msg = self.mover.move_to_pose(x, y, z)

        with self._state_lock:
            self._state = State.PLANNING

        if success:
            # Wait for camera to update after the move
            self._img_event.clear()
            self._img_event.wait(timeout=4.0)
            return (f'Moved to ({x:.3f}, {y:.3f}, {z:.3f}) m.{note} '
                    f'New image captured.')
        else:
            return (f'Move to ({x:.3f}, {y:.3f}, {z:.3f}) failed: {msg}. '
                    f'Try a different position.')

    def _exec_adjust(self, dx: float, dy: float, dz: float) -> str:
        current = self.mover.get_current_ee_pose()
        if not current:
            return ('Cannot read current pose. '
                    'Use move_to_viewpoint with absolute coordinates instead.')
        return self._exec_move(
            current['x'] + dx,
            current['y'] + dy,
            current['z'] + dz,
        )

    def _exec_scan_orbit(
        self,
        radius:          float,
        num_points:      int,
        start_angle_deg: float,
        sweep_deg:       float,
        height,          # float or None
    ) -> tuple[str, bool]:
        import math
        import os
        from datetime import datetime

        pose = self._latest_pose
        if pose is None:
            return 'Object not detected — cannot compute orbit. Try again after detection.', False

        obj_x = pose.pose.position.x
        obj_y = pose.pose.position.y
        obj_z = pose.pose.position.z

        # Safety clamps on radius
        radius = max(0.10, min(radius, 0.45))

        # Start angle: -1 means "use current EE position" to avoid repositioning
        if start_angle_deg < -0.5:
            ee = self.mover.get_current_ee_pose()
            if ee:
                start_rad = coverage_angles_for_label(
                    ee['x'], ee['y'], obj_x, obj_y)
            else:
                start_rad = 0.0
        else:
            start_rad = math.radians(start_angle_deg)

        sweep_rad = math.radians(min(sweep_deg, 360.0))

        viewpoints = generate_orbit(
            obj_x       = obj_x,
            obj_y       = obj_y,
            obj_z       = obj_z,
            radius      = radius,
            height      = float(height) if height is not None else None,
            num_points  = num_points,
            start_angle = start_rad,
            sweep       = sweep_rad,
        )

        self.get_logger().info(
            f'Executing orbit: {num_points} waypoints, r={radius:.2f} m, '
            f'sweep={sweep_deg:.0f}°, start={math.degrees(start_rad):.0f}°')

        # Save images to disk so we can inspect them later
        ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'/tmp/ur5e_orbit/{ts}'
        os.makedirs(save_dir, exist_ok=True)

        with self._state_lock:
            self._state = State.MOVING

        # Collect an image at every successfully-reached waypoint
        orbit_images = []   # list of (angle_deg, cv2 BGR image)

        def _on_waypoint(idx, vp):
            self._img_event.clear()
            self._img_event.wait(timeout=3.0)
            img = self._latest_img
            if img is not None:
                img_copy  = img.copy()
                angle_deg = math.degrees(vp.yaw)
                orbit_images.append((angle_deg, img_copy))
                path = os.path.join(save_dir, f'wp{idx+1:02d}_{angle_deg:.0f}deg.jpg')
                cv2.imwrite(path, img_copy)
                self.get_logger().info(f'  Saved: {path}')

        succeeded, total = self.mover.execute_orbit(
            viewpoints, on_waypoint_reached=_on_waypoint)

        with self._state_lock:
            self._state = State.PLANNING

        # Also grab the final resting image
        self._img_event.clear()
        self._img_event.wait(timeout=4.0)
        if self._latest_img is not None:
            img_copy  = self._latest_img.copy()
            angle_deg = math.degrees(viewpoints[-1].yaw) if viewpoints else 0.0
            orbit_images.append((angle_deg, img_copy))
            cv2.imwrite(os.path.join(save_dir, 'final.jpg'), img_copy)

        self.get_logger().info(
            f'Orbit done: {succeeded}/{total} waypoints, '
            f'{len(orbit_images)} images → {save_dir}')

        if not orbit_images:
            return (f'Orbit complete: {succeeded}/{total} waypoints '
                    f'but no images captured.'), False

        # Send ALL images to the LLM at once — much more reliable than
        # navigating one image at a time.
        return self._analyze_orbit_images(orbit_images, succeeded, total)

    def _analyze_orbit_images(
        self,
        orbit_images: list,   # [(angle_deg, cv2_bgr), ...]
        succeeded:    int,
        total:        int,
    ) -> tuple[str, bool]:
        """Send all orbit images to the LLM simultaneously and ask it to read the label."""
        content = [
            {
                'type': 'text',
                'text': (
                    f'I orbited a robot arm around an angle grinder and captured '
                    f'{len(orbit_images)} images from different angles '
                    f'({succeeded}/{total} waypoints reached). '
                    f'Please examine EVERY image carefully.\n\n'
                    f'Look for: brand name, model number, power rating (watts), '
                    f'speed (RPM), any certification marks, or any other text on the tool.\n\n'
                    f'The label could be on ANY face — check all images.'
                ),
            }
        ]

        for i, (angle_deg, img) in enumerate(orbit_images):
            _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            b64    = base64.b64encode(buf).decode('utf-8')
            content.append({
                'type': 'text',
                'text': f'— Image {i + 1}/{len(orbit_images)}  (view angle {angle_deg:.0f}°) —',
            })
            content.append({
                'type': 'image_url',
                'image_url': {'url': f'data:image/jpeg;base64,{b64}'},
            })

        content.append({
            'type': 'text',
            'text': (
                'Which image shows text most clearly? '
                'Call report_label with ALL readable text (brand, model, specs, etc.). '
                'If no text is legible in any image, briefly describe what each image shows.'
            ),
        })

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': (
                            'You are analyzing photos of an angle grinder to find and '
                            'transcribe its label. Look for brand names, model numbers, '
                            'power (W), speed (RPM), voltage, and any other text. '
                            'Call report_label with everything you can read.'
                        ),
                    },
                    {'role': 'user', 'content': content},
                ],
                tools=TOOLS,
                tool_choice='auto',
            )
        except Exception as e:
            self.get_logger().error(f'Multi-image LLM call failed: {e}')
            return (f'Orbit complete ({len(orbit_images)} images). '
                    f'LLM analysis failed: {e}'), False

        msg = response.choices[0].message

        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == 'report_label':
                    try:
                        args   = json.loads(tc.function.arguments)
                        result = self._exec_report(
                            args.get('text', ''),
                            args.get('confidence', 0.0),
                        )
                        return result, True   # ← terminal: label found
                    except Exception as e:
                        self.get_logger().error(f'report_label parse error: {e}')

        # LLM described what it sees but couldn't read a label
        description = msg.content or 'No text identified in any orbit image.'
        self.get_logger().info(f'Orbit analysis: {description}')
        return (
            f'Orbit complete ({len(orbit_images)} images analyzed). '
            f'Observation: {description}\n'
            f'Try a closer orbit (radius=0.12) or different angle.'
        ), False

    def _exec_report(self, text: str, confidence: float) -> str:
        self.get_logger().info(
            f'\n{"=" * 55}\n'
            f'  LABEL READ: "{text}"\n'
            f'  Confidence: {confidence:.0%}\n'
            f'{"=" * 55}')
        with self._state_lock:
            self._state = State.DONE
        return f'Label reported: "{text}" ({confidence:.0%}). Task complete.'

    # ------------------------------------------------------------------
    # Message builder
    # ------------------------------------------------------------------

    def _build_user_message(self) -> dict:
        """
        Build the multimodal user message:
          - current camera image encoded as base64 JPEG
          - end-effector XYZ position
          - detected object XYZ position
        """
        img = self._latest_img

        # Encode image as base64 JPEG (quality 85 keeps tokens reasonable)
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64    = base64.b64encode(buf).decode('utf-8')

        # End-effector position
        ee = self.mover.get_current_ee_pose()
        ee_str = (f"x={ee['x']:.3f} y={ee['y']:.3f} z={ee['z']:.3f} m"
                  if ee else 'unavailable')

        # Object position
        pose = self._latest_pose
        if pose:
            p      = pose.pose.position
            obj_str = f'x={p.x:.3f} y={p.y:.3f} z={p.z:.3f} m (base frame)'
        else:
            obj_str = 'not detected'

        context = (
            f'Current end-effector position: {ee_str}\n'
            f'Detected object position: {obj_str}\n\n'
            'Here is the current camera image. '
            'What can you see? Move to a better viewpoint or report the label.'
        )

        return {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': context},
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{b64}'},
                },
            ],
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = LLMPlanner()
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
