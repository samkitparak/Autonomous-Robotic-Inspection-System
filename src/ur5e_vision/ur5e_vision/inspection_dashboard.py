"""
inspection_dashboard.py
-----------------------
Flask web dashboard + ROS2 node for UR5e label inspection.

Replaces llm_planner.py with a clean, demo-friendly interface:
  - Serves a web UI on port 5000
  - Streams live YOLO-annotated video
  - "Start Inspection" button triggers an orbit + Ollama label read
  - Real-time status updates via SSE
  - Displays final label result

Orbit strategy (fixed, no LLM-in-the-loop):
  1. Robot orbits at 6 waypoints (radius=0.20 m)
  2. At each waypoint: stop, wait 1 s, capture one photo
  3. All 6 photos sent to Ollama in one call
  4. If unreadable, retry at radius=0.12 m (closer)
  5. Result shown on dashboard

Run: ros2 run ur5e_vision inspection_dashboard
"""

import base64
import os
import queue
import threading
import time
from enum import Enum, auto

import cv2
import rclpy
import rclpy.executors
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from cv_bridge import CvBridge
from flask import Flask, Response, render_template_string, stream_with_context
from geometry_msgs.msg import PoseStamped
import requests
from sensor_msgs.msg import Image

from ur5e_vision.robot_mover import RobotMover
from ur5e_vision.viewpoint_generator import generate_orbit, coverage_angles_for_label


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class State(Enum):
    IDLE       = auto()   # ready, waiting for user to press Start
    INSPECTING = auto()   # orbit + capture in progress
    ANALYZING  = auto()   # Ollama call in progress
    DONE       = auto()   # result shown; press Start again to re-run


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UR5e Inspection System</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:            #04060f;
    --glass:         rgba(255,255,255,0.035);
    --glass-border:  rgba(255,255,255,0.07);
    --text:          #dde2f0;
    --text-dim:      rgba(221,226,240,0.38);
    --blue:          #4d8df5;
    --blue-glow:     rgba(77,141,245,0.28);
    --green:         #00dfa0;
    --green-glow:    rgba(0,223,160,0.22);
    --purple:        #a96df5;
    --purple-glow:   rgba(169,109,245,0.22);
    --amber:         #f5c043;
    --red:           #f55060;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, 'SF Pro Display', 'Segoe UI', system-ui, sans-serif;
    height: 100vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  /* Background: subtle radial blobs + fine grid */
  .bg-layer {
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background:
      radial-gradient(ellipse 70% 55% at 15%  8%, rgba(77,141,245,0.07)  0%, transparent 65%),
      radial-gradient(ellipse 55% 45% at 85% 90%, rgba(169,109,245,0.06) 0%, transparent 60%),
      radial-gradient(ellipse 45% 50% at 55% 50%, rgba(0,223,160,0.03)   0%, transparent 65%);
  }
  .grid-layer {
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
      linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px);
    background-size: 44px 44px;
  }

  /* ── Header ── */
  header {
    position: relative; z-index: 10; flex-shrink: 0;
    height: 62px;
    display: flex; align-items: center; gap: 14px;
    padding: 0 26px;
    background: rgba(4,6,15,0.75);
    border-bottom: 1px solid var(--glass-border);
    backdrop-filter: blur(24px);
  }

  .logo-mark {
    width: 34px; height: 34px; flex-shrink: 0;
    border-radius: 9px;
    background: linear-gradient(140deg, var(--blue), var(--purple));
    box-shadow: 0 0 18px var(--blue-glow);
    display: flex; align-items: center; justify-content: center;
  }
  .logo-mark svg { width: 18px; height: 18px; }

  .header-titles { display: flex; flex-direction: column; gap: 1px; }
  .header-titles strong {
    font-size: 0.92rem; font-weight: 650; letter-spacing: 0.01em;
    color: var(--text);
  }
  .header-titles span {
    font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 0.09em; color: var(--text-dim);
  }

  .h-sep { width: 1px; height: 26px; background: var(--glass-border); margin: 0 2px; }
  .h-spacer { flex: 1; }

  /* Status pill */
  .status-pill {
    display: flex; align-items: center; gap: 7px;
    padding: 5px 14px; border-radius: 20px;
    font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.09em;
    border: 1px solid transparent;
    transition: background 0.35s, border-color 0.35s, color 0.35s;
  }
  .status-dot {
    width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
    transition: background 0.35s;
  }

  .pill-IDLE       { background: rgba(77,141,245,0.1);  border-color: rgba(77,141,245,0.28);  color: var(--blue);   }
  .pill-IDLE       .status-dot { background: var(--blue); }
  .pill-INSPECTING { background: rgba(0,223,160,0.1);   border-color: rgba(0,223,160,0.3);    color: var(--green);  }
  .pill-INSPECTING .status-dot { background: var(--green);  animation: blink .9s infinite; }
  .pill-ANALYZING  { background: rgba(169,109,245,0.1); border-color: rgba(169,109,245,0.3);  color: var(--purple); }
  .pill-ANALYZING  .status-dot { background: var(--purple); animation: blink .6s infinite; }
  .pill-DONE       { background: rgba(0,223,160,0.1);   border-color: rgba(0,223,160,0.3);    color: var(--green);  }
  .pill-DONE       .status-dot { background: var(--green); }

  @keyframes blink {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:.3; transform:scale(.65); }
  }

  /* ── Main layout ── */
  .main {
    position: relative; z-index: 1; flex: 1;
    display: flex; gap: 14px;
    padding: 14px 18px 18px;
    overflow: hidden;
  }

  /* ── Video panel ── */
  .video-wrap {
    flex: 1; min-width: 0;
    position: relative;
    border-radius: 14px; overflow: hidden;
    border: 1px solid var(--glass-border);
    background: rgba(0,0,0,0.5);
  }
  .video-wrap img {
    width: 100%; height: 100%;
    object-fit: contain; display: block;
  }

  /* Viewfinder corner brackets */
  .vf-corner {
    position: absolute; width: 22px; height: 22px;
    border-color: var(--blue); border-style: solid; opacity: .55;
    transition: opacity .3s;
  }
  .vf-tl { top:13px;    left:13px;  border-width:2px 0 0 2px; border-radius:3px 0 0 0; }
  .vf-tr { top:13px;    right:13px; border-width:2px 2px 0 0; border-radius:0 3px 0 0; }
  .vf-bl { bottom:13px; left:13px;  border-width:0 0 2px 2px; border-radius:0 0 0 3px; }
  .vf-br { bottom:13px; right:13px; border-width:0 2px 2px 0; border-radius:0 0 3px 0; }

  .video-wrap.active .vf-corner { opacity: .9; border-color: var(--green); }

  /* Camera label */
  .cam-label {
    position: absolute; top: 16px; left: 50%; transform: translateX(-50%);
    font-size: 0.62rem; letter-spacing: .12em; text-transform: uppercase;
    color: var(--text-dim);
    background: rgba(4,6,15,0.65);
    padding: 3px 11px; border-radius: 5px;
    border: 1px solid var(--glass-border);
    backdrop-filter: blur(8px);
  }

  /* Scan line during INSPECTING */
  .scan-line {
    position: absolute; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--green) 40%, var(--blue) 60%, transparent 100%);
    opacity: 0; pointer-events: none; top: 10%;
  }
  .video-wrap.active .scan-line {
    animation: scan-sweep 2.8s ease-in-out infinite;
  }
  @keyframes scan-sweep {
    0%   { top:8%;  opacity:0; }
    8%   { opacity:.7; }
    92%  { opacity:.7; }
    100% { top:92%; opacity:0; }
  }

  /* ── Side panel ── */
  .side-panel {
    width: 332px; flex-shrink: 0;
    display: flex; flex-direction: column; gap: 10px;
  }

  /* Glass card */
  .card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 13px;
    backdrop-filter: blur(18px);
    padding: 13px 15px;
  }

  .card-header {
    font-size: 0.62rem; text-transform: uppercase;
    letter-spacing: .1em; color: var(--text-dim);
    margin-bottom: 10px;
    display: flex; align-items: center; gap: 8px;
  }
  .card-header::after {
    content: ''; flex: 1; height: 1px;
    background: var(--glass-border);
  }

  /* Log */
  .log-card { flex: 1; display: flex; flex-direction: column; min-height: 0; }
  .log {
    flex: 1; overflow-y: auto; min-height: 0;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 0.72rem; color: #8a96b0; line-height: 1.85;
    scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.08) transparent;
  }
  .log::-webkit-scrollbar { width: 3px; }
  .log::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.09); border-radius: 2px; }

  .log-row {
    display: flex; gap: 8px;
    animation: row-in .22s ease;
  }
  @keyframes row-in {
    from { opacity:0; transform:translateY(5px); }
    to   { opacity:1; transform:translateY(0); }
  }
  .log-ts  { color: rgba(255,255,255,0.17); flex-shrink: 0; }
  .log-txt { }
  .log-txt.ok  { color: var(--green);  }
  .log-txt.err { color: var(--red); }
  .log-txt.dim { color: rgba(255,255,255,0.25); }

  /* Result card */
  .result-card {
    display: none;
    border-color: rgba(0,223,160,0.18);
    background: rgba(0,223,160,0.04);
  }
  .result-card.visible {
    display: block;
    animation: row-in .3s ease;
  }
  .result-body {
    font-size: 0.8rem; color: var(--text);
    white-space: pre-wrap; line-height: 1.65;
    max-height: 110px; overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: rgba(255,255,255,0.08) transparent;
  }
  .result-body::-webkit-scrollbar { width: 3px; }
  .result-body::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.09); border-radius: 2px; }

  /* CTA button */
  .btn-start {
    flex-shrink: 0;
    width: 100%; padding: 15px;
    border: none; border-radius: 12px;
    font-size: 0.9rem; font-weight: 650; letter-spacing: .04em;
    cursor: pointer; position: relative; overflow: hidden;
    transition: transform .18s, box-shadow .18s, background .3s;
    background: linear-gradient(130deg, var(--blue) 0%, var(--purple) 100%);
    color: #fff;
    box-shadow: 0 4px 22px var(--blue-glow);
  }
  .btn-start:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(77,141,245,.42);
  }
  .btn-start:active:not(:disabled) { transform: translateY(0); }
  .btn-start:disabled {
    background: rgba(255,255,255,0.06);
    color: rgba(255,255,255,.22); cursor: not-allowed; box-shadow: none;
  }
  /* Ripple */
  .ripple {
    position: absolute; border-radius: 50%;
    background: rgba(255,255,255,.22);
    transform: scale(0); pointer-events: none;
    animation: ripple-out .55s linear forwards;
  }
  @keyframes ripple-out { to { transform: scale(5); opacity: 0; } }
</style>
</head>
<body>
<div class="bg-layer"></div>
<div class="grid-layer"></div>

<header>
  <div class="logo-mark">
    <svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="10" cy="10" r="3" fill="white" opacity="0.9"/>
      <path d="M10 2 L10 5" stroke="white" stroke-width="1.8" stroke-linecap="round" opacity="0.7"/>
      <path d="M10 15 L10 18" stroke="white" stroke-width="1.8" stroke-linecap="round" opacity="0.7"/>
      <path d="M2 10 L5 10" stroke="white" stroke-width="1.8" stroke-linecap="round" opacity="0.7"/>
      <path d="M15 10 L18 10" stroke="white" stroke-width="1.8" stroke-linecap="round" opacity="0.7"/>
      <path d="M4.22 4.22 L6.34 6.34" stroke="white" stroke-width="1.6" stroke-linecap="round" opacity="0.45"/>
      <path d="M13.66 13.66 L15.78 15.78" stroke="white" stroke-width="1.6" stroke-linecap="round" opacity="0.45"/>
      <path d="M15.78 4.22 L13.66 6.34" stroke="white" stroke-width="1.6" stroke-linecap="round" opacity="0.45"/>
      <path d="M6.34 13.66 L4.22 15.78" stroke="white" stroke-width="1.6" stroke-linecap="round" opacity="0.45"/>
    </svg>
  </div>
  <div class="header-titles">
    <strong>UR5e Inspection System</strong>
    <span>Visual Label Recognition</span>
  </div>
  <div class="h-sep"></div>
  <div class="status-pill pill-IDLE" id="pill">
    <div class="status-dot"></div>
    <span id="pill-label">Idle</span>
  </div>
  <div class="h-spacer"></div>
</header>

<div class="main">
  <div class="video-wrap" id="video-wrap">
    <img src="/video_feed" alt="Live camera feed">
    <div class="vf-corner vf-tl"></div>
    <div class="vf-corner vf-tr"></div>
    <div class="vf-corner vf-bl"></div>
    <div class="vf-corner vf-br"></div>
    <div class="cam-label">Live Feed &mdash; YOLO</div>
    <div class="scan-line"></div>
  </div>

  <div class="side-panel">
    <!-- Log -->
    <div class="card log-card">
      <div class="card-header">System Log</div>
      <div class="log" id="log"></div>
    </div>

    <!-- Result -->
    <div class="card result-card" id="result-card">
      <div class="card-header">Label Text</div>
      <div class="result-body" id="result-body"></div>
    </div>

    <!-- CTA -->
    <button class="btn-start" id="btn" onclick="startInspection(event)">
      Start Inspection
    </button>
  </div>
</div>

<script>
  const log        = document.getElementById('log');
  const btn        = document.getElementById('btn');
  const pill       = document.getElementById('pill');
  const pillLabel  = document.getElementById('pill-label');
  const resultCard = document.getElementById('result-card');
  const resultBody = document.getElementById('result-body');
  const videoWrap  = document.getElementById('video-wrap');

  const STATE_LABELS = { IDLE:'Idle', INSPECTING:'Inspecting', ANALYZING:'Analyzing', DONE:'Complete' };

  function addLog(text, cls) {
    const now = new Date().toLocaleTimeString('en-GB');
    const row = document.createElement('div');
    row.className = 'log-row';
    const msgClass = cls ? 'log-txt ' + cls : 'log-txt';
    row.innerHTML =
      '<span class="log-ts">' + now + '</span>' +
      '<span class="' + msgClass + '">' + text + '</span>';
    log.appendChild(row);
    log.scrollTop = log.scrollHeight;
  }

  function setState(s) {
    pill.className = 'status-pill pill-' + s;
    pillLabel.textContent = STATE_LABELS[s] || s;
    btn.disabled = (s === 'INSPECTING' || s === 'ANALYZING');
    btn.textContent = (s === 'DONE') ? 'Inspect Again' : 'Start Inspection';
    if (s === 'INSPECTING') {
      videoWrap.classList.add('active');
    } else {
      videoWrap.classList.remove('active');
    }
  }

  const es = new EventSource('/events');
  es.addEventListener('status', e => addLog(e.data));
  es.addEventListener('state',  e => setState(e.data));
  es.addEventListener('result', e => {
    resultCard.classList.add('visible');
    resultBody.textContent = e.data;
    addLog('Label reading complete.', 'ok');
  });
  es.onerror = () => addLog('Connection lost — refresh to reconnect.', 'err');

  function startInspection(evt) {
    // Ripple effect
    const r = document.createElement('span');
    r.className = 'ripple';
    const rect = btn.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    r.style.cssText = 'width:' + size + 'px;height:' + size + 'px;' +
      'left:' + (evt.clientX - rect.left - size/2) + 'px;' +
      'top:'  + (evt.clientY - rect.top  - size/2) + 'px;';
    btn.appendChild(r);
    r.addEventListener('animationend', () => r.remove());

    resultCard.classList.remove('visible');
    fetch('/start', { method: 'POST' })
      .then(res => res.json())
      .then(d => { if (!d.ok) addLog('Error: ' + d.error, 'err'); })
      .catch(e => addLog('Request failed: ' + e, 'err'));
  }
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Dashboard node
# ---------------------------------------------------------------------------

class InspectionDashboard(Node):

    ORBIT_RADIUS_1 = 0.20   # first orbit standoff (m)
    ORBIT_RADIUS_2 = 0.12   # retry orbit — closer (m)
    NUM_WAYPOINTS  = 6
    SETTLE_TIME    = 1.0    # seconds to wait after robot stops before photo

    def __init__(self):
        super().__init__('inspection_dashboard')

        # ---- parameters ----
        self.declare_parameter('llm_server', 'http://172.22.132.20:8001')
        self.declare_parameter('move_speed', 0.15)
        self.declare_parameter('port',       5000)

        self.llm_server = self.get_parameter('llm_server').value.rstrip('/')
        speed           = self.get_parameter('move_speed').value
        self.port       = self.get_parameter('port').value

        # ---- state ----
        self._state      = State.IDLE
        self._state_lock = threading.Lock()

        self._latest_img  = None        # cv2 BGR
        self._img_lock    = threading.Lock()
        self._img_event   = threading.Event()

        self._latest_pose = None        # PoseStamped (base frame)
        self._pose_lock   = threading.Lock()

        # SSE: one queue per connected browser tab
        self._sse_queues: list[queue.SimpleQueue] = []
        self._sse_lock = threading.Lock()

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

        # ---- Flask ----
        self.app = Flask(__name__)
        self._register_routes()
        threading.Thread(
            target=lambda: self.app.run(
                host='0.0.0.0', port=self.port,
                threaded=True, use_reloader=False),
            daemon=True,
        ).start()

        self.get_logger().info(
            f'Inspection dashboard at http://0.0.0.0:{self.port}  '
            f'llm_server={self.llm_server}')

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _image_cb(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self._img_lock:
            self._latest_img = img
        self._img_event.set()

    def _pose_cb(self, msg: PoseStamped):
        with self._pose_lock:
            self._latest_pose = msg

    # ------------------------------------------------------------------
    # Flask routes
    # ------------------------------------------------------------------

    def _register_routes(self):
        app = self.app

        @app.route('/')
        def index():
            return render_template_string(HTML)

        @app.route('/video_feed')
        def video_feed():
            return Response(
                self._gen_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route('/events')
        def events():
            q = queue.SimpleQueue()
            with self._sse_lock:
                self._sse_queues.append(q)
            # Push current state immediately so the badge is correct on load
            with self._state_lock:
                current = self._state.name
            q.put(f'event: state\ndata: {current}\n\n')

            def generate():
                try:
                    while True:
                        try:
                            yield q.get(timeout=15.0)
                        except queue.Empty:
                            yield ': heartbeat\n\n'
                finally:
                    with self._sse_lock:
                        try:
                            self._sse_queues.remove(q)
                        except ValueError:
                            pass

            return Response(
                stream_with_context(generate()),
                mimetype='text/event-stream',
                headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

        @app.route('/start', methods=['POST'])
        def start():
            with self._state_lock:
                if self._state in (State.INSPECTING, State.ANALYZING):
                    return {'ok': False,
                            'error': 'Inspection already in progress'}
                with self._pose_lock:
                    pose = self._latest_pose
                if pose is None:
                    return {'ok': False,
                            'error': 'No object detected yet — '
                                     'wait for YOLO to detect the part'}
                self._state = State.INSPECTING

            self._push_state('INSPECTING')
            threading.Thread(
                target=self._run_inspection, daemon=True).start()
            return {'ok': True}

    def _gen_frames(self):
        """MJPEG stream for /video_feed (~15 fps)."""
        while True:
            with self._img_lock:
                img = self._latest_img
            if img is None:
                time.sleep(0.05)
                continue
            ok, buf = cv2.imencode('.jpg', img,
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + buf.tobytes() + b'\r\n')
            time.sleep(1.0 / 15)

    # ------------------------------------------------------------------
    # SSE helpers
    # ------------------------------------------------------------------

    def _push_event(self, event: str, data: str):
        msg = f'event: {event}\ndata: {data}\n\n'
        with self._sse_lock:
            for q in list(self._sse_queues):
                q.put(msg)

    def _push_status(self, text: str):
        self.get_logger().info(text)
        self._push_event('status', text)

    def _push_state(self, name: str):
        self._push_event('state', name)

    # ------------------------------------------------------------------
    # Inspection orchestration (background thread)
    # ------------------------------------------------------------------

    def _run_inspection(self):
        try:
            self._push_status('Inspection started.')

            with self._pose_lock:
                pose = self._latest_pose

            obj_x = pose.pose.position.x
            obj_y = pose.pose.position.y
            obj_z = pose.pose.position.z
            self._push_status(
                f'Object at ({obj_x:.3f}, {obj_y:.3f}, {obj_z:.3f}) m.')

            # --- First orbit ---
            images = self._do_orbit(obj_x, obj_y, obj_z,
                                    self.ORBIT_RADIUS_1)
            if not images:
                self._push_status('No images captured — aborting.')
                self._finish(State.IDLE, 'IDLE')
                return

            # --- Ollama ---
            self._set_state(State.ANALYZING)
            self._push_state('ANALYZING')
            self._push_status(f'Sending {len(images)} photos to Ollama…')
            text = self._analyze_images(images)

            if text and 'unable' not in text.lower():
                self._push_event('result', text)
                self._push_status('Label read successfully.')
                self._finish(State.DONE, 'DONE')
                return

            # --- Retry with closer orbit ---
            self._push_status(
                f'Label unclear at {self.ORBIT_RADIUS_1} m. '
                f'Retrying at {self.ORBIT_RADIUS_2} m…')
            self._set_state(State.INSPECTING)
            self._push_state('INSPECTING')

            images2 = self._do_orbit(obj_x, obj_y, obj_z,
                                     self.ORBIT_RADIUS_2)
            result_text = 'Unable to read label.'
            if images2:
                self._set_state(State.ANALYZING)
                self._push_state('ANALYZING')
                self._push_status(
                    f'Sending {len(images2)} closer photos to Ollama…')
                text2 = self._analyze_images(images2)
                if text2:
                    result_text = text2

            self._push_event('result', result_text)
            if 'unable' in result_text.lower():
                self._push_status('Could not read label after two orbits.')
            else:
                self._push_status('Label read successfully.')
            self._finish(State.DONE, 'DONE')

        except Exception as e:
            self.get_logger().error(f'Inspection error: {e}')
            self._push_status(f'ERROR: {e}')
            self._finish(State.IDLE, 'IDLE')

    def _finish(self, state: State, name: str):
        self._set_state(state)
        self._push_state(name)

    # ------------------------------------------------------------------
    # Orbit executor
    # ------------------------------------------------------------------

    def _do_orbit(
        self,
        obj_x: float, obj_y: float, obj_z: float,
        radius: float,
    ) -> list:
        """
        Move through NUM_WAYPOINTS around the object at the given radius.
        At each waypoint: wait SETTLE_TIME seconds, then capture one photo.
        Returns list of cv2 BGR images.
        """
        self._push_status(
            f'Orbit r={radius:.2f} m, {self.NUM_WAYPOINTS} waypoints…')

        ee = self.mover.get_current_ee_pose()
        start_angle = (coverage_angles_for_label(
                           ee['x'], ee['y'], obj_x, obj_y)
                       if ee else 0.0)

        viewpoints = generate_orbit(
            obj_x=obj_x, obj_y=obj_y, obj_z=obj_z,
            radius=radius,
            num_points=self.NUM_WAYPOINTS,
            start_angle=start_angle,
        )

        captured: list = []

        def _on_waypoint(idx: int, vp):
            self._push_status(
                f'Waypoint {idx + 1}/{self.NUM_WAYPOINTS} — '
                f'settling {self.SETTLE_TIME:.0f} s…')
            time.sleep(self.SETTLE_TIME)
            # Grab a fresh frame after settling
            self._img_event.clear()
            if not self._img_event.wait(timeout=3.0):
                self._push_status(
                    f'  Warning: no image at waypoint {idx + 1}.')
                return
            with self._img_lock:
                img = self._latest_img
            if img is not None:
                captured.append(img.copy())
                self._push_status(f'  Photo {len(captured)} captured.')

        succeeded, total = self.mover.execute_orbit(
            viewpoints, on_waypoint_reached=_on_waypoint)

        self._push_status(
            f'Orbit complete: {succeeded}/{total} waypoints, '
            f'{len(captured)} photos.')
        return captured

    # ------------------------------------------------------------------
    # Gemini analysis
    # ------------------------------------------------------------------

    def _analyze_images(self, images: list) -> str:
        """Send orbit photos to /read-label, return the best result by confidence."""
        url = f'{self.llm_server}/read-label'
        best_text = ''
        best_conf = -1.0

        for i, img in enumerate(images):
            _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            b64 = base64.b64encode(buf).decode('utf-8')
            try:
                r = requests.post(url, json={'image_b64': b64}, timeout=60)
                r.raise_for_status()
                data = r.json()
                text = data.get('label_text', '')
                conf = data.get('confidence', 0.0)
                self._push_status(
                    f'  Photo {i + 1}/{len(images)}: conf={conf:.2f} — {text[:60]}')
                if conf > best_conf:
                    best_conf = conf
                    best_text = text
            except Exception as e:
                self.get_logger().error(f'LLM server error (photo {i + 1}): {e}')
                self._push_status(f'  Photo {i + 1} error: {e}')

        return best_text or 'Unable to read label'

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_state(self, state: State):
        with self._state_lock:
            self._state = state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = InspectionDashboard()
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
