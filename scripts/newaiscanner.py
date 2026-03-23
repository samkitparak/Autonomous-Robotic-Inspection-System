import os
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import cv2
import threading
import numpy as np
import base64
import json
import time
import sys
from flask import Flask, Response, jsonify
from ultralytics import YOLO
from openai import OpenAI

# --- CONFIGURATION ---
MODEL_PATH = "best.pt"
LLM_MODEL = "google/gemini-2.5-flash"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
CONFIDENCE_THRESHOLD = 0.85
CAM_INDEX = 2

# --- ROS 2 DIRECT COMMANDER ---
class UR5Commander(Node):
    def __init__(self):
        super().__init__('ur5_ai_commander')
        
        # Publisher to the Driver (Direct Trajectory Control)
        self.publisher_ = self.create_publisher(
            JointTrajectory, 
            '/scaled_joint_trajectory_controller/joint_trajectory', 
            10
        )
        
        # Subscriber to read start position
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        
        self.current_joints = {}
        self.received_first_state = False
        self.get_logger().info("✅ UR5 Commander Started. Waiting for robot state...")

    def joint_state_callback(self, msg):
        for i, name in enumerate(msg.name):
            self.current_joints[name] = msg.position[i]
        self.received_first_state = True

    def execute_inspection_move(self, dx=0.0, dy=0.0, dz=0.0):
        """
        Maps Cartesian offsets from Gemini to Joint-Space nudges.
        In base_link:
        - dz > 0 is UP (Further)
        - dz < 0 is DOWN (Closer)
        """
        if not self.received_first_state:
            self.get_logger().error("❌ Cannot move: No joint states received.")
            return False

        ordered_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        # Get current position
        target_positions = []
        try:
            for name in ordered_names:
                target_positions.append(self.current_joints[name])
        except KeyError:
            return False

        # --- MAPPING LOGIC (Corrected Z-Axis) ---
        # Gemini provides offsets in meters. We convert to Radian nudges.
        
        # 1. Horizontal (Pan): dx moves the base
        target_positions[0] += (dx * 1.5)  
        
        # 2. Vertical (Lift): dz moves the shoulder. 
        # Since dz < 0 is 'Closer' (Down), adding a negative dz will lower the shoulder.
        target_positions[1] += (dz * 1.2)  
        
        # 3. Depth (Reach): dy moves the elbow to reach forward/backward
        target_positions[2] += (dy * 1.5)  

        # --- BUILD MESSAGE ---
        msg = JointTrajectory()
        msg.header.frame_id = "base_link"
        msg.joint_names = ordered_names
        
        point = JointTrajectoryPoint()
        point.positions = target_positions
        point.time_from_start.sec = 4  # Slow, smooth 4-second move
        
        msg.points.append(point)

        self.get_logger().info(f"🚀 EXECUTING NUDGE: dx={dx}, dy={dy}, dz={dz}")
        
        # Publish multiple times to ensure driver catches it
        for _ in range(3):
            self.publisher_.publish(msg)
            time.sleep(0.05)
            
        return True

# --- FLASK & AI APP SETUP ---
app = Flask(__name__)
lock = threading.Lock()
yolo_lock = threading.Lock() 
outputFrame = None
RAW_FRAME_ANALYSIS = None 
commander_node = None 

# Load AI
print("1. Loading YOLO...")
try:
    YOLO_MODEL = YOLO(MODEL_PATH)
    OPENAI_CLIENT = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

# Camera Finder
def find_and_open_camera():
    indices = [4, 2, 0, 1, 3, 5]
    for i in indices:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret: return cap, i
            cap.release()
    sys.exit(1)

CAMERA, CAM_INDEX = find_and_open_camera()

def get_base64(image_arr):
    _, buffer = cv2.imencode('.jpg', image_arr)
    return base64.b64encode(buffer).decode('utf-8')

# Camera Thread
def camera_loop():
    global outputFrame, RAW_FRAME_ANALYSIS
    CAMERA.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    CAMERA.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    CAMERA.set(cv2.CAP_PROP_FPS, 10)
    while True:
        success, frame = CAMERA.read()
        if not success:
            time.sleep(0.1); continue
        try:
            RAW_FRAME_ANALYSIS = frame.copy()
            display_frame = cv2.resize(frame, (640, 480))
            with yolo_lock:
                results = YOLO_MODEL.predict(display_frame, verbose=False, device='cpu')
            annotated_frame = results[0].plot()
            with lock:
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if ret: outputFrame = buffer.tobytes()
        except: pass
        time.sleep(0.1)

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
             @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            .console-font {{ font-family: 'Inter', monospace; font-size: 0.9rem; }}
            .custom-bg {{ background-color: #121212; color: #f0f0f0; font-family: 'Inter', sans-serif; }}
            #videoFeed {{ max-width: 100%; height: 360px; object-fit: contain; }}
            .box {{ background-color: #1e293b; border: 1px solid #334155; }}
            .right-panel {{ height: 380px; display: flex; flex-direction: column; gap: 1rem; }}
            #output-container {{ flex-grow: 1; }}
            #output {{ height: 100%; overflow: auto; }}
        </style>
    </head>
    <body class="custom-bg min-h-screen p-6">
        <div class="max-w-7xl mx-auto space-y-8">
            <h1 class="text-4xl font-extrabold tracking-tight border-b border-gray-700 pb-4 text-gray-100">
                ACTIVE PERCEPTION INSPECTOR
            </h1>
            <div class="flex flex-col lg:flex-row space-y-6 lg:space-y-0 lg:space-x-8 items-start">
                <div class="lg:w-3/4 w-full box p-3 rounded-xl shadow-2xl">
                    <img id="videoFeed" class="w-full h-auto rounded-lg" src="/video_feed">
                </div>
                <div class="lg:w-1/4 w-full right-panel">
                    <div class="box p-5 rounded-xl shadow-md">
                        <button id="btn" onclick="startAnalysis()" class="w-full bg-green-700 hover:bg-green-600 transition duration-150 rounded-lg shadow-lg">
                            <span id="buttonText" class="font-bold text-lg">📸 SCAN NOW</span>
                        </button>
                    </div>
                    <div id="output-container" class="box p-5 rounded-xl shadow-md">
                        <pre id="output" class="console-font bg-gray-900 p-3 rounded-lg text-gray-200 border-gray-700 border">System Ready.</pre>
                    </div>
                </div>
            </div>
        </div>
        <script>
            async function startAnalysis() {{
                const btn = document.getElementById('btn');
                const out = document.getElementById('output');
                btn.disabled = true;
                out.innerText = "Analyzing View...";
                try {{
                    const res = await fetch('/analyze', {{ method: 'POST' }});
                    const data = await res.json();
                    if(data.status === 'move_triggered') {{
                         out.innerText = `🤖 IMAGE BLURRY: Adjusted Camera...\\n\\nReason: ${{data.reason}}\\nMove: ${{data.move_command}}`;
                         setTimeout(startAnalysis, 5500); 
                    }} else if (data.status === 'success') {{
                         out.innerText = JSON.stringify(JSON.parse(data.result), null, 2);
                    }} else {{
                         out.innerText = data.result;
                    }}
                }} catch(e) {{ out.innerText = "Error: " + e; }}
                btn.disabled = false;
            }}
        </script>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                if outputFrame: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + outputFrame + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze', methods=['POST'])
def analyze():
    with lock: 
        if RAW_FRAME_ANALYSIS is None: return jsonify({'status': 'error', 'result': 'Cam Error'})
        frame = RAW_FRAME_ANALYSIS.copy()

    try:
        with yolo_lock: results = YOLO_MODEL.predict(frame, verbose=False, device='cpu')
    except: return jsonify({'status': 'error', 'result': 'YOLO fail'})

    target_box = None
    max_conf = 0.0
    for r in results:
        for box in r.boxes:
            cls = YOLO_MODEL.names[int(box.cls)].lower()
            if "housing" in cls or "motor" in cls:
                target_box = box.xyxy[0].cpu().numpy().astype(int)
                max_conf = box.conf.item()
                break
        if target_box is not None: break
    
    if target_box is None: return jsonify({'status': 'YOLO_FAIL', 'result': 'No Part Found'})

    # --- ACTIVE PERCEPTION (CORRECTED COORDINATES) ---
    if max_conf < CONFIDENCE_THRESHOLD:
        x1, y1, x2, y2 = target_box
        crop = frame[y1:y2, x1:x2]
        img_str = get_base64(crop)
        
        try:
            # COORDINATE SYSTEM GUIDE FOR GEMINI:
            # dx: Left/Right (0.05 is Right)
            # dy: Forward/Backward (0.05 is Forward)
            # dz: Up/Down (-0.05 is CLOSER/DOWN, +0.05 is FURTHER/UP)
            prompt = (
                "The inspection image is blurry. You must move the camera CLOSER to read the technical label. "
                "In the robot's coordinate system, dz < 0 moves DOWN (CLOSER) and dz > 0 moves UP (FURTHER). "
                "Recommend a move to get closer. Output JSON: {'action': 'move', 'dx': 0.0, 'dy': 0.05, 'dz': -0.05, 'reason': 'Moving closer and forward to read label.'}"
            )
            resp = OPENAI_CLIENT.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                ]}],
                response_format={"type": "json_object"}
            )
            move_data = json.loads(resp.choices[0].message.content)
            
            if commander_node:
                t = threading.Thread(target=commander_node.execute_inspection_move, kwargs={
                    'dx': move_data.get('dx', 0), 
                    'dy': move_data.get('dy', 0), 
                    'dz': move_data.get('dz', 0)
                })
                t.start()
            
            return jsonify({'status': 'move_triggered', 'move_command': str(move_data), 'reason': move_data.get('reason')})
        except Exception as e: return jsonify({'status': 'error', 'result': str(e)})

    # --- OCR EXECUTION ---
    x1, y1, x2, y2 = target_box
    crop = frame[y1:y2, x1:x2]
    img_str = get_base64(crop)
    try:
        resp = OPENAI_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Extract specs to JSON: {'Serial': 'str', 'Model': 'str', 'Specs': 'str'}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
            ]}],
            response_format={"type": "json_object"}
        )
        return jsonify({'status': 'success', 'result': resp.choices[0].message.content})
    except Exception as e: return jsonify({'status': 'error', 'result': str(e)})

def main():
    global commander_node
    rclpy.init()
    commander_node = UR5Commander()
    ros_thread = threading.Thread(target=lambda: rclpy.spin(commander_node), daemon=True)
    ros_thread.start()
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()
    print("🌍 SYSTEM READY: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    main()
