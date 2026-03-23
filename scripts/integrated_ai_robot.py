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
LLM_MODEL = "moonshotai/kimi-k2.5"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
CONFIDENCE_THRESHOLD = 0.85
CAM_INDEX = 2

# --- ROS 2 DIRECT COMMANDER ---
class UR5Commander(Node):
    def __init__(self):
        super().__init__('ur5_ai_commander')
        self.publisher_ = self.create_publisher(
            JointTrajectory, 
            '/scaled_joint_trajectory_controller/joint_trajectory', 
            10
        )
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.current_joints = {}
        self.received_first_state = False
        self.get_logger().info("✅ Robot Control Link Active.")

    def joint_state_callback(self, msg):
        for i, name in enumerate(msg.name):
            self.current_joints[name] = msg.position[i]
        self.received_first_state = True

    def execute_visual_servo_move(self, dx=0.0, dy=0.0, dz=0.0):
        if not self.received_first_state: 
            self.get_logger().error("❌ IGNORING MOVE: Robot state not yet received.")
            return False

        ordered_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        target_positions = []
        try:
            for name in ordered_names:
                target_positions.append(self.current_joints[name])
        except KeyError: return False

        # --- SERVO GAINS ---
        # Adjusted for smoother, safer "visual servoing"
        target_positions[0] += (dx * 2.0)  # Pan (Horizontal)
        target_positions[1] += (dz * 1.5)  # Lift (Vertical/Depth compensation)
        target_positions[2] += (dy * 1.8)  # Reach (Zoom)

        msg = JointTrajectory()
        msg.header.frame_id = "base_link"
        msg.joint_names = ordered_names
        point = JointTrajectoryPoint()
        point.positions = target_positions
        point.time_from_start.sec = 3 
        msg.points.append(point)

        self.get_logger().info(f"🤖 VISUAL SERVO: dx:{dx:.2f} dy:{dy:.2f} dz:{dz:.2f}")
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

print("1. Loading YOLO Vision...")
try:
    YOLO_MODEL = YOLO(MODEL_PATH)
    OPENAI_CLIENT = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
except Exception as e:
    print(f"Init Error: {e}"); sys.exit(1)

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

def camera_loop():
    global outputFrame, RAW_FRAME_ANALYSIS
    CAMERA.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    CAMERA.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        success, frame = CAMERA.read()
        if not success: time.sleep(0.1); continue
        try:
            RAW_FRAME_ANALYSIS = frame.copy()
            display_frame = cv2.resize(frame, (640, 480))
            with yolo_lock:
                results = YOLO_MODEL.predict(display_frame, verbose=False, device='cpu')
            annotated_frame = results[0].plot()
            # Draw crosshair
            cv2.line(annotated_frame, (320, 220), (320, 260), (255, 255, 255), 1)
            cv2.line(annotated_frame, (300, 240), (340, 240), (255, 255, 255), 1)
            with lock:
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if ret: outputFrame = buffer.tobytes()
        except: pass
        time.sleep(0.1)

# --- UI ---
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
            .console-font {{ font-family: 'Inter', monospace; font-size: 0.85rem; }}
            .custom-bg {{ background-color: #020617; color: #f8fafc; font-family: 'Inter', sans-serif; }}
            #videoFeed {{ width: 100%; height: 420px; object-fit: contain; background: #000; }}
            .box {{ background-color: #0f172a; border: 1px solid #1e293b; }}
        </style>
    </head>
    <body class="custom-bg min-h-screen p-8">
        <div class="max-w-6xl mx-auto space-y-6">
            <div class="flex justify-between items-end border-b border-slate-800 pb-6">
                <div>
                    <h1 class="text-4xl font-black text-white tracking-tighter uppercase">AI Visual Servo</h1>
                    <p class="text-slate-400 font-medium tracking-wide">YOLO-Targeted Autonomous Zoom & Inspection</p>
                </div>
            </div>
            
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div class="lg:col-span-8 box p-2 rounded-2xl shadow-2xl overflow-hidden">
                    <img id="videoFeed" src="/video_feed" class="rounded-xl">
                </div>
                
                <div class="lg:col-span-4 flex flex-col gap-4">
                    <button id="btn" onclick="startAnalysis()" class="w-full py-6 bg-indigo-600 hover:bg-indigo-500 rounded-2xl font-black text-xl shadow-lg transition-all active:scale-95">
                        <span id="buttonText">🚀 TRIGGER ZOOM & SCAN</span>
                    </button>
                    
                    <div class="box p-6 rounded-2xl flex-grow flex flex-col min-h-[300px]">
                        <h3 class="text-xs font-bold uppercase tracking-widest text-slate-500 mb-4">Targeting Logic</h3>
                        <pre id="output" class="console-font text-indigo-300 flex-grow overflow-auto whitespace-pre-wrap leading-relaxed">System Ready.</pre>
                    </div>
                </div>
            </div>
        </div>
        <script>
            async function startAnalysis() {{
                const btn = document.getElementById('btn');
                const out = document.getElementById('output');
                btn.disabled = true;
                out.innerText = ">>> LOCKING TARGET...\\n";
                
                try {{
                    const res = await fetch('/analyze', {{ method: 'POST' }});
                    const data = await res.json();
                    
                    if(data.status === 'move_triggered') {{
                         out.innerText += `\\n[YOLO LOCK] Center Offset: ${{data.offset}}\\n[AI DECISION] ${{data.reason}}\\n\\n>>> EXECUTING VISUAL SERVO...`;
                         setTimeout(startAnalysis, 4500); 
                    }} else if (data.status === 'success') {{
                         out.innerText = "✅ TARGET CHARACTERIZED:\\n" + JSON.stringify(JSON.parse(data.result), null, 2);
                    }} else {{
                         out.innerText = "⚠️ " + data.result;
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
    except: return jsonify({'status': 'error', 'result': 'YOLO Error'})

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
    
    if target_box is None: return jsonify({'status': 'YOLO_FAIL', 'result': 'Searching for part...'})

    # --- PIXEL MATH FOR TARGETING ---
    h, w, _ = frame.shape
    x1, y1, x2, y2 = target_box
    
    # Calculate offset from center
    norm_center_x = ((x1 + x2) / 2) / w
    norm_center_y = ((y1 + y2) / 2) / h
    error_x = norm_center_x - 0.5
    error_z = norm_center_y - 0.5 
    
    object_width_pct = (x2 - x1) / w

    crop = frame[y1:y2, x1:x2]
    img_str = get_base64(crop)

    # --- DYNAMIC PROMPT ---
    prompt = f"""
    You are a visual-servoing robot controller. 
    TARGET INFO:
    - Center Offset: X:{error_x:.2f}, Y:{error_z:.2f}
    - Screen Coverage: {object_width_pct:.2f} (Target is 0.50)
    
    GUIDELINES:
    1. If abs(X) or abs(Y) > 0.05, suggest a move to center the object.
    2. If Coverage < 0.45, suggest a positive 'dy' move to ZOOM IN.
    3. Use the pixel offsets provided to calculate precise 'dx' and 'dz'.
    4. Only return 'success' if the Serial and Model numbers are crystal clear.

    COORDINATES:
    - dx: Horizontal (Right=Pos, Left=Neg)
    - dy: Zoom/Reach (Forward=Pos, Back=Neg)
    - dz: Vertical (Up=Pos, DOWN TOWARDS TABLE=Neg)

    JSON OUTPUT:
    {{"action": "move", "dx": float, "dy": float, "dz": float, "reason": "str"}} 
    OR 
    {{"action": "success", "Serial": "str", "Model": "str"}}
    """

    try:
        resp = OPENAI_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                ]}
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(resp.choices[0].message.content)
        
        if data.get('action') == 'move':
            if commander_node:
                t = threading.Thread(target=commander_node.execute_visual_servo_move, kwargs={
                    'dx': data.get('dx', 0.0), 
                    'dy': data.get('dy', 0.0), 
                    'dz': data.get('dz', 0.0)
                })
                t.start()
            return jsonify({'status': 'move_triggered', 'offset': f"X:{error_x:.2f} Y:{error_z:.2f}", 'move_command': f"NAV: {data.get('dx')}, {data.get('dy')}, {data.get('dz')}", 'reason': data.get('reason')})
        
        return jsonify({'status': 'success', 'result': json.dumps(data)})
    except Exception as e:
        return jsonify({'status': 'error', 'result': str(e)})

def main():
    global commander_node
    rclpy.init()
    commander_node = UR5Commander()
    ros_thread = threading.Thread(target=lambda: rclpy.spin(commander_node), daemon=True)
    ros_thread.start()
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    main()
