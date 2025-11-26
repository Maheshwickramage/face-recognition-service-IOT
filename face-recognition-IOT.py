#!/usr/bin/env python3
"""
Optimized Face Recognition Service with Live Video Stream
Key improvements:
1. Skip frame processing (only process every Nth frame)
2. Separate capture and processing threads
3. Adjustable detection frequency
4. Camera buffer management
"""
import os
import time
import json
import ssl
import cv2
import faulthandler
import pickle
import logging
import threading
import tempfile
import requests
import datetime
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify
from dotenv import load_dotenv
import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError
import paho.mqtt.client as mqtt

# ---- Optional hardware libs ----
try:
    import RPi.GPIO as GPIO
    from gpiozero import Servo, LED, Buzzer
    HAS_GPIO = True
except Exception:
    GPIO = None
    Servo = LED = Buzzer = None
    HAS_GPIO = False

import face_recognition

faulthandler.enable()
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [FACE-SERVICE] %(message)s"
)

app = Flask(__name__)

# Config
KNOWN_FACES_FILE = os.environ.get("KNOWN_FACES_FILE", "known_faces.pkl")
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
PORT = int(os.environ.get('FACE_SERVICE_PORT', '5001'))
VIDEO_WIDTH = int(os.environ.get('VIDEO_WIDTH', '640'))
VIDEO_HEIGHT = int(os.environ.get('VIDEO_HEIGHT', '480'))
RECOGNITION_TOLERANCE = float(os.environ.get('RECOGNITION_TOLERANCE', '0.4'))
IDENTIFY_COOLDOWN = int(os.environ.get('IDENTIFY_COOLDOWN', '30'))

# NEW: Performance tuning parameters
PROCESS_EVERY_N_FRAMES = int(os.environ.get('PROCESS_EVERY_N_FRAMES', '3'))  # Only process every 3rd frame
VIDEO_SCALE = float(os.environ.get('VIDEO_SCALE', '0.25'))  # Smaller = faster (0.25 = 1/4 size)
CAMERA_FPS = int(os.environ.get('CAMERA_FPS', '30'))
FACE_UPSAMPLE = int(os.environ.get('FACE_UPSAMPLE', '0'))  # 0 = fastest

PROCESS_START_TIME = datetime.datetime.utcnow().isoformat()

# State
known_faces = {}
known_faces_lock = threading.RLock()
access_events = []
access_events_lock = threading.Lock()
last_seen = {}
last_identified = {'user_id': None, 'timestamp': None, 'distance': None, 'method': 'FACE'}
last_identified_lock = threading.Lock()

# Hardware placeholders
servo = None
led = None
buzzer = None

if HAS_GPIO:
    try:
        GPIO.setwarnings(False)
        SERVO_PIN = int(os.environ.get('SERVO_PIN', '17'))
        LED_PIN = int(os.environ.get('LED_PIN', '22'))
        BUZZER_PIN = int(os.environ.get('BUZZER_PIN', '23'))
        servo = Servo(SERVO_PIN, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
        servo.value = -1
        led = LED(LED_PIN)
        buzzer = Buzzer(BUZZER_PIN)
        logging.info("✅ Hardware components initialized")
    except Exception:
        logging.exception("❌ Hardware initialization failed")
        HAS_GPIO = False
else:
    logging.info("⚠️ Running without GPIO (development mode)")

face_recognition_enabled = os.environ.get('ENABLE_FACE_RECOGNITION', 'true').lower() not in ('0', 'false', 'no')

# Camera capture
camera_lock = threading.Lock()
camera = None

# NEW: Shared frame buffer for threading
current_frame = None
current_frame_lock = threading.Lock()
last_processed_results = {'locations': [], 'names': []}
last_processed_lock = threading.Lock()

# === Utility functions (unchanged) ===
def save_known_faces():
    try:
        with known_faces_lock:
            tmp_path = KNOWN_FACES_FILE + ".tmp"
            with open(tmp_path, 'wb') as f:
                pickle.dump(known_faces, f)
            os.replace(tmp_path, KNOWN_FACES_FILE)
            marker = {'pid': os.getpid(), 'timestamp': int(time.time()), 'keys': list(known_faces.keys())}
            with open(KNOWN_FACES_FILE + ".marker.json.tmp", 'w') as mf:
                json.dump(marker, mf)
            os.replace(KNOWN_FACES_FILE + ".marker.json.tmp", KNOWN_FACES_FILE + ".marker.json")
            logging.info("✅ Saved known faces (%d)", len(known_faces))
    except Exception:
        logging.exception("Failed to save known faces")

def load_known_faces():
    global known_faces
    try:
        with known_faces_lock:
            if os.path.exists(KNOWN_FACES_FILE):
                with open(KNOWN_FACES_FILE, 'rb') as f:
                    known_faces = pickle.load(f)
                    logging.info("✅ Loaded %d known faces", len(known_faces))
            else:
                known_faces = {}
    except Exception:
        logging.exception("Failed to load known faces")
        known_faces = {}

def record_access_event(method, identifier, granted):
    event = {'timestamp': datetime.datetime.utcnow().isoformat(), 'method': method, 'identifier': identifier, 'granted': granted}
    with access_events_lock:
        access_events.append(event)
        if len(access_events) > 200:
            access_events.pop(0)

def _should_notify(uid: str) -> bool:
    now = time.time()
    last = last_seen.get(uid)
    if last is None or (now - last) >= IDENTIFY_COOLDOWN:
        last_seen[uid] = now
        return True
    return False

def _record_identified(uid: str, distance: float):
    ts = datetime.datetime.utcnow().isoformat()
    with last_identified_lock:
        last_identified['user_id'] = uid
        last_identified['timestamp'] = ts
        last_identified['distance'] = distance
        last_identified['method'] = 'FACE'
    logging.info("Recorded identification %s dist=%.4f", uid, distance)

def grant_access(identifier=""):
    logging.info("✅ FACE ACCESS GRANTED: %s", identifier)
    if HAS_GPIO and led and buzzer and servo:
        try:
            led.on()
            buzzer.on()
            time.sleep(0.2)
            buzzer.off()
            logging.info("🔓 Opening door...")
            servo.value = 0
            time.sleep(5)
            servo.value = -1
            logging.info("🔒 Door closed")
            led.off()
        except Exception:
            logging.exception("Hardware control failed")
    else:
        logging.info("🔓 [SIMULATED] Door unlocked for 5 seconds")
    record_access_event("FACE", identifier, True)

def deny_access(identifier=""):
    logging.info("❌ FACE ACCESS DENIED: %s", identifier)
    if HAS_GPIO and led and buzzer:
        try:
            for _ in range(3):
                led.on(); buzzer.on(); time.sleep(0.1)
                led.off(); buzzer.off(); time.sleep(0.1)
        except Exception:
            logging.exception("Hardware control failed during deny")
    else:
        logging.info("🚫 [SIMULATED] Access denied beeps")
    record_access_event("FACE", identifier, False)

def download_image_to_temp(url_or_s3, bucket=None, key=None):
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    try:
        if url_or_s3 and url_or_s3.lower().startswith('http'):
            r = requests.get(url_or_s3, timeout=30)
            r.raise_for_status()
            with open(tmp.name, 'wb') as f:
                f.write(r.content)
            return tmp.name

        if (url_or_s3 and url_or_s3.lower().startswith('s3://')) or (bucket and key):
            if url_or_s3 and url_or_s3.lower().startswith('s3://'):
                s3_no_scheme = url_or_s3[5:]
                parts = s3_no_scheme.split('/', 1)
                if len(parts) != 2:
                    raise ValueError("Invalid s3 URL; expected s3://bucket/key")
                bucket, key = parts[0], parts[1]
            if not bucket or not key:
                raise ValueError("Missing bucket or key for S3 download")

            s3_client = boto3.client(
                's3',
                region_name=os.environ.get('AWS_REGION', 'us-east-1'),
                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                config=BotoConfig(retries={'max_attempts': 3, 'mode': 'standard'})
            )
            s3_client.download_file(bucket, key, tmp.name)
            return tmp.name

        raise ValueError("Unsupported URL scheme; expected http(s) or s3://")
    except (BotoCoreError, ClientError, requests.RequestException, Exception):
        logging.exception("Failed downloading image: %s", url_or_s3 or f"s3://{bucket}/{key}")
        try:
            os.remove(tmp.name)
        except Exception:
            pass
        raise


face_recognition_lock = threading.Lock()
def process_registration(name, image_urls):
    logging.info("Processing registration for %s (%d images)", name, len(image_urls))
    encodings = []
    for url in image_urls:
        try:
            tmpname = download_image_to_temp(url)
            image = face_recognition.load_image_file(tmpname)
            
            # ADD LOCK HERE to prevent concurrent face_recognition calls
            with face_recognition_lock:
                face_locations = face_recognition.face_locations(image, model='hog', number_of_times_to_upsample=FACE_UPSAMPLE)
                face_encs = face_recognition.face_encodings(image, face_locations)
            
            if face_encs:
                encodings.append(face_encs[0])
            try: os.remove(tmpname)
            except Exception: pass
        except Exception:
            logging.exception("Failed processing image %s", url)
            continue
    if not encodings:
        raise ValueError("No faces detected in provided images")

    encodings = [np.array(e, dtype=np.float64) for e in encodings]
    with known_faces_lock:
        known_faces[name] = known_faces.get(name, []) + encodings
    save_known_faces()
    logging.info("Saved known face for %s", name)
    return len(encodings)

def delete_known_face(user_id: str):
    with known_faces_lock:
        if user_id in known_faces:
            try:
                del known_faces[user_id]
                save_known_faces()
                logging.info("Deleted known face for %s", user_id)
                return True
            except Exception:
                logging.exception("Failed to delete known face for %s", user_id)
                return False
    return False

# === Flask endpoints ===
@app.route('/')
def index():
    html = """
    <!doctype html>
    <html>
    <head>
      <title>Face Service - Live Stream</title>
      <style>
        body { font-family: -apple-system, Roboto, "Segoe UI", Arial; background:#111; color:#eee; text-align:center; }
        .container { max-width:900px; margin: 12px auto; }
        .stream { border-radius:8px; overflow:hidden; display:inline-block; box-shadow: 0 8px 24px rgba(0,0,0,0.6); }
        .status { margin-top:10px; text-align:left; background:#151515; padding:12px; border-radius:8px; }
        .badge { display:inline-block; padding:4px 8px; border-radius:6px; background:#222; margin-right:8px; }
        a { color:#66d9ef; text-decoration:none; }
        .perf { color:#aaa; font-size:0.9em; }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>📷 Face Recognition - Live Stream</h1>
        <div class="stream">
          <img src="{{ url_for('video_feed') }}" width="640" height="480" />
        </div>
        <div class="status">
          <div><span class="badge">Service:</span> face_recognition</div>
          <div><span class="badge">Running:</span> {{ 'Yes' if running else 'No' }}</div>
          <div><span class="badge">Known Faces:</span> {{ face_count }}</div>
          <div><span class="badge">Last Identified:</span> {{ last_identified.user_id or '—' }} @ {{ last_identified.timestamp or '—' }}</div>
          <div class="perf"><span class="badge">Performance:</span> Processing every {{ process_n }}th frame at {{ scale }}x scale</div>
          <div class="perf"><span class="badge">Threshold:</span> {{ threshold }} ({{ threshold_level }})</div>
          <div style="margin-top:8px;"><a href="/known">/known</a> • <a href="/known/details">/known/details</a> • <a href="/access_log">/access_log</a> • <a href="/threshold">/threshold</a></div>
          <div style="margin-top:12px;">
            <h3>Adjust Recognition Threshold</h3>
            <input type="range" id="thresholdSlider" min="0.3" max="0.7" step="0.05" value="{{ threshold }}" style="width:300px;">
            <span id="thresholdValue">{{ threshold }}</span>
            <button onclick="updateThreshold()" style="margin-left:10px; padding:5px 15px; cursor:pointer;">Update</button>
            <div style="font-size:0.85em; color:#888; margin-top:5px;">
              Lower = Stricter (fewer false matches) | Higher = Looser (easier to match)
            </div>
          </div>
        </div>
      </div>
      <script>
        const slider = document.getElementById('thresholdSlider');
        const valueDisplay = document.getElementById('thresholdValue');
        
        slider.addEventListener('input', function() {
          valueDisplay.textContent = this.value;
        });
        
        async function updateThreshold() {
          const newThreshold = parseFloat(slider.value);
          try {
            const response = await fetch('/threshold', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({threshold: newThreshold})
            });
            const data = await response.json();
            if (response.ok) {
              alert('Threshold updated to ' + newThreshold + '\\n' + data.note);
              location.reload();
            } else {
              alert('Error: ' + data.error);
            }
          } catch (error) {
            alert('Failed to update threshold: ' + error);
          }
        }
      </script>
    </body>
    </html>
    """
    with known_faces_lock:
        face_count = len(known_faces)
    
    threshold_level = 'strict' if RECOGNITION_TOLERANCE < 0.45 else 'moderate' if RECOGNITION_TOLERANCE < 0.55 else 'loose'
    
    return render_template_string(html, running=face_recognition_enabled, face_count=face_count, 
                                   last_identified=last_identified, process_n=PROCESS_EVERY_N_FRAMES, 
                                   scale=VIDEO_SCALE, threshold=RECOGNITION_TOLERANCE, threshold_level=threshold_level)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    name = data.get("name")
    image_urls = data.get("image_urls", [])
    if not name or not image_urls:
        return jsonify({"error": "Missing name or image_urls"}), 400
    try:
        count = process_registration(name, image_urls)
        return jsonify({"message": f"Registered {count} images for {name}"}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception:
        logging.exception("Registration failed")
        return jsonify({"error": "Server error"}), 500

@app.route('/delete_user/<user_id>', methods=['DELETE'])
def delete_user_route(user_id):
    logging.info('Received delete_user for %s', user_id)
    threading.Thread(target=lambda: delete_known_face(user_id), daemon=True).start()
    return jsonify({'accepted': True}), 202

@app.route('/known', methods=['GET'])
def known():
    with known_faces_lock:
        guests = list(known_faces.keys())
    return jsonify({"guests": guests})

@app.route('/known/details', methods=['GET'])
def known_details():
    with known_faces_lock:
        details = {uid: len(encs) for uid, encs in known_faces.items()}
    return jsonify({'details': details})

@app.route('/reload_known', methods=['POST'])
def reload_known_route():
    load_known_faces()
    return jsonify({'reloaded': True}), 200

@app.route('/status', methods=['GET'])
def status():
    with known_faces_lock:
        face_count = len(known_faces)
    return jsonify({
        'service': 'face_recognition',
        'running': face_recognition_enabled,
        'known_faces_count': face_count,
        'hardware_available': HAS_GPIO,
        'process_start_time': PROCESS_START_TIME,
        'process_every_n_frames': PROCESS_EVERY_N_FRAMES,
        'video_scale': VIDEO_SCALE,
        'recognition_tolerance': RECOGNITION_TOLERANCE
    })

@app.route('/last_identified', methods=['GET'])
def last_identified_route():
    with last_identified_lock:
        return jsonify(last_identified)

@app.route('/access_log', methods=['GET'])
def access_log():
    limit = int(request.args.get('limit', 50))
    with access_events_lock:
        events = access_events[-limit:]
    return jsonify({'events': events, 'count': len(events)})

@app.route('/threshold', methods=['GET', 'POST'])
def threshold_route():
    """Get or update the face recognition threshold."""
    global RECOGNITION_TOLERANCE
    
    if request.method == 'POST':
        data = request.get_json() or {}
        new_threshold = data.get('threshold')
        
        if new_threshold is None:
            return jsonify({'error': 'Missing threshold value'}), 400
        
        try:
            new_threshold = float(new_threshold)
            if new_threshold < 0 or new_threshold > 1:
                return jsonify({'error': 'Threshold must be between 0 and 1'}), 400
            
            RECOGNITION_TOLERANCE = new_threshold
            logging.info("✅ Updated recognition threshold to %.3f", new_threshold)
            return jsonify({
                'message': 'Threshold updated successfully',
                'threshold': RECOGNITION_TOLERANCE,
                'note': 'Lower = stricter matching (less false positives), Higher = looser matching (more false positives)'
            }), 200
            
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid threshold value'}), 400
    
    # GET request - return current threshold
    return jsonify({
        'threshold': RECOGNITION_TOLERANCE,
        'recommended_range': '0.4 - 0.6',
        'current_setting': 'strict' if RECOGNITION_TOLERANCE < 0.45 else 'moderate' if RECOGNITION_TOLERANCE < 0.55 else 'loose',
        'note': 'Lower = stricter (fewer false positives), Higher = looser (more false positives)'
    })

@app.route('/notify', methods=['POST'])
def notify():
    data = request.get_json() or {}
    event = data.get('event')
    user_id = data.get('user_id') or data.get('name')
    if event == 'new_face_registered':
        image_urls = data.get('image_urls') or []
        if not user_id or not image_urls:
            return jsonify({'error': 'missing user_id or image_urls'}), 400
        threading.Thread(target=lambda: process_registration(user_id, image_urls), daemon=True).start()
        return jsonify({'message': 'processing'}), 202
    elif event == 'face_deleted':
        if not user_id:
            return jsonify({'error': 'missing user_id'}), 400
        threading.Thread(target=lambda: delete_known_face(user_id), daemon=True).start()
        return jsonify({'message': 'deletion processing'}), 202
    else:
        return jsonify({'message': 'ignored'}), 200

# === OPTIMIZED camera and processing ===
def init_camera():
    """Initialize cv2 VideoCapture with optimizations."""
    global camera
    with camera_lock:
        if camera is None:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            
            # NEW: Optimize camera settings
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce lag
            
            if not cap.isOpened():
                logging.error("❌ Cannot open camera device")
                camera = None
            else:
                camera = cap
                logging.info("✅ Camera initialized (w=%d h=%d fps=%d)", VIDEO_WIDTH, VIDEO_HEIGHT, CAMERA_FPS)
        return camera

def release_camera():
    global camera
    with camera_lock:
        if camera:
            try:
                camera.release()
            except Exception:
                pass
            camera = None

def encode_frame_to_jpeg(frame):
    # NEW: Faster JPEG encoding with lower quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # 85% quality = faster
    ret, buffer = cv2.imencode('.jpg', frame, encode_param)
    return buffer.tobytes() if ret else None

def match_face(face_encoding):
    with known_faces_lock:
        items = list(known_faces.items())
    best_name = None
    best_dist = None
    for name, enc_list in items:
        encs = [np.array(e, dtype=np.float64) for e in enc_list]
        if not encs:
            continue
        distances = face_recognition.face_distance(encs, face_encoding)
        if len(distances) == 0:
            continue
        min_dist = float(np.min(distances))
        if best_dist is None or min_dist < best_dist:
            best_dist = min_dist
            best_name = name
    if best_dist is not None and best_dist <= RECOGNITION_TOLERANCE:
        return best_name, best_dist
    return None, None

def process_faces_thread():
    """
    Background thread that processes face recognition at a reduced rate.
    Updates shared last_processed_results which the video stream uses.
    """
    frame_count = 0
    while True:
        time.sleep(0.05)  # Don't hog CPU
        
        # Get current frame
        with current_frame_lock:
            if current_frame is None:
                continue
            frame_to_process = current_frame.copy()
        
        frame_count += 1
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            continue  # Skip this frame
        
        try:
            # Resize for faster processing
            small_frame = cv2.resize(frame_to_process, (0, 0), fx=VIDEO_SCALE, fy=VIDEO_SCALE)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = []
            face_names = []
            
            if face_recognition_enabled:
                with face_recognition_lock:
                    face_locations = face_recognition.face_locations(rgb_small, model='hog', number_of_times_to_upsample=FACE_UPSAMPLE)
                    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
                
                for face_encoding in face_encodings:
                    matched_name, matched_dist = match_face(face_encoding)
                    if matched_name:
                        face_names.append((matched_name, matched_dist))
                        if _should_notify(matched_name):
                            _record_identified(matched_name, matched_dist)
                            threading.Thread(target=grant_access, args=(matched_name,), daemon=True).start()
                    else:
                        face_names.append((None, None))
            
            # Update shared results
            with last_processed_lock:
                last_processed_results['locations'] = face_locations
                last_processed_results['names'] = face_names
                
        except Exception:
            logging.exception("Error in face processing thread")

def gen_frames():
    """
    Optimized generator that yields frames quickly.
    Face detection happens in separate thread at reduced rate.
    """
    global current_frame
    
    cam = init_camera()
    if cam is None:
        blank = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        cv2.putText(blank, "No camera available", (20, VIDEO_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        jpeg = encode_frame_to_jpeg(blank)
        if jpeg:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        return

    while True:
        # Capture frame
        with camera_lock:
            ret, frame = cam.read()
        
        if not ret or frame is None:
            logging.warning("⚠️ Failed to read frame from camera")
            time.sleep(0.1)
            continue
        
        # Update shared frame for processing thread
        with current_frame_lock:
            current_frame = frame.copy()
        
        # Draw boxes from last processed results
        with last_processed_lock:
            face_locations = last_processed_results['locations']
            face_names = last_processed_results['names']
        
        # Draw rectangles and labels
        for (top, right, bottom, left), (name, dist) in zip(face_locations, face_names):
            # Scale back to original size
            top = int(top / VIDEO_SCALE)
            right = int(right / VIDEO_SCALE)
            bottom = int(bottom / VIDEO_SCALE)
            left = int(left / VIDEO_SCALE)
            
            if name:
                label = f"{name} ({dist:.2f})"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (w, h), _ = cv2.getTextSize(label, font, font_scale, 1)
            cv2.rectangle(frame, (left, top - h - 10), (left + w + 4, top), color, -1)
            cv2.putText(frame, label, (left + 2, top - 4), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw timestamp
        cv2.putText(frame, datetime.datetime.utcnow().isoformat(timespec='seconds'),
                    (10, VIDEO_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        
        # Encode and yield
        jpeg = encode_frame_to_jpeg(frame)
        if jpeg:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')

# === MQTT (unchanged) ===
def _handle_face_event(payload: dict):
    event = payload.get('event')
    user_id = payload.get('user_id') or payload.get('name')
    image_urls = payload.get('image_urls') or []
    if event == 'new_face_registered':
        if not user_id or not image_urls:
            logging.warning("new_face_registered missing user_id or image_urls")
            return
        logging.info("MQTT: new_face_registered for %s (%d images)", user_id, len(image_urls))
        threading.Thread(target=lambda: process_registration(user_id, image_urls), daemon=True).start()
    elif event == 'face_deleted':
        if not user_id:
            logging.warning("face_deleted missing user_id")
            return
        logging.info("MQTT: face_deleted for %s", user_id)
        threading.Thread(target=lambda: delete_known_face(user_id), daemon=True).start()
    else:
        logging.info("MQTT: ignored event %s", event)

def start_mqtt_listener():
    broker = os.environ.get('MQTT_BROKER_URL')
    port = int(os.environ.get('MQTT_PORT', '8883'))
    username = os.environ.get('MQTT_USERNAME')
    password = os.environ.get('MQTT_PASSWORD')
    topic_faces = os.environ.get('MQTT_TOPIC', 'wildwaves/faces')

    if not broker:
        logging.info("MQTT disabled (no broker URL)")
        return

    client_id = f"face-service-{os.getpid()}-{int(time.time())}"
    client = mqtt.Client(client_id=client_id, clean_session=True)

    if username and password:
        client.username_pw_set(username, password)

    client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    client.tls_insecure_set(False)

    def on_connect(c, userdata, flags, rc):
        if rc == 0:
            logging.info("✅ MQTT connected")
            try:
                c.subscribe(topic_faces, qos=1)
                logging.info("MQTT subscribed to %s", topic_faces)
            except Exception:
                logging.exception("Failed subscribing to %s", topic_faces)
        else:
            logging.error("❌ MQTT connect failed rc=%s", rc)

    def on_message(c, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
        except Exception:
            logging.exception("MQTT message JSON decode failed on topic %s", msg.topic)
            return
        try:
            _handle_face_event(payload)
        except Exception:
            logging.exception("Error handling MQTT face event")

    client.on_connect = on_connect
    client.on_message = on_message

    def _run():
        while True:
            try:
                logging.info("Connecting to MQTT %s:%d ...", broker, port)
                client.connect(broker, port=port, keepalive=60)
                client.loop_forever()
            except Exception:
                logging.exception("MQTT connection loop error; retrying in 5s")
                time.sleep(5)

    threading.Thread(target=_run, daemon=True).start()

# === Main startup ===
if __name__ == '__main__':
    try:
        load_known_faces()
        logging.info('='*60)
        logging.info('🚀 FACE RECOGNITION SERVICE STARTING (OPTIMIZED)')
        logging.info('Start time: %s', PROCESS_START_TIME)
        logging.info('Face Recognition Enabled: %s', face_recognition_enabled)
        logging.info('Known faces: %d', len(known_faces))
        logging.info('Camera size: %dx%d @ %d fps', VIDEO_WIDTH, VIDEO_HEIGHT, CAMERA_FPS)
        logging.info('Processing every %d frames at %.2fx scale', PROCESS_EVERY_N_FRAMES, VIDEO_SCALE)
        logging.info('='*60)

        init_camera()
        
        # Start background face processing thread
        threading.Thread(target=process_faces_thread, daemon=True).start()
        logging.info("✅ Background face processing thread started")
        
        start_mqtt_listener()

        app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
    finally:
        release_camera()