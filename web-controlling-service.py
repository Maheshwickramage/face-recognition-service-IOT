#!/usr/bin/env python3
"""
Web API Service for Raspberry Pi
Handles HTTP API requests, MQTT notifications, and hardware control via web app
Does NOT include camera (face recognition) or RFID scanning (those are separate services)
"""
from flask import Flask, request, jsonify
import ssl
import boto3
import os
import pickle
import tempfile
import threading
import time
import logging
import json
import requests
import faulthandler
import numpy as np
import datetime
from dotenv import load_dotenv

# Try importing hardware modules
try:
    from gpiozero import Servo
    HAS_GPIO = True
except Exception:
    Servo = None
    HAS_GPIO = False

try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

# Load environment variables
load_dotenv()
faulthandler.enable()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Flask app
app = Flask(__name__)
KNOWN_FACES_FILE = os.environ.get("KNOWN_FACES_FILE", "known_faces.pkl")
PROCESS_START_TIME = datetime.datetime.now(datetime.timezone.utc).isoformat()

# Load known faces if exists
if os.path.exists(KNOWN_FACES_FILE):
    with open(KNOWN_FACES_FILE, "rb") as f:
        known_faces = pickle.load(f)
else:
    known_faces = {}

known_faces_lock = threading.RLock()

# AWS S3 client
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
s3 = boto3.client('s3', region_name=AWS_REGION)

# ==================== SERVO CONFIGURATION ====================
# Map itemCode to servo GPIO pins
SERVO_PINS = {
    "1": int(os.environ.get('SERVO_PIN_1', '27')),
    "D933": int(os.environ.get('SERVO_PIN_2', '17')),  # ItemCode 2 → GPIO 17
    "3": int(os.environ.get('SERVO_PIN_3', '22')),
    "4": int(os.environ.get('SERVO_PIN_4', '5')),
    "5": int(os.environ.get('SERVO_PIN_5', '6')),
    "6": int(os.environ.get('SERVO_PIN_6', '13')),
    "7": int(os.environ.get('SERVO_PIN_7', '19')),
    "8": int(os.environ.get('SERVO_PIN_8', '26')),
}

# Dictionary to store servo objects
servos = {}  # {itemCode: Servo object}

# Initialize servos
if HAS_GPIO:
    try:
        logging.info('=' * 60)
        logging.info('🔧 INITIALIZING SERVOS')
        logging.info('=' * 60)
        
        for item_code, servo_pin in SERVO_PINS.items():
            try:
                servos[item_code] = Servo(
                    servo_pin, 
                    min_pulse_width=0.5/1000, 
                    max_pulse_width=2.5/1000
                )
                servos[item_code].value = -1  # Start in closed position
                logging.info('✅ ItemCode "%s" → GPIO Pin %d (READY)', item_code, servo_pin)
            except Exception as e:
                logging.error('❌ Failed to initialize servo for itemCode "%s" on pin %d: %s', 
                            item_code, servo_pin, e)
        
        logging.info('=' * 60)
        logging.info('✅ Initialized %d servos successfully', len(servos))
        logging.info('=' * 60)
    except Exception as e:
        logging.error('❌ Servo initialization failed: %s', e)
        HAS_GPIO = False
else:
    logging.warning('⚠️  Running without GPIO (development/testing mode)')

# Access events tracking
access_events = []
access_events_lock = threading.Lock()

# ==================== DOOR CONTROL FUNCTION ====================
def control_door(item_code, lock_status, door_name="", room_name=""):
    """
    Control a specific door servo based on itemCode
    Uses 90-degree rotation: 0° (closed) to 90° (open)
    """
    # Convert to string to ensure consistency
    item_code = str(item_code)
    
    logging.info("=" * 70)
    logging.info("🚪 DOOR CONTROL REQUEST")
    logging.info("ItemCode: %s | Door: %s | Room: %s | Action: %s", 
                 item_code, door_name or "N/A", room_name or "N/A", 
                 "UNLOCK" if lock_status else "LOCK")
    logging.info("=" * 70)
    
    # Check if we have a servo pin for this itemCode
    if item_code not in SERVO_PINS:
        logging.error('❌ NO SERVO PIN for itemCode "%s"', item_code)
        logging.error('Available itemCodes: %s', list(SERVO_PINS.keys()))
        return False
    
    servo_pin = SERVO_PINS[item_code]
    logging.info("🎯 Selected: ItemCode '%s' on GPIO Pin %d", item_code, servo_pin)
    
    try:
        if lock_status:
            # UNLOCK - Open the door (90-degree rotation)
            logging.info("🔓 UNLOCKING door (itemCode: %s, pin: %d)", item_code, servo_pin)
            
            if HAS_GPIO:
                # Close old servo if exists
                if item_code in servos:
                    try:
                        servos[item_code].close()
                    except:
                        pass
                
                # Re-initialize servo for clean state
                logging.info("⚙️  Re-initializing servo")
                servo = Servo(
                    servo_pin, 
                    min_pulse_width=0.5/1000, 
                    max_pulse_width=2.5/1000
                )
                servos[item_code] = servo
                
                # Start at 0° (closed position)
                logging.info("⚙️  Setting to CLOSED position (0°, value = -1)")
                servo.value = -1  # 0° position
                time.sleep(0.8)
                
                # Rotate 90° (open position)
                logging.info("⚙️  Opening door - Rotating 90° (value = 0)")
                servo.value = 0  # 90° position (middle of servo range)
                
                logging.info("⏳ Door open for 5 seconds...")
                time.sleep(5)
                
                # Return to 0° (closed position)
                logging.info("⚙️  Closing door - Returning to 0° (value = -1)")
                servo.value = -1  # Back to 0°
                time.sleep(0.8)
                
                # Cleanup
                logging.info("⚙️  Detaching servo")
                servo.detach()
                
                logging.info("✅ Door cycle complete (90° rotation)")
            else:
                logging.info("🔓 [SIMULATION] Door unlocked (90° rotation)")
                time.sleep(5)
                logging.info("🔒 [SIMULATION] Door locked")
        else:
            # LOCK - Ensure door is at 0° (closed)
            logging.info("🔒 LOCKING door")
            
            if HAS_GPIO:
                # Close old servo if exists
                if item_code in servos:
                    try:
                        servos[item_code].close()
                    except:
                        pass
                
                # Re-initialize
                servo = Servo(
                    servo_pin, 
                    min_pulse_width=0.5/1000, 
                    max_pulse_width=2.5/1000
                )
                servos[item_code] = servo
                
                # Lock at 0°
                logging.info("⚙️  Locking at 0° (value = -1)")
                servo.value = -1  # 0° position
                time.sleep(0.8)
                servo.detach()
                
                logging.info("✅ Door locked (0° position)")
            else:
                logging.info("🔒 [SIMULATION] Door locked")
        
        # Record the event
        record_access_event("WEB_APP", f"{door_name} ({room_name}) - ItemCode {item_code}", lock_status)
        
        logging.info("=" * 70)
        return True
        
    except Exception as e:
        logging.exception('❌ FAILED to control servo: %s', e)
        logging.info("=" * 70)
        return False
    """
    Control a specific door servo based on itemCode
    
    Args:
        item_code: The door identifier (e.g., "1", "2", "3")
        lock_status: True to unlock, False to lock
        door_name: Name of the door (for logging)
        room_name: Name of the room (for logging)
    """
    # Convert to string to ensure consistency
    item_code = str(item_code)
    
    logging.info("=" * 70)
    logging.info("🚪 DOOR CONTROL REQUEST")
    logging.info("ItemCode: %s | Door: %s | Room: %s | Action: %s", 
                 item_code, door_name or "N/A", room_name or "N/A", 
                 "UNLOCK" if lock_status else "LOCK")
    logging.info("=" * 70)
    
    # Check if we have a servo for this itemCode
    if item_code not in servos:
        logging.error('❌ NO SERVO FOUND for itemCode "%s"', item_code)
        logging.error('Available itemCodes: %s', list(servos.keys()))
        return False
    
    servo = servos[item_code]
    servo_pin = SERVO_PINS.get(item_code, 0)
    
    logging.info("🎯 Selected: ItemCode '%s' on GPIO Pin %d", item_code, servo_pin)
    
    try:
        if lock_status:
            # UNLOCK - Open the door
            logging.info("🔓 UNLOCKING door (itemCode: %s, pin: %d)", item_code, servo_pin)
            
            if HAS_GPIO:
                # Step 1: Ensure we start from closed position
                logging.info("⚙️  Step 1: Ensuring CLOSED position (value = -1)")
                servo.value = -1
                time.sleep(0.5)  # Give servo time to reach position
                
                # Step 2: Move to neutral/mid position briefly
                logging.info("⚙️  Step 2: Moving to NEUTRAL position (value = None)")
                servo.mid()  # or servo.value = None to detach
                time.sleep(0.3)
                
                # Step 3: Open the door
                logging.info("⚙️  Step 3: Setting servo to OPEN position (value = 1)")
                servo.value = 1  # Try 1 instead of 0 for full range
                
                logging.info("⏳ Door will remain open for 5 seconds...")
                time.sleep(5)
                
                # Step 4: Return to neutral briefly
                logging.info("⚙️  Step 4: Moving to NEUTRAL position")
                servo.mid()
                time.sleep(0.3)
                
                # Step 5: Close the door
                logging.info("⚙️  Step 5: Setting servo to CLOSED position (value = -1)")
                servo.value = -1
                time.sleep(0.5)  # Ensure it reaches closed position
                
                # Step 6: Detach servo to prevent jitter
                logging.info("⚙️  Step 6: Detaching servo to prevent jitter")
                servo.detach()
                
                logging.info("✅ Door cycle complete for itemCode %s", item_code)
            else:
                logging.info("🔓 [SIMULATION] Door unlocked (GPIO not available)")
                time.sleep(5)
                logging.info("🔒 [SIMULATION] Door locked")
        else:
            # LOCK - Ensure door is closed
            logging.info("🔒 LOCKING door (itemCode: %s, pin: %d)", item_code, servo_pin)
            
            if HAS_GPIO:
                # Ensure servo is attached
                servo.min()  # Re-attach if detached
                time.sleep(0.3)
                
                logging.info("⚙️  Setting servo to CLOSED position (value = -1)")
                servo.value = -1
                time.sleep(0.5)
                
                # Detach to prevent jitter
                servo.detach()
                logging.info("✅ Door locked for itemCode %s", item_code)
            else:
                logging.info("🔒 [SIMULATION] Door locked (GPIO not available)")
        
        # Record the event
        record_access_event("WEB_APP", f"{door_name} ({room_name}) - ItemCode {item_code}", lock_status)
        
        logging.info("=" * 70)
        return True
        
    except Exception as e:
        logging.exception('❌ FAILED to control servo for itemCode %s: %s', item_code, e)
        logging.info("=" * 70)
        return False
def record_access_event(method, identifier, granted):
    """Record access attempt for audit log"""
    event = {
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'method': method,
        'identifier': identifier,
        'granted': granted
    }
    with access_events_lock:
        access_events.append(event)
        if len(access_events) > 100:
            access_events.pop(0)

# ==================== KNOWN FACES MANAGEMENT ====================
def save_known_faces():
    """Persist known_faces safely to disk"""
    try:
        with known_faces_lock:
            tmp_path = KNOWN_FACES_FILE + ".tmp"
            with open(tmp_path, 'wb') as f:
                pickle.dump(known_faces, f)
            try:
                os.replace(tmp_path, KNOWN_FACES_FILE)
            except Exception:
                os.rename(tmp_path, KNOWN_FACES_FILE)
            logging.info("✅ Saved known faces (%d)", len(known_faces))
    except Exception:
        logging.exception('Failed to save known faces')

def load_known_faces():
    """Load known_faces from disk if present"""
    global known_faces
    try:
        with known_faces_lock:
            if os.path.exists(KNOWN_FACES_FILE):
                with open(KNOWN_FACES_FILE, 'rb') as f:
                    known_faces = pickle.load(f)
                    logging.info('✅ Loaded %d known faces', len(known_faces))
            else:
                known_faces = {}
    except Exception:
        logging.exception('Failed to load known faces')
        known_faces = {}

def download_image_to_temp(url_or_s3, bucket=None, key=None):
    """Download from http(s) or s3. Returns temp filename."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    try:
        if url_or_s3 and url_or_s3.lower().startswith('http'):
            r = requests.get(url_or_s3, timeout=30)
            r.raise_for_status()
            with open(tmp.name, 'wb') as f:
                f.write(r.content)
            return tmp.name
        elif url_or_s3 and url_or_s3.lower().startswith('s3://'):
            path = url_or_s3.replace('s3://', '').split('/', 1)
            if len(path) == 2:
                b, k = path
                s3.download_file(b, k, tmp.name)
            else:
                raise ValueError('Invalid s3 url')
        elif bucket and key:
            s3.download_file(bucket, key, tmp.name)
        else:
            raise ValueError('Unsupported url')
        return tmp.name
    except Exception:
        try:
            os.remove(tmp.name)
        except Exception:
            pass
        raise

def process_registration(name, image_urls):
    """Download images, extract face encodings, store them"""
    import face_recognition
    logging.info('Processing registration for %s (%d images)', name, len(image_urls))
    
    encodings = []
    for url in image_urls:
        try:
            tmpname = download_image_to_temp(url)
            image = face_recognition.load_image_file(tmpname)
            upsample = int(os.environ.get('FACE_UPSAMPLE', '0'))
            face_locations = face_recognition.face_locations(image, model='hog', number_of_times_to_upsample=upsample)
            face_encs = face_recognition.face_encodings(image, face_locations)
            
            if face_encs:
                encodings.append(face_encs[0])
            
            try:
                os.remove(tmpname)
            except Exception:
                pass
        except Exception:
            logging.exception('Failed processing image %s', url)
            continue
    
    if not encodings:
        raise ValueError('No faces detected in provided images')
    
    encodings = [np.array(e, dtype=np.float64) for e in encodings]
    with known_faces_lock:
        known_faces[name] = encodings
    save_known_faces()
    logging.info('✅ Saved known face for %s', name)
    return len(encodings)

def delete_known_face(user_id: str):
    """Remove user's encodings from known_faces"""
    with known_faces_lock:
        if user_id in known_faces:
            try:
                del known_faces[user_id]
                save_known_faces()
                logging.info('✅ Deleted known face for %s', user_id)
                return True
            except Exception:
                logging.exception('Failed to delete known face for %s', user_id)
                return False
    logging.info('No known face entry for %s', user_id)
    return False

# ==================== FLASK ROUTES ====================
@app.route('/register', methods=['POST'])
def register():
    """Register a guest with face images from S3."""
    data = request.get_json()
    name = data.get("name")
    image_urls = data.get("image_urls", [])

    if not name or not image_urls:
        return jsonify({"error": "Missing name or image URLs"}), 400

    try:
        count = process_registration(name, image_urls)
        return jsonify({"message": f"Registered {count} images for {name}"}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception:
        logging.exception('Registration failed')
        return jsonify({"error": "Server error"}), 500

@app.route('/delete_user/<user_id>', methods=['DELETE'])
def delete_user_route(user_id):
    """Delete user face from database"""
    logging.info('Received HTTP delete_user for %s', user_id)
    threading.Thread(target=lambda: delete_known_face(user_id), daemon=True).start()
    return jsonify({'accepted': True}), 202

@app.route('/known', methods=['GET'])
def known():
    """Get list of known faces"""
    with known_faces_lock:
        guests = list(known_faces.keys())
    return jsonify({"guests": guests})

@app.route('/known/details', methods=['GET'])
def known_details():
    """Get detailed list of known faces"""
    with known_faces_lock:
        details = {uid: len(encs) for uid, encs in known_faces.items()}
    return jsonify({'details': details})

@app.route('/reload_known', methods=['POST'])
def reload_known_route():
    """Reload known faces from disk"""
    logging.info('Reloading known faces from disk')
    load_known_faces()
    return jsonify({'reloaded': True}), 200

@app.route('/status', methods=['GET'])
def status():
    """Get service status"""
    with known_faces_lock:
        face_count = len(known_faces)
    
    return jsonify({
        'service': 'web_api',
        'known_faces_count': face_count,
        'hardware_available': HAS_GPIO,
        'servos_initialized': len(servos),
        'available_item_codes': list(servos.keys()),
        'process_start_time': PROCESS_START_TIME
    })

@app.route('/access_log', methods=['GET'])
def access_log():
    """Return recent access events"""
    limit = int(request.args.get('limit', 50))
    with access_events_lock:
        events = access_events[-limit:]
    return jsonify({'events': events, 'count': len(events)})

@app.route('/notify', methods=['POST'])
def notify():
    """Webhook endpoint for MQTT notifications"""
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

@app.route('/test_servo/<item_code>', methods=['POST'])
def test_servo(item_code):
    """Test endpoint to manually trigger a servo"""
    logging.info('🧪 TEST: Manual servo test requested for itemCode: %s', item_code)
    
    result = control_door(
        item_code=item_code,
        lock_status=True,
        door_name=f"Test Door {item_code}",
        room_name="Test Room"
    )
    
    if result:
        return jsonify({'success': True, 'message': f'Servo {item_code} tested'}), 200
    else:
        return jsonify({'success': False, 'error': 'Servo control failed'}), 500

# ==================== MQTT LISTENER ====================
def on_mqtt_message(client, userdata, msg):
    """Handle incoming MQTT messages"""
    try:
        logging.info('=' * 70)
        logging.info('📨 MQTT MESSAGE RECEIVED')
        logging.info('Topic: %s', msg.topic)
        
        payload = json.loads(msg.payload.decode())
        logging.info('Payload: %s', json.dumps(payload, indent=2))
        
        event = payload.get('event')
        user_id = payload.get('user_id')
        
        if event == 'new_face_registered':
            image_urls = payload.get('image_urls', [])
            if user_id and image_urls:
                logging.info('👤 Processing face registration for: %s (%d images)', user_id, len(image_urls))
                threading.Thread(target=lambda: process_registration(user_id, image_urls), daemon=True).start()
            else:
                logging.warning('⚠️ Missing user_id or image_urls')
                
        elif event == 'face_deleted':
            if user_id:
                logging.info('🗑️ Processing face deletion for: %s', user_id)
                threading.Thread(target=lambda: delete_known_face(user_id), daemon=True).start()
            else:
                logging.warning('⚠️ Missing user_id')
                
        elif event == 'door_lock_changed':
            # Extract door information
            door_id = payload.get('door_id')
            lock_status = payload.get('lock_status')
            door_name = payload.get('door_name', 'Unknown Door')
            room_name = payload.get('room_name', 'Unknown Room')
            item_code = payload.get('item_code') or payload.get('itemCode')
            
            if item_code is not None and lock_status is not None:
                # Control the door in a separate thread
                threading.Thread(
                    target=lambda: control_door(item_code, lock_status, door_name, room_name),
                    daemon=True
                ).start()
            else:
                logging.warning('⚠️ Missing item_code or lock_status in door_lock_changed event')
                
        else:
            logging.info('ℹ️ Event ignored: %s', event)
            
        logging.info('=' * 70)
        
    except json.JSONDecodeError:
        logging.exception('❌ Failed to parse MQTT message as JSON')
    except Exception:
        logging.exception('❌ Failed to handle MQTT message')

def start_mqtt_listener():
    """Start MQTT listener"""
    broker_url = os.environ.get('MQTT_BROKER_URL') or os.environ.get('MQTT_BROKER')
    topic = os.environ.get('MQTT_TOPIC', 'wildwaves/faces')
    
    if broker_url and ':' in broker_url:
        broker_host, broker_port = broker_url.rsplit(':', 1)
        broker_port = int(broker_port)
    else:
        broker_host = broker_url
        broker_port = int(os.environ.get('MQTT_PORT', '8883'))
    
    logging.info('=' * 70)
    logging.info('🔌 MQTT CONFIGURATION')
    logging.info('Broker: %s:%d', broker_host, broker_port)
    logging.info('Topic: %s', topic)
    logging.info('=' * 70)
    
    if not broker_host or mqtt is None:
        logging.warning('⚠️ MQTT listener not started')
        return

    try:
        try:
            from paho.mqtt.client import CallbackAPIVersion
            client = mqtt.Client(CallbackAPIVersion.VERSION1)
        except (ImportError, AttributeError):
            client = mqtt.Client()
        
        username = os.environ.get('MQTT_USERNAME')
        password = os.environ.get('MQTT_PASSWORD')
        if username:
            client.username_pw_set(username, password)
        
        if broker_port == 8883:
            logging.info('🔒 Configuring TLS/SSL')
            client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS)
        
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                logging.info('=' * 70)
                logging.info('✅ MQTT CONNECTED')
                logging.info('=' * 70)
                client.subscribe(topic)
                logging.info('✅ SUBSCRIBED: %s', topic)
            else:
                error_messages = {
                    1: 'incorrect protocol version',
                    2: 'invalid client identifier',
                    3: 'server unavailable',
                    4: 'bad username or password',
                    5: 'not authorized'
                }
                logging.error('❌ MQTT FAILED: %s', error_messages.get(rc, f'Unknown code: {rc}'))
        
        def on_disconnect(client, userdata, rc):
            if rc != 0:
                logging.warning('⚠️ MQTT DISCONNECTED')
        
        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_message = on_mqtt_message
        
        logging.info('🔌 Connecting to MQTT broker...')
        client.connect(broker_host, broker_port, keepalive=60)
        client.loop_start()
        logging.info('✅ MQTT listener started')
        
    except Exception:
        logging.exception('❌ Failed to start MQTT listener')

# ==================== MAIN STARTUP ====================
if __name__ == '__main__':
    load_known_faces()

    logging.info('=' * 70)
    logging.info('🚀 WEB API SERVICE STARTING')
    logging.info('=' * 70)
    logging.info('Hardware GPIO: %s', '✅ AVAILABLE' if HAS_GPIO else '❌ NOT AVAILABLE')
    logging.info('Known Faces: %d', len(known_faces))
    logging.info('Servos Configured: %d', len(SERVO_PINS))
    logging.info('Servos Initialized: %d', len(servos))
    
    if servos:
        logging.info('Available ItemCodes:')
        for code, pin in SERVO_PINS.items():
            status = "✅" if code in servos else "❌"
            logging.info('  %s ItemCode "%s" → GPIO Pin %d', status, code, pin)
    
    logging.info('=' * 70)

    # Start MQTT listener
    start_mqtt_listener()

    # Start Flask server
    port = int(os.environ.get('WEB_CONTROL_PORT', '5003'))
    logging.info('🌐 Starting Flask API server on port %d', port)
    app.run(host='0.0.0.0', port=port, debug=False)