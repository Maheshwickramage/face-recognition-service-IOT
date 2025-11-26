#!/usr/bin/env python3
"""
RFID Scanner Service
Handles RFID card scanning and door access control
"""

from flask import Flask, request, jsonify
import ssl
import os
import threading
import time
import logging
import json
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# GPIO imports
try:
    import RPi.GPIO as GPIO
    from mfrc522 import SimpleMFRC522
    from gpiozero import Servo, LED, Buzzer, OutputDevice
    HAS_GPIO = True
except Exception as e:
    GPIO = None
    SimpleMFRC522 = None
    Servo = LED = Buzzer = OutputDevice = None
    HAS_GPIO = False

try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [RFID-SERVICE] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Flask App
app = Flask(__name__)
PROCESS_START_TIME = datetime.datetime.utcnow().isoformat()

# RFID Configuration
AUTHORIZED_UIDS = list(map(int, os.environ.get('AUTHORIZED_UIDS', '1027153469128').split(',')))

# GPIO Pins
SERVO_PIN = int(os.environ.get('SERVO_PIN', '17'))
LED_PIN = int(os.environ.get('LED_PIN', '22'))
BUZZER_PIN = int(os.environ.get('BUZZER_PIN', '23'))

# Hardware components
servo = None
led = None
buzzer = None
rfid_reader = None

if HAS_GPIO:
    try:
        GPIO.setwarnings(False)
        servo = Servo(SERVO_PIN, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
        servo.value = -1
        led = LED(LED_PIN)
        buzzer = Buzzer(BUZZER_PIN)
        rfid_reader = SimpleMFRC522()
        logging.info('✅ Hardware components initialized')
    except Exception as e:
        logging.error('❌ Hardware initialization failed: %s', e)
        HAS_GPIO = False
else:
    logging.warning('⚠️  Running without GPIO (development mode)')

# Access tracking
access_events = []
access_events_lock = threading.Lock()

# Last identified
last_identified = {
    'uid': None,
    'timestamp': None,
    'method': 'RFID'
}
last_identified_lock = threading.Lock()


def grant_access(identifier=""):
    """Grant access via RFID"""
    logging.info("✅ RFID ACCESS GRANTED: %s", identifier)
    
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
        except Exception as e:
            logging.exception('❌ Hardware control failed: %s', e)
    else:
        logging.info('🔓 [SIMULATED] Door unlocked for 5 seconds')
        time.sleep(5)
        logging.info('🔒 [SIMULATED] Door locked')
    
    record_access_event("RFID", identifier, True)


def deny_access(identifier=""):
    """Deny access"""
    logging.info("❌ RFID ACCESS DENIED: %s", identifier)
    
    if HAS_GPIO and led and buzzer:
        try:
            for _ in range(3):
                led.on()
                buzzer.on()
                time.sleep(0.1)
                led.off()
                buzzer.off()
                time.sleep(0.1)
        except Exception as e:
            logging.exception('Hardware control failed during access deny')
    else:
        logging.info('🚫 [SIMULATED] Access denied beeps')
    
    record_access_event("RFID", identifier, False)


def record_access_event(method, identifier, granted):
    """Record access attempt"""
    event = {
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'method': method,
        'identifier': identifier,
        'granted': granted
    }
    with access_events_lock:
        access_events.append(event)
        if len(access_events) > 100:
            access_events.pop(0)


def _record_identified(uid: int):
    """Record identification event"""
    ts = datetime.datetime.utcnow().isoformat()
    with last_identified_lock:
        last_identified['uid'] = uid
        last_identified['timestamp'] = ts
        last_identified['method'] = 'RFID'
    logging.info('Recorded RFID identification: %d', uid)


# Flask Routes
@app.route('/status', methods=['GET'])
def status():
    """Get service status"""
    return jsonify({
        'service': 'rfid_scanner',
        'running': rfid_scanner_enabled,
        'authorized_uids_count': len(AUTHORIZED_UIDS),
        'hardware_available': HAS_GPIO,
        'rfid_reader_available': rfid_reader is not None,
        'process_start_time': PROCESS_START_TIME
    })


@app.route('/last_identified', methods=['GET'])
def last_identified_route():
    """Get last identified card"""
    with last_identified_lock:
        return jsonify(last_identified)


@app.route('/access_log', methods=['GET'])
def access_log():
    """Get recent access events"""
    limit = int(request.args.get('limit', 50))
    with access_events_lock:
        events = access_events[-limit:]
    return jsonify({'events': events, 'count': len(events)})


@app.route('/authorized_uids', methods=['GET'])
def get_authorized_uids():
    """Get list of authorized RFID UIDs"""
    return jsonify({'uids': AUTHORIZED_UIDS})


@app.route('/authorized_uids', methods=['POST'])
def add_authorized_uid():
    """Add new authorized RFID UID"""
    data = request.get_json()
    uid = data.get('uid')
    if not uid:
        return jsonify({'error': 'Missing uid'}), 400
    
    try:
        uid = int(uid)
        if uid not in AUTHORIZED_UIDS:
            AUTHORIZED_UIDS.append(uid)
            logging.info('Added authorized UID: %d', uid)
            # Save to file
            save_authorized_uids()
            return jsonify({'message': 'UID added', 'uids': AUTHORIZED_UIDS}), 200
        else:
            return jsonify({'message': 'UID already exists', 'uids': AUTHORIZED_UIDS}), 200
    except ValueError:
        return jsonify({'error': 'Invalid UID format'}), 400


@app.route('/authorized_uids/<int:uid>', methods=['DELETE'])
def delete_authorized_uid(uid):
    """Remove authorized RFID UID"""
    if uid in AUTHORIZED_UIDS:
        AUTHORIZED_UIDS.remove(uid)
        logging.info('Removed authorized UID: %d', uid)
        # Save to file
        save_authorized_uids()
        return jsonify({'message': 'UID removed', 'uids': AUTHORIZED_UIDS}), 200
    else:
        return jsonify({'error': 'UID not found', 'uids': AUTHORIZED_UIDS}), 404


def save_authorized_uids():
    """Save authorized UIDs to file"""
    try:
        uids_file = os.environ.get('AUTHORIZED_UIDS_FILE', 'authorized_uids.json')
        with open(uids_file, 'w') as f:
            json.dump({'uids': AUTHORIZED_UIDS}, f)
        logging.info('Saved authorized UIDs to %s', uids_file)
    except Exception:
        logging.exception('Failed to save authorized UIDs')


def load_authorized_uids():
    """Load authorized UIDs from file"""
    global AUTHORIZED_UIDS
    try:
        uids_file = os.environ.get('AUTHORIZED_UIDS_FILE', 'authorized_uids.json')
        if os.path.exists(uids_file):
            with open(uids_file, 'r') as f:
                data = json.load(f)
                AUTHORIZED_UIDS = data.get('uids', AUTHORIZED_UIDS)
            logging.info('Loaded %d authorized UIDs', len(AUTHORIZED_UIDS))
    except Exception:
        logging.exception('Failed to load authorized UIDs')


# RFID Scanner Thread
def rfid_scanner():
    """Continuously scan for RFID cards"""
    logging.info("🔖 Starting RFID scanner...")
    
    if not HAS_GPIO or rfid_reader is None:
        logging.error("❌ Cannot start RFID scanner - hardware not available")
        return
    
    logging.info("✅ RFID scanner ready")
    logging.info("Authorized UIDs: %s", AUTHORIZED_UIDS)
    
    while rfid_scanner_enabled:
        try:
            id, _ = rfid_reader.read_no_block()
            if id:
                logging.info("📇 RFID Card Scanned - UID: %s", id)
                
                if id in AUTHORIZED_UIDS:
                    grant_access(identifier=str(id))
                    _record_identified(id)
                else:
                    deny_access(identifier=str(id))
                
                time.sleep(1)  # Prevent multiple reads
            
            time.sleep(0.1)
        except Exception:
            logging.exception('Error in RFID scanner loop')
            time.sleep(1)


# MQTT Listener
def on_mqtt_message(client, userdata, msg):
    """Handle MQTT messages"""
    try:
        logging.info('📨 MQTT: %s', msg.topic)
        payload = json.loads(msg.payload.decode())
        event = payload.get('event')
        
        if event == 'rfid_uid_added':
            uid = payload.get('uid')
            if uid:
                try:
                    uid = int(uid)
                    if uid not in AUTHORIZED_UIDS:
                        AUTHORIZED_UIDS.append(uid)
                        save_authorized_uids()
                        logging.info('➕ Added UID via MQTT: %d', uid)
                except ValueError:
                    logging.error('Invalid UID format: %s', uid)
        
        elif event == 'rfid_uid_removed':
            uid = payload.get('uid')
            if uid:
                try:
                    uid = int(uid)
                    if uid in AUTHORIZED_UIDS:
                        AUTHORIZED_UIDS.remove(uid)
                        save_authorized_uids()
                        logging.info('➖ Removed UID via MQTT: %d', uid)
                except ValueError:
                    logging.error('Invalid UID format: %s', uid)
    
    except Exception as e:
        logging.exception('Failed to handle MQTT message: %s', e)


def start_mqtt_listener():
    """Start MQTT listener"""
    broker_url = os.environ.get('MQTT_BROKER_URL') or os.environ.get('MQTT_BROKER')
    topic = os.environ.get('MQTT_TOPIC_RFID', 'wildwaves/rfid')
    
    if broker_url and ':' in broker_url:
        broker_host, broker_port = broker_url.rsplit(':', 1)
        broker_port = int(broker_port)
    else:
        broker_host = broker_url
        broker_port = int(os.environ.get('MQTT_PORT', '1883'))
    
    logging.info('🔌 MQTT Config: %s:%d topic=%s', broker_host, broker_port, topic)
    
    if not broker_host or mqtt is None:
        logging.warning('⚠️ MQTT not configured')
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
                logging.info('✅ MQTT Connected')
                client.subscribe(topic)
            else:
                logging.error('❌ MQTT Connection failed: %s', rc)
        
        client.on_connect = on_connect
        client.on_message = on_mqtt_message
        
        client.connect(broker_host, broker_port, keepalive=60)
        client.loop_start()
        logging.info('✅ MQTT listener started')
    
    except Exception as e:
        logging.exception('❌ MQTT listener failed: %s', e)


# Main
if __name__ == '__main__':
    load_authorized_uids()
    
    rfid_scanner_enabled = os.environ.get('ENABLE_RFID_SCANNER', 'true').lower() not in ('0', 'false', 'no')
    
    logging.info('=' * 60)
    logging.info('🚀 RFID SCANNER SERVICE STARTING')
    logging.info('=' * 60)
    logging.info('RFID Scanner: %s', '✅ ENABLED' if rfid_scanner_enabled else '❌ DISABLED')
    logging.info('Hardware GPIO: %s', '✅ AVAILABLE' if HAS_GPIO else '❌ NOT AVAILABLE')
    logging.info('Authorized UIDs: %d', len(AUTHORIZED_UIDS))
    logging.info('UIDs: %s', AUTHORIZED_UIDS)
    logging.info('=' * 60)
    
    if rfid_scanner_enabled:
        rfid_thread = threading.Thread(target=rfid_scanner, daemon=True)
        rfid_thread.start()
    
    start_mqtt_listener()
    
    port = int(os.environ.get('RFID_SERVICE_PORT', '5002'))
    logging.info('🌐 Starting Flask server on port %d', port)
    app.run(host='0.0.0.0', port=port, debug=False)