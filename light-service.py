#!/usr/bin/env python3
"""
Raspberry Pi 4 MQTT Light Control with MOSFET (PWM Dimming)
Converted from ESP32 version - Uses GPIO for PWM control
"""

import ssl
import time
import json
import random
import paho.mqtt.client as mqtt
import RPi.GPIO as GPIO

# WiFi not needed - Pi uses Ethernet/WiFi via OS
# Just ensure your Pi is connected to network

# HiveMQ Cloud Settings
MQTT_SERVER = "32b18ba4678b4f3db60a560db8a90c86.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "wildwave"
MQTT_PASSWORD = "Wild@123"

# THIS RASPBERRY PI IS BOUND TO THIS ITEM CODE ONLY
MY_ITEM_CODE = "LT643"  # Change this for different devices

# MQTT Topics
current_light_id = ""
current_item_code = ""
mqtt_topic = ""

# GPIO PWM Configuration for Raspberry Pi 4
# GPIO Pin Mapping (BCM numbering)
MOSFET_PIN = 18  # GPIO 18 (Physical Pin 12) - Hardware PWM capable
PWM_FREQ = 5000  # 5kHz PWM frequency
PWM_DUTY_MAX = 100  # Percentage (0-100)

# State variables
current_brightness = 0
current_status = False

# PWM object
pwm = None

def setup_gpio():
    """Initialize GPIO and PWM"""
    global pwm
    
    GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering
    GPIO.setwarnings(False)
    GPIO.setup(MOSFET_PIN, GPIO.OUT)
    
    # Initialize PWM on the MOSFET pin
    pwm = GPIO.PWM(MOSFET_PIN, PWM_FREQ)
    pwm.start(0)  # Start with 0% duty cycle (light OFF)
    
    print(f"GPIO initialized - PWM on GPIO {MOSFET_PIN} (BCM)")
    print(f"PWM Frequency: {PWM_FREQ} Hz")

def apply_light(brightness, status):
    """Apply brightness and status to the light"""
    global current_brightness, current_status
    
    current_brightness = brightness
    current_status = status
    
    actual_brightness = brightness
    
    # If status is OFF, force brightness to 0
    if not status:
        actual_brightness = 0
    
    # Constrain brightness to 0-100 range
    actual_brightness = max(0, min(100, actual_brightness))
    
    # Set PWM duty cycle (0-100%)
    pwm.ChangeDutyCycle(actual_brightness)
    
    print(f"Status: {'ON' if status else 'OFF'} | "
          f"Brightness Setting: {brightness}% | "
          f"Actual Output: {actual_brightness}% (PWM: {actual_brightness}%)")

def update_subscription(client, light_id):
    """Update MQTT subscription when lightId changes"""
    global current_light_id, mqtt_topic
    
    # Unsubscribe from old topic if necessary
    if current_light_id:
        client.unsubscribe(mqtt_topic)
    
    # Update the current lightId
    current_light_id = light_id
    
    # Create new topic string
    mqtt_topic = f"wildwaves/lights/{current_light_id}"
    
    # Subscribe to new topic
    client.subscribe(mqtt_topic)
    print(f"Subscribed to topic: {mqtt_topic}")

def on_connect(client, userdata, flags, rc):
    """Callback when connected to MQTT broker"""
    if rc == 0:
        print("Connected to MQTT broker!")
        
        # Subscribe to wildcard topic to receive messages for all lights
        client.subscribe("wildwaves/lights/+")
        print("Subscribed to: wildwaves/lights/+")
        print(f"Listening for itemCode: {MY_ITEM_CODE}")
    else:
        error_messages = {
            1: "MQTT_CONNECT_BAD_PROTOCOL",
            2: "MQTT_CONNECT_BAD_CLIENT_ID",
            3: "MQTT_CONNECT_UNAVAILABLE",
            4: "MQTT_CONNECT_BAD_CREDENTIALS",
            5: "MQTT_CONNECT_UNAUTHORIZED"
        }
        print(f"Failed to connect, return code {rc} ({error_messages.get(rc, 'Unknown error')})")

def on_message(client, userdata, msg):
    """Callback when MQTT message is received"""
    global current_item_code
    
    print(f"\n=== Message received on topic: {msg.topic} ===")
    
    try:
        # Parse JSON payload
        payload = json.loads(msg.payload.decode())
        print(f"Raw JSON: {json.dumps(payload)}")
        
        # Extract itemCode first - THIS IS CRITICAL
        item_code = payload.get("itemCode")
        
        # CHECK IF THIS MESSAGE IS FOR THIS RASPBERRY PI
        if item_code:
            if item_code != MY_ITEM_CODE:
                print(f"Ignoring message - itemCode '{item_code}' "
                      f"does not match MY_ITEM_CODE '{MY_ITEM_CODE}'")
                return  # EXIT - This message is NOT for this device
            
            current_item_code = item_code
            print(f"✓ itemCode matches! Processing message for: {MY_ITEM_CODE}")
        else:
            print("Warning: No itemCode in message, ignoring...")
            return
        
        # Extract the lightId and subscribe if needed
        light_id = payload.get("lightId")
        if light_id and (not current_light_id or current_light_id != light_id):
            update_subscription(client, light_id)
        
        # Extract event
        event = payload.get("event")
        
        if event == "light_control":
            # Get status - handle both boolean and string formats
            status = current_status  # Default to current status
            
            if "status" in payload:
                status_value = payload["status"]
                if isinstance(status_value, bool):
                    status = status_value
                elif isinstance(status_value, str):
                    status = status_value.upper() in ["ON", "TRUE"]
            
            # Get brightness - use current if not provided
            brightness = current_brightness
            
            if "brightness" in payload:
                brightness = int(payload["brightness"])
                brightness = max(0, min(100, brightness))
            
            # Get access control
            access = payload.get("access", True)
            
            print(f"Parsed - Status: {'ON' if status else 'OFF'} | "
                  f"Brightness: {brightness}% | "
                  f"Access: {'GRANTED' if access else 'DENIED'}")
            
            # Check access control
            if not access:
                print("⚠ Access DENIED! Forcing light OFF")
                apply_light(0, False)
                return
            
            # Apply the control
            apply_light(brightness, status)
        else:
            print("Unknown event or missing event field")
    
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
    except Exception as e:
        print(f"Error processing message: {e}")
    
    print("=== End of message processing ===\n")

def on_disconnect(client, userdata, rc):
    """Callback when disconnected from MQTT broker"""
    if rc != 0:
        print(f"Unexpected disconnection. Return code: {rc}")
        print("Attempting to reconnect...")

def main():
    """Main function"""
    print("\n========================================")
    print("Raspberry Pi MQTT Light Controller")
    print("========================================")
    print(f"Bound to itemCode: {MY_ITEM_CODE}")
    print(f"PWM on GPIO {MOSFET_PIN} (BCM)")
    print("========================================\n")
    
    # Setup GPIO
    setup_gpio()
    
    # Create MQTT client
    client_id = f"RaspberryPi-{MY_ITEM_CODE}-{random.randint(0, 0xFFFF):04x}"
    client = mqtt.Client(client_id=client_id)
    
    # Set username and password
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    
    # Configure TLS/SSL
    client.tls_set(cert_reqs=ssl.CERT_NONE)
    client.tls_insecure_set(True)
    
    # Set callbacks
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    
    # Connect to broker
    print(f"Connecting to MQTT broker at {MQTT_SERVER}:{MQTT_PORT}...")
    try:
        client.connect(MQTT_SERVER, MQTT_PORT, 60)
    except Exception as e:
        print(f"Failed to connect: {e}")
        GPIO.cleanup()
        return
    
    print("\nSetup complete! Waiting for MQTT messages...")
    print(f"Will only respond to itemCode: {MY_ITEM_CODE}")
    print("========================================\n")
    print("Press Ctrl+C to exit\n")
    
    # Start MQTT loop
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Cleanup
        pwm.stop()
        GPIO.cleanup()
        client.disconnect()
        print("Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main()