from flask import Flask, render_template, Response, request, jsonify
import os
import random
import cv2
import numpy as np
import tensorflow as tf
import time
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)

# Load the pre-trained model for gesture recognition
model = tf.keras.models.load_model('recognition_model_for_sesotho.keras')

# Initialize hand detector
detector = HandDetector(maxHands=2)

# Set parameters for gesture detection
image_size = 100  # The size to which images were resized during training
offset = 20       # Offset for cropping the hand region

# Video capture from webcam
video_cap = cv2.VideoCapture(0)

# Initialize variables to store the constructed word and detected gestures
constructed_word = ""
detected_gestures = []  # To store all detected gestures

# Define gesture mapping for letters
gesture_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'J', 9: 'K',
    10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T',
    19: 'U', 20: 'Y', 21: 'KEA U RAPELA', 22: 'KEA U RATA'
}

# Timer for controlling gesture detection intervals
last_detection_time = time.time()
detection_interval = 5  # 5 seconds between detections

# Function to generate video frames with gesture recognition
def generate_frames():
    global constructed_word, last_detection_time, detected_gestures

    while True:
        success, image = video_cap.read()
        if not success:
            break

        hands, image = detector.findHands(image)

        current_time = time.time()
        if hands and (current_time - last_detection_time >= detection_interval):
            last_detection_time = current_time

            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Prepare white background for hand image
            image_white = np.ones((image_size, image_size, 3), np.uint8) * 255
            image_crop = image[y - offset: y + h + offset, x - offset: x + w + offset]

            aspect_ratio = h / w
            if aspect_ratio > 1:
                constant = image_size / h
                width_calculated = int(constant * w)
                image_resize = cv2.resize(image_crop, (width_calculated, image_size))
                width_gap = (image_size - width_calculated) // 2
                image_white[:, width_gap: width_gap + width_calculated] = image_resize
            else:
                constant = image_size / w
                height_calculated = int(constant * h)
                image_resize = cv2.resize(image_crop, (image_size, height_calculated))
                height_gap = (image_size - height_calculated) // 2
                image_white[height_gap: height_gap + height_calculated, :] = image_resize

            # Normalize and expand dimensions for model input
            image_white_normalized = image_white / 255.0
            image_input = np.expand_dims(image_white_normalized, axis=0)

            # Predict the gesture
            prediction = model.predict(image_input)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_gesture = gesture_mapping.get(predicted_class, "Unknown")

            # Update constructed word and detected gestures
            constructed_word += predicted_gesture
            detected_gestures.append(predicted_gesture)

        # Encode the frame as JPEG and yield for rendering in HTML
        ret, buffer = cv2.imencode('.jpg', image)
        image = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

# Route to render the main page
@app.route('/')
def home():
    return render_template('home.html')

# Route for video feed display
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to handle gesture-to-text detection results
@app.route('/get_constructed_word')
def get_constructed_word():
    global detected_gestures
    gestures_to_send = detected_gestures.copy()  # Send a copy of the current gestures
    detected_gestures.clear()  # Clear after sending to avoid re-sending the same gestures
    return jsonify({'word': constructed_word, 'gestures': gestures_to_send})

# Function to get gesture images for each letter in the text
def get_gesture_images(text):
    gesture_urls = []
    for letter in text:
        gesture_folder = f'static/gesture_images/{letter}'  # Folder for each letter
        if os.path.exists(gesture_folder):
            gesture_images = [f for f in os.listdir(gesture_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if gesture_images:
                selected_image = random.choice(gesture_images)  # Select a random gesture image
                gesture_urls.append(f'/static/gesture_images/{letter}/{selected_image}')  # Use absolute URL
    return gesture_urls

# Route to handle text-to-gesture translation
@app.route('/translate', methods=['POST'])
def translate():
    text = request.form['text']
    
    # Get gesture image URLs for the input text
    gesture_urls = get_gesture_images(text)
    
    return jsonify({'gesture_urls': gesture_urls})

if __name__ == '__main__':
    app.run(debug=True)
