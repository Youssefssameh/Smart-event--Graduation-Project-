import threading
from flask import Flask, jsonify, request
import cv2
import numpy as np
import json
from flask_cors import CORS
import time
from mtcnn import MTCNN
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import requests
 
# Initialize Flask
app = Flask(__name__)
CORS(app)

# Load trained model
cnn_model = load_model("best_model122.h5")

# Emotion labels
class_dict = {
    1: "surprise",
    2: "unsatisfied",
    3: "happy",
    4: "unsatisfied",
    5: "angry",
    6: "neutral"
}

# Global stop signal
stop_signal = False

# Face preprocessing
def preprocess_face(face, target_size=(100, 100)):
    face = cv2.resize(face, target_size)
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    face = face / 255.0
    return face

# Analyze single frame
def analyze_frame(frame):
    detector = MTCNN()
    results = detector.detect_faces(frame)

    if not results:
        return {}

    emotions_count = {emotion: 0 for emotion in class_dict.values()}
    total_faces = 0

    for result in results:
        x, y, width, height = result["box"]
        x, y = max(0, x), max(0, y)
        face = frame[y:y + height, x:x + width]
        processed_face = preprocess_face(face)
        prediction = cnn_model.predict(processed_face)
        emotion_label = np.argmax(prediction) + 1

        if emotion_label in [2, 4]:
            predicted_emotion = "unsatisfied"
        else:
            predicted_emotion = class_dict.get(emotion_label, "Unknown")

        emotions_count[predicted_emotion] += 1
        total_faces += 1

    if total_faces > 0:
        for emotion in emotions_count:
            emotions_count[emotion] = round((emotions_count[emotion] / total_faces) * 100, 2)

    return emotions_count

# Start camera and analyze periodically
def start_camera_analysis(ip_camera_url, event_id):
    global stop_signal
    stop_signal = False  # Reset signal at start

    event_emotions = []
    cap = cv2.VideoCapture(ip_camera_url)
    start_time = time.time()

    if not cap.isOpened():
        print("Failed to open camera")
        return

    print("🎥 Camera started successfully.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        if (current_time - start_time) >= 2:  # every 5 minutes
            emotions_data = analyze_frame(frame)
            if emotions_data:
                event_emotions.append(emotions_data)
                print("📸 Captured and analyzed a new frame.")
            start_time = current_time

        # Check stop signal
        if stop_signal:
            print("🛑 Received stop signal from backend.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Final report
    final_report = {emotion: 0 for emotion in class_dict.values()}

    if event_emotions:
        for data in event_emotions:
            for emotion in data:
                final_report[emotion] += data[emotion]

        for emotion in final_report:
            final_report[emotion] = round(final_report[emotion] / len(event_emotions), 2)

    dominant_emotion = max(final_report, key=final_report.get)
    final_report = {emotion: f"{value:.2f}%" for emotion, value in final_report.items()}

    report = {
        "event_id": event_id,
        "event_summary": {
            "dominant_emotion": dominant_emotion
        },
        "emotion_distribution": final_report
    }

    try:
        laravel_url = 'http://192.168.191.146:8000/api/receive-emotional'  # Replace with your Laravel API endpoint
        response = requests.post(laravel_url, json=report)

        if response.status_code == 200:
            print("Report sent successfully to Laravel backend.")
        else:
            print("Failed to send report to backend.")
    except Exception as e:
        print(str(e))

# Route to start analyzing
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    print("📥 Received JSON:", data)
    ip_camera_url = data.get("camera_ip")
    event_id = data.get("event_id")

    if ip_camera_url == "0":
        ip_camera_url = 0

    if ip_camera_url is None or event_id is None:
        return jsonify({"message": "Missing IP camera URL or event ID"}), 400

    # Start the camera analysis in a separate thread
    camera_thread = threading.Thread(target=start_camera_analysis, args=(ip_camera_url, event_id))
    camera_thread.start()

    # Return immediate response to the front-end
    return jsonify({"message": "Camera started successfully, analysis in progress."}), 200

# Endpoint to stop the camera
@app.route('/stop', methods=['POST'])
def stop():
    global stop_signal
    stop_signal = True
    return jsonify({"message": "Camera stop signal received"}), 200

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
