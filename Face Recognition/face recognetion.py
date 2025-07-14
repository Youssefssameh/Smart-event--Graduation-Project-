import cv2
import face_recognition
import requests
import numpy as np
from datetime import datetime

# Load user data from the backend
response = requests.get("http://192.168.191.146:8000/api/send-face-encodings")
if response.status_code == 200:
    known_faces = response.json().get("users", [])
else:
    print(" Failed to load data from the backend")
    exit()

# Ensure data is not empty
if not known_faces:
    print(" No users found in the database")
    exit()

# Extract encodings and user IDs
known_encodings = [np.array(user["encoding"]) for user in known_faces if "encoding" in user]
known_ids = [user["user_id"] for user in known_faces if "user_id" in user]

# Start the camera
video_capture = cv2.VideoCapture(0)

# Set to store registered users
registered_users = set()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize the frame to speed up processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

        name = "Unknown"
        user_id = None

        if best_match_index is not None and face_distances[best_match_index] < 0.5:  # Adjusted threshold
            user_id = known_ids[best_match_index]
            name = f"User {user_id}"

            # Check if the user is already registered
            if user_id not in registered_users:
                # Send attendance data
                attendance_data = {
                    "user_id": user_id,
                    "event_id": event_id,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                try:
                    response = requests.post("http://192.168.191.146:8000/api/mark-attendance", json=attendance_data)
                    if response.status_code == 200:
                        print(f" Attendance recorded for user {user_id}")
                        registered_users.add(user_id)
                    else:
                        print(f" Failed to mark attendance for user {user_id}")
                except requests.RequestException as e:
                    print(f" Error connecting to server: {e}")

        # Draw a rectangle around the face
        top, right, bottom, left = [coord * 2 for coord in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()