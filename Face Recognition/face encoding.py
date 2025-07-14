from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import os

app = Flask(__name__)


@app.route('/process_image', methods=['POST'])
def process_image():
    # Check if an image is provided
    if 'image' not in request.files:
        return jsonify({"error": "No image was sent"}), 400

    # Receive the image and user_iduser_id
    image_file = request.files['image']

    user_id = request.form.get('user_id', 'unknown')  #  Default to "unknown" if user_id is not provided


    # Load the image and convert it into an array
    image = face_recognition.load_image_file(image_file)

    #  Extract face encoding
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return jsonify({"error": "No face detected in the image"}), 400

    #  Convert encoding into a regular list that can be stored in JSON
    encoding_list = encodings[0].tolist()

    # Send the data back to Laravel
    response_data = {
        "user_id": user_id,
        "encoding": encoding_list
    }

    return jsonify(response_data), 200


if __name__ == '__main__':
    app.run(host='192.168.191.16', port=5050, debug=True)



