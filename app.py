from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for communication with Flutter

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="mob_isl_gesture_model.tflite")
interpreter.allocate_tensors()

# Get model input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label encoder for decoding predictions
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def extract_hand_keypoints(image):
    """Extracts hand keypoints using MediaPipe Hands."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    keypoints = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            keypoints.extend([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

    # Ensure fixed length of 126 (42 keypoints, 3D each)
    if len(keypoints) < 42:
        keypoints.extend([(0, 0, 0)] * (42 - len(keypoints)))  # Pad with zeros

    return np.array(keypoints).flatten().astype(np.float32)  # Ensure float32 dtype

@app.route('/predict', methods=['POST'])
def predict():
    """Receives a frame from Flutter, processes it, and returns the predicted gesture."""
    if 'frame' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['frame']
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        print("Error: Could not decode image")
        return jsonify({'error': 'Invalid image data'}), 400

    # Extract keypoints
    keypoints = extract_hand_keypoints(frame)
    keypoints = np.expand_dims(keypoints, axis=0)  # Ensure correct shape

    # Run inference with TFLite model
    interpreter.set_tensor(input_details[0]['index'], keypoints)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_label = label_encoder.inverse_transform([np.argmax(output_data)])[0]
    print(f"Predicted Gesture: {predicted_label}")  # Debugging info

    return jsonify({'gesture': predicted_label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
