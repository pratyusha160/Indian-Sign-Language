import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import collections
import time
import threading

# Load Keras Model
keras_model = tf.keras.models.load_model("mob_isl_gesture_model.h5")

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path="mob_isl_gesture_model.tflite")
interpreter.allocate_tensors()

# Get input and output details for TFLite
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Mediapipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Gesture Buffer for Stable Predictions
gesture_buffer_keras = collections.deque(maxlen=5)
gesture_buffer_tflite = collections.deque(maxlen=5)

# Gesture Labels
gesture_labels = {0: "GoodAfternoon", 1: "GoodEvening", 2: "GoodMorning", 3: "Hello", 4: "No", 5: "ThankYou",6:"Yes"}

# Video Capture
cap = cv2.VideoCapture(0)

# Check if camera opens successfully
if not cap.isOpened():
    print("Error: Camera not accessible!")
    exit()

# Frame Rate Counter
fps = 0
prev_time = time.time()

# Global variables for threading
frame = None
stop_thread = False

# Thread function to capture frames asynchronously
def capture_frames():
    global frame, stop_thread
    while not stop_thread:
        ret, new_frame = cap.read()
        if ret:
            frame = cv2.flip(new_frame, 1)

# Start frame capture thread
thread = threading.Thread(target=capture_frames)
thread.start()

def extract_hand_keypoints(image):
    """Extracts hand landmarks using Mediapipe and flattens them into an array."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            return keypoints
    return np.zeros(126)  # Return empty array if no hand detected

while True:
    if frame is None:
        continue  # Wait for the first frame

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (320, 240))

    # Extract hand keypoints
    keypoints = extract_hand_keypoints(small_frame)
    
    if keypoints.shape[0] != 126:
        continue  # Skip if no valid keypoints

    keypoints = keypoints.reshape(1, -1).astype(np.float32)

    # Keras Model Prediction
    keras_pred = keras_model.predict(keypoints, verbose=0)
    keras_index = np.argmax(keras_pred)

    # TFLite Model Prediction
    interpreter.set_tensor(input_details[0]['index'], keypoints)
    interpreter.invoke()
    tflite_pred = interpreter.get_tensor(output_details[0]['index'])
    tflite_index = np.argmax(tflite_pred)

    # Store Predictions
    gesture_buffer_keras.append(keras_index)
    gesture_buffer_tflite.append(tflite_index)

    # Stabilized Gesture Recognition
    keras_gesture = gesture_labels.get(keras_index, "Unknown") if gesture_buffer_keras.count(keras_index) >= 3 else "Waiting..."
    tflite_gesture = gesture_labels.get(tflite_index, "Unknown") if gesture_buffer_tflite.count(tflite_index) >= 3 else "Waiting..."

    # Display Predictions on Frame
    cv2.putText(frame, f"Keras: {keras_gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"TFLite: {tflite_gesture}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Gesture Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_thread = True
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
thread.join()
