import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
import pyttsx3  # Text-to-Speech Library
import time  # For cooldown between speech outputs

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="mob_isl_gesture_model.tflite")
interpreter.allocate_tensors()

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Text-to-Speech Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)  # Set speech speed

def speak(text):
    """Convert text to speech."""
    tts_engine.say(text)
    tts_engine.runAndWait()

def extract_hand_keypoints(image):
    """Extracts hand keypoints for gesture recognition (126 values)."""
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
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

# Start Video Capture
cap = cv2.VideoCapture(0)

last_prediction = None  # Store last prediction
prediction_streak = 0  # Counter for stable predictions
streak_threshold = 10  # Minimum frames needed to confirm a prediction
cooldown_time = 3  # Wait time (in seconds) before speaking the next word
last_speak_time = time.time() - cooldown_time  # Ensures speaking starts immediately

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame
    frame = cv2.flip(frame, 1)

    # Extract keypoints
    keypoints = extract_hand_keypoints(frame)
    keypoints = keypoints.reshape(1, -1)  # Reshape for TFLite input

    # Run inference with TFLite model
    interpreter.set_tensor(input_details[0]['index'], keypoints)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_label = label_encoder.inverse_transform([np.argmax(output_data)])[0]

    # Stability check: Speak only if the same prediction is consistent
    if predicted_label == last_prediction:
        prediction_streak += 1
    else:
        prediction_streak = 0
        last_prediction = predicted_label  # Reset last prediction

    # Display result
    cv2.putText(frame, f"Prediction: {predicted_label}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Speak the gesture only if it's stable and cooldown is over
    if prediction_streak >= streak_threshold and (time.time() - last_speak_time) > cooldown_time:
        print(f"Speaking: {predicted_label}")  # Debugging info
        speak(predicted_label)
        last_speak_time = time.time()  # Reset cooldown
        prediction_streak = 0  # Reset streak after speaking

    cv2.imshow("ISL Gesture Recognition (TFLite)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


