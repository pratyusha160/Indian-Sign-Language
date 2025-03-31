import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
import pyttsx3  # Text-to-Speech Library
import time  # For adding delays

# Load trained model
model = tf.keras.models.load_model("mob_isl_gesture_model.h5")

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
    """Extracts keypoints from an image and ensures fixed-length (126 values)."""
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

        return np.array(keypoints).flatten()  # Always return 126 values

# Start Video Capture
cap = cv2.VideoCapture(0)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if FPS is not available

# Define Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter('session_record1.mp4', fourcc, fps, (frame_width, frame_height))

last_prediction = None  # Store last prediction
prediction_streak = 0  # Counter to ensure stable predictions
streak_threshold = 10  # Minimum frames needed to confirm a prediction
cooldown_time = 3  # Wait time (in seconds) after speaking
last_speak_time = time.time() - cooldown_time  # Ensures it can speak at the start

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame (optional)
    frame = cv2.flip(frame, 1)

    # Extract keypoints
    keypoints = extract_hand_keypoints(frame)

    # Make prediction
    keypoints = keypoints.reshape(1, -1)  # Reshape for model input
    prediction = model.predict(keypoints)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    # Stability check: Only update if the same prediction is made consistently
    if predicted_label == last_prediction:
        prediction_streak += 1
    else:
        prediction_streak = 0
        last_prediction = predicted_label  # Reset last prediction

    # Display result
    cv2.putText(frame, f"Prediction: {predicted_label}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Speak out the predicted label only if it is stable and cooldown is over
    if prediction_streak >= streak_threshold and (time.time() - last_speak_time) > cooldown_time:
        print(f"Speaking: {predicted_label}")  # Debugging info
        speak(predicted_label)
        last_speak_time = time.time()  # Update last speak time
        prediction_streak = 0  # Reset streak after speaking

    # Write the frame to the video file
    out.write(frame)

    # Show video feed
    cv2.imshow("ISL Gesture Recognition", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
