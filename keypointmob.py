import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_hand_keypoints(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        keypoints = []
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                keypoints.extend([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

        # Ensure fixed size of 126 (42 keypoints, 3D each)
        if len(keypoints) < 42:  
            keypoints.extend([(0, 0, 0)] * (42 - len(keypoints)))  # Pad with zeros
        
        return np.array(keypoints).flatten()  # Always return 126 values

input_dir = "mobframes"
output_dir = "mobkeypoints"
os.makedirs(output_dir, exist_ok=True)

for sentence in os.listdir(input_dir):
    sentence_path = os.path.join(input_dir, sentence)
    keypoint_path = os.path.join(output_dir, sentence)
    os.makedirs(keypoint_path, exist_ok=True)
    
    for image_file in os.listdir(sentence_path):
        image_path = os.path.join(sentence_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Couldn't read {image_path}")
            continue  # Skip corrupted images
        
        keypoints = extract_hand_keypoints(image)
        np.save(os.path.join(keypoint_path, f"{image_file[:-4]}.npy"), keypoints)
