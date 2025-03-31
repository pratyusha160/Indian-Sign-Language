import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
keypoints_dir = "mobkeypoints"
X, y = [], []

# Read keypoints and labels
for label in os.listdir(keypoints_dir):
    label_path = os.path.join(keypoints_dir, label)
    
    for keypoint_file in os.listdir(label_path):
        keypoint_path = os.path.join(label_path, keypoint_file)
        keypoints = np.load(keypoint_path)
        
        X.append(keypoints)
        y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Encode labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build Neural Network Model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(set(y_encoded)), activation='softmax')  # Output layer for classification
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Save Model
model.save("mob_isl_gesture_model.h5")

# Save Label Encoder for Decoding Predictions
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Model training complete and saved.")




