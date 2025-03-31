import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model("mob_isl_gesture_model.h5")

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("mob_isl_gesture_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to TFLite format.")

