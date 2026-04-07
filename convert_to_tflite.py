import tensorflow as tf

# =====================
# LOAD TRAINED MODEL
# =====================

model = tf.keras.models.load_model("animal_model_v1.keras")

# =====================
# TFLITE CONVERSION
# =====================

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimization for smaller size and better performance
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert model
tflite_model = converter.convert()

# =====================
# SAVE TFLITE MODEL
# =====================

with open("animal_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model successfully saved as animal_model.tflite")