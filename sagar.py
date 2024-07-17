import tensorflow as tf
from tensorflow.keras.models import load_model

print("TensorFlow version:", tf.__version__)

# Path to your model file
model_path = 'Model_Plant.h5'

try:
    model = load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)
