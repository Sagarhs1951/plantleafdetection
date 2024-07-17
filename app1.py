import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your model
model_path = "Model_Plant.h5"  # Adjust the path to your model
model = load_model(model_path)

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust the size as per your model's requirement
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions using the loaded model
def make_prediction(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction[0]

# Streamlit app
def main():
    st.set_page_config(page_title="Plant Leaf Disease Prediction", layout="wide")

    st.title("Plant Leaf Disease Prediction")
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

    st.header("Upload a Plant Leaf Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button("Predict"):
            prediction = make_prediction(image)
            st.success(f"Prediction: {prediction}")

    st.sidebar.header("About")
    st.sidebar.info("""
    This application predicts plant leaf diseases based on uploaded images.
    The model processes the image and outputs the prediction of the disease.
    """)

if __name__ == "__main__":
    main()
