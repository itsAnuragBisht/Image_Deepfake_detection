import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("Authentiseeeee.h5")  # Change filename as needed
    return model

model = load_model()

# Page Configuration
# st.set_page_config(page_title="Authentisee", page_icon="🔍", layout="wide")

# Title and Subtitle
st.title("Authentisee")
st.subheader("A Deepfake Detection System")

# Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload & Analyze", "About"])

if page == "Home":
    st.write("### Welcome to Authentisee!")
    st.write("This tool helps detect deepfake images and videos using advanced machine learning models.")
    st.image("https://via.placeholder.com/800x400", caption="Deepfake Detection in Action")

elif page == "Upload & Analyze":
    st.write("### Upload an Image for Analysis")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess Image
        img_array = np.array(image.resize((224, 224))) / 255.0  # Adjust size as per model input
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)
        result = "Real" if prediction[0][0] > 0.5 else "Deepfake"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
        
        st.write(f"### Prediction: {result}")
        st.write(f"Confidence: {confidence:.2f}")

elif page == "About":
    st.write("### About Authentisee")
    st.write("Authentisee is a deepfake detection system that analyzes images to determine their authenticity.")
    st.write("Developed using Streamlit and machine learning techniques, it helps users detect manipulated content easily.")