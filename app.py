import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import tempfile
import pandas as pd

# Load model and labels
model = tf.keras.models.load_model("keras_model.h5")
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

st.set_page_config(page_title="Wood Type Classifier", layout="centered")
st.title("🪵 Wood Type Classifier")

st.markdown("Upload an image or take a photo to classify the wood type and estimate quality.")

# Image uploader and camera input
option = st.radio("Choose input method:", ["Upload Image", "Use Webcam"])

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
elif option == "Use Webcam":
    picture = st.camera_input("Take a picture")
    if picture:
        image = Image.open(picture)

if image:
    st.image(image, caption="Input Image", use_container_width=True)

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = np.asarray(img).astype(np.float32)
    img_array = (img_array / 127.5) - 1  # Normalize to [-1, 1]
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    # Show raw prediction scores as a table
    
    st.subheader("📊 Raw Prediction Scores:")
    df = pd.DataFrame({
        "Defect Type": class_names,
        "Confidence Score (%)": prediction * 100
    })
    st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)

    # Determine quality percentage
    if predicted_class.lower() == "good":
        quality_percent = confidence
    else:
        quality_percent = 100 - confidence

    # Display final prediction
    st.subheader("🔍 Prediction")
    if predicted_class.lower() == "good":
        st.write(f"🪵 **Wood Condition:** No Defect")
    else:
        st.write(f"🪵 **Defect Type:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.write(f"**Estimated Quality:** {quality_percent:.2f}%")


