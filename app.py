import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cattle_breed_model.h5')

model = load_model()


dataset_path = 'dataset/specialized'

if os.path.exists(dataset_path):
    CLASS_NAMES = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
else:
    st.error(f"Could not find dataset folder at {dataset_path}. Please check your path.")
    CLASS_NAMES = [] 


st.title("🐄 AI Cattle Breed Classifier")
st.write(f"Model is trained to detect **{len(CLASS_NAMES)}** breeds.")

file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', width=300)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class_index = np.argmax(predictions)
    
    if predicted_class_index < len(CLASS_NAMES):
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = 100 * np.max(score)
        
        st.success(f"Result: **{predicted_class}**")
        st.info(f"Confidence: {confidence:.2f}%")
    else:
        st.error(f"Error: Model predicted class index {predicted_class_index}, but we only have {len(CLASS_NAMES)} class names.")