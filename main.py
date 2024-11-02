import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from util import classify

# Set title
st.title('Pneumonia Classification')

# Set header
st.header('Please upload a chest X-ray or CT image')

# Upload file
file = st.file_uploader('Upload Image', type=['jpeg', 'jpg', 'png'])

# Select model
model_choice = st.selectbox('Select Model:', ('X-ray Model', 'CT Model'))

# Load models based on user choice
if 'model' not in st.session_state or st.session_state.model_choice != model_choice:
    if model_choice == 'X-ray Model':
        try:
            st.session_state.model = load_model('./models/xray_classification_model.h5')
            st.session_state.model_type = 'xray'
        except Exception as e:
            st.error(f"Error loading X-ray model: {str(e)}")
    else:
        try:
            st.session_state.model = load_model('./models/CT_classification_model.h5')
            st.session_state.model_type = 'ct'
        except Exception as e:
            st.error(f"Error loading CT model: {str(e)}")

    st.session_state.model_choice = model_choice

# Load class names
if 'class_names' not in st.session_state:
    model_type = st.session_state.get('model_type', 'xray')  # Default to xray
    with open(f'./models/{model_type}_labels.txt', 'r') as f:
        st.session_state.class_names = [line.strip() for line in f.readlines()]

# Display image and classify
if file is not None:
    try:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        with st.spinner("Classifying..."):
            class_name, conf_score = classify(image, st.session_state.model, st.session_state.class_names)

        st.write("## Prediction: {}".format(class_name))
        st.write("### Confidence Score: {:.2f}%".format(conf_score * 100))
    except Exception as e:
        st.error("Error processing the image: {}".format(str(e)))
