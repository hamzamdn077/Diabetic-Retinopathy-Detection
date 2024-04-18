import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
import cv2

model = load_model('my_model.h5')

def preprocessing(image):
    # Convert image to RGB format if it's not already in RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Convert PIL image to numpy array
    image_array = np.array(image)    
    # Convert RGB to LAB color space
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    # Split LAB channels
    l, a, b = cv2.split(lab)
    # Apply CLAHE to enhance L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    # Merge LAB channels back to RGB
    enhanced_lab = cv2.merge((enhanced_l, a, b))
    processed_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return processed_image

def predict(image):
    processed_image = preprocessing(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    prediction = model.predict(processed_image)
    return prediction

st.markdown("<h1 style='text-align: left; color: #ff5733;'>👁️ Retinopathy Diabetic Detection</h1>", unsafe_allow_html=True)

main_container = st.container()

with main_container:
    st.write("Upload an image to check for retinopathy")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        processed_image = preprocessing(image)
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption='Uploaded Image', width=200)
            with col2:
                st.image(processed_image, caption='Processed Image', width=200)
        if st.button('Predict'):
            with st.spinner('Making prediction...'):
                prediction = predict(image)
                predicted_class_index = np.argmax(prediction)
                classes = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']
                predicted_class_label = classes[predicted_class_index]
                st.markdown(f"<h2 style='color: #ff5733; text-align: left;'>Predicted Level of Retinopathy: {predicted_class_label}</h2>", unsafe_allow_html=True)

st.markdown("<div style='text-align: right;'>©️ Created by <a href='https://www.linkedin.com/in/hamza-moudden/' target='_blank'>Moudden Hamza</a></div>", unsafe_allow_html=True)
