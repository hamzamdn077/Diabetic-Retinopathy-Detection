import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
import cv2
from skimage.filters import frangi

model = load_model('binary_classifier.h5')

def preprocessing(image, scale_range=(1, 10), scale_step=2, beta=0.5):
    image_array = np.array(image)    
    resized_image = cv2.resize(image_array, (224, 224))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    frangi_image = frangi(gray_image, scale_range=scale_range, scale_step=scale_step, beta=beta)
    frangi_image = cv2.normalize(frangi_image, None, 0, 1, cv2.NORM_MINMAX)
    return frangi_image

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
                st.markdown(prediction)
                if prediction < 0.5:
                    st.markdown("<h2 style='color: #ff5733; text-align: left;'>⚠️ Retinopathy Diabetic Detected </h2>", unsafe_allow_html=True)
                else:
                    st.markdown("<h2 style='color: #0080ff; text-align: left;'>🟢 You are safe. No diabetic retinopathy detected.</h2>", unsafe_allow_html=True)
st.markdown("<div style='text-align: right;'>©️ Created by <a href='https://www.linkedin.com/in/hamza-moudden/' target='_blank'>Moudden Hamza</a></div>", unsafe_allow_html=True)