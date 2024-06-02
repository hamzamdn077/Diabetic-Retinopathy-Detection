import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import cv2
import time
import matplotlib.pyplot as plt

model = load_model('resnet.h5')
st.success('Model loaded successfully!')

def preprocessing(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_array = np.array(image)    
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    enhanced_lab = cv2.merge((enhanced_l, a, b))
    processed_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    processed_image = cv2.resize(processed_image, (224, 224))
    return processed_image
def predict(image):
    processed_image = preprocessing(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    start_time = time.time()
    prediction = model.predict(processed_image)
    prediction_time = time.time() - start_time  
    return prediction, prediction_time
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
                prediction, prediction_time = predict(image)
                predicted_class_index = np.argmax(prediction)
                classes = ['Normal', 'Mild', 'Moderate', 'Proliferate', 'Severe']
                predicted_class_label = classes[predicted_class_index]
                st.markdown(f"<h2 style='color: #ff5733; text-align: left;'>Predicted Level of Retinopathy: {predicted_class_label}</h2>", unsafe_allow_html=True)
                st.write(f"Time taken for prediction: {prediction_time:.2f} seconds")
                st.write("Prediction Probabilities:")
                fig, ax = plt.subplots(figsize=(5, 3))  
                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')
                ax.bar(classes, prediction[0], color=['#ff5733', '#33ff57', '#3357ff', '#ff33a6', '#a633ff'])
                ax.set_xlabel('Classes', color='white')
                ax.set_ylabel('Probabilities', color='white')
                ax.set_title('Class Probabilities', color='white')
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_edgecolor('white')
                st.pyplot(fig)
