import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from tensorflow.keras.optimizers import Adam
import cv2
import zipfile
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    # Preprocess the image
    processed_image = preprocessing(image)
    
    processed_image = np.expand_dims(processed_image, axis=0)
    
   
    prediction = model.predict(processed_image)
    return prediction

def preprocess_image(img):
    
    img = array_to_img(img)
    img = preprocessing(img)
    img = img_to_array(img)
    img /= 255.0  
    return img

def retrain_model(data_dir, model, epochs=5):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        preprocessing_function=preprocess_image  
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs
    )
    
    return model


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
                
                
                classes = ['Normal', 'Mild', 'Moderate', 'Severe', 'Very Severe']
                
               
                predicted_class_label = classes[predicted_class_index]
                
                
                st.markdown(f"<h2 style='color: #ff5733; text-align: left;'>Predicted Level of Retinopathy: {predicted_class_label}</h2>", unsafe_allow_html=True)
    st.write("Upload a zip file containing new training data")
    training_file = st.file_uploader("Choose a zip file...", type=["zip"])

    if training_file is not None:
        with zipfile.ZipFile(training_file, 'r') as zip_ref:
            zip_ref.extractall("training_data")
        
        data_dir = "training_data"
        
        if st.button('Retrain Model'):
            with st.spinner('Retraining model...'):
                model = retrain_model(data_dir, model)
                model.save('resnet_retrained.h5')
                st.success('Model retrained and saved successfully!')
