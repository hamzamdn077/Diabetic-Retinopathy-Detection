import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from tensorflow.keras.optimizers import Adam
import cv2
import zipfile
import os
import bcrypt
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

hashed_password = bcrypt.hashpw('admin'.encode(), bcrypt.gensalt())

def check_credentials(username, password):
    return username == 'admin' and bcrypt.checkpw(password.encode(), hashed_password)

st.markdown("""
    <style>
        .main-title {
            font-size: 2.5em;
            color: #ff5733;
            text-align: center;
            margin-bottom: 20px;
        }
        .sidebar .sidebar-content {
            width: 100%;
        }
        .stButton>button {
            background-color: #ff5733;
            color: white;
            border: None;
        }
        .stButton>button:hover {
            background-color: #ff3311;
        }
        .stSpinner {
            font-size: 1.2em;
            color: #ff5733;
        }
        .stAlert {
            font-size: 1.1em;
        }
    </style>
""", unsafe_allow_html=True)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<h1 class='main-title'>👁️ Retinopathy Diabetic Detection</h1>", unsafe_allow_html=True)
    st.sidebar.subheader("Login to access more features")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if check_credentials(username, password):
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.sidebar.error("Invalid username or password")
else:
    st.sidebar.markdown("## Navigation")
    st.sidebar.write("Upload a zip file containing new training data")
    training_file = st.sidebar.file_uploader("Choose a zip file...", type=["zip"])

    if training_file is not None:
        with zipfile.ZipFile(training_file, 'r') as zip_ref:
            zip_ref.extractall("training_data")

        data_dir = "training_data"

        if st.sidebar.button('Retrain Model'):
            with st.spinner('Retraining model...'):
                model = retrain_model(data_dir, model)
                model.save('resnet_retrained.h5')
                st.sidebar.success('Model retrained and saved successfully!')

    st.markdown("<h1 class='main-title'>👁️ Retinopathy Diabetic Detection</h1>", unsafe_allow_html=True)
    st.markdown("### Use the sidebar to upload training data and retrain the model.")
