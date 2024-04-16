# DiabVision Repository

DiabVision is a sophisticated tool for analyzing retinal images to detect signs of diabetic retinopathy. This repository hosts several key files:

1. **diabVision_binary.ipynb:**  
   This file contains the code for the binary classification model. Given a retinal image as input, the model predicts whether the image exhibits signs of diabetic retinopathy or not. The model is trained to provide a binary output, indicating the presence or absence of the condition.

2. **diabVision.ipynb:**  
   This Jupyter Notebook extends the functionality of the binary classification model. In addition to predicting the presence or absence of diabetic retinopathy, it also provides an assessment of the severity or level of the condition. This enhanced model offers more detailed insights into the retinal images. In this notebook, pretrained models including EfficientNetB0, ResNet, DenseNet, and VGG16 are utilized for the analysis.

3. **binary_classification_app.py:**  
   This file contains a user interface for the DiabVision binary classification model. Users can interact with the model through this interface to input retinal images and obtain predictions regarding the presence of diabetic retinopathy.

The DiabVision repository serves as a valuable resource for healthcare professionals and researchers working in the field of diabetic retinopathy detection and diagnosis.
