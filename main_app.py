# library imports
import numpy as np
import streamlit as st
import cv2 
from keras.models import load_model

#loading the Model
model = load_model('plant_disease_prediction_model.keras')

CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

#title of App
st.title('Plant Disease Detection by Shoaib Hoque')
st.markdown('Upload an image of the plant leaf')

#Uploading the plant image
plant_image = st.file_uploader("Choose an image (jpg format)...", type='jpg')
submit = st.button('Predict')

if submit:

    if plant_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)

        #resizing the image
        opencv_image = cv2.resize(opencv_image, (256,256))

        #Convert image to 4 Dimension
        opencv_image.shape = (1,256,256,3)

        #Make Prediction
        y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(y_pred)]
        st.title(str(f"This is {result.split('-')[0]} leaf with {result.split('-')[1]}"))