import streamlit as st
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image


st.set_page_config(page_title="Diabetic Retinopathy Classification", page_icon=":eyes:")
st.title("Diabetic Retinopathy Classification")
st.sidebar.title("Menu")

menu = ["Home", "Upload Image", "About"]
choice = st.sidebar.selectbox("Select an option", menu)

if choice == "Home":
    st.write("Welcome to the Diabetic Retinopathy Classification app.")
    
    
    

if choice == "Upload Image":
    model = load_model('cnn.h5')
    class_labels = ['DR', 'NO-DR']

    def predict(image):
        img = Image.open(image).convert('RGB')
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        predictions = model.predict(img)
        label = np.argmax(predictions[0])
        return class_labels[label]

    def main():
        uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.subheader("Please wait uploaded file is analysing")
            with st.spinner('Almost there....'):
                time.sleep(5)
            st.image(image, caption='Uploaded Image',use_column_width=False)
            label = predict(uploaded_file)
            if label == 'DR':
                st.warning("The uploaded image is classified as : DR ")
            else:
                st.success("The uploaded image is classified as : NO-DR ")
                st.balloons()

    if __name__ == '__main__':
        main()
            
        
        
        
        
        

if choice == "About":
    st.subheader("About")
    st.write("This app uses machine learning algorithms to detect diabetic retinopathy.")

