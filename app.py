from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import streamlit as st
import os
from pathlib import Path
import numpy as np


# this is for saving images and prediction
def save_image(uploaded_file):
    if uploaded_file is not None:
        save_path = os.path.join("images", "input.jpeg")
        os.makedirs('images', exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        # st.success(f"Image saved to {save_path}")
        st.image(uploaded_file,)

        model = load_model(Path("artifacts\ModelTrainings\model.h5"))

        # test_image = image.load_img(Path("artifacts\img.jpeg"), target_size = (224,224))
        test_image = image.load_img(uploaded_file, target_size=(224, 224))

        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        prediction = np.argmax(model.predict(test_image), axis=1)
        print(prediction)

        if (prediction == 0):
            print('Normal')
            st.text_area(label="Prediction:", value="Normal", height=100)
        if (prediction == 1):
            print('PNEUMONIA')
            st.text_area(label="Prediction:", value="PNEUMONIA", height=100)


if __name__ == "__main__":
    st.title("Medical Image classifier")
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "png", "jpeg"])
    save_image(uploaded_file)
