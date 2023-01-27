import numpy as np
from PIL import Image


# Keras
from tensorflow.keras.models import load_model

# Steamlit
import streamlit as st

st.title("Image classification")
st.text("Please upload chinhuahua or muffin's image for classification")

@st.cache(allow_output_mutation=True)
def classification_model():
  model = load_model("model_vgg19.h5")
  return model
with st.spinner('Model is being loaded..'):
  model= classification_model()

uploaded_image = st.file_uploader("upload the image: ", type = ["jpg", "png", "jpeg"])
if uploaded_image is not None:
    img = Image.open(uploaded_image)
    img = img.resize((224,224))
    img = np.asarray(img)
    img = img/255.0
    img = img.reshape(1,224,224,3)


    def model_predict(image, model):
        preds = model.predict(image)
        preds = np.argmax(preds,axis = 1)
        if preds == 0:
            preds = "This is a chinhuahua"
        if preds == 1:
            preds = "This is a muffin"
        return preds


    output = model_predict(img, model)

    col1, col2 = st.columns(2)

    with col1:
        st.header("Input image")
        st.image(uploaded_image)

    with col2:
        st.header("Prediction")
        st.write(output)






