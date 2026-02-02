import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

st.set_page_config(page_title="Fashion MNIST Classifier", layout="centered")

st.title("Fashion MNIST — Візуалізація роботи нейронної мережі")

model_type = st.selectbox(
    "Оберіть модель:",
    ("CNN", "VGG16")
)

@st.cache_resource
def load_selected_model(model_type):
    if model_type == "CNN":
        return load_model("cnn_fashion_mnist.h5")
    else:
        return load_model("vgg16_fashion_mnist.h5")

model = load_selected_model(model_type)

uploaded_file = st.file_uploader(
    "Завантажте зображення (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.subheader("Вхідне зображення")
    st.image(image, width=200)

    if model_type == "CNN":
        image = image.resize((28, 28))
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
    else:
        image = image.resize((96, 96))
        img_array = np.array(image) / 255.0
        img_array = np.stack([img_array]*3, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0]

    st.subheader("Результат класифікації")
    st.success(f"Передбачений клас: **{class_names[predicted_class]}**")

    st.subheader("Ймовірності для кожного класу")

    fig, ax = plt.subplots()
    ax.barh(class_names, confidence)
    ax.set_xlabel("Ймовірність")
    ax.set_xlim(0, 1)

    st.pyplot(fig)

st.subheader("Графіки навчання моделі")

st.markdown(""" Для відображення loss / accuracy необхідно зберігати історію навчання (`history.history`) під час тренування.""")