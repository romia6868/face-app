import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# פונקציה של ה-Lambda layer
def euclidean_distance(vectors):
    x, y = vectors
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

# טעינת המודל
@st.cache_resource
def load_face_model():

    model = load_model(
        "my_siamese2_model.h5",
        compile=False,
        custom_objects={
            "euclidean_distance": euclidean_distance
        }
    )

    return model

model = load_face_model()


st.title("Face Recognition App")

st.success("Model loaded successfully")
