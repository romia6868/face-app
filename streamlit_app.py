import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# פונקציית המרחק של ה-Siamese
def euclidean_distance(vectors):
    x, y = vectors
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

@st.cache_resource
def load_face_model():

    model = tf.keras.models.load_model(
        "my_siamese2_model.h5",
        compile=False,
        custom_objects={
            "euclidean_distance": euclidean_distance
        },
        safe_mode=False   # חשוב ב-Keras 3 כשיש Lambda layer
    )

    return model

model = load_face_model()

st.write("Model loaded successfully")



st.title("Face Recognition App")

st.success("Model loaded successfully")
