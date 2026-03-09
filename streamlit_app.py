import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# אם יש Lambda layer צריך להגדיר את הפונקציה שלה
def euclidean_distance(vectors):
    x, y = vectors
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

# טעינת המודל עם cache
@st.cache_resource
def load_face_model():

    model = load_model(
        "/content/drive/MyDrive/Presence_Project/my_siamese2_model.h5",                 # שם קובץ המודל שלך
        compile=False,              # מונע בעיות תאימות
        custom_objects={
            "euclidean_distance": euclidean_distance
        }                           # מאפשר Lambda layer
    )

    return model


# טעינת המודל
model = load_face_model()

st.title("Face Recognition App")

st.success("Model loaded successfully")
