import streamlit as st
import tensorflow as tf

class L2Normalize(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "face_encoder.keras",
        custom_objects={"L2Normalize": L2Normalize},
        compile=False
    )
    return model

model = load_model()

st.title("Face Recognition App")
st.success("Model loaded successfully")


