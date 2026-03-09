import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

class L2Normalize(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

def build_pro_embedding():

    base_model = MobileNetV2(
        input_shape=(128,128,3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),

        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),

        layers.Dense(128),
        L2Normalize()
    ])

    return model


@st.cache_resource
def load_model():
    model = build_pro_embedding()
    model.load_weights("face_encoder.weights.h5")
    return model


model = load_model()

st.title("Face Recognition")
st.success("Model loaded")
