import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np
import os

# -------------------------
# הגדרות דף
# -------------------------
st.set_page_config(page_title="מערכת נוכחות חכמה", layout="wide")
st.title("📸 מערכת נוכחות - Siamese Network")

# -------------------------
# רשימת תלמידים
# -------------------------
STUDENT_ROSTER = ['Maayan', 'Tomer', 'Roei', 'Zohar', 'Ilay']
REFERENCE_DIR = "My_Classmates"
st.write("Loaded students:", list(reference_embeddings.keys()))
# -------------------------
# שכבת נרמול
# -------------------------
class L2Normalize(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

# -------------------------
# בניית מודל embedding
# -------------------------
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

# -------------------------
# טעינת מודל
# -------------------------
@st.cache_resource
def load_model():
    model = build_pro_embedding()
    model.load_weights("face_encoder.weights.h5")
    return model

model = load_model()

st.success("Model loaded")

# -------------------------
# עיבוד תמונה
# -------------------------
def preprocess_image(img):

    img = img.convert("RGB").resize((128,128))
    arr = np.array(img).astype(np.float32) / 255.0

    return np.expand_dims(arr, axis=0)

# -------------------------
# יצירת embeddings לתלמידים
# -------------------------
@st.cache_data
def load_reference_embeddings():

    embeddings = {}

    for name in STUDENT_ROSTER:

        path = os.path.join(REFERENCE_DIR, f"{name}.jpg")

        if os.path.exists(path):

            img = Image.open(path)

            emb = model.predict(
                preprocess_image(img),
                verbose=0
            )[0]

            embeddings[name] = emb

    return embeddings


reference_embeddings = load_reference_embeddings()

# -------------------------
# הגדרות צד
# -------------------------
with st.sidebar:

    st.header("הגדרות")

    threshold = st.slider(
        "Similarity Threshold",
        0.0,
        1.0,
        0.5
    )

    st.write("רשימת הכיתה")
    st.write(STUDENT_ROSTER)

# -------------------------
# קלט משתמש
# -------------------------
st.subheader("העלי וקטור תמונות פנים")

test_files = st.file_uploader(
    "Upload faces",
    accept_multiple_files=True,
    type=["jpg","jpeg","png"]
)

# -------------------------
# זיהוי
# -------------------------
if st.button("בדוק נוכחות"):

    if not test_files:
        st.warning("יש להעלות תמונות פנים")
        st.stop()

    present_students = {}

    for file in test_files:

        img = Image.open(file)

        emb = model.predict(
            preprocess_image(img),
            verbose=0
        )[0]

        best_name = None
        best_dist = 1.0

        for name, ref_emb in reference_embeddings.items():

            dist = 1 - np.dot(emb, ref_emb)

            if dist < best_dist and dist < threshold:

                best_dist = dist
                best_name = name

        if best_name and best_name not in present_students:

            present_students[best_name] = img

    # -------------------------
    # תלמידים חסרים
    # -------------------------
    missing_students = [
        s for s in STUDENT_ROSTER
        if s not in present_students
    ]

    st.divider()

    # -------------------------
    # נוכחים
    # -------------------------
    st.header(f"✅ נוכחים ({len(present_students)})")

    cols = st.columns(4)

    for i,(name,img) in enumerate(present_students.items()):

        with cols[i % 4]:

            st.image(img, caption=name)

    # -------------------------
    # חסרים
    # -------------------------
    st.header(f"❌ חסרים ({len(missing_students)})")

    if missing_students:

        st.error(", ".join(missing_students))

    else:

        st.success("כולם נוכחים")
