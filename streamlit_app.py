import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown
import os

# -------------------------
# הגדרות דף
# -------------------------
st.set_page_config(page_title="מערכת נוכחות חכמה", layout="wide")

st.title("📸 מערכת נוכחות - Siamese Network")

# -------------------------
# רשימת תלמידים קבועה
# -------------------------
STUDENT_ROSTER = ['Maayan', 'Tomer', 'Roei', 'Zohar', 'Ilay']

REFERENCE_DIR = "reference_images"

# -------------------------
# טעינת מודל
# -------------------------
import streamlit as st
import gdown
import tensorflow as tf
import os

import streamlit as st
import tensorflow as tf
import gdown
import os


# הפונקציה שהייתה בתוך Lambda
def l2_normalize(x):
    return tf.math.l2_normalize(x, axis=1)


@st.cache_resource
def load_model():

    file_id = "1fMz79YX4wACoHw_iSuS13qTWBNvywZGK"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "embedding_model.keras"

    # הורדה מהדרייב אם הקובץ לא קיים
    if not os.path.exists(output):
        with st.spinner("Downloading model..."):
            gdown.download(url, output, quiet=False)

    # טעינת המודל
    model = tf.keras.models.load_model(
    "embedding_model_clean.keras",
    compile=False
)

    return model
model = load_model()

# -------------------------
# עיבוד תמונה
# -------------------------
def preprocess_image(pil_image):

    img = pil_image.convert("RGB").resize((224,224))
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
            emb = model.predict(preprocess_image(img), verbose=0)

            embeddings[name] = emb

    return embeddings

reference_embeddings = load_reference_embeddings()

# -------------------------
# הגדרות צד
# -------------------------
with st.sidebar:

    st.header("הגדרות")

    threshold = st.slider(
        "Threshold",
        0.0,
        2.0,
        0.6
    )

    st.write("רשימת הכיתה:")
    st.write(STUDENT_ROSTER)

# -------------------------
# קלט משתמש
# -------------------------
st.subheader("העלי וקטור תמונות פנים מהכיתה")

test_files = st.file_uploader(
    "Upload faces",
    accept_multiple_files=True,
    type=["jpg","png","jpeg"]
)

# -------------------------
# זיהוי
# -------------------------
if st.button("בצע בדיקת נוכחות"):

    if not test_files:

        st.warning("יש להעלות תמונות פנים")
        st.stop()

    present_students = {}
    
    for face_file in test_files:

        img = Image.open(face_file)

        emb = model.predict(
            preprocess_image(img),
            verbose=0
        )

        best_name = None
        min_dist = float("inf")

        for name, ref_emb in reference_embeddings.items():

            dist = np.linalg.norm(emb - ref_emb)

            if dist < min_dist and dist < threshold:

                min_dist = dist
                best_name = name

        if best_name:

            present_students[best_name] = img

    # -------------------------
    # חסרים
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
