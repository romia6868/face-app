import streamlit as st
import PIL.Image as Image
import numpy as np
import tensorflow as tf
import os

# --- הגדרות דף ---
st.set_page_config(page_title="מערכת נוכחות חכמה", layout="wide")

# --- שלב 1: טעינת המודל מהנתיב המלא בדרייב ---
@st.cache_resource
def load_my_model():
    # הנתיב המלא שסיפקת
    model_path = '/content/drive/MyDrive/Presence_Project/my_siamese2_model.h5'
    
    if not os.path.exists(model_path):
        st.error(f"לא נמצא קובץ בנתיב: {model_path}. וודאי שהדרייב מחובר (Mounted).")
        return None
    
    try:
        # טעינת מודל Keras
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"שגיאה בטעינת המודל: {e}")
        return None

model = load_my_model()

# --- שלב 2: פונקציית עיבוד תמונה ---
def preprocess_image(pil_image):
    # התאמה לרזולוציית האימון (224x224 היא הסטנדרטית)
    img = pil_image.convert('RGB').resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- שלב 3: רשימת התלמידים הקבועה ---
STUDENT_ROSTER = ['Maayan', 'Tomer', 'Roei', 'Zohar', 'Ilay']

st.title("📸 מערכת רישום נוכחות - פרויקט סיאמי")

if model:
    # פאנל צדי
    with st.sidebar:
        st.header("הגדרות")
        threshold = st.slider("סף רגישות (Threshold)", 0.0, 1.5, 0.6)
        st.write("רשימת תלמידים לבדיקה:", STUDENT_ROSTER)

    # העלאת קבצים
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. העלאת תמונות מקור")
        ref_files = st.file_uploader("העלי תמונות (שם הקובץ = שם התלמיד)", accept_multiple_files=True)

    with col2:
        st.subheader("2. העלאת תמונות מהכיתה")
        test_files = st.file_uploader("העלי את וקטור התמונות לזיהוי", accept_multiple_files=True)

    if st.button("בצע זיהוי והצלבה"):
        if ref_files and test_files:
            # חילוץ וקטורי ייחוס
            known_embeddings = {}
            for ref in ref_files:
                name = os.path.splitext(ref.name)[0]
                img = Image.open(ref)
                emb = model.predict(preprocess_image(img), verbose=0)
                known_embeddings[name] = emb

            # זיהוי בתמונות הכיתה
            present_found = {}
            
            for test_f in test_files:
                t_img = Image.open(test_f)
                t_emb = model.predict(preprocess_image(t_img), verbose=0)
                
                best_match = None
                min_dist = float('inf')
                
                for name, r_emb in known_embeddings.items():
                    dist = np.linalg.norm(t_emb - r_emb)
                    if dist < min_dist and dist < threshold:
                        min_dist = dist
                        best_match = name
                
                if best_match:
                    present_found[best_match] = t_img

            # הצגת תוצאות
            absent_list = [n for n in STUDENT_ROSTER if n not in present_found]
            
            st.divider()
            
            # נוכחים
            st.header(f"✅ נוכחים ({len(present_found)})")
            c = st.columns(4)
            for idx, (name, img) in enumerate(present_found.items()):
                with c[idx % 4]:
                    st.image(img, caption=name)

            # חסרים
            st.header(f"❌ חסרים ({len(absent_list)})")
            if absent_list:
                st.error(", ".join(absent_list))
            else:
                st.success("כולם נמצאו!")
        else:
            st.warning("אנא העלי את כל הקבצים הדרושים.")
