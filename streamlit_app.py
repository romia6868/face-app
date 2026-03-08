import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image

# טען את המודל שלך
model = load_model('/content/drive/MyDrive/Presence_Project/my_siamese2_model.h5')

# הכנס את רשימת השמות של התלמידים
student_names = ['Maayan', 'Tomer', 'Roei', 'Zohar', 'Ilay']


def identify_students(image, model, student_names):
    # עיבוד התמונה
    img = cv2.resize(image, (224, 224))
    img = np.expand_dims(img, axis=0)

    # זיהוי התלמידים באמצעות המודל
    predictions = model.predict(img)

    # השוואת התוצאות לרשימת השמות
    present = []
    absent = []
    for i, pred in enumerate(predictions[0]):
        if pred > 0.5:
            present.append(student_names[i])
        else:
            absent.append(student_names[i])

    return present, absent

def run_app():
    st.title("Student Attendance Tracker")

    # אפשר לאפליק לקלט תמונה
    uploaded_file = st.file_uploader("Choose a class photo", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Class Photo', use_column_width=True)

        # זיהוי התלמידים בתמונה
        present, absent = identify_students(np.array(image), model, student_names)

        # הצגת התוצאות
        st.subheader("Present Students:")
        for student in present:
            st.write(student)

        st.subheader("Absent Students:")
        for student in absent:
            st.write(student)

if __name__ == '__main__':
    run_app()
