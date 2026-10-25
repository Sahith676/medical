
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Risk Prediction")
st.write("Enter the clinical measurements below.")

MODEL_PATH = Path(__file__).parent / "heart_disease_best_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("Could not load the saved model.")
    st.code(str(e))
    st.stop()

with st.form("prediction_form"):
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest pain type", [1, 2, 3, 4])
    trestbps = st.number_input("Resting blood pressure", 50, 250, 120)
    chol = st.number_input("Cholesterol", 50, 700, 200)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG result", [0, 1, 2])
    thalach = st.number_input("Maximum heart rate", 50, 250, 150)
    exang = st.selectbox("Exercise-induced angina", [0, 1])
    oldpeak = st.number_input("ST depression", 0.0, 10.0, 1.0, 0.1)
    slope = st.selectbox("ST segment slope", [1, 2, 3])
    ca = st.selectbox("Major vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia category", [3, 6, 7])

    submitted = st.form_submit_button(
        "Predict", type="primary", use_container_width=True
    )

if submitted:
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    try:
        prediction = int(model.predict(input_df)[0])

        st.subheader("Prediction result")

        if prediction == 0:
            st.success("Model prediction: No heart disease indicated")
        else:
            st.warning("Model prediction: Heart disease indicated")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            st.write("Model class probabilities:")
            for label, probability in zip(model.classes_, probabilities):
                st.write(f"Class {label}: {probability:.1%}")

        st.caption(
            "Educational use only; this result is not a medical diagnosis."
        )

    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))
