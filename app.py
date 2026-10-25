
import streamlit as st
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "heart_disease_best_model.joblib"

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

st.title("❤️ Heart Disease Risk Prediction")
st.write(
    "Enter the clinical measurements below to obtain "
    "a model-based prediction."
)

try:
    bundle = load_bundle()
except Exception as e:
    st.error("Unable to load the model file.")
    st.code(str(e))
    st.stop()

# Support a saved sklearn Pipeline or a dictionary bundle.
if isinstance(bundle, dict):
    predictor = (
        bundle.get("pipeline")
        or bundle.get("model")
        or bundle.get("best_model")
    )
    preprocessor = bundle.get("preprocessor")
    feature_names = bundle.get("feature_names")
else:
    predictor = bundle
    preprocessor = None
    feature_names = None

if predictor is None:
    st.error(
        "The saved bundle does not contain a recognized "
        "pipeline or model. Check how the model was saved."
    )
    st.stop()

# Cleveland UCI Heart Disease features
default_features = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

if feature_names is None:
    feature_names = default_features

# Standard Cleveland feature inputs
with st.form("patient_form"):
    st.subheader("Patient information")

    age = st.number_input("Age (years)", 1, 120, 50)
    sex = st.selectbox("Sex", [0, 1],
                       format_func=lambda x:
                       "Female (0)" if x == 0 else "Male (1)")

    cp = st.selectbox("Chest pain type", [1, 2, 3, 4])
    trestbps = st.number_input(
        "Resting blood pressure (mm Hg)", 50, 250, 120
    )
    chol = st.number_input(
        "Serum cholesterol (mg/dl)", 50, 700, 200
    )
    fbs = st.selectbox(
        "Fasting blood sugar > 120 mg/dl", [0, 1]
    )
    restecg = st.selectbox("Resting ECG result", [0, 1, 2])
    thalach = st.number_input(
        "Maximum heart rate achieved", 50, 250, 150
    )
    exang = st.selectbox(
        "Exercise-induced angina", [0, 1]
    )
    oldpeak = st.number_input(
        "ST depression induced by exercise",
        min_value=0.0, max_value=10.0,
        value=1.0, step=0.1
    )
    slope = st.selectbox("ST segment slope", [1, 2, 3])
    ca = st.selectbox(
        "Number of major vessels (0–3)", [0, 1, 2, 3]
    )
    thal = st.selectbox(
        "Thalassemia category", [3, 6, 7]
    )

    submitted = st.form_submit_button(
        "Predict", type="primary", use_container_width=True
    )

if submitted:
    values = {
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
    }

    try:
        import pandas as pd

        input_df = pd.DataFrame([
            {name: values[name] for name in feature_names}
        ])

        # A saved Pipeline performs its own preprocessing.
        # A separate model requires its saved preprocessor.
        if preprocessor is not None:
            processed = preprocessor.transform(input_df)
            prediction = predictor.predict(processed)
        else:
            prediction = predictor.predict(input_df)

        result = int(np.asarray(prediction).ravel()[0])

        st.subheader("Prediction result")

        if result == 0:
            st.success("Model prediction: No heart disease detected")
        else:
            st.warning("Model prediction: Heart disease indicated")

        # Show probability only when supported by the model.
        if hasattr(predictor, "predict_proba"):
            if preprocessor is not None:
                probability_input = processed
            else:
                probability_input = input_df

            probabilities = predictor.predict_proba(
                probability_input
            )[0]

            st.write("Model class probabilities:")
            for class_label, probability in zip(
                predictor.classes_, probabilities
            ):
                st.write(
                    f"Class {class_label}: {probability:.1%}"
                )

        st.caption(
            "Educational demonstration only. This prediction "
            "is not a medical diagnosis. Consult a qualified "
            "healthcare professional for medical assessment."
        )

    except Exception as e:
        st.error(
            "Prediction failed. The model's expected input "
            "features or preprocessing may differ from this app."
        )
        st.code(str(e))
