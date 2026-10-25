import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Risk Prediction")
st.write(
    "Enter the clinical measurements below to obtain "
    "a prediction from the trained machine learning model."
)

st.info(
    "Educational project only. This application does not "
    "provide a medical diagnosis."
)

# --------------------------------------------------
# 2. MODEL PATH
# --------------------------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "heart_disease_best_model.joblib"
)

# --------------------------------------------------
# 3. LOAD MODEL ONCE AND CACHE IT
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(model_path):
    return joblib.load(model_path)


if not MODEL_PATH.is_file():
    st.error(
        "Model file not found. Make sure "
        "'heart_disease_best_model.joblib' is in the "
        "same folder as app.py."
    )
    st.stop()

try:
    with st.spinner("Loading prediction model..."):
        start_time = time.perf_counter()
        model = load_model(str(MODEL_PATH))
        load_seconds = time.perf_counter() - start_time

    st.caption(
        f"Model ready. Loading time: {load_seconds:.2f} seconds."
    )

except Exception as e:
    st.error("The model could not be loaded.")
    st.code(str(e))
    st.info(
        "Check that the model was saved in Colab using "
        "scikit-learn 1.6.1 and that requirements.txt "
        "uses a compatible version."
    )
    st.stop()

# --------------------------------------------------
# 4. PATIENT INPUT FORM
# --------------------------------------------------

st.subheader("Patient Information")

with st.form("heart_disease_form"):

    age = st.number_input(
        "Age (years)", min_value=1, max_value=120, value=50
    )

    sex = st.selectbox(
        "Sex", options=[0, 1],
        format_func=lambda x: "Female (0)" if x == 0 else "Male (1)"
    )

    cp = st.selectbox(
        "Chest pain type",
        options=[1, 2, 3, 4]
    )

    trestbps = st.number_input(
        "Resting blood pressure (mm Hg)",
        min_value=50, max_value=250, value=120
    )

    chol = st.number_input(
        "Serum cholesterol (mg/dl)",
        min_value=50, max_value=700, value=200
    )

    fbs = st.selectbox(
        "Fasting blood sugar > 120 mg/dl",
        options=[0, 1]
    )

    restecg = st.selectbox(
        "Resting ECG result",
        options=[0, 1, 2]
    )

    thalach = st.number_input(
        "Maximum heart rate achieved",
        min_value=50, max_value=250, value=150
    )

    exang = st.selectbox(
        "Exercise-induced angina",
        options=[0, 1]
    )

    oldpeak = st.number_input(
        "ST depression induced by exercise",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "ST segment slope",
        options=[1, 2, 3]
    )

    ca = st.selectbox(
        "Number of major vessels (0-3)",
        options=[0, 1, 2, 3]
    )

    thal = st.selectbox(
        "Thalassemia category",
        options=[3, 6, 7]
    )

    submitted = st.form_submit_button(
        "Predict Heart Disease",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# 5. PREDICTION
# --------------------------------------------------

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
        with st.spinner("Analyzing patient information..."):
            prediction_start = time.perf_counter()

            prediction = int(model.predict(input_df)[0])

            prediction_seconds = (
                time.perf_counter() - prediction_start
            )

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 0:
            st.success(
                "Model prediction: No heart disease indicated."
            )
        else:
            st.warning(
                "Model prediction: Heart disease indicated."
            )

        st.caption(
            f"Prediction completed in {prediction_seconds:.3f} seconds."
        )

        # ------------------------------------------
        # 6. CLASS PROBABILITIES, IF AVAILABLE
        # ------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            st.subheader("Model Class Probabilities")

            for label, probability in zip(
                model.classes_, probabilities
            ):
                class_name = (
                    "No heart disease"
                    if int(label) == 0
                    else "Heart disease"
                )

                st.write(
                    f"**{class_name}:** {probability:.1%}"
                )

                st.progress(float(probability))

            st.caption(
                "These are model-estimated probabilities, "
                "not a clinically validated assessment of risk."
            )

        # ------------------------------------------
        # 7. INPUT SUMMARY
        # ------------------------------------------

        with st.expander("Review entered information"):
            st.dataframe(
                input_df,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "This educational application is not a substitute "
            "for professional medical assessment. If you have "
            "symptoms or concerns, consult a qualified clinician."
        )

    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))
        st.write(
            "Verify that the saved model contains the complete "
            "preprocessing pipeline and expects these 13 features."
        )

# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "Heart Disease Risk Prediction | Machine Learning Project"
)