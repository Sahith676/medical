
import gradio as gr
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "heart_disease_best_model.joblib"
model = joblib.load(MODEL_PATH)

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

def predict_heart_disease(
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
):
    values = [
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]

    patient = pd.DataFrame([values], columns=FEATURES)
    prediction = int(model.predict(patient)[0])

    result = (
        "No heart disease indicated by the model."
        if prediction == 0
        else "Heart disease indicated by the model."
    )

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(patient)[0]
        details = "\n".join(
            f"Class {label}: {prob:.1%}"
            for label, prob in zip(model.classes_, probs)
        )
        return result + "\n\nModel class probabilities:\n" + details

    return result

demo = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Age", value=50),
        gr.Dropdown([0, 1], label="Sex"),
        gr.Dropdown([1, 2, 3, 4], label="Chest pain type"),
        gr.Number(label="Resting blood pressure", value=120),
        gr.Number(label="Cholesterol", value=200),
        gr.Dropdown([0, 1], label="Fasting blood sugar > 120"),
        gr.Dropdown([0, 1, 2], label="Resting ECG"),
        gr.Number(label="Maximum heart rate", value=150),
        gr.Dropdown([0, 1], label="Exercise-induced angina"),
        gr.Number(label="ST depression", value=1.0),
        gr.Dropdown([1, 2, 3], label="ST slope"),
        gr.Dropdown([0, 1, 2, 3], label="Major vessels"),
        gr.Dropdown([3, 6, 7], label="Thalassemia")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Heart Disease Risk Prediction",
    description=(
        "Educational machine-learning demo. "
        "Not a medical diagnosis."
    )
)

demo.launch()
