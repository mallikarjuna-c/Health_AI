"""
Level 1: Heart disease risk API
POST /predict  -> risk probability and a plain-language message
GET  /         -> simple HTML form
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "..", "QA", "model.joblib")

SCREENING_THRESHOLD = 0.18
HIGH_RISK_THRESHOLD = 0.50

bundle = joblib.load(MODEL_PATH)
model, scaler, columns = bundle["model"], bundle["scaler"], bundle["columns"]

app = FastAPI(title="Heart Disease Risk API")


class Patient(BaseModel):
    age: int
    sex: int          # 1 = male, 0 = female
    cp: int           # chest pain type 1-4
    trestbps: int     # resting blood pressure (mm Hg)
    chol: int         # serum cholesterol (mg/dl)
    fbs: int          # fasting blood sugar > 120 mg/dl (1 = yes)
    restecg: int      # resting ECG result 0-2
    thalach: int      # max heart rate achieved
    exang: int        # exercise-induced angina (1 = yes)
    oldpeak: float    # ST depression from exercise
    slope: int        # slope of peak exercise ST segment 1-3
    ca: int           # number of major vessels 0-3
    thal: int         # 3 = normal, 6 = fixed defect, 7 = reversible defect


def to_features(patient):
    row = pd.DataFrame([patient.model_dump()])
    row = pd.get_dummies(row, columns=["cp", "restecg", "slope", "thal"], dtype=int)
    row = row.reindex(columns=columns, fill_value=0)
    return scaler.transform(row)


def risk_message(prob):
    if prob >= HIGH_RISK_THRESHOLD:
        return "High risk", "The model shows a high chance of heart disease. Please see a cardiologist soon."
    if prob >= SCREENING_THRESHOLD:
        return "Moderate risk", "Some warning signs found. A check-up with a doctor is recommended."
    return "Low risk", "No strong signs of heart disease. Keep up regular check-ups."


@app.post("/predict")
def predict(patient: Patient):
    prob = float(model.predict_proba(to_features(patient))[0, 1])
    level, message = risk_message(prob)
    return {"risk_probability": round(prob, 3), "risk_level": level, "message": message}


@app.get("/")
def home():
    return FileResponse(os.path.join(HERE, "index.html"))
