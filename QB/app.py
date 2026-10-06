"""
Heart disease risk API
POST /predict  -> risk probability and a plain-language message (saved to SQLite)
GET  /stats    -> total requests, average risk, share of high-risk results
GET  /         -> simple HTML form
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

import db

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "..", "QA", "model.joblib")

SCREENING_THRESHOLD = 0.18
HIGH_RISK_THRESHOLD = 0.50

bundle = joblib.load(MODEL_PATH)
model, scaler, columns = bundle["model"], bundle["scaler"], bundle["columns"]

app = FastAPI(title="Heart Disease Risk API")
db.init_db()

# allowed values for each field, used in the error messages
RULES = {
    "age": "a whole number from 1 to 120",
    "sex": "0 (female) or 1 (male)",
    "cp": "a chest pain type from 1 to 4",
    "trestbps": "a blood pressure from 50 to 250 mm Hg",
    "chol": "a cholesterol value from 100 to 600 mg/dl",
    "fbs": "0 (no) or 1 (yes)",
    "restecg": "an ECG result from 0 to 2",
    "thalach": "a heart rate from 60 to 220",
    "exang": "0 (no) or 1 (yes)",
    "oldpeak": "a number from 0 to 7",
    "slope": "a slope type from 1 to 3",
    "ca": "a number of vessels from 0 to 3",
    "thal": "3 (normal), 6 (fixed defect) or 7 (reversible defect)",
}


class Patient(BaseModel):
    age: int = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)          # 1 = male, 0 = female
    cp: int = Field(ge=1, le=4)           # chest pain type
    trestbps: int = Field(ge=50, le=250)  # resting blood pressure (mm Hg)
    chol: int = Field(ge=100, le=600)     # serum cholesterol (mg/dl)
    fbs: int = Field(ge=0, le=1)          # fasting blood sugar > 120 mg/dl
    restecg: int = Field(ge=0, le=2)      # resting ECG result
    thalach: int = Field(ge=60, le=220)   # max heart rate achieved
    exang: int = Field(ge=0, le=1)        # exercise-induced angina
    oldpeak: float = Field(ge=0, le=7)    # ST depression from exercise
    slope: int = Field(ge=1, le=3)        # slope of peak exercise ST segment
    ca: int = Field(ge=0, le=3)           # number of major vessels
    thal: int                             # 3 = normal, 6 = fixed, 7 = reversible

    @field_validator("thal")
    @classmethod
    def check_thal(cls, value):
        if value not in (3, 6, 7):
            raise ValueError("invalid thal")
        return value


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # one clear sentence per wrong field instead of pydantic's technical output
    messages = []
    for err in exc.errors():
        field = str(err["loc"][-1])
        if err["type"] == "missing":
            messages.append(f"{field} is required.")
        elif field in RULES:
            messages.append(f"{field} must be {RULES[field]} (you sent {err.get('input')!r}).")
        else:
            messages.append(f"{field}: {err['msg']}")
    return JSONResponse(status_code=422, content={"detail": messages})


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
    db.save_prediction(patient.model_dump(), prob, level)
    return {"risk_probability": round(prob, 3), "risk_level": level, "message": message}


@app.get("/stats")
def stats():
    return db.get_stats()


@app.get("/")
def home():
    return FileResponse(os.path.join(HERE, "index.html"))
