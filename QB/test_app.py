"""
Level 2: automated tests.
Run from inside the QB folder:  python -m pytest -v
"""

import pytest
from fastapi.testclient import TestClient

import db
from app import app

VALID_PATIENT = {
    "age": 55, "sex": 1, "cp": 4, "trestbps": 130, "chol": 240, "fbs": 0,
    "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 2,
    "ca": 0, "thal": 3,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    # each test uses its own empty database, so predictions.db is not touched
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    return TestClient(app)


def test_predict_valid_patient(client):
    res = client.post("/predict", json=VALID_PATIENT)
    assert res.status_code == 200
    body = res.json()
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert body["risk_level"] in ("Low risk", "Moderate risk", "High risk")


def test_bad_input_gives_clear_error(client):
    bad = dict(VALID_PATIENT, age=150, chol="abc")
    res = client.post("/predict", json=bad)
    assert res.status_code == 422
    errors = " ".join(res.json()["detail"])
    assert "age must be a whole number from 1 to 120" in errors
    assert "chol must be" in errors


def test_stats_after_requests(client):
    low = dict(VALID_PATIENT, cp=3, thalach=180, oldpeak=0.0, slope=1)
    high = dict(VALID_PATIENT, ca=3, thal=7, exang=1, oldpeak=3.0)
    client.post("/predict", json=low)
    client.post("/predict", json=high)
    client.post("/predict", json=dict(VALID_PATIENT, age=0))   # rejected, should not be saved

    stats = client.get("/stats").json()
    assert stats["total_requests"] == 2
    assert stats["high_risk_share"] == 0.5
    assert 0.0 < stats["average_risk"] < 1.0
