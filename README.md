# Health AI - Heart Disease Risk Prediction and App

Technical Assignment: AI for Personal Health and Wellness

**Questions answered:** A (Predict a health risk) and B (Turn a model into a usable app)

**Seed S = 3025** (last four digits of USN 1DA23AI025). Used as `random_state` in every
train/test split and every model.

## Project structure

```
Health_AI/
├── QA/                          Question A - Predict a health risk
│   ├── get_data.py              downloads UCI Heart Disease data -> heart_disease.csv
│   ├── level1_train.py          Level 1: cleaning, Logistic Regression + Random Forest, saves model.joblib
│   ├── numpy_model.py           Level 2: logistic regression and confusion matrix in NumPy
│   ├── level3_prediction.md     Level 3: prediction (committed before running)
│   ├── level3_threshold.py      Level 3: threshold experiment
│   └── level3_results.md        Level 3: results and reasoning
├── QB/                          Question B - Turn the model into an app
│   ├── app.py                   FastAPI: /predict, /stats, /health, / (form)
│   ├── db.py                    SQLite storage with hand-written SQL
│   ├── index.html               front end
│   ├── test_app.py              pytest tests
│   ├── level3_prediction.md     Level 3: prediction (committed before testing)
│   └── level3_results.md        Level 3: break / fix results, scaling for 100 users
├── PERSONAL_INTELLIGENCE.md     decision log and AI usage declaration
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Run steps

### Question A

```bash
python QA/get_data.py            # download the dataset (303 rows)
python QA/level1_train.py        # Level 1 - also saves QA/model.joblib for Question B
python QA/numpy_model.py         # Level 2
python QA/level3_threshold.py    # Level 3
```

### Question B

`QA/model.joblib` must exist first (created by `QA/level1_train.py`).

```bash
cd QB
python -m uvicorn app:app --reload
```

Open http://127.0.0.1:8000 for the form, http://127.0.0.1:8000/stats for statistics and
http://127.0.0.1:8000/docs for the API documentation.

Run the tests (from inside `QB`):

```bash
python -m pytest -v
```

## Results summary (seed 3025, 61 test patients)

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Logistic Regression (scikit-learn) | 0.820 | 0.870 | 0.714 |
| Random Forest | 0.754 | 0.810 | 0.607 |
| Logistic Regression (NumPy, Level 2) | 0.820 | 0.870 | 0.714 |
| Logistic Regression, threshold 0.18 (Level 3) | 0.738 | 0.650 | 0.929 |

## Sources and credits

- **Dataset:** UCI Machine Learning Repository - Heart Disease (Cleveland), Janosi, Steinbrunn,
  Pfisterer and Detrano (1988), https://archive.ics.uci.edu/dataset/45/heart+disease,
  loaded with the `ucimlrepo` package.
- **Libraries:** NumPy, pandas, scikit-learn, joblib, FastAPI, Uvicorn, Pydantic, pytest, httpx.
- **AI assistance:** Claude (Anthropic) - see `PERSONAL_INTELLIGENCE.md` for details.

This is a learning project and not a medical diagnosis tool.
