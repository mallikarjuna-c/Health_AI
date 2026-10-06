"""
Level 1: 
Cleaning the UCI Heart Disease data and training Logistic Regression and Random Forest.
Seed S = 3025.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

SEED = 3025
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heart_disease.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")


def load_and_clean(path):
    df = pd.read_csv(path)

    for col in ["ca", "thal"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    df = pd.get_dummies(df, columns=["cp", "restecg", "slope", "thal"], drop_first=True, dtype=int)
    return df


def report(name, y_true, y_pred):
    print(f"{name:<20} accuracy={accuracy_score(y_true, y_pred):.3f}  "
          f"precision={precision_score(y_true, y_pred):.3f}  "
          f"recall={recall_score(y_true, y_pred):.3f}")


def main():
    df = load_and_clean(DATA_PATH)
    X = df.drop(columns="target")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows, Features: {X.shape[1]}\n")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000, random_state=SEED)
    log_reg.fit(X_train_s, y_train)
    report("Logistic Regression", y_test, log_reg.predict(X_test_s))

    joblib.dump({"model": log_reg, "scaler": scaler, "columns": X.columns.tolist()}, MODEL_PATH)

    forest = RandomForestClassifier(n_estimators=200, random_state=SEED)
    forest.fit(X_train, y_train)
    report("Random Forest", y_test, forest.predict(X_test))


if __name__ == "__main__":
    main()
