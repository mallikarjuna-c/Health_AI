"""
Level 3: Reason
Lower the decision threshold until recall reaches 0.9 .
Seed S = 3025.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from level1_train import load_and_clean, DATA_PATH, SEED
from numpy_model import confusion_matrix, metrics

TARGET_RECALL = 0.9


def main():
    df = load_and_clean(DATA_PATH)
    X = df.drop(columns="target")
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(X_train_s, y_train)
    probs = model.predict_proba(X_test_s)[:, 1]

    print(f"Test set: {len(y_test)} patients, {y_test.sum()} with heart disease\n")
    print(f"{'threshold':>9} {'TP':>4} {'FN':>4} {'FP':>4} {'TN':>4} "
          f"{'accuracy':>9} {'precision':>10} {'recall':>7}")

    chosen = None
    for t in np.arange(0.50, 0.04, -0.05):
        y_pred = (probs >= t).astype(int)
        tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
        acc, prec, rec = metrics(tp, tn, fp, fn)
        print(f"{t:>9.2f} {tp:>4} {fn:>4} {fp:>4} {tn:>4} "
              f"{acc:>9.3f} {prec:>10.3f} {rec:>7.3f}")

    for t in np.arange(0.50, 0.00, -0.01):
        y_pred = (probs >= t).astype(int)
        tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
        acc, prec, rec = metrics(tp, tn, fp, fn)
        if rec >= TARGET_RECALL:
            chosen = (t, tp, tn, fp, fn, acc, prec, rec)
            break

    t, tp, tn, fp, fn, acc, prec, rec = chosen
    print(f"\nHighest threshold with recall >= {TARGET_RECALL}: {t:.2f}")
    print(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"  accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}")

    baseline_acc = np.mean(y_test == 0)
    print(f"\nAlways predicting 'no disease': accuracy={baseline_acc:.3f}, recall=0.000")


if __name__ == "__main__":
    main()
