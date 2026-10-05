"""
Level 2:
Logistic regression (sigmoid, loss, gradient descent) and a confusion matrix
Seed S = 3025.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from level1_train import load_and_clean, DATA_PATH, SEED


# Logistic regression

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def log_loss(y, p, w, l2):
    eps = 1e-12
    bce = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    return bce + (l2 / 2) * np.sum(w ** 2)


def train_logistic(X, y, lr=0.1, epochs=5000, l2=0.0):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    history = []

    for epoch in range(epochs):
        p = sigmoid(X @ w + b)
        error = p - y

        # gradients of the loss 
        dw = (X.T @ error) / n_samples + l2 * w
        db = np.mean(error)

        w -= lr * dw
        b -= lr * db

        if epoch % 1000 == 0 or epoch == epochs - 1:
            history.append((epoch, log_loss(y, p, w, l2)))

    return w, b, history


def predict(X, w, b, threshold=0.5):
    return (sigmoid(X @ w + b) >= threshold).astype(int)


# Confusion matrix

def confusion_matrix(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return accuracy, precision, recall


def print_confusion(name, tp, tn, fp, fn):
    print(f"\n{name}")
    print("                 Pred 0   Pred 1")
    print(f"  Actual 0 (no)   {tn:>5}    {fp:>5}")
    print(f"  Actual 1 (yes)  {fn:>5}    {tp:>5}")
    acc, prec, rec = metrics(tp, tn, fp, fn)
    print(f"  accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}")


def main():
    df = load_and_clean(DATA_PATH)
    X = df.drop(columns="target")
    y = df["target"].values
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    l2 = 1.0 / (1.0 * len(X_train_s))

    w, b, history = train_logistic(X_train_s, y_train, lr=0.1, epochs=5000, l2=l2)

    print("Training loss (my model):")
    for epoch, loss in history:
        print(f"  epoch {epoch:>4}: loss = {loss:.4f}")

    my_pred = predict(X_test_s, w, b)

    sk_model = LogisticRegression(max_iter=1000, random_state=SEED)
    sk_model.fit(X_train_s, y_train)
    sk_pred = sk_model.predict(X_test_s)

    print_confusion("My logistic regression (NumPy)", *confusion_matrix(y_test, my_pred))
    print_confusion("scikit-learn LogisticRegression", *confusion_matrix(y_test, sk_pred))

    agree = np.mean(my_pred == sk_pred)
    print(f"\nPredictions that match between the two models: {agree:.1%}")

    sk_w = sk_model.coef_[0]
    top3 = np.argsort(-np.abs(w))[:3]
    print("\nTop 3 features by |weight|:")
    print(f"  {'feature':<12} {'my weight':>10} {'sklearn':>10}")
    for i in top3:
        print(f"  {feature_names[i]:<12} {w[i]:>10.4f} {sk_w[i]:>10.4f}")
    print(f"  {'bias':<12} {b:>10.4f} {sk_model.intercept_[0]:>10.4f}")


if __name__ == "__main__":
    main()
