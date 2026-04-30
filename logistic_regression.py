import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from dataset import load_data, split_data, normalize_data


def apply_poly(X_train, X_val, X_test, continuous_cols, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)

    X_train_cont = poly.fit_transform(X_train[:, continuous_cols])
    X_val_cont   = poly.transform(X_val[:, continuous_cols])
    X_test_cont  = poly.transform(X_test[:, continuous_cols])

    n_orig = len(continuous_cols)
    X_train_new = X_train_cont[:, n_orig:]
    X_val_new   = X_val_cont[:,   n_orig:]
    X_test_new  = X_test_cont[:,  n_orig:]

    return (
        np.hstack([X_train, X_train_new]),
        np.hstack([X_val,   X_val_new]),
        np.hstack([X_test,  X_test_new])
    )


X, y, feature_names = load_data("heart.csv")
feature_names = [f.strip() for f in feature_names]

continuous_cols = [feature_names.index(f) for f in [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak'
]]

X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(X, y)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def score(X, w, b):
    return X.dot(w) + b

def loss(X, y, w, b, reg='none', lam=0.0):
    y_pred = np.clip(sigmoid(score(X, w, b)), 1e-7, 1 - 1e-7)
    bce = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    if reg == 'lasso':
        return bce + lam * np.sum(np.abs(w))
    return bce

def gradient(X, y, w, b, reg='none', lam=0.0):
    y_pred = sigmoid(score(X, w, b))
    error = y_pred - y
    dw = X.T.dot(error) / len(y)
    db = np.mean(error)
    if reg == 'lasso':
        dw += lam * np.sign(w)
    return dw, db

def predict(X, w, b, threshold=0.5):
    return (sigmoid(score(X, w, b)) >= threshold).astype(int)

def compute_metrics(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    accuracy    = (tp + tn) / len(y_true)
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return accuracy, precision, recall, specificity

def train_lr(X_train, y_train, X_val, y_val, iter_num=3000, alpha=0.01, reg='none', lam=0.0):
    w = np.zeros(X_train.shape[1])
    b = 0.0
    best_w, best_b = w.copy(), b
    best_val_loss = float('inf')

    for cnt in range(iter_num):
        dw, db = gradient(X_train, y_train, w, b, reg=reg, lam=lam)
        w -= alpha * dw
        b -= alpha * db

        val_loss = loss(X_val, y_val, w, b)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_w, best_b = w.copy(), b

    return best_w, best_b


degrees = [1, 2, 3, 4]
lambdas = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]

print("Degree\tLambda\tAccuracy\tPrecision\tRecall\t\tSpecificity")
print("-" * 75)

plt.figure(figsize=(10, 6))

for degree in degrees:
    if degree == 1:
        X_tr, X_v, X_te = normalize_data(X_train_raw, X_val_raw, X_test_raw)
    else:
        X_tr, X_v, X_te = apply_poly(X_train_raw, X_val_raw, X_test_raw, continuous_cols, degree=degree)
        X_tr, X_v, X_te = normalize_data(X_tr, X_v, X_te)

    accuracies = []

    for lam in lambdas:
        reg = 'lasso' if lam > 0.0001 else 'none'
        w, b = train_lr(X_tr, y_train, X_v, y_val, reg=reg, lam=lam)
        acc, prec, rec, spec = compute_metrics(y_test, predict(X_te, w, b))
        accuracies.append(acc * 100)
        print(f"{degree}\t{lam}\t{acc*100:.2f}\t{prec*100:.2f}\t{rec*100:.2f}\t{spec*100:.2f}")

    plt.plot(lambdas, accuracies, marker='o', label=f'Degree {degree}')

plt.xlabel("Lambda (λ)")
plt.ylabel("Accuracy (%)")
plt.title("Logistic Regression (Lasso) Accuracy vs Lambda by Degree")
plt.legend()
plt.xscale('log')
plt.xticks(lambdas, [str(l) for l in lambdas], rotation=45, ha='right')
plt.ylim(70, 100)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("logistic_lasso_comparison.png", dpi=150, bbox_inches='tight')
plt.show()