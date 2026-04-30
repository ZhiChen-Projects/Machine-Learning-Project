import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

from dataset import load_data, split_data, normalize_data

X, y, feature_names = load_data("heart.csv")
feature_names = [f.strip() for f in feature_names]
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def score(X, w, b):
    return X.dot(w) + b

def loss(X, y, w, b, reg='none', lam=0.0):
    y_pred = sigmoid(score(X, w, b))
    bce = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    if reg == 'ridge':
        return bce + lam * np.sum(w**2)
    elif reg == 'lasso':
        return bce + lam * np.sum(np.abs(w))
    return bce

def gradient(X, y, w, b, reg='none', lam=0.0):
    y_pred = sigmoid(score(X, w, b))
    error = y_pred - y
    dw = X.T.dot(error) / len(y)
    db = np.mean(error)
    if reg == 'ridge':
        dw += 2 * lam * w
    elif reg == 'lasso':
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

def print_metrics(label, y_true, y_pred):
    accuracy, precision, recall, specificity = compute_metrics(y_true, y_pred)
    print(f"\n--- {label} ---")
    print(f"Accuracy:    {accuracy    * 100:.2f}%")
    print(f"Precision:   {precision   * 100:.2f}%")
    print(f"Recall:      {recall      * 100:.2f}%")
    print(f"Specificity: {specificity * 100:.2f}%")

def train_lr(X_train, y_train, X_val, y_val, iter_num=3000, alpha=0.01, reg='none', lam=0.0):
    w = np.zeros(X_train.shape[1])
    b = 0.0
    train_cost_func = []
    val_cost_func = []

    for cnt in range(iter_num):
        if cnt % 500 == 0:
            print(f"Iteration {cnt}/{iter_num}")

        dw, db = gradient(X_train, y_train, w, b, reg=reg, lam=lam)
        w -= alpha * dw
        b -= alpha * db

        train_cost_func.append(loss(X_train, y_train, w, b, reg=reg, lam=lam))
        val_cost_func.append(loss(X_val, y_val, w, b))

    return w, b, train_cost_func, val_cost_func


print("Training standard...")
w_std,   b_std,   train_cost_std,   val_cost_std   = train_lr(X_train, y_train, X_val, y_val, reg='none')

print("Training ridge...")
w_ridge, b_ridge, train_cost_ridge, val_cost_ridge = train_lr(X_train, y_train, X_val, y_val, reg='ridge', lam=0.01)

print("Training lasso...")
w_lasso, b_lasso, train_cost_lasso, val_cost_lasso = train_lr(X_train, y_train, X_val, y_val, reg='lasso', lam=0.01)

print_metrics("Standard", y_test, predict(X_test, w_std,   b_std))
print_metrics("Ridge",    y_test, predict(X_test, w_ridge, b_ridge))
print_metrics("Lasso",    y_test, predict(X_test, w_lasso, b_lasso))

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, train_cost, val_cost, title in zip(
    axes,
    [train_cost_std,   train_cost_ridge,  train_cost_lasso],
    [val_cost_std,     val_cost_ridge,    val_cost_lasso],
    ["Standard",       "Ridge (λ=0.01)",  "Lasso (λ=0.01)"]
):
    ax.plot(train_cost, label="Train")
    ax.plot(val_cost,   label="Val")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend()

plt.suptitle("Logistic Regression Learning Curves")
plt.tight_layout()
plt.show()