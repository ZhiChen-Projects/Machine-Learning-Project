import matplotlib.pyplot as plt
import numpy as np
import numpy.random as r
from sklearn.metrics import accuracy_score

from dataset import load_data, split_data, normalize_data

X, y, feature_names = load_data("heart.csv")
feature_names = [f.strip() for f in feature_names]
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)

y_train = y_train.reshape(-1, 1)
y_val   = y_val.reshape(-1, 1)
y_test  = y_test.reshape(-1, 1)

def f(z):
    return 1 / (1 + np.exp(-z))

def f_deriv(z):
    return f(z) * (1 - f(z))

def setup_and_init_weights(nn_structure):
    W, b = {}, {}
    for l in range(1, len(nn_structure)):
        scale = np.sqrt(2.0 / nn_structure[l-1])
        W[l] = r.randn(nn_structure[l], nn_structure[l-1]) * scale
        b[l] = np.zeros((1, nn_structure[l]))
    return W, b

def feed_forward(X, W, b, dropout_rate=0.0, training=True):
    a = {1: X}
    z = {}
    masks = {}
    for l in range(1, len(W) + 1):
        z[l+1] = a[l].dot(W[l].T) + b[l]
        a[l+1] = f(z[l+1])
        if training and dropout_rate > 0.0 and l < len(W):
            mask = (r.random_sample(a[l+1].shape) > dropout_rate).astype(float)
            a[l+1] *= mask / (1.0 - dropout_rate)
            masks[l+1] = mask
    return a, z, masks

def train_nn(nn_structure, X_train, y_train, X_val, y_val, iter_num=3000, alpha=2.5, dropout_rate=0.0):
    W, b = setup_and_init_weights(nn_structure)
    train_cost_func = []
    val_cost_func = []
    N = len(y_train)
    L = len(nn_structure)

    for cnt in range(iter_num):
        if cnt % 500 == 0:
            print(f"Iteration {cnt}/{iter_num}")

        a, z, masks = feed_forward(X_train, W, b, dropout_rate=dropout_rate, training=True)

        delta = {}
        delta[L] = -(y_train - a[L]) * f_deriv(z[L])

        for l in range(L - 1, 1, -1):
            delta[l] = delta[l+1].dot(W[l]) * f_deriv(z[l])
            if l in masks:
                delta[l] *= masks[l] / (1.0 - dropout_rate)

        for l in range(1, L):
            W[l] -= alpha * (delta[l+1].T.dot(a[l]) / N)
            b[l] -= alpha * (delta[l+1].mean(axis=0, keepdims=True))

        train_cost_func.append(np.mean(np.linalg.norm(y_train - a[L], axis=1)))

        a_val, _, _ = feed_forward(X_val, W, b, dropout_rate=0.0, training=False)
        val_cost_func.append(np.mean(np.linalg.norm(y_val - a_val[L], axis=1)))

    return W, b, train_cost_func, val_cost_func

def predict_y(W, b, X):
    a, _, _ = feed_forward(X, W, b, dropout_rate=0.0, training=False)
    L = max(a.keys())
    return (a[L] >= 0.5).astype(int).flatten()

def compute_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    accuracy    = (tp + tn) / len(y_true)
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return accuracy, precision, recall, specificity


nn_structure = [13, 9, 1]

W, b, train_cost, val_cost = train_nn(nn_structure, X_train, y_train, X_val, y_val, dropout_rate=0.0)
y_pred = predict_y(W, b, X_test)

accuracy, precision, recall, specificity = compute_metrics(y_test, y_pred)
print(f"Accuracy:    {accuracy    * 100:.2f}%")
print(f"Precision:   {precision   * 100:.2f}%")
print(f"Recall:      {recall      * 100:.2f}%")
print(f"Specificity: {specificity * 100:.2f}%")

plt.plot(train_cost, label="Train")
plt.plot(val_cost, label="Val")
plt.xlabel("Iteration")
plt.ylabel("Avg Cost")
plt.title("Learning Curve")
plt.legend()
plt.show()