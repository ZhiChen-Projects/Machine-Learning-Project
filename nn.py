import matplotlib.pyplot as plt
import numpy as np
import numpy.random as r

from dataset import load_data, split_data, normalize_data

X, y, feature_names = load_data("heart.csv")
feature_names = [f.strip() for f in feature_names]
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)

y_train = y_train.reshape(-1, 1)
y_val   = y_val.reshape(-1, 1)
y_test  = y_test.reshape(-1, 1)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

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
    n_layers = len(W)
    for l in range(1, n_layers + 1):
        z[l+1] = a[l].dot(W[l].T) + b[l]
        if l < n_layers:
            a[l+1] = relu(z[l+1])
        else:
            a[l+1] = sigmoid(z[l+1])
        if training and dropout_rate > 0.0 and l < n_layers:
            mask = (r.random_sample(a[l+1].shape) > dropout_rate).astype(float)
            a[l+1] *= mask / (1.0 - dropout_rate)
            masks[l+1] = mask
    return a, z, masks

def train_nn(nn_structure, X_train, y_train, X_val, y_val, iter_num=3000, alpha=2.5, dropout_rate=0.0):
    W, b = setup_and_init_weights(nn_structure)
    N = len(y_train)
    L = len(nn_structure)
    best_W, best_b = None, None
    best_val_loss = float('inf')

    for cnt in range(iter_num):
        a, z, masks = feed_forward(X_train, W, b, dropout_rate=dropout_rate, training=True)

        delta = {}
        delta[L] = -(y_train - a[L]) * sigmoid_deriv(z[L])

        for l in range(L - 1, 1, -1):
            delta[l] = delta[l+1].dot(W[l]) * relu_deriv(z[l])
            if l in masks:
                delta[l] *= masks[l] / (1.0 - dropout_rate)

        for l in range(1, L):
            W[l] -= alpha * (delta[l+1].T.dot(a[l]) / N)
            b[l] -= alpha * (delta[l+1].mean(axis=0, keepdims=True))

        a_val, _, _ = feed_forward(X_val, W, b, dropout_rate=0.0, training=False)
        val_loss = np.mean(np.linalg.norm(y_val - a_val[L], axis=1))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_W = {l: W[l].copy() for l in W}
            best_b = {l: b[l].copy() for l in b}

    return best_W, best_b

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


dropout_rates  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
architectures  = {
    '1 hidden layer':  [13, 9, 1],
    '2 hidden layers': [13, 9, 9, 1],
    '3 hidden layers': [13, 9, 9, 9, 1],
}

print("Architecture\t\tDropout\tAccuracy\tPrecision\tRecall\t\tSpecificity")
print("-" * 90)

plt.figure(figsize=(10, 6))

for label, nn_structure in architectures.items():
    accuracies = []
    for dropout_rate in dropout_rates:
        W, b = train_nn(nn_structure, X_train, y_train, X_val, y_val,
                        dropout_rate=dropout_rate, iter_num=3000)
        y_pred = predict_y(W, b, X_test)
        accuracy, precision, recall, specificity = compute_metrics(y_test, y_pred)
        accuracies.append(accuracy * 100)
        print(f"{label}\t\t{dropout_rate}\t{accuracy*100:.2f}\t\t{precision*100:.2f}\t\t{recall*100:.2f}\t\t{specificity*100:.2f}")

    plt.plot(dropout_rates, accuracies, marker='o', label=label)

plt.xlabel("Dropout Rate")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Dropout Rate by Architecture")
plt.legend()
plt.xticks(dropout_rates)
plt.ylim(70, 100)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("nn_architecture_dropout.png", dpi=150, bbox_inches='tight')
plt.show()