import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def one_vs_rest_encoding(y, digit=1):
    return np.where(y == digit, 1, -1)
def score(X, w):
    return X.dot(w)

def svm_objective(w, X, y, lambda1=0.08):
    """Computes the Primal SVM objective function: 
    L(w) = lambda/2 * ||w||^2 + hinge loss"""
    result = lambda1/2.0 * np.dot(w,w) + np.mean(np.maximum(0, 1 - y * score(X, w)))
    return result

def pegasos(X_train, y_train, lambda1=0.08, num_iters=3):
    N = X_train.shape[0]
    d = X_train.shape[1]
    t = 0
    # Initial weight vector w as a vector of ones
    w = np.ones((d,))

    for iter in range(num_iters):
        for i in range(N):
            t = t + 1
            # Step size and regularization scaling
            w = (1 - 1/t) * w
            
            # Subgradient update if margin is violated (y * score < 1)
            if y_train[i] * score(X_train[i], w) < 1:
                w = w + (1/(lambda1*t)) * y_train[i] * X_train[i]
        
    return w

def SVM(X, y):
    y_encoded = one_vs_rest_encoding(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    w_final = pegasos(X_train_scaled, y_train, lambda1=0.08, num_iters=15)

    test_scores = score(X_test_scaled, w_final)
    predictions = np.where(test_scores > 0, 1, -1)

    # Calculate Accuracy
    accuracy = np.mean(predictions == y_test)

    # Calculate Precision and Recall manually
    tp = np.sum((predictions == 1) & (y_test == 1))
    fp = np.sum((predictions == 1) & (y_test == -1))
    fn = np.sum((predictions == -1) & (y_test == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return accuracy, precision, recall