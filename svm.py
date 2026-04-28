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

def rbf_kernel(x1, x2, gamma=0.1):
    # Calculates similarity between points
    if x1.ndim == 1 and x2.ndim == 1:
        return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
    # Batch calculation for efficiency
    sq_dist = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
    return np.exp(-gamma * sq_dist)

def kernel_pegasos(X_train, y_train, lambda1=0.08, gamma=0.1, num_iters=3):
    N = X_train.shape[0]
    alpha = np.zeros(N)
    t = 0

    for _ in range(num_iters):
        for i in range(N):
            t += 1
            # score = sum(alpha_j * y_j * K(x_j, x_i))
            support_indices = np.where(alpha > 0)[0]
            
            if len(support_indices) == 0:
                current_score = 0
            else:
                kernels = rbf_kernel(X_train[support_indices], X_train[i], gamma)
                current_score = np.sum(alpha[support_indices] * y_train[support_indices] * kernels)
            
            # y_i * f(x_i) < 1
            if y_train[i] * (current_score / (lambda1 * t)) < 1:
                alpha[i] += 1
                
    return alpha, X_train, y_train

def kernel_predict(X_test, alpha, X_train, y_train, lambda1, gamma):
    N_test = X_test.shape[0]
    predictions = np.zeros(N_test)
    
    support_indices = np.where(alpha > 0)[0]
    
    for i in range(N_test):
        # f(x) = (1 / lambda * T) * sum(alpha_j * y_j * K(x_j, x_test))
        kernels = rbf_kernel(X_train[support_indices], X_test[i], gamma)
        score = np.sum(alpha[support_indices] * y_train[support_indices] * kernels)
        predictions[i] = 1 if score > 0 else -1
        
    return predictions

def SVM_with_Kernel(X, y):
    y_encoded = one_vs_rest_encoding(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameters
    lmbda = 0.08
    gamma = 0.1 
    
    alpha, X_train_sv, y_train_sv = kernel_pegasos(X_train_scaled, y_train, lambda1=lmbda, gamma=gamma, num_iters=5)

    predictions = kernel_predict(X_test_scaled, alpha, X_train_sv, y_train_sv, lmbda, gamma)

    accuracy = np.mean(predictions == y_test)

    tp = np.sum((predictions == 1) & (y_test == 1))
    fp = np.sum((predictions == 1) & (y_test == -1))
    fn = np.sum((predictions == -1) & (y_test == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return accuracy, precision, recall

