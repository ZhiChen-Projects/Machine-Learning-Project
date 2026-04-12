import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Load data
path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(path, 'heart.csv')

df = pd.read_csv(file_path)
X = df.drop('target', axis=1).values
y = df['target'].values 

# A) One-vs-Rest Encoding for binary classification
def one_vs_rest_encoding(y, digit=1):
    return np.where(y == digit, 1, -1)

y_encoded = one_vs_rest_encoding(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scaling features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Helper function to calculate the score (w^T x) for a given weight vector w and input X
def score(X, w):
    return X.dot(w)

def svm_objective(w, X, y, lambda1=0.08):
    result = lambda1/2.0 * np.dot(w,w) + np.mean(np.maximum(0, 1 - y * score(X, w)))
    return result

def pegasos(X_train, y_train, lambda1=0.08, num_iters=3):
    N = X_train.shape[0] 
    d = X_train.shape[1]
    t = 0
    w = np.ones((d,))

    for iter in range(num_iters):
        print('Iteration %d. J: %.6f' % (iter, svm_objective(w, X_train, y_train, lambda1)))

        for i in range(N):
            t = t + 1
            # Learning rate and regularization scaling
            w = (1 - 1/t) * w   
            
            # Subgradient update if margin is violated
            if y_train[i] * score(X_train[i], w) < 1:
                w = w + (1/(lambda1*t)) * y_train[i] * X_train[i]
    
    return w

# Training
lambda1 = 0.08
num_iters = 15
w_final = pegasos(X_train_scaled, y_train, lambda1=lambda1, num_iters=num_iters)

# Binary prediction based on the sign of the score
test_scores = score(X_test_scaled, w_final)
predictions = np.where(test_scores > 0, 1, -1)

eval1 = (predictions == y_test)
accuracy = np.mean(eval1)

tp = np.sum((predictions == 1) & (y_test == 1))
fp = np.sum((predictions == 1) & (y_test == -1))
fn = np.sum((predictions == -1) & (y_test == 1))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print("\Results:\n")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")