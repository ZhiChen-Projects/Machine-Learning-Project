from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from dataset import load_data, split_data, normalize_data

X, y, feature_names = load_data("heart.csv")
feature_names = [f.strip() for f in feature_names]

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)

clf = SVC(kernel=' ')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")