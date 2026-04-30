from dataset import load_data
from svm import SVM_with_Kernel
X, y, feature_names = load_data("heart.csv")
feature_names = [f.strip() for f in feature_names]

accuracy, precision, recall = SVM_with_Kernel(X, y)
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")