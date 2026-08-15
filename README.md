# Machine Learning Project

CS-UY 4563: Intro to Machine Learning, Section B  
Professor Linda M Sellie  
Final Project Written Report  
Zhi Chen, Adithya Somashekhar  
Predicting Heart Disease in Patients  
April 27, 2026


# Introduction:

This project focuses on using machine learning models to predict heart diseases in
patients. The dataset was taken from Kaggle and included personal information that can have an
impact on the prediction of heart disease. This dataset has 1,025 entries, and 13 features in this
dataset. There is also a target value, which in this dataset is the target column, which is a binary
indicator to determine whether they have heart disease or not. For the sake of testing and training
our model, we have to remove the target label from our dataset to prepare the training.
The dataset is a binary classification due to the fact that our models will be predicting
whether a patient has heart disease based on their medical details. Such a simple diagnosis can be
a key to combating heart disease. Patients can take proper treatment and can do a daily check-up
to keep track of their heart disease. In retrospect, these predictive diagnoses will assist patients in
improving the survival rates against heart diseases.
In this project, three different binary classifications will be used: Logistic Regression,
Support Vector Machine (SVM), and Neural Network (NN). Each model will have a baseline
model and will have three different feature transformations performed on it. During each feature
transformation, different hyperparameters will be altered as well to prevent the models from
overfitting and underfitting. Since our goal is to predict heart diseases in patients, our model will
care about the accuracy and the precision of the model. The model wants to have a high accuracy
and recall since it is better to incorrectly predict a patient as a heart disease haver rather than
ignoring the possibility that they could be an actual heart disease patient. Thus, we want to
maintain a high recall while having a high accuracy without the models overfitting to the
validation set and training set.


# Data Preparation:

This data set was really simple and easy to understand. There was no missing dataset, nor
did we have to clean any of the datasets, as each feature was a key component of finding the
target values. The dataset already had the one-hot-encoding for us and cleaned the dataset up for
us, as it had already been removed of the patients' names and social security numbers, as it was
irrelevant to the prediction, and for the sake of the patients’ privacy.
The important features consist of the following: age, sex, chest pain type, resting blood
pressure, serum cholesterol, fasting blood sugar > 120 mg/dl, resting electrocardiographic results
(values 0,1,2), maximum heart rate achieved, exercise-induced angina, oldpeak = ST depression
induced by exercise relative to rest, the slope of the peak exercise ST segment, number of major
vessels (0-3) colored by flourosopy, thallium stress test results (thal). These features would be
tested against the target
Figure 1: Histograms of Disease vs No Disease Across Features


From the correlation matrix, we obtained how much each feature has an impact on
determining whether a patient has heart disease or not (Figure 2).
Figure 2: Correlation Matrix of the Feature
Using the correlation matrix, we can determine the key features of the dataset and figure
out the relationship between those features and the target value (Figure 3).


Figure 3: Relationship between Key Features and Target
Although there are key features, each feature is important, such as age, blood pressure,
cholesterol, sex, resting blood pressure, and more. There are no missing values in this dataset,
which makes it easy to use, which is why there was no sort of preprocessing. This dataset was
really nice and nicely constructed.
The data was scaled using the function StandardScaler from sklearn.preprocessing
library. The data was split into training, validation, and test subsets with an 80/10/10 split using
the same library. The data was standardized using Z-score normalization that fit the training data
in order to speed up the process, as it would converge faster during training. No encoding was
necessary as the dataset had already been processed to encode values as integers, and they were
fair.


# Logistic Regression

The first model evaluated for heart disease prediction was Logistic Regression. A baseline model
without regularization or feature transformations was evaluated with the dataset.
**Accuracy Precision Recall**
78.64% 75.41% 86.79%
Table 1: Baseline Metrics on the Validation Set

## N-Degree Feature Transformations

To discover and capture potential non-linear relationships within the data, the age, resting
blood pressure (trestbps), serum cholesterol (chol), maximum heart rate achieved during exercise
(thalach), and ST depression induced by exercise relative to rest (oldpeak) features were
transformed using the sklearn PolynomialFeatures utility.

## Lasso and Ridge Regularization

A risk of adding more features and increasing model complexity is the possibility of
overfitting. To reduce overfitting, regularization can be used to penalize large weights in the
model. The two methods used were: Lasso, which applies a constant reduction in weights
regardless of magnitude, and Ridge, which penalizes weights as a proportional to their
magnitude, reducing large weights to a greater extent. The influence of regularization can be


controlled with the hyperparameter, 𝛌. All logistic regression models were evaluated using both
Lasso and Ridge using 𝛌 = 0.01 to evaluate any benefits from regularization.
**Lambda (** 𝛌) **Accuracy Precision Recall**
0 (Baseline) 78.64% 75.41% 86.79%
0.001 78.64% 75.41% 86.79%
0.005 78.64% 75.41% 86.79%
0.01 77.67% 74.19% 86.79%
0.05 77.67% 74.19% 86.79%
0.1 74.76% 72.88% 81.13%
Table 2: L1 Regularization Comparison on Baseline
The results of the baseline model indicate that at 𝛌 values 0.005 and below there is
negligible difference in the performance of the model. More telling is that at 𝛌 of 0.01 and
greater the model performance degrades. This indicates that the model wasn’t overfitting and that
there is no benefit in trying to reduce variance and model complexity as was expected with the
baseline model.
**Lambda
(** 𝛌)
**Accuracy Precision Recall**
0 77.67% 75% 84.91%
0.001 77.67% 75% 84.91%


##### 0.005 77.67% 75% 84.91%

##### 0.01 76.7% 73.77% 84.91%

##### 0.05 77.67% 74.19% 86.79%

##### 0.1 74.76% 72.88% 81.13%

Table 3: L1 Regularization Comparison on Degree 2 Transformation
From these results a similar case can be seen as in the baseline model where the lower
lambda values have little impact. At 𝛌 of 0.01 the precision and recall degrade slightly and at
0.05 they improve a bit. At 𝛌 the model has a more measurable reduction in performance
indicating that regularization didn’t improve the model. The model overall performs slightly
worse than the baseline.
**Lambda (** 𝛌) **Accuracy**^ **Precision**^ **Recall**^
0 79.61% 75.81% 88.68%
0.001 79.61% 75.81% 88.68%
0.005 77.67% 75% 84.91%
0.01 76.7% 73.77% 84.91%
0.05 77.67% 74.19% 86.79%
0.1 74.76% 72.88% 81.13%
Table 4: L1 Regularization Comparison on Degree 3 Transformation


The performance of the degree 3 model is greater than that of both degree 2 and baseline.
Like in the other two models regularization had a negative impact on performance.
**Lambda (** 𝛌) **Accuracy**^ **Precision**^ **Recall**^
0 80.58% 77.05% 88.68%
0.001 79.61% 75.81% 88.68%
0.005 78.64% 75.41% 86.79%
0.01 77.67% 74.19% 86.79%
0.05 77.67% 74.19% 86.79%
0.1 74.76% 72.88% 81.13%
Table 5: L1 Regularization Comparison on Degree 4 Transformation
The performance of the degree 4 model is slightly better than that of the degree 3 and
follows the same trend of performing worse as 𝛌 is increased.


Figure 4: Graph of Each Model’s Accuracy vs Lambda Value
The models generally performed poorly as there was a greater 𝛌 indicating that regularization
didn’t improve model performance. This is not unexpected as the dimensionality of the features
was low and the model is simple leading to low variance and less need for regularization.

# Support Vector Machines

Support Vector Machines (SVM) were used here due to their freedom of hyperparameters
and ability to handle non-linearly separable data.
The baseline model (1st degree polynomial) was used to calculate our data on the
80/10/10, achieving the best λ by using the λ values in Table 6. It achieved a validation accuracy
for the corresponding value, which was then used to find the accuracy on the overall test.


**λ Accuracy Precision Recall**
0.001 82.35% 78.57% 88.00%
0.01 83.33% 77.05% 94.00%
0.1 83.33% 76.19% 96.00%
1.0 82.35% 75.81% 94.00%
10.0 76.47% 70.31% 90.00%
100.0 76.47% 70.31% 90.00%
Table 6: Baseline’s Accuracy on the Validation Set
However, the table lists two λ values, 0.01 and 0.1, both of which yield an accuracy of
83.33%. Both values were then tested to obtain the accuracy, precision, and recall for each λ
value in Table 7. Given that the accuracy was slightly higher, the precision slightly lower, and the
recall significantly better, λ = 0.01 is preferred, with an accuracy of 77.67%. This dataset prefers
a higher recall since it would at least declare that a patient has heart disease, and it becomes a
false negative rather than a false positive, where they would be diagnosed as having no heart
disease, while in reality, it’s the opposite.
**λ Accuracy Precision Recall**
0.001 78.64% 81.63% 75.47%
0.01 76.70% 74.58% 83.02%
0.1 77.67% 74.19% 86.79%
1.0 78.64% 75.41% 86.79%


##### 10.0 79.61% 78.57% 83.02%

##### 100.0 79.61% 78.57% 83.02%

```
Table 7: Base Model’s Comparison on the Test Set
```
### RBF Kernel

This experiment uses the Radial Basis Function (RBF) Kernel since it can be used
universally, as it can be transferred into an infinite degree z-space. This was done through the dot
products in the higher dimensions without the explicit feature expansion. The motivation behind
this feature was to identify complex decision boundaries.
Using the algorithm implemented, the RBF Kernel Model achieved an accuracy of 99.
with an λ of 0.001 on the validation set (Table 8).
**λ Accuracy Precision Recall**
0.001 90.20% 84.48% 98.00%
0.01 85.29% 79.66% 94.00%
0.1 77.45% 69.01% 98.00%
1.0 77.45% 69.01% 98.00%
10.0 77.45% 69.01% 98.00%
100.0 77.45% 69.01% 98.00%
Table 8: RBF Kernel’s Accuracy on the Validation Set
This could be a sign of overfitting, but since this is on the validation set, it should be safe
to use it. In fact, since the precision and recall are so high, that means the model is really well


done. Therefore, the best λ value for testing on the testing set would be 0.001. This would result
in a final accuracy of 94.17%, a final precision of 91.23%, and a final recall of 98.11% on the
test (Table 9).
**λ Accuracy Precision Recall**
0.001 94.17% 91.23% 98.11%
0.01 82.52% 79.66% 88.68%
0.1 78.64% 74.60% 88.68%
1.0 78.64% 74.60% 88.68%
10.0 78.64% 74.60% 88.68%
100.0 78.64% 74.60% 88.68%
Table 9: RBF Kernel’s Comparison on the Test Set
When the λ becomes bigger, the cost of the function is higher, which causes the penalty
to be enormous. When the penalty is too heavy, the update would be insignificant; hence, the
model sees a consistent number in accuracy, precision, and recall when 0.1 ≤ λ ≤ 100.
When λ is small, in this dataset, λ = .001, it obtained the highest accuracy, precision, and
recall. The RBF model works really well for our dataset, hence that would be our best value so
far.

### 2nd Degree Polynomial Kernel

To address the issue of the possibility of overfitting, we used a polynomial kernel for our
second transformation. Sklearn has the polynomial library and function, so we used those for


simplicity. Instead of turning the dimensions into infinite, we scaled them down to the second
degree to test if the RBF is overfitting or it’s accurate.
This model utilized the SVC function from the sklearn library to obtain the accuracy on
the validation set. This model saw an accuracy of 98.04% when C = 100 (Table 11), while the
rest were at a lower accuracy, lower precision, and lower recall.
**C Accuracy Precision Recall**
0.001 49.02% 49.02% 100.00%
0.01 80.39% 72.06% 98.00%
0.1 83.33% 77.05% 94.00%
1.0 85.29% 80.70% 92.00%
10.0 91.18% 87.27% 96.00%
100.0 98.04% 98.00% 98.00%
Table 10: 2nd Degree Polynomial’s Accuracy on the Validation Set
As C increases from 0.001 to 0.01, the accuracy increases from 49% to 80%, meaning
that the model is doing well on the validation set. When C = 100, the model obtained an accuracy
of 94.17%, a precision of 92.73%, and a recall of 96.23% (Table 12).
**C Accuracy Precision Recall**
0.001 51.46% 51.46% 100%


##### 0.01 78.64% 74.60% 88.68%

##### 0.1 80.58% 77.05% 88.68%

##### 1.0 87.38% 84.48% 92.45 %

##### 10.0 90.29% 87.72% 94.34%

##### 100.0 94.17% 92.73% 96.23 %

Table 11: 2nd Degree Polynomials’ Comparison on the Test Set
When C is at 100, it also has a high recall, which means the model has declared a patient
as having heart disease even when they don’t, but in this scenario, it is not a bad case. It is better
to detect whether they have heart disease even when they don’t, rather than telling someone that
they don’t have heart disease, in reality, they do. This means C would be our best hyperparameter
for this dataset.

### 3rd Degree Polynomial Kernel

### To see if the second-degree polynomial is underfitting, we want to use the 3rd degree

polynomial to experiment with our model. Similar to the second-degree polynomial kernel, we
use the sklearn library along with the SVC function. The steps are the same, except for the
degrees.
Increasing C from 0.001 to 0.01, the accuracy on the validation set increased by roughly
15%. It progressively increased between C = 0.01 and 1.0, until it reached 10.0 and 100.0, where
it had an accuracy of 99.02% for both of those C values (Table 13).


```
C Accuracy Precision Recall
```
#### 0.001 66.67% 82.35% 100.0%

##### 0.01 82.35% 72.24% 98.0%

##### 0.1 87.25% 82.46% 94.00%

##### 1.0 93.14% 89.09% 98.00%

##### 10.0 99.02% 100.00% 98.00%

##### 100.0 99.02% 100.00% 98.00%

Table 12: 3rd Degree Polynomial’s Accuracy on the Validation Set
However, from this example, when C is 10 and 100, we get the same exact values. This is
not a good sign, as this was our third feature transformation, and our dataset is a bit weird. We
suspect that the training data has a class imbalance, which may have very skewed predictions.
However, that still means that for our data set, when C = 10 or C = 100, it will obtain a high
accuracy, precision, and recall. (Table 14).
**C Accuracy Precision Recall**
0.001 67.96% 61.90% 98.11%
0.01 79.61% 75.81% 88.68%
0.1 86.41% 84.21% 90.57%
1.0 93.20% 91.07% 96.23%
10.0 98.06% 100.00% 96.23%


##### 100.0 98.06% 100.00% 96.23%

Table 13: 3rd Degree Polynomials’ Comparison on the Test Set
The base model has a good standpoint for the dataset. However, after applying the three
transformations, the dataset can obtain a lot better accuracy with more accurate precision and
recall. The fact that the higher polynomial degree kernel transformation has produced the best
model before overfitting means that the data is not linearly separable. (Table 15).
Baseline RBF 2nd Degree 3rd Degree
Accuracy 77.67% 95.15% 94.17% 98.06%
Table 14: Highest Final Accuracy for SVM
The graph shows the overall final accuracy for each training model and hyperparameter,
with hyperparameters kept consistent throughout training (Figure 5).


```
Figure 5: Graph of the overall Test Accuracy
```
# Result:

```
λ Accuracy Precision Recall
0.001 82.35% 78.57% 88.00%
0.01 83.33% 77.05% 94.00%
0.1 83.33% 76.19% 96.00%
1.0 82.35% 75.81% 94.00%
10.0 76.47% 70.31% 90.00%
```

##### 100.0 76.47% 70.31% 90.00%

# Table 15: Baseline’s Accuracy on the Validation Set

**λ Accuracy Precision Recall**
0.001 78.64% 81.63% 75.47%
0.01 76.70% 74.58% 83.02%
0.1 77.67% 74.19% 86.79%
1.0 78.64% 75.41% 86.79%
10.0 79.61% 78.57% 83.02%
100.0 79.61% 78.57% 83.02%
Table 16: Base Model’s Comparison on the Test Set
**λ Accuracy Precision Recall**
0.001 90.20% 84.48% 98.00%
0.01 85.29% 79.66% 94.00%
0.1 77.45% 69.01% 98.00%
1.0 77.45% 69.01% 98.00%
10.0 77.45% 69.01% 98.00%
100.0 77.45% 69.01% 98.00%
Table 17: RBF Kernel’s Accuracy on the Validation Set


**λ Accuracy Precision Recall**
0.001 94.17% 91.23% 98.11%
0.01 82.52% 79.66% 88.68%
0.1 78.64% 74.60% 88.68%
1.0 78.64% 74.60% 88.68%
10.0 78.64% 74.60% 88.68%
100.0 78.64% 74.60% 88.68%
Table 18: RBF Kernel’s Comparison on the Test Set
**C Accuracy Precision Recall**
0.001 49.02% 49.02% 100.00%
0.01 80.39% 72.06% 98.00%
0.1 83.33% 77.05% 94.00%
1.0 85.29% 80.70% 92.00%
10.0 91.18% 87.27% 96.00%
100.0 98.04% 98.00% 98.00%
Table 19: 2nd Degree Polynomial’s Accuracy on the Validation Set
**C Accuracy Precision Recall**
0.001 51.46% 51.46% 100%
0.01 78.64% 74.60% 88.68%


##### 0.1 80.58% 77.05% 88.68%

##### 1.0 87.38% 84.48% 92.45 %

##### 10.0 90.29% 87.72% 94.34%

##### 100.0 94.17% 92.73% 96.23 %

```
Table 20: 2nd Degree Polynomials’ Comparison on the Test Set
C Accuracy Precision Recall
```
#### 0.001 66.67% 82.35% 100.0%

##### 0.01 82.35% 72.24% 98.0%

##### 0.1 87.25% 82.46% 94.00%

##### 1.0 93.14% 89.09% 98.00%

##### 10.0 99.02% 100.00% 98.00%

##### 100.0 99.02% 100.00% 98.00%

Table 21: 3rd Degree Polynomial’s Accuracy on the Validation Set
**C Accuracy Precision Recall**
0.001 67.96% 61.90% 98.11%
0.01 79.61% 75.81% 88.68%
0.1 86.41% 84.21% 90.57%
1.0 93.20% 91.07% 96.23%
10.0 98.06% 100.00% 96.23%


##### 100.0 98.06% 100.00% 96.23%

```
Table 22: 3rd Degree Polynomials’ Comparison on the Test Set
Baseline RBF 2nd Degree 3rd Degree
Accuracy 77.67% 95.15% 94.17% 98.06%
Table 23: Highest Final Accuracy for SVM
```
# Neural Networks

Neural Networks are much better at capturing complex nonlinear relationships than the
previous models. Unlike logistic regression, which has a strictly linear decision boundary and
requires feature transformation to capture nonlinear relationships, neural networks can learn
nonlinear decision boundaries through nonlinear activation functions. Unlike SVMs, which
require prior assumptions about the data beforehand to apply kernels, NNs can automatically
capture hierarchies of complex features via layers of neurons. The baseline model was
implemented as an NN with one hidden layer with nine neurons and a binary classification
output. The neural network uses ReLU activation functions and a sigmoid output function.



```
Figure 6: Neural Network
Accuracy Precision Recall
92.23 94.12 90.57
Table 24: 1 Hidden Layer Comparison on the Test Set
```
### More Hidden Layers

The strength of the neural networks is their ability to learn nonlinear relationships by
using multiple layers with nonlinear activation functions. To determine the NN's performance


with additional layers, a model with two hidden layers of 9 neurons and 3 hidden layers of 9
neurons was evaluated as well.

## Dropout

To prevent the network from relying on particular neurons or paths, some activations are
randomly set to 0 during training. This effectively removes the neuron and forces the model to
use multiple paths for inference.
**Dropout Rate Accuracy Precision Recall**
0 92.23 94.12 90.57
0.1 92.23 95.92 88.68
0.2 89.32 90.38 88.68
0.3 87.38 85.71 90.57
0.4 88.35 88.68 88.68
0.5 88.35 88.68 88.68
Table 25: 1 Hidden Layer Comparison on the Test Set
The neural network baseline model performed much better than the baselines for the
other models. However, like the other models, the performance did not improve with increasing
regularization, although this is to be expected with the baseline model.
**Dropout Rate Accuracy Precision Recall**


0 93.2 96 90.57
0.1 92.23 94.12 90.57
0.2 91.26 92.31 90.57
0.3 87.38 87.04 88.68
0.4 90.29 92.16 88.68
0.5 90.29 93.88 86.79
Table 26: 2 Hidden Layers Comparison on the Test Set
The neural network performed better across all dropout rates with two layers than with
one layer. Interestingly, the model’s performance generally reduces as regularization increases,
but improves slightly after the dropout rate increases from 0.3.
**Dropout Rate Accuracy Precision Recall**
0 92.23 97.87 86.79
0.1 94.17 92.73 96.23
0.2 91.26 94 88.68
0.3 88.35 88.68 88.68
0.4 87.38 87.04 88.68
0.5 87.38 90 84.91
Table 27: 3 Hidden Layers Comparison on the Test Set
With the three layers the model’s performance is even greater though it still falls as regularization
increases.


Figure 7: Graph of Neural Network’s Performance
Overall, the neural networks generally improve with more complex models but degrade
with greater regularization. This may indicate there is room for capturing more complex
relationships in the data.

# Conclusion:

After calculating each of the models of the logistic, support vector machine, and neural
network, and doing the corresponding feature transformation, we have obtained the best test
result for each model. The best performing model was SVM with a 3rd Degree Polynomial
Kernel.


Logistic SVM NN
Accuracy 80.58% 98.06% 94.17%
Table 28: Best Test Results Across Models
The dataset is nonlinearly separable, and the relationship between the patient’s healthcare
features and heart disease is consistent. Our model indicates this analysis, given the fact that our
model improves with increased polynomial feature transformation and more complex models.
Things that we could improve on is doing the preprocessing. During the training for the
SVM, there were times when our model did not change by much. This made me realize that there
was a possibility that our dataset had a class imbalance. This could’ve been avoided if we used
SMOTE and the undersampling technique in the preprocessing rather than doing scaling.
Furthermore, we could have also not applied StandardScaling to all of the features as it could
have drastically changed some values. Many models have similar results for certain values of
accuracy, precision, or recall. This may indicate that there are a few points that are impossible to
classify with a generalized model. Regularization generally negatively impacted the models
possibly indicating that more complex models were needed to see any of its benefits. Overall
reducing bias improved the models while attempting to reduce variance degraded them. The next
goal would be to try to find a more complex set of features and evaluate more complex models to
see if they would reach a point of overfitting.



**Work Cited**
https://scikit-learn.org/stable/modules/svm.html
https://scikit-learn.org/stable/api/sklearn.svm.html
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html


