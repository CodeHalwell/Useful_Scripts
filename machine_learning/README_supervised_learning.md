# supervised_learning.py

This file provides lightweight wrappers around scikit-learn classifiers for quick experimentation.

## Provided Functions

- **`train_logistic_regression`** – constructs a `LogisticRegression` model and fits it on the data.
  Logistic regression models the log-odds of the target class as a linear function of the features.
- **`train_random_forest`** – trains a `RandomForestClassifier`, which combines many decision trees to reduce variance
  and capture nonlinear relationships.
- **`evaluate_model`** – runs prediction on test data and prints accuracy.

## Python Highlights

`**kwargs` let callers pass any scikit-learn options, such as `C` for regularization or `n_estimators` for forest size.
The `evaluate_model` function uses f-strings for formatting the accuracy with four decimal places.

## Theoretical Background

Logistic regression is a linear model that uses the sigmoid function to map inputs to probabilities. It is widely used
for binary classification. Random forests are an ensemble method where each tree is built on a bootstrap sample of the
training data; predictions are averaged to provide robustness. Accuracy measures the proportion of correct predictions
and serves as a baseline metric for classification tasks.
