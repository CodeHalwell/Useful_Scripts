# Preprocessing.py

This module gathers basic preprocessing utilities using `pandas` and `scikit-learn`. Proper preparation of data is
crucial for any machine learning workflow, and these helpers cover the common steps of splitting, scaling, and encoding.

## Functions

- **`split_features_labels`** – separates a DataFrame into feature matrix `X` and target vector `y`.
- **`basic_train_test_split`** – wraps `train_test_split` to obtain train and test sets from a DataFrame.
- **`scale_numeric_features`** – fits a `StandardScaler` on the training data and applies it to both train and test
  sets, returning the transformed arrays and the fitted scaler.
- **`encode_categorical_features`** – performs one-hot encoding on specified categorical columns using
  `OneHotEncoder`, then concatenates the encoded features back to the rest of the DataFrame.

## Python Usage

The functions rely on pandas for DataFrame manipulation. Notice the use of keyword defaults (`test_size=0.2`) which let
users override behaviour. Returning multiple values (for example, `(X_train_processed, X_test_processed, encoder)`)
illustrates how Python supports tuple unpacking.

## Theory

Data should be split into training and testing sets to evaluate models on unseen data. Standardization scales features
to zero mean and unit variance, a common prerequisite for many algorithms that assume similarly scaled inputs. One-hot
encoding converts categorical variables into numerical binary features so that algorithms can treat each category as a
separate dimension.
