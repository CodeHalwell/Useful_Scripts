import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_features_labels(df: pd.DataFrame, target: str):
    """Split a DataFrame into features and target."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def basic_train_test_split(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    """Split DataFrame into train and test sets."""
    X, y = split_features_labels(df, target)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_numeric_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scale numeric features with ``StandardScaler``."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def encode_categorical_features(X_train: pd.DataFrame, X_test: pd.DataFrame, categorical_columns):
    """One-hot encode categorical columns and return transformed DataFrames."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_encoded = encoder.fit_transform(X_train[categorical_columns])
    test_encoded = encoder.transform(X_test[categorical_columns])
    X_train_rest = X_train.drop(columns=categorical_columns).reset_index(drop=True)
    X_test_rest = X_test.drop(columns=categorical_columns).reset_index(drop=True)
    X_train_processed = pd.concat(
        [X_train_rest, pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out())],
        axis=1,
    )
    X_test_processed = pd.concat(
        [X_test_rest, pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out())],
        axis=1,
    )
    return X_train_processed, X_test_processed, encoder
