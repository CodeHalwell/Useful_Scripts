from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_logistic_regression(X_train, y_train, **kwargs):
    """Train a ``LogisticRegression`` model."""
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, **kwargs):
    """Train a ``RandomForestClassifier`` model."""
    model = RandomForestClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Predict on ``X_test`` and return predictions and accuracy."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    return preds, acc
