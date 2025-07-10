import json
import os
import random
from functools import wraps
from time import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix)


def set_seed(seed: int = 42) -> None:
    """Set random seed for Python, NumPy and, if available, PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def flatten_list(nested_list):
    """Flatten a nested list into a single list."""
    return [item for sub in nested_list for item in sub]


def chunk_list(iterable, chunk_size):
    """Yield successive chunks from an iterable."""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]


def download_file(url: str, destination: str) -> None:
    """Download a file from ``url`` to ``destination``."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def timeit(func):
    """Decorator that prints the execution time of the function it wraps."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"{func.__name__} executed in {end - start:.2f}s")
        return result

    return wrapper


def dataframe_memory_usage(df: pd.DataFrame) -> float:
    """Return memory usage of a ``pandas`` DataFrame in megabytes."""
    return df.memory_usage(deep=True).sum() / 1e6


def save_json(data, path: str, **kwargs) -> None:
    """Save a Python object as a JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, **kwargs)


def load_json(path: str):
    """Load a JSON file and return the corresponding Python object."""
    with open(path) as f:
        return json.load(f)


def classification_report(y_true, y_pred) -> dict:
    """Return a dictionary with common classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """Plot and return a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    return fig, ax
