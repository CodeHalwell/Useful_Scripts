# helpers.py

This module contains a variety of utility functions useful in day-to-day data science and software projects. The
functions demonstrate core Python features such as decorators, comprehensions, and standard library modules.

## Key Utilities

- **`set_seed`** – sets the random seed for Python, NumPy and optionally PyTorch so that experiments are reproducible.
- **`flatten_list`** – demonstrates list comprehensions by flattening a list of lists into a single list.
- **`chunk_list`** – yields successive chunks from an iterable using a generator with the `yield` keyword.
- **`download_file`** – shows how to stream a download with `requests` and write the bytes to disk in chunks.
- **`timeit`** – an example of a decorator that measures execution time for any wrapped function.
- **`dataframe_memory_usage`** – computes memory consumption of a `pandas` DataFrame in megabytes.
- **`save_json` / `load_json`** – save Python objects to disk as JSON and read them back.
- **`classification_report`** – wraps common scikit-learn metrics (accuracy, precision, recall, f1).
- **`plot_confusion_matrix`** – visualises a confusion matrix using Matplotlib.

## Python Syntax Notes

The code uses type hints like `def set_seed(seed: int) -> None:` to clarify argument and return types. Decorators
(`timeit`) employ `functools.wraps` to preserve metadata of the wrapped function. Comprehensions are used to create
lists concisely, and context managers handle file operations in `download_file`, `save_json`, and `load_json`.

## Underlying Concepts

Reproducibility through seeding ensures that random operations (e.g., model training) yield the same results across
runs. JSON serialization enables easy sharing of parameters or predictions. Accuracy, precision, recall and F1-score
are fundamental metrics for evaluating classification models, while a confusion matrix provides a richer breakdown of
true vs predicted labels.
