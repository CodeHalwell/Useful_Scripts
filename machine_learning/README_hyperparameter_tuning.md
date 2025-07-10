# hyperparameter_tuning.py

Machine learning models often expose numerous hyperparameters that control training behaviour. This script wraps
scikit-learn's `GridSearchCV` and `RandomizedSearchCV` to help find good parameter combinations.

## Helpers

- **`grid_search`** – exhaustively explores the provided `param_grid` for a model using cross-validation.
- **`random_search`** – samples combinations from `param_distributions` for a fixed number of iterations.

Both functions print the best parameters and return the fitted search object so you can inspect results further.

## Python Features

Notice how `**kwargs` are forwarded to scikit-learn classes via direct instantiation. `f"Best score: {search.best_score_}"`
illustrates f-strings for formatting. Type hints like `cv: int = 5` document expected argument types and defaults.

## Theory

Hyperparameters control aspects such as regularization strength or tree depth. Grid search systematically evaluates all
possible combinations but can be expensive. Random search selects random combinations, often reaching good solutions
faster when the hyperparameter space is large. Cross-validation averages results across folds to estimate generalization
performance.
