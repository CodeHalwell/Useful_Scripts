from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def grid_search(model, param_grid, X_train, y_train, cv: int = 5, scoring=None):
    """Perform ``GridSearchCV`` and return the fitted object."""
    search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
    search.fit(X_train, y_train)
    print(f"Best parameters: {search.best_params_}")
    print(f"Best score: {search.best_score_}")
    return search


def random_search(model, param_distributions, X_train, y_train, n_iter: int = 10, cv: int = 5, scoring=None):
    """Perform ``RandomizedSearchCV`` and return the fitted object."""
    search = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, cv=cv, scoring=scoring)
    search.fit(X_train, y_train)
    print(f"Best parameters: {search.best_params_}")
    print(f"Best score: {search.best_score_}")
    return search
