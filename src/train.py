import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, Any
import numpy as np


def train_model(X_train: np.ndarray, y_train: pd.Series) -> Any:
    param_grid = {'n_neighbors': list(range(1, 21, 2))}

    knn = KNeighborsClassifier()

    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
    )

    print("Model training and hyperparameter optimisation are starting...")
    grid_search.fit(X_train, y_train)

    print("--> Optimisation Complete!")
    print(f" --> Best k Value: {grid_search.best_params_['n_neighbors']}")
    print(f"--> Cross-Validation Best Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_