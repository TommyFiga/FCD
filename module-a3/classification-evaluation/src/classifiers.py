from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd


def train_decision_tree(X: pd.DataFrame, y: pd.Series, max_depth: int = None) -> tuple[pd.Series, np.ndarray]:
    pipeline = Pipeline([
        ('scaller', StandardScaler()),
        ('classifier', DecisionTreeClassifier(max_depth=max_depth))
    ])

    y_pred = cross_val_predict(pipeline, X, y, cv=10, method='predict')

    return y, y_pred


def train_knn(X: pd.DataFrame, y: pd.Series, k: int = 5) -> tuple[pd.Series, np.ndarray]:
    pipeline = Pipeline([
        ('scaller', StandardScaler()),
        ('classifier', KNeighborsClassifier(n_neighbors=k))
    ])

    y_pred = cross_val_predict(pipeline, X, y, cv=10, method='predict')

    return y, y_pred


def train_svm(X: pd.DataFrame, y: pd.Series, kernel: str = 'rbf', C: float = 1.0) -> tuple[pd.Series, np.ndarray]:
    pipeline = Pipeline([
        ('scaller', StandardScaler()),
        ('classifier', SVC(kernel=kernel, C=C))
    ])

    y_pred = cross_val_predict(pipeline, X, y, cv=10, method='predict')

    return y, y_pred