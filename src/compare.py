from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from typing import Dict, Any
import numpy as np
import pandas as pd


def compare_models(X_train: np.ndarray, X_test: np.ndarray, y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
    """
    Trains Random Forest and SVM models and reports their accuracy scores on the test set.

    Args:
        X_train, X_test, y_train, y_test: Datasets.

    Returns:
        Dict[str, float]: Dictionary containing model names and their accuracy scores.
    """

    results = {}

    print("\n--- Starting Model Comparison ---")

    # 1. Random Forest (RF) Model
    print("-> 1. Training Random Forest Model (n_estimators=100)...")
    # n_estimators=100 (100 trees), random_state=42 ensures reproducibility.
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    results['Random Forest'] = rf_accuracy
    print(f"-> Random Forest Test Accuracy: {rf_accuracy:.4f}")

    # 2. Support Vector Machine (SVM) Model
    print("-> 2. Training Support Vector Machine (SVM) Model (RBF Kernel)...")
    # SVC with 'rbf' (Radial Basis Function) kernel enables powerful non-linear classification.
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)

    svm_pred = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    results['SVM (RBF Kernel)'] = svm_accuracy
    print(f"-> SVM (RBF) Test Accuracy: {svm_accuracy:.4f}")

    return results
