from preprocessing import preprocess_data
from train import train_model
from compare import compare_models
from sklearn.metrics import accuracy_score
import pandas as pd
from typing import Dict, Any


def main():
    """
    Main function that manages the entire Machine Learning process.
    """

    data_file_path = "../data/raw/eeg-headset.csv"

    # 1. DATA PREPROCESSING STAGE
    print("--- 1. DATA PREPROCESSING STAGE ---")
    X_train, X_test, y_train, y_test = preprocess_data(data_file_path)
    print("\n")

    # 2. MODEL TRAINING & OPTIMIZATION STAGE (K-NN)
    print("--- 2. K-NN TRAINING STAGE (with GridSearch) ---")
    # train_model returns the best optimized K-NN model
    final_knn_model = train_model(X_train, y_train)
    print("\n")

    # 3. FINAL EVALUATION OF K-NN MODEL
    print("--- 3. FINAL EVALUATION OF K-NN MODEL ---")

    y_pred = final_knn_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"Final Model: {final_knn_model}")
    print(f"Final Test Accuracy (K-NN k=1): {test_accuracy:.4f}")
    print("\n")
    # 4. NEW MODEL COMPARISON STAGE
    print("--- 4. RANDOM FOREST & SVM COMPARISON ---")

    # Get the scores of two new models from compare_models
    comparison_results = compare_models(X_train, X_test, y_train, y_test)

    # Add the K-NN score to the comparison dictionary
    comparison_results['K-NN (k=1, Optimized)'] = test_accuracy
    # 5. FINAL REPORT
    print("\n--- OVERALL MODEL PERFORMANCE REPORT ---")

    # Sort and print results by accuracy score
    final_ranking = pd.Series(comparison_results).sort_values(ascending=False)

    for model_name, score in final_ranking.items():
        print(f"| {model_name:<25} | Accuracy: {score:.4f} |")

if __name__ == "__main__":
    main()
