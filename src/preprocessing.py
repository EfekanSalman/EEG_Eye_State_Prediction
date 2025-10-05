import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple


def preprocess_data(file_path: str) -> Tuple:
    df = pd.read_csv(file_path)

    df_clean = df.copy()
    eeg_cols = df.drop(columns=["eye_state"]).columns

    for col in eeg_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    X = df_clean.drop(columns=["eye_state"])
    y = df_clean["eye_state"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"[{file_path} The file has been successfully processed.]")
    print(f"Cleaned Data Size (Training): {X_train.shape[0]} rows.")

    return X_train, X_test, y_train, y_test