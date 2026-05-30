"""
data/prepare_data.py
--------------------
Loads and preprocesses the Credit Card Fraud dataset.
Think of this like sorting your Lego bricks before building — 
we clean and organise the data so our model can learn from it.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os


def load_and_prepare(csv_path: str = "data/creditcard.csv"):
    """
    Load the Kaggle Credit Card Fraud dataset and return train/test splits.
    
    The dataset has:
    - 284,807 transactions
    - 492 frauds (only 0.17%!)  ← very imbalanced, like finding 1 bad apple in 578
    - Features V1-V28 (PCA-transformed for privacy), Amount, Time
    - Class: 0 = normal, 1 = fraud
    """
    print("📂 Loading dataset...")
    
    if not os.path.exists(csv_path):
        print(f"⚠️  Dataset not found at '{csv_path}'")
        print("   Generating synthetic data for demonstration...")
        print("   (Download real data: https://www.kaggle.com/mlg-ulb/creditcardfraud)")
        return _generate_synthetic_data()

    df = pd.read_csv(csv_path)
    print(f"   ✅ Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"   Fraud cases  : {df['Class'].sum():,}  ({df['Class'].mean()*100:.2f}%)")
    print(f"   Normal cases : {(df['Class']==0).sum():,}")

    # ── Features & labels ──────────────────────────────────────────────────────
    X = df.drop(columns=["Class", "Time"])   # drop label + Time (not useful)
    y = df["Class"].values

    # ── Scale Amount (V1-V28 are already scaled via PCA) ──────────────────────
    scaler = StandardScaler()
    X["Amount"] = scaler.fit_transform(X[["Amount"]])

    X = X.values.astype(np.float32)

    # ── Train / Test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── SMOTE: oversample fraud in training set ───────────────────────────────
    # Like making photocopies of rare stamps so our model sees enough examples
    print("\n🔄 Applying SMOTE to balance training data...")
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    print(f"   After SMOTE — fraud: {y_train_bal.sum():,} | normal: {(y_train_bal==0).sum():,}")

    return X_train_bal, X_test, y_train_bal, y_test, X_train, y_train


def _generate_synthetic_data(n_samples: int = 50000, fraud_ratio: float = 0.0017,
                              random_state: int = 42) -> tuple:
    """
    Generate synthetic credit card transaction data that mimics the Kaggle dataset.
    Used automatically when creditcard.csv is not available (demo / CI mode).
    """
    rng = np.random.default_rng(random_state)

    n_fraud  = max(int(n_samples * fraud_ratio), 50)
    n_normal = n_samples - n_fraud

    # V1-V28: PCA-transformed features — approximate with Gaussian
    V_normal = rng.standard_normal((n_normal, 28)).astype(np.float32)
    V_fraud  = rng.standard_normal((n_fraud,  28)).astype(np.float32)

    # Fraud transactions look statistically different in several dimensions
    V_fraud[:, :5]  += rng.uniform(-3, 3, (n_fraud, 5))   # shift a few key components
    V_fraud[:, 5:10] *= 2.5

    # Amount: log-normal, slightly higher for fraud
    amt_normal = rng.lognormal(mean=3.5, sigma=1.5, size=n_normal).astype(np.float32)
    amt_fraud  = rng.lognormal(mean=4.5, sigma=1.2, size=n_fraud ).astype(np.float32)

    X_normal = np.hstack([V_normal, amt_normal.reshape(-1, 1)])
    X_fraud  = np.hstack([V_fraud,  amt_fraud .reshape(-1, 1)])

    X = np.vstack([X_normal, X_fraud])
    y = np.hstack([np.zeros(n_normal, dtype=np.int32),
                   np.ones (n_fraud,  dtype=np.int32)])

    # Shuffle
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    print(f"   ✅ Synthetic data: {len(y):,} rows  |  "
          f"fraud: {y.sum():,}  ({y.mean()*100:.2f}%)")

    # Scale Amount (last column)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X[:, -1] = scaler.fit_transform(X[:, -1].reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n🔄 Applying SMOTE to balance training data...")
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    print(f"   After SMOTE — fraud: {y_train_bal.sum():,} | "
          f"normal: {(y_train_bal==0).sum():,}")

    return X_train_bal, X_test, y_train_bal, y_test, X_train, y_train


if __name__ == "__main__":
    load_and_prepare()
