# src/models/train_model.py

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--input-path", default="data/processed/ridewise_churn_modeling_dataset.csv")
    p.add_argument("--model-out", default="artifacts/churn_model.pkl")
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--split-out", default="artifacts/test_split.pkl")
    return p.parse_args()


def main():
    args = parse_args()

    # Load modeling dataset
    df = pd.read_csv(args.input_path)

    # Separate target
    y = df["churned"]
    X = df.drop(columns=["churned"])

    # Identify column types
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols)])

    # Selected Model - tuned Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        max_features=0.5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1)

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("rf", rf),
        ])

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Save the exact test split indices (best practice)
    Path(args.split_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"test_index": X_test.index.tolist()}, args.split_out)

    # Fit final model
    model.fit(X_train, y_train)

    # Save model artifact WITH threshold
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "threshold": args.threshold,
            "features": X.columns.tolist(),
        },
        args.model_out)


if __name__ == "__main__":
    main()
