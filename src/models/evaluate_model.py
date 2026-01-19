# src/models/evaluate_model.py

import argparse
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-path", default="data/processed/ridewise_churn_modeling_dataset.csv")
    p.add_argument("--model-path", default="artifacts/churn_model.pkl")
    p.add_argument("--split-path", default="artifacts/test_split.pkl")
    return p.parse_args()


def main():
    args = parse_args()

    # Load dataset
    df = pd.read_csv(args.input_path)

    y_true = df["churned"].astype(int)
    X = df.drop(columns=["churned"])

    # Load trained model + threshold
    artifact = joblib.load(args.model_path)
    model = artifact["model"]
    threshold = artifact["threshold"]

    split = joblib.load(args.split_path)
    test_idx = split["test_index"]

    X_test = X.loc[test_idx]
    y_test = y_true.loc[test_idx]

    # Predict probabilities and apply threshold
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # 4) Report metrics
    print(f"Evaluation Threshold: {threshold}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC (Average Preci   sion): {average_precision_score(y_test, y_proba):.4f}")


if __name__ == "__main__":
    main()