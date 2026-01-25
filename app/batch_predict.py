# src/batch/batch_predict.py

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-path", required=True, help="CSV containing customer records to score")
    p.add_argument("--out-path", required=True, help="CSV path to save churn predictions")
    return p.parse_args()


def main():
    args = parse_args()

    # Project paths
    project_root = Path(__file__).resolve().parents[2]
    artifacts_dir = project_root / "artifacts"

    churn_path = artifacts_dir / "churn_model.pkl"
    seg_path = artifacts_dir / "segmentation_model.pkl"

    # Load artifacts
    churn_artifact = joblib.load(churn_path)
    seg_artifact = joblib.load(seg_path)

    model = churn_artifact["model"]
    threshold = float(churn_artifact["threshold"])
    features = churn_artifact["features"]

    scaler = seg_artifact["scaler"]
    kmeans = seg_artifact["kmeans"]
    seg_features = seg_artifact["seg_features"]

    # Load input data
    input_path = project_root / args.input_path
    df = pd.read_csv(input_path)

    # Compute segment
    X_seg = df[seg_features].copy()
    X_scaled = scaler.transform(X_seg)
    df["segment"] = kmeans.predict(X_scaled).astype(int)

    # Predict churn
    X = df[features]
    churn_proba = model.predict_proba(X)[:, 1]

    df["churn_probability"] = churn_proba
    df["churn_pred"] = (churn_proba >= threshold).astype(int)
    df["threshold"] = threshold

    # Save output
    out_path = project_root / args.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Batch prediction complete.")
    print(f"Rows scored: {len(df)}")
    print(f"Output saved to: {out_path}")


if __name__ == "__main__":
    main()
