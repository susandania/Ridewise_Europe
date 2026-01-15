

# src/batch_predict.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Offline Batch Inference (Production Batch Prediction Job)
DEFAULT_THRESHOLD = 0.45


def _read_json_features(path: Path) -> Optional[list[str]]:
    if path.exists():
        data = json.loads(path.read_text())
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
    return None


def _find_features_in_dict(d: dict) -> Optional[list[str]]:
    # Try common keys first
    for k in ("features", "feature_list", "columns", "cols", "selected_features", "input_features"):
        v = d.get(k)
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return v

    # Fallback: find any list[str] in values
    for v in d.values():
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return v

    return None


def _unwrap_model(bundle: Any, *, purpose: str) -> Tuple[Any, Optional[list[str]]]:
    """
    purpose: "seg" or "churn" (helps error messages)
    Supports:
    - direct model object
    - dict bundle containing model under unknown key
    """
    if not isinstance(bundle, dict):
        return bundle, None

    features = _find_features_in_dict(bundle)

    # First try common model keys
    common_keys = (
        "model", "pipeline", "estimator", "final_model", "trained_model", "clf",
        "kmeans", "cluster_model", "preprocess_model", "full_pipeline"
    )
    for k in common_keys:
        v = bundle.get(k)
        if hasattr(v, "predict"):
            return v, features

    # Last resort: scan everything for something that can predict
    for v in bundle.values():
        if hasattr(v, "predict"):
            return v, features

    raise ValueError(
        f"Loaded {purpose} artifact is a dict but no model-like object was found. "
        f"Keys available: {list(bundle.keys())}"
    )


def load_artifacts(artifacts_dir: Path):
    churn_bundle = joblib.load(artifacts_dir / "churn_model.pkl")
    seg_bundle = joblib.load(artifacts_dir / "segmentation_model.pkl")

    churn_model, churn_features_in_bundle = _unwrap_model(churn_bundle, purpose="churn")
    seg_model, seg_features_in_bundle = _unwrap_model(seg_bundle, purpose="segmentation")

    # Prefer JSON feature lists if present, else use bundle features (if any)
    churn_features = _read_json_features(artifacts_dir / "churn_features.json") or churn_features_in_bundle
    seg_features = _read_json_features(artifacts_dir / "segment_features.json") or seg_features_in_bundle

    return churn_model, seg_model, churn_features, seg_features


def ensure_columns(df: pd.DataFrame, required: list[str], fill_value: float = 0.0) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    for c in missing:
        df[c] = fill_value
    return df[required]


def predict_proba(model: Any, X: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        return pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="churn_proba")

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        proba = 1 / (1 + np.exp(-scores))
        return pd.Series(proba, index=X.index, name="churn_proba")

    raise ValueError("Churn model has neither predict_proba nor decision_function.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input CSV/Parquet")
    ap.add_argument("--output", required=True, help="Path to output CSV/Parquet")
    ap.add_argument("--artifacts-dir", required=True, help="Folder containing saved models")
    ap.add_argument("--id-col", default="customer_id", help="Customer identifier column")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    churn_model, seg_model, churn_features, seg_features = load_artifacts(artifacts_dir)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    if args.id_col not in df.columns:
        raise ValueError(f"Missing id column: {args.id_col}")

    # bool -> int to avoid sklearn surprises
    bool_cols = df.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # --- Segmentation ---
    if seg_features is not None:
        X_seg = ensure_columns(df.copy(), seg_features, fill_value=0.0).fillna(0.0)
    else:
        X_seg = df.select_dtypes(include="number").fillna(0.0)

    df["segment"] = seg_model.predict(X_seg).astype(int)

    # --- Churn scoring ---
    if churn_features is not None:
        X_churn = ensure_columns(df.copy(), churn_features, fill_value=0.0).fillna(0.0)
    else:
        X_churn = df.select_dtypes(include="number").fillna(0.0)

    df["churn_proba"] = predict_proba(churn_model, X_churn)
    df["will_churn"] = (df["churn_proba"] >= args.threshold).astype(int)
    df["threshold_used"] = float(args.threshold)

    # Key outputs first
    front_cols = [args.id_col, "segment", "churn_proba", "will_churn", "threshold_used"]
    remaining = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + remaining]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"Saved batch predictions -> {out_path}")


if __name__ == "__main__":
    main()




# Production-grade API hardening