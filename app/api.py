# src/api/app.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import hashlib
import json


# Paths
project_root = Path(__file__).resolve().parents[1]
artifacts = project_root / "models/artifacts"

churn_path = artifacts/ "churn_model.pkl"
seg_path =  artifacts/ "segmentation_model.pkl"

# Load model artifacts
churn_artifact = joblib.load(churn_path)
segment_artifact = joblib.load(seg_path)

model = churn_artifact["model"]
thresholds = float(churn_artifact["threshold"])
features = churn_artifact["features"]

scaler = segment_artifact["scaler"]
kmeans = segment_artifact["kmeans"]
segment_features = segment_artifact["seg_features"]  


app = FastAPI(title="RideWise Europe - Churn Prediction API")


class PredictRequest(BaseModel):
    record: Dict[str, Any]


class PredictResponse(BaseModel):
    churn_probability: float
    churn_pred: bool
    threshold: float
    segment: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    # Convert input to dataFrame
    df = pd.DataFrame([payload.record])

    # Compute segment
    X_seg = df[segment_features].copy()
    X_scaled = scaler.transform(X_seg)
    segment = int(kmeans.predict(X_scaled)[0])

    # 3) add segment to record
    df["segment"] = segment

    # Enforce feature order used in training
    X = df[features]

    # Predict
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = bool(proba >= thresholds)

    return {
    "churn_probability": float(proba),
    "churn_pred": bool(pred),
    "threshold": float(thresholds),
    "segment": int(segment),
    }

# Add metadata
def _artifact_fingerprint(artifact: dict) -> str:
    payload = {
        "threshold": artifact.get("threshold"),
        "features": artifact.get("features"),
    }
    s = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:12]


@app.get("/metadata")
def metadata():
    return {
        "model_type": type(model).__name__,
        "threshold": thresholds,
        "n_features": len(features),
        "features": features,
        "segment_features": segment_features,
        "artifact_id": _artifact_fingerprint(churn_artifact),
        "server_time": datetime.utcnow().isoformat() + "Z",
    }


