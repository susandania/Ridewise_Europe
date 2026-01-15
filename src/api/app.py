"""
RideWise Churn Prediction API (Learning Version)

Goal (for now):
- Start a FastAPI server successfully
- Expose a health endpoint to prove the server is running
- Define request/response schemas (so you understand how /predict will work later)
"""

# Import the framework (FastAPI), schema tool (Pydantic) and other libraries
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from typing import Optional, Any, Dict, List
import joblib
import os
import pandas as pd
import argparse
import json
from pathlib import Path



# Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
seg_model_path = os.path.join(BASE_DIR, "models", "segmentation_model.pkl")
churn_model_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")

# Deployment Decision Rule
segmentation_pipeline = None
seg_features = None

churn_pipeline = None
churn_threshold = None

# Create the FastAPI application object "server app" that will receive HTTP requests
app = FastAPI(
    title="RideWise Churn Prediction API",
    version="0.1.0",
    description="API Service for Churn Prediction (Deployment & MLOps Phase)"
)


@app.on_event("startup")
def load_artifacts():
    global segmentation_pipeline, seg_features, churn_pipeline, churn_threshold

    # Load segmentation artifacts
    seg_bundle = joblib.load(seg_model_path)
    segmentation_pipeline = seg_bundle["segmentation_pipeline"]
    seg_features = seg_bundle["seg_features"]

    # Load churn artifacts
    churn_bundle = joblib.load(churn_model_path)
    churn_pipeline = churn_bundle["pipeline"]
    churn_threshold = churn_bundle.get("threshold", 0.45)



# Define schemas (data contracts) to teach the pattern.
class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool = Field(..., examples=[True])

# Define Risk Band for churn rate
def risk_band(prob: float, threshold: float) -> str:
    if prob >= 0.70:
        return "high"
    elif prob >= threshold:
        return "medium"
    else:
        return "low"


# Create an endpoint: GET /health to confirm the API server is alive and responding during health check
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ok = segmentation_pipeline is not None and churn_pipeline is not None
    return HealthResponse(status="ok", model_loaded=ok)


# Add /predict
class PredictRequest(BaseModel):
    age: int
    gender: str
    cancelled_trips: int
    days_since_last_trip: int
    signup_channel: str
    income_level: str
    age_band: str
    total_cancelled_trips: int
    net_platform_revenue: float
    avg_duration_min: float
    promo_used_avg: float
    total_promo_cost: float
    avg_discount_value: float
    avg_driver_burden: float
    driver_borne_total: float
    driver_burden_ratio: float
    promo_used_rate: float
    promo_dependency: bool
    revenue_quartile: str
    promo_intensity_q: str
    signup_month: str
    last_trip_month: str
    is_inactive_20d: int
    is_inactive_30d: int
    recency_bucket: str
    is_high_value: int
    high_value_and_inactive: int
    low_engagement_and_recent: int
    usage_intensity_score: float


class PredictResponse(BaseModel):
    churn_probability: float
    threshold_used: float
    churn_label: int
    segment: int
    risk_band: str



@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:

    # Convert incoming JSON to DataFrame (ONE ROW)
    data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    df = pd.DataFrame([data])

    # Compute segment from selected segmentation features
    X_seg = df[seg_features]
    seg_value = int(segmentation_pipeline.predict(X_seg)[0])

    # Add segment for churn prediction
    df["segment"] = seg_value

    # Convert boolean columns to int
    bool_cols = df.select_dtypes(include="bool").columns
    for c in bool_cols:
        df[c] = df[c].astype(int)

    # Get churn probability from trained pipeline
    proba = float(churn_pipeline.predict_proba(df)[0, 1])

    threshold = churn_threshold if churn_threshold is not None else 0.45


    # Apply trained threshold (0.45)
    label = int(proba >= threshold)

    # Apply risk band
    band = risk_band(proba, threshold)

    # Return response
    return PredictResponse(
    churn_probability=proba,
    threshold_used=threshold,
    churn_label=label,
    segment=int(seg_value),
    risk_band=band
)


# Run via python src/api/app.py
if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)





# Bulk / Batch API Inference (Synchronous Batch Scoring via API)
class BatchPredictRequest(BaseModel):
    records: List[PredictRequest]

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    outputs = []
    for rec in payload.records:
        df = pd.DataFrame([rec.model_dump()])

        # segment
        X_seg = df[seg_features]
        segment = int(segmentation_pipeline.predict(X_seg)[0])
        df["segment"] = segment

        # bool -> int
        for col in df.select_dtypes(include="bool").columns:
            df[col] = df[col].astype(int)

        # churn
        proba = float(churn_pipeline.predict_proba(df)[0, 1])
        label = int(proba >= churn_threshold)
        band = risk_band(proba, churn_threshold)

        outputs.append(PredictResponse(
            churn_probability=proba,
            threshold_used=churn_threshold,
            churn_label=label,
            segment=segment,
            risk_band=band
        ))

    return BatchPredictResponse(results=outputs)




