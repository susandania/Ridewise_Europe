from fastapi.testclient import TestClient

# import path from FastAPI app object.
from app.api import app 


client = TestClient(app)

def _default_value_for_feature(name: str):
        n = name.lower()

    # categorical defaults
        if n in {"gender"}:
            return "Female"
        if n in {"signup_channel"}:
            return "organic"
        if n in {"income_level"}:
            return "low"
        if "recency_bucket" in n:
            return "Inactive"
        if "revenue_quartile" in n:
            return "Q1"
        if "promo_intensity_q" in n:
            return "High"
        if n == "last_trip_month":
            return "2025-12"
        if n == "signup_month":
            return "2024-01"
        if n.startswith("is_") or n.endswith("_flag"):
            return 0
        if n == "segment":
            return 0
        return 0


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)


def test_metadata_ok_and_has_expected_keys():
    r = client.get("/metadata")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)

    # To learn features + threshold info from metadata
    assert "threshold" in data
    assert "features" in data
    assert isinstance(data["features"], list)
    assert len(data["features"]) > 0


def test_predict_smoke_uses_metadata_features():
    meta = client.get("/metadata").json()
    features = meta["features"]

    record = {f: _default_value_for_feature(f) for f in features}
    payload = {"record": record}

    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()

    # Required response contract
    assert "churn_probability" in data
    assert "churn_pred" in data
    assert "threshold" in data
    assert "segment" in data

    assert isinstance(data["churn_probability"], (int, float))
    assert isinstance(data["churn_pred"], bool)
    assert data["threshold"] == 0.45
