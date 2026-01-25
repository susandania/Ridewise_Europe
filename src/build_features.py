# src/data/build_features.py

import argparse
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-path", default="data/processed/ridewise_eda_data.csv")
    p.add_argument("--seg-out", default="data/processed/ridewise_cus_seg.csv")
    p.add_argument("--model-out", default="data/processed/ridewise_churn_modeling_dataset.csv")
    p.add_argument("--seg-model-out", default="models/artifacts/segmentation_model.pkl")
    return p.parse_args()


cus_seg_cols = [
 'age', 'gender', 'cancelled_trips', 'days_since_last_trip', 'churned',
 'signup_channel', 'income_level','total_cancelled_trips', 'net_platform_revenue', 
 'avg_duration_min', 'promo_used_avg', 'total_promo_cost', 'avg_discount_value',
 'avg_driver_burden', 'driver_borne_total', 'driver_burden_ratio',
 'promo_used_rate', 'promo_dependency', 'revenue_quartile',
 'promo_intensity_q', 'signup_month', 'last_trip_month',
 'is_inactive_20d', 'is_inactive_30d', 'recency_bucket', 'is_high_value',
 'high_value_and_inactive', 'low_engagement_and_recent',
 'usage_intensity_score'
]

seg_features = ["usage_intensity_score", "low_engagement_and_recent", "net_platform_revenue", 
                "is_high_value", "high_value_and_inactive", "days_since_last_trip", "is_inactive_20d",
                "is_inactive_30d", "promo_used_rate", "promo_used_avg", "total_promo_cost", 
                "avg_discount_value", "promo_dependency", "avg_duration_min", 
                "total_cancelled_trips", "avg_driver_burden", "driver_borne_total", "driver_burden_ratio"]


def main():
    args = parse_args()

    # Load dataset
    df = pd.read_csv(args.input_path)


    # Fill numeric NaNs from left joins with 0
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0)

    # Feature Engineering
    df["is_inactive_20d"] = (df["days_since_last_trip"] >= 20).astype(int)
    df["is_inactive_30d"] = (df["days_since_last_trip"] >= 30).astype(int)

    df["recency_bucket"] = pd.cut(
        df["days_since_last_trip"],
        bins=[-1, 7, 14, 27, np.inf],
        labels=["Recent", "Mid-Recent", "Mid-Inactive", "Inactive"])
    
    rev_75 = df["total_gross_revenue"].quantile(0.75)
    df["is_high_value"] = (df["total_gross_revenue"] >= rev_75).astype(int)
    df["high_value_and_inactive"] = ((df["is_high_value"] == 1) & (df["is_inactive_20d"] == 1)).astype(int)

    df["promo_dependency"] = (df["promo_used_rate"] > df["promo_used_rate"].median()).astype(int)
    df["promo_intensity_q"] = pd.qcut(
        df["promo_used_rate"],
        q=4,
        labels=["Low", "Mid-Low", "Mid-High", "High"],
        duplicates="drop")
    
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df["last_trip_date"] = pd.to_datetime(df["last_trip_date"], errors="coerce")
    df["signup_month"] = df["signup_date"].dt.to_period("M").astype(str)
    df["last_trip_month"] = df["last_trip_date"].dt.to_period("M").astype(str)

    df["revenue_quartile"] = pd.qcut(
        df["net_platform_revenue"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop")
    
    low_eng_cut = df["engagement_score"].quantile(0.25)
    df["low_engagement_and_recent"] = (
        (df["engagement_score"] <= low_eng_cut) & (df["days_since_last_trip"] <= 14)
    ).astype(int)

    mm = MinMaxScaler()
    df[["active_days_norm", "total_trips_norm"]] = mm.fit_transform(df[["active_days", "total_trips"]])
    df["usage_intensity_score"] = 0.5 * df["active_days_norm"] + 0.5 * df["total_trips_norm"]


    df_seg = df[cus_seg_cols].copy()

    # Save seg dataset
    Path(args.seg_out).parent.mkdir(parents=True, exist_ok=True)
    df_seg.to_csv(args.seg_out, index=False)

    # Standard scaling
    X = df_seg[seg_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df_seg["segment"] = kmeans.fit_predict(X_scaled)

    # Save modeling dataset
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    df_seg.to_csv(args.model_out, index=False)

    # Save segmentation artifact
    Path(args.seg_model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"scaler": scaler, "kmeans": kmeans, "seg_features": seg_features},
        args.seg_model_out)


if __name__ == "__main__":
    main()
