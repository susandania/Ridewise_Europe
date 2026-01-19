# src/data/ingest_data.py

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out-path", default="data/processed/ridewise_eda_data.csv")
    return p.parse_args()

def main():
    args = parse_args()
    raw = Path(args.raw_dir)

    # Load raw tables
    customer = pd.read_csv(raw / "customers.csv")
    df_t = pd.read_csv(raw / "trips.csv")
    df_ca = pd.read_csv(raw / "customer_activity.csv")
    promos = pd.read_csv(raw / "promotions.csv")
    redemption = pd.read_csv(raw / "promo_redemptions.csv")

    # Merge promotions + redemption
    df_pf = promos.merge(redemption, on="promo_id", how="left")

    # Trips: fill missing with median
    df_t["distance_km"] = df_t["distance_km"].fillna(df_t["distance_km"].median())
    df_t["duration_min"] = df_t["duration_min"].fillna(df_t["duration_min"].median())

    # Convert datetime
    df_t['trip_date'] = pd.to_datetime(df_t['trip_datetime']).dt.date
    df_t['trip_date'] = pd.to_datetime(df_t['trip_date'])


    # Aggregate customer activity
    ca_agg = (
        df_ca.groupby("customer_id")
        .agg(
            app_open_events=("event_type", lambda x: (x == "app_open").sum()),
            search_events=("event_type", lambda x: (x == "search").sum()),
            ride_request_events=("event_type", lambda x: (x == "ride_request").sum()),
            ride_completed_events=("event_type", lambda x: (x == "ride_completed").sum()),
        )
        .reset_index()
    )

    ca_agg["ride_conversion_rate"] = (
        ca_agg["ride_completed_events"] / ca_agg["ride_request_events"]
    ).fillna(0)

    ca_agg["request_to_search_ratio"] = (
        ca_agg["ride_request_events"] / ca_agg["search_events"]).fillna(0)

    ca_agg["search_to_open_ratio"] = (
        ca_agg["search_events"] / ca_agg["app_open_events"]).fillna(0)

    activity_recency = (
        df_ca.groupby("customer_id")["event_date"]
        .max()
        .reset_index(name="last_event_date"))

    active_days = (
        df_ca.groupby("customer_id")["event_date"]
        .nunique()
        .reset_index(name="active_days"))

    ca_agg = (
        ca_agg.merge(activity_recency, on="customer_id", how="left")
              .merge(active_days, on="customer_id", how="left"))


    # Aggregate trips
    t_agg = (
        df_t.groupby("customer_id")
        .agg(
            total_completed_trips=("trip_status", lambda x: (x == "completed").sum()),
            total_cancelled_trips=("trip_status", lambda x: (x == "cancelled").sum()),
            total_gross_revenue=("fare_amount", "sum"),
            avg_gross_revenue=("fare_amount", "mean"),
            net_platform_revenue=("platform_revenue", "sum"),
            avg_distance_km=("distance_km", "mean"),
            avg_duration_min=("duration_min", "mean"),
            total_driver_payout=("driver_payout", "sum"),
            total_trips=("trip_status", "count"),
            first_trip_date=("trip_date", "min"),
            last_trip_date=("trip_date", "max"),
            promo_used_avg=("promo_used", "mean"),
        )
        .reset_index()
    )

    t_agg["cancellation_rate"] = (
        t_agg["total_cancelled_trips"] / t_agg["total_trips"]).fillna(0)

    t_agg["conversion_rate"] = (
        t_agg["total_completed_trips"] / t_agg["total_trips"]).where(t_agg["total_trips"] > 0, 0)


    # Aggregate promotions
    df_pf["driver_borne_discount"] = (df_pf["discount_value"] - df_pf["promo_cost"]).clip(lower=0)

    pf_agg = (
    df_pf.groupby("customer_id")
    .agg(
        promo_used_count=("trip_id", "nunique"),
        total_promo_cost=("promo_cost", "sum"),
        avg_promo_cost=("promo_cost", "mean"),
        avg_discount_value=("discount_value", "mean"),
        avg_driver_burden=("driver_borne_discount", "mean"),
        driver_borne_total=("driver_borne_discount", "sum")).reset_index())

    # Totals used for ratios
    pf_agg["discount_total_fixed"] = pf_agg["driver_borne_total"] + pf_agg["total_promo_cost"]

    pf_agg["driver_burden_ratio"] = (
    pf_agg["driver_borne_total"] / pf_agg["discount_total_fixed"]).fillna(0)

    pf_agg["platform_burden_ratio"] = (
    pf_agg["total_promo_cost"] / pf_agg["discount_total_fixed"]).fillna(0)

    # Promo used rate relative to total trips
    trip_counts = (
    df_t.groupby("customer_id")["trip_id"]
    .nunique()
    .rename("total_trips")
    .reset_index())

    pf_agg = pf_agg.merge(trip_counts, on="customer_id", how="left")
    pf_agg["promo_used_rate"] = (pf_agg["promo_used_count"] / pf_agg["total_trips"]).fillna(0)

    # Drop helper column
    pf_agg = pf_agg.drop(columns=["total_trips"])



    # Merge all aggregates with customer table
    df = (
    customer
    .merge(ca_agg, on="customer_id", how="left")
    .merge(t_agg, on="customer_id", how="left")
    .merge(pf_agg, on="customer_id", how="left")
)

    # Fill numeric NaNs from left joins
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0)

    # Fix duplicate total_trips columns
    if "total_trips_y" in df.columns:
        df = df.drop(columns=["total_trips_y"])
    if "total_trips_x" in df.columns:
        df = df.rename(columns={"total_trips_x": "total_trips"})


    # Save
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()

