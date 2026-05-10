from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.services.data_store import get_enriched_df, get_trend_city, get_trend_summary

router = APIRouter(prefix="/api", tags=["stats"])


@router.get("/stats/{city}")
def city_stats(city: str):
    df = get_enriched_df()
    city_df = df[df["Ville"] == city]
    if city_df.empty:
        raise HTTPException(status_code=404, detail="City not found")

    price_series = city_df["Prix_DH"].values
    prix_m2_series = city_df["Prix_m2"].values

    q1 = float(np.percentile(price_series, 25))
    median = float(np.percentile(price_series, 50))
    q3 = float(np.percentile(price_series, 75))

    type_group = (
        city_df.groupby("Type_Bien")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="count")
    )
    type_distribution = [
        {"type": row["Type_Bien"], "count": int(row["count"])} for _, row in type_group.iterrows()
    ]

    quartier_group = (
        city_df.groupby("Quartier")["Prix_m2"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .reset_index(name="avg_prix_m2")
    )
    top_quartiers = [
        {"name": row["Quartier"], "avg_prix_m2": float(row["avg_prix_m2"])}
        for _, row in quartier_group.iterrows()
    ]

    trend = get_trend_city(city)
    trend_summary = get_trend_summary().get(city, {})

    return {
        "city": city,
        "listing_count": int(len(city_df)),
        "avg_prix_m2": float(prix_m2_series.mean()),
        "avg_prix_total": float(price_series.mean()),
        "avg_surface": float(city_df["Surface_m2"].mean()),
        "investment_score": float(trend_summary.get("investment_score", 0.0)),
        "price_range": {
            "min": float(price_series.min()),
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": float(price_series.max()),
        },
        "type_distribution": type_distribution,
        "top_quartiers": top_quartiers,
        "trend": trend.get("historical", []),
    }
