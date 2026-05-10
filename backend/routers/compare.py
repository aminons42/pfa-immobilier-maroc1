from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.data_store import get_enriched_df, get_trend_summary

router = APIRouter(prefix="/api", tags=["compare"])


class CompareRequest(BaseModel):
    city_a: str
    city_b: str
    type_bien: str | None = None


@router.post("/compare")
def compare_cities(payload: CompareRequest):
    df = get_enriched_df()
    trend_summary = get_trend_summary()

    def city_metrics(city: str):
        city_df = df[df["Ville"] == city]
        if city_df.empty:
            return None
        prix_m2 = float(city_df["Prix_m2"].mean())
        prix_total = float(city_df["Prix_DH"].mean())
        surface = float(city_df["Surface_m2"].mean())
        count = int(len(city_df))
        growth = float(trend_summary.get(city, {}).get("annual_growth_rate", 0.0))
        investment = float(trend_summary.get(city, {}).get("investment_score", 0.0))
        return {
            "city": city,
            "avg_prix_m2": prix_m2,
            "avg_prix_total": prix_total,
            "avg_surface": surface,
            "listing_count": count,
            "annual_growth_rate": growth,
            "investment_score": investment,
        }

    metrics_a = city_metrics(payload.city_a)
    metrics_b = city_metrics(payload.city_b)
    if not metrics_a or not metrics_b:
        raise HTTPException(status_code=404, detail="City not found")

    winner_growth = metrics_a if metrics_a["annual_growth_rate"] >= metrics_b["annual_growth_rate"] else metrics_b
    winner_volume = metrics_a if metrics_a["listing_count"] >= metrics_b["listing_count"] else metrics_b

    verdict = {
        "long_term": winner_growth["city"],
        "budget": winner_volume["city"],
        "growth_rate": winner_growth["annual_growth_rate"],
        "listing_count": winner_volume["listing_count"],
    }

    return {
        "city_a": metrics_a,
        "city_b": metrics_b,
        "verdict": verdict,
    }
