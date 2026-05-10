from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.services.data_store import get_trend_city, get_trend_summary

router = APIRouter(prefix="/api", tags=["trend"])


@router.get("/trend/{city}")
def get_trend(city: str, type_bien: str | None = None):
    try:
        payload = get_trend_city(city)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="City trend not found") from exc

    return {
        "city": city,
        "type_bien": type_bien,
        "historical": payload.get("historical", []),
        "forecast": payload.get("forecast", []),
        "annual_growth_rate": payload.get("annual_growth_rate", 0.0),
        "investment_score": payload.get("investment_score", 0.0),
    }


@router.get("/trend/{city}/investment")
def get_investment(city: str):
    summary = get_trend_summary()
    if city not in summary:
        raise HTTPException(status_code=404, detail="City not found")

    score = summary[city].get("investment_score", 0.0)
    growth = summary[city].get("annual_growth_rate", 0.0)

    if score >= 8:
        verdict = "Excellent investissement"
    elif score >= 6.5:
        verdict = "Bon investissement"
    elif score >= 4:
        verdict = "Investissement modéré"
    else:
        verdict = "Investissement risqué"

    return {
        "city": city,
        "investment_score": round(score, 2),
        "annual_growth_rate": growth,
        "verdict": verdict,
        "breakdown": {
            "growth_component": round(min(4.0, max(0.0, growth * 4)), 2),
            "volume_component": "included",
            "volatility_component": "included",
        },
    }
