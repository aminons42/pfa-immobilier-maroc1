from __future__ import annotations

from datetime import datetime

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database.models_db import City, PredictionLog, Quartier
from backend.routers.deps import get_db
from backend.services.data_store import build_feature_vector, compute_percentile, get_enriched_df, get_price_models, get_trend_summary

router = APIRouter(prefix="/api", tags=["predict"])


class PredictRequest(BaseModel):
    ville: str
    quartier: str
    type_bien: str
    surface: float
    nb_chambres: int
    nb_salles_bain: int
    etat_bien: str
    etage: int


@router.post("/predict")
def predict_price(payload: PredictRequest, db: Session = Depends(get_db)):
    models = get_price_models()
    feature_columns = models["feature_columns"]
    vector = build_feature_vector(payload.model_dump(), feature_columns, models["scaler"])

    xgb = models["xgb"]
    rf = models["rf"]

    central = float(xgb.predict(vector)[0])
    tree_preds = np.stack([estimator.predict(vector)[0] for estimator in rf.estimators_])
    std = float(np.std(tree_preds))

    lower = max(0.0, central - 1.645 * std)
    upper = max(lower, central + 1.645 * std)

    prix_m2_estime = central / max(payload.surface, 1)

    city_row = db.query(City).filter(City.name == payload.ville).first()
    quartier_row = None
    if city_row:
        quartier_row = (
            db.query(Quartier)
            .filter(Quartier.city_id == city_row.id)
            .filter(Quartier.name == payload.quartier)
            .first()
        )

    avg_city = city_row.avg_prix_m2 if city_row else 0.0
    avg_quartier = quartier_row.avg_prix_m2 if quartier_row else avg_city

    position = "above_market" if prix_m2_estime >= avg_quartier else "below_market"

    enriched_df = get_enriched_df()
    percentile = compute_percentile(enriched_df, prix_m2_estime, payload.ville, payload.quartier, payload.type_bien)

    trend_summary = get_trend_summary()
    investment_score = float(trend_summary.get(payload.ville, {}).get("investment_score", 0.0))

    if investment_score >= 8:
        verdict = "Excellent investissement"
    elif investment_score >= 6.5:
        verdict = "Bon investissement"
    elif investment_score >= 4:
        verdict = "Investissement modéré"
    else:
        verdict = "Investissement risqué"

    log_row = PredictionLog(
        timestamp=datetime.utcnow(),
        city=payload.ville,
        quartier=payload.quartier,
        type_bien=payload.type_bien,
        surface=payload.surface,
        nb_chambres=payload.nb_chambres,
        etat_bien=payload.etat_bien,
        predicted_min=lower,
        predicted_central=central,
        predicted_max=upper,
    )
    db.add(log_row)
    db.commit()

    return {
        "estimation": round(central),
        "range_min": round(lower),
        "range_max": round(upper),
        "prix_m2_estime": round(prix_m2_estime),
        "confidence": "90%",
        "market_context": {
            "avg_prix_m2_city": round(avg_city, 2),
            "avg_prix_m2_quartier": round(avg_quartier, 2),
            "position": position,
            "percentile": percentile,
        },
        "investment_score": round(investment_score, 2),
        "investment_verdict": verdict,
    }
