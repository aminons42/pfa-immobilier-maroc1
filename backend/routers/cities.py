from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database.models_db import City, Quartier
from backend.routers.deps import get_db

router = APIRouter(prefix="/api", tags=["cities"])


@router.get("/cities")
def get_cities(db: Session = Depends(get_db)):
    cities = db.query(City).order_by(City.listing_count.desc()).all()
    return [
        {
            "name": city.name,
            "lat": city.lat,
            "lng": city.lng,
            "listing_count": city.listing_count,
            "avg_prix_m2": round(city.avg_prix_m2, 2),
            "avg_surface": round(city.avg_surface, 2),
            "investment_score": round(city.investment_score, 2),
        }
        for city in cities
    ]


@router.get("/cities/{city}/quartiers")
def get_quartiers(city: str, db: Session = Depends(get_db)):
    city_row = db.query(City).filter(City.name == city).first()
    if not city_row:
        return []
    quartiers = db.query(Quartier).filter(Quartier.city_id == city_row.id).all()
    return [
        {
            "name": quartier.name,
            "listing_count": quartier.listing_count,
            "avg_prix_m2": round(quartier.avg_prix_m2, 2),
        }
        for quartier in quartiers
    ]


@router.get("/heatmap")
def get_heatmap(db: Session = Depends(get_db)):
    cities = db.query(City).all()
    if not cities:
        return {"min": 0, "max": 0, "cities": []}
    min_price = min(city.avg_prix_m2 for city in cities)
    max_price = max(city.avg_prix_m2 for city in cities)
    return {
        "min": round(min_price, 2),
        "max": round(max_price, 2),
        "cities": [
            {
                "name": city.name,
                "lat": city.lat,
                "lng": city.lng,
                "listing_count": city.listing_count,
                "avg_prix_m2": round(city.avg_prix_m2, 2),
                "avg_surface": round(city.avg_surface, 2),
                "investment_score": round(city.investment_score, 2),
            }
            for city in cities
        ],
    }
