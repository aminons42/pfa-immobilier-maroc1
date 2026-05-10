from __future__ import annotations

from pathlib import Path

import pandas as pd

from .db import Base, SessionLocal, engine
from .models_db import City, PropertyType, Quartier

CITY_COORDINATES = {
    "Casablanca": {"lat": 33.5731, "lng": -7.5898},
    "Rabat": {"lat": 33.9716, "lng": -6.8498},
    "Marrakech": {"lat": 31.6295, "lng": -7.9811},
    "Tanger": {"lat": 35.7595, "lng": -5.8330},
    "Agadir": {"lat": 30.4278, "lng": -9.5981},
    "Fes": {"lat": 34.0181, "lng": -5.0078},
    "Meknes": {"lat": 33.8935, "lng": -5.5547},
    "Kenitra": {"lat": 34.2610, "lng": -6.5802},
    "Tetouan": {"lat": 35.5785, "lng": -5.3684},
    "Sale": {"lat": 34.0531, "lng": -6.7985},
    "Oujda": {"lat": 34.6814, "lng": -1.9086},
    "Safi": {"lat": 32.2994, "lng": -9.2372},
    "El Jadida": {"lat": 33.2316, "lng": -8.5007},
    "Beni Mellal": {"lat": 32.3373, "lng": -6.3498},
    "Laayoune": {"lat": 27.1536, "lng": -13.2033},
}


def seed_database(csv_path: Path, trend_summary: dict) -> None:
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as session:
        if session.query(City).count() > 0:
            return

        df = pd.read_csv(csv_path)
        df["Surface_m2"] = pd.to_numeric(df.get("Surface_m2"), errors="coerce")
        df["Prix_DH"] = pd.to_numeric(df.get("Prix_DH"), errors="coerce")
        df = df.dropna(subset=["Surface_m2", "Prix_DH", "Ville", "Quartier", "Type_Bien"])
        df = df[df["Surface_m2"] > 0]
        df["Prix_m2"] = df["Prix_DH"] / df["Surface_m2"]

        top_cities = df["Ville"].value_counts().head(15).index.tolist()
        df = df[df["Ville"].isin(top_cities)]

        for city in top_cities:
            city_df = df[df["Ville"] == city]
            listing_count = int(len(city_df))
            avg_prix_m2 = float(city_df["Prix_m2"].mean())
            avg_prix_total = float(city_df["Prix_DH"].mean())
            avg_surface = float(city_df["Surface_m2"].mean())
            coords = CITY_COORDINATES.get(city, {})
            investment_score = float(trend_summary.get(city, {}).get("investment_score", 0.0))

            city_row = City(
                name=city,
                lat=coords.get("lat"),
                lng=coords.get("lng"),
                listing_count=listing_count,
                avg_prix_m2=avg_prix_m2,
                avg_prix_total=avg_prix_total,
                avg_surface=avg_surface,
                investment_score=investment_score,
            )
            session.add(city_row)
            session.flush()

            quartier_group = city_df.groupby("Quartier").agg(
                listing_count=("Quartier", "count"),
                avg_prix_m2=("Prix_m2", "mean"),
            )
            for quartier_name, row in quartier_group.iterrows():
                quartier_row = Quartier(
                    city_id=city_row.id,
                    name=str(quartier_name),
                    listing_count=int(row["listing_count"]),
                    avg_prix_m2=float(row["avg_prix_m2"]),
                )
                session.add(quartier_row)

            type_group = city_df.groupby("Type_Bien").agg(
                count=("Type_Bien", "count"),
                avg_prix_m2=("Prix_m2", "mean"),
                avg_surface=("Surface_m2", "mean"),
            )
            for type_name, row in type_group.iterrows():
                percentage = float(row["count"] / listing_count * 100) if listing_count else 0.0
                type_row = PropertyType(
                    city_id=city_row.id,
                    type_name=str(type_name),
                    count=int(row["count"]),
                    percentage=percentage,
                    avg_prix_m2=float(row["avg_prix_m2"]),
                    avg_surface=float(row["avg_surface"]),
                )
                session.add(type_row)

        session.commit()
