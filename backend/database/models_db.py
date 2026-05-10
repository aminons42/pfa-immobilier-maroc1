from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class City(Base):
    __tablename__ = "cities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lng: Mapped[float | None] = mapped_column(Float, nullable=True)
    listing_count: Mapped[int] = mapped_column(Integer)
    avg_prix_m2: Mapped[float] = mapped_column(Float)
    avg_prix_total: Mapped[float] = mapped_column(Float)
    avg_surface: Mapped[float] = mapped_column(Float)
    investment_score: Mapped[float] = mapped_column(Float)

    quartiers: Mapped[list["Quartier"]] = relationship("Quartier", back_populates="city", cascade="all, delete-orphan")
    property_types: Mapped[list["PropertyType"]] = relationship(
        "PropertyType", back_populates="city", cascade="all, delete-orphan"
    )


class Quartier(Base):
    __tablename__ = "quartiers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    city_id: Mapped[int] = mapped_column(ForeignKey("cities.id"))
    name: Mapped[str] = mapped_column(String, index=True)
    listing_count: Mapped[int] = mapped_column(Integer)
    avg_prix_m2: Mapped[float] = mapped_column(Float)

    city: Mapped[City] = relationship("City", back_populates="quartiers")


class PropertyType(Base):
    __tablename__ = "property_types"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    city_id: Mapped[int] = mapped_column(ForeignKey("cities.id"))
    type_name: Mapped[str] = mapped_column(String)
    count: Mapped[int] = mapped_column(Integer)
    percentage: Mapped[float] = mapped_column(Float)
    avg_prix_m2: Mapped[float] = mapped_column(Float)
    avg_surface: Mapped[float] = mapped_column(Float)

    city: Mapped[City] = relationship("City", back_populates="property_types")


class PredictionLog(Base):
    __tablename__ = "predictions_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    city: Mapped[str] = mapped_column(String)
    quartier: Mapped[str] = mapped_column(String)
    type_bien: Mapped[str] = mapped_column(String)
    surface: Mapped[float] = mapped_column(Float)
    nb_chambres: Mapped[int] = mapped_column(Integer)
    etat_bien: Mapped[str] = mapped_column(String)
    predicted_min: Mapped[float] = mapped_column(Float)
    predicted_central: Mapped[float] = mapped_column(Float)
    predicted_max: Mapped[float] = mapped_column(Float)
