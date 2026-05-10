from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "backend" / "models_saved"
ENRICHED_CSV = BASE_DIR / "avito_data_enriched.csv"

_cache: Dict[str, Any] = {}


def get_enriched_df() -> pd.DataFrame:
    if "enriched_df" not in _cache:
        df = pd.read_csv(ENRICHED_CSV)
        df["Surface_m2"] = pd.to_numeric(df.get("Surface_m2"), errors="coerce")
        df["Prix_DH"] = pd.to_numeric(df.get("Prix_DH"), errors="coerce")
        df = df.dropna(subset=["Surface_m2", "Prix_DH", "Ville", "Quartier", "Type_Bien"])
        df = df[df["Surface_m2"] > 0]
        df["Prix_m2"] = df["Prix_DH"] / df["Surface_m2"]
        _cache["enriched_df"] = df
    return _cache["enriched_df"].copy()


def get_price_models() -> dict:
    if "price_models" not in _cache:
        xgb_model = joblib.load(MODELS_DIR / "xgb_price_model.pkl")
        rf_model = joblib.load(MODELS_DIR / "rf_price_model.pkl")
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        feature_columns = json_load(MODELS_DIR / "feature_columns.json")
        _cache["price_models"] = {
            "xgb": xgb_model,
            "rf": rf_model,
            "scaler": scaler,
            "feature_columns": feature_columns,
        }
    return _cache["price_models"]


def get_trend_summary() -> dict:
    if "trend_summary" not in _cache:
        summary_path = MODELS_DIR / "trend_summary.json"
        _cache["trend_summary"] = json_load(summary_path)
    return _cache["trend_summary"]


def json_load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_trend_city(city: str) -> dict:
    cache_key = f"trend_{city}"
    if cache_key not in _cache:
        payload = joblib.load(MODELS_DIR / f"trend_{city}.pkl")
        _cache[cache_key] = payload
    return _cache[cache_key]


def build_feature_vector(payload: dict, feature_columns: list, scaler) -> np.ndarray:
    numeric_cols = ["Surface_m2", "nb_chambres", "nb_salles_bain", "etage"]
    numeric_values = np.array([[
        float(payload["surface"]),
        float(payload["nb_chambres"]),
        float(payload["nb_salles_bain"]),
        float(payload["etage"]),
    ]])
    scaled_numeric = scaler.transform(numeric_values)[0]

    vector = np.zeros(len(feature_columns))
    feature_index = {name: idx for idx, name in enumerate(feature_columns)}

    for col, value in zip(numeric_cols, scaled_numeric):
        if col in feature_index:
            vector[feature_index[col]] = value

    month = datetime.utcnow().month
    if month in {12, 1, 2}:
        season = "Hiver"
    elif month in {3, 4, 5}:
        season = "Printemps"
    elif month in {6, 7, 8}:
        season = "Ete"
    else:
        season = "Automne"

    categorical_map = {
        "Ville": payload["ville"],
        "Quartier": payload["quartier"],
        "Type_Bien": payload["type_bien"],
        "etat_bien": payload["etat_bien"],
        "Saison": season,
    }

    for prefix, value in categorical_map.items():
        key = f"{prefix}_{value}"
        if key in feature_index:
            vector[feature_index[key]] = 1.0

    return vector.reshape(1, -1)


def compute_percentile(df: pd.DataFrame, value: float, city: str, quartier: str, type_bien: str) -> int:
    subset = df[(df["Ville"] == city) & (df["Quartier"] == quartier) & (df["Type_Bien"] == type_bien)]
    if len(subset) < 30:
        subset = df[(df["Ville"] == city) & (df["Type_Bien"] == type_bien)]
    if len(subset) < 30:
        subset = df[df["Ville"] == city]
    if subset.empty:
        subset = df
    prix_m2 = subset["Prix_m2"].values
    percentile = int(round(100 * (prix_m2 <= value).sum() / max(len(prix_m2), 1)))
    return max(1, min(99, percentile))
