from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Surface_m2"] = pd.to_numeric(df.get("Surface_m2"), errors="coerce")
    df["Prix_DH"] = pd.to_numeric(df.get("Prix_DH"), errors="coerce")
    df = df.dropna(subset=["Surface_m2", "Prix_DH", "Ville", "Date_Annonce"])
    df = df[df["Surface_m2"] > 0]
    df["Date_Annonce"] = pd.to_datetime(df["Date_Annonce"], errors="coerce")
    df = df.dropna(subset=["Date_Annonce"])
    df["Prix_m2"] = df["Prix_DH"] / df["Surface_m2"]
    df["Year"] = df["Date_Annonce"].dt.year
    df["Month"] = df["Date_Annonce"].dt.month
    return df


def _fit_trend(series: pd.DataFrame) -> dict:
    series = series.sort_values(["Year", "Month"]).reset_index(drop=True)
    time_index = np.arange(len(series)).reshape(-1, 1)
    y = series["Prix_m2"].values

    model = LinearRegression()
    model.fit(time_index, y)
    trend = model.predict(time_index)
    residuals = y - trend

    series["Month_of_year"] = series["Month"]
    seasonal = series.assign(residual=residuals).groupby("Month_of_year")["residual"].mean().to_dict()

    residual_std = np.std(residuals) if len(residuals) else 0.0

    return {
        "model": model,
        "trend": trend,
        "residuals": residuals,
        "seasonal": seasonal,
        "residual_std": residual_std,
    }


def train_trend_model(input_csv: Path, models_dir: Path, top_cities: List[str]) -> None:
    df = pd.read_csv(input_csv)
    df = _prepare_data(df)
    df = df[df["Ville"].isin(top_cities)]

    models_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, dict] = {}

    city_counts = df["Ville"].value_counts().to_dict()

    city_volatility: Dict[str, float] = {}
    city_growth: Dict[str, float] = {}

    for city in top_cities:
        city_df = df[df["Ville"] == city]
        grouped = city_df.groupby(["Year", "Month"]).median(numeric_only=True).reset_index()
        if grouped.empty:
            continue

        trend_data = _fit_trend(grouped)
        model = trend_data["model"]

        time_index = np.arange(len(grouped)).reshape(-1, 1)
        trend = trend_data["trend"]
        residual_std = trend_data["residual_std"]
        seasonal = trend_data["seasonal"]

        historical = []
        for idx, row in grouped.iterrows():
            month = int(row["Month"])
            seasonal_adj = seasonal.get(month, 0.0)
            value = float(trend[idx] + seasonal_adj)
            lower = float(value - 1.645 * residual_std)
            upper = float(value + 1.645 * residual_std)
            historical.append({
                "year": int(row["Year"]),
                "month": month,
                "value": round(value, 2),
                "low": round(lower, 2),
                "high": round(upper, 2),
            })

        forecast = []
        last_index = len(grouped) - 1
        for i in range(1, 121):
            future_index = last_index + i
            future_time = np.array([[future_index]])
            future_trend = float(model.predict(future_time)[0])
            future_month = int((grouped.iloc[-1]["Month"] + i - 1) % 12 + 1)
            seasonal_adj = seasonal.get(future_month, 0.0)
            value = future_trend + seasonal_adj
            lower = value - 1.645 * residual_std
            upper = value + 1.645 * residual_std
            year = int(grouped.iloc[-1]["Year"] + (grouped.iloc[-1]["Month"] + i - 1) // 12)
            forecast.append({
                "year": year,
                "month": future_month,
                "value": round(value, 2),
                "low": round(lower, 2),
                "high": round(upper, 2),
            })

        last_trend_value = trend[-1] if len(trend) else 0.0
        annual_growth_rate = (model.coef_[0] * 12) / max(last_trend_value, 1.0)
        annual_growth_rate = float(annual_growth_rate)

        volatility = float(np.std(grouped["Prix_m2"].values))
        city_volatility[city] = volatility
        city_growth[city] = annual_growth_rate

        investment_score_placeholder = 0.0

        payload = {
            "city": city,
            "historical": historical,
            "forecast": forecast,
            "annual_growth_rate": annual_growth_rate,
            "residual_std": residual_std,
            "investment_score": investment_score_placeholder,
        }
        joblib.dump(payload, models_dir / f"trend_{city}.pkl")

    max_count = max(city_counts.values()) if city_counts else 1
    max_growth = max(city_growth.values()) if city_growth else 1
    min_growth = min(city_growth.values()) if city_growth else 0
    max_volatility = max(city_volatility.values()) if city_volatility else 1
    min_volatility = min(city_volatility.values()) if city_volatility else 0

    for city in top_cities:
        growth_rate = city_growth.get(city, 0.0)
        growth_score = 4 * (growth_rate - min_growth) / max(max_growth - min_growth, 1e-6)
        volume_score = 3 * city_counts.get(city, 0) / max_count
        volatility = city_volatility.get(city, max_volatility)
        volatility_score = 3 * (max_volatility - volatility) / max(max_volatility - min_volatility, 1e-6)
        investment_score = float(np.clip(growth_score + volume_score + volatility_score, 0, 10))

        model_path = models_dir / f"trend_{city}.pkl"
        if model_path.exists():
            payload = joblib.load(model_path)
            payload["investment_score"] = investment_score
            joblib.dump(payload, model_path)

        summary[city] = {
            "annual_growth_rate": growth_rate,
            "investment_score": investment_score,
        }

    with open(models_dir / "trend_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    input_csv = base_dir / "avito_data_enriched.csv"
    models_dir = base_dir / "backend" / "models_saved"

    top_cities_path = models_dir / "top15_cities.json"
    if top_cities_path.exists():
        with open(top_cities_path, "r", encoding="utf-8") as f:
            top_cities = json.load(f)
    else:
        df = pd.read_csv(input_csv)
        top_cities = df["Ville"].value_counts().head(15).index.tolist()

    train_trend_model(input_csv, models_dir, top_cities)


if __name__ == "__main__":
    main()
