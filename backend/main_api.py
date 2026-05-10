from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.database.seed import seed_database
from backend.models.generate_synthetic import generate_enriched_csv
from backend.models.train_price_model import train_price_model
from backend.models.train_trend_model import train_trend_model
from backend.routers import cities, compare, predict, stats, trend

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

app = FastAPI(title="IMMO MAROC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cities.router)
app.include_router(predict.router)
app.include_router(trend.router)
app.include_router(stats.router)
app.include_router(compare.router)

frontend_dir = ROOT_DIR / "frontend"
app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend")


@app.on_event("startup")
def startup_tasks() -> None:
    enriched_csv = ROOT_DIR / "avito_data_enriched.csv"
    if not enriched_csv.exists():
        generate_enriched_csv(ROOT_DIR / "avito_data_2022_collection.csv", enriched_csv)

    models_dir = ROOT_DIR / "backend" / "models_saved"
    if not (models_dir / "xgb_price_model.pkl").exists():
        train_price_model(enriched_csv, models_dir)

    if not (models_dir / "trend_summary.json").exists():
        with open(models_dir / "top15_cities.json", "r", encoding="utf-8") as f:
            top_cities = __import__("json").load(f)
        train_trend_model(enriched_csv, models_dir, top_cities)

    trend_summary_path = models_dir / "trend_summary.json"
    trend_summary = __import__("json").loads(trend_summary_path.read_text(encoding="utf-8"))
    seed_database(enriched_csv, trend_summary)
