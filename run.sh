#!/usr/bin/env bash
set -e

python -m pip install -r backend/requirements.txt

if [ ! -f avito_data_enriched.csv ]; then
  python backend/models/generate_synthetic.py
fi

if [ ! -f backend/models_saved/xgb_price_model.pkl ]; then
  python backend/models/train_price_model.py
fi

if [ ! -f backend/models_saved/trend_summary.json ]; then
  python backend/models/train_trend_model.py
fi

echo "System running at http://localhost:8000/app"
uvicorn backend.main_api:app --reload --host 0.0.0.0 --port 8000
