$ErrorActionPreference = "Stop"

python -m pip install --default-timeout 120 -r backend/requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-Not (Test-Path "avito_data_enriched.csv")) {
  python backend/models/generate_synthetic.py
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-Not (Test-Path "backend/models_saved/xgb_price_model.pkl")) {
  python backend/models/train_price_model.py
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-Not (Test-Path "backend/models_saved/trend_summary.json")) {
  python backend/models/train_trend_model.py
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "System running at http://localhost:8000/app"
python -m uvicorn backend.main_api:app --reload --host 0.0.0.0 --port 8000
