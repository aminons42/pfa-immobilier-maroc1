from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conint, confloat, constr
from typing import Optional, Literal
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import math
from typing import Annotated
from pydantic import  Field
# ── Ajouter après les imports, avant la définition de l'app ──────────────────

# Taux de croissance annuels estimés par ville (source: HCP + indices Avito/Mubawab 2019-2024)
# Ces taux sont facilement ajustables sans re-entraîner le modèle
TAUX_CROISSANCE = {
    "Casablanca":  0.055,   # +5.5%/an — marché le plus dynamique
    "Rabat":       0.048,
    "Marrakech":   0.062,   # Tourisme + investissement étranger
    "Tanger":      0.071,   # Zone franche — croissance forte
    "Agadir":      0.045,
    "Fès":         0.038,
    "Meknès":      0.032,
    "Oujda":       0.028,
    "Kénitra":     0.042,
    "Tétouan":     0.038,
    "El Jadida":   0.035,
    "Essaouira":   0.040,
    "Mohammedia":  0.048,
    "Temara":      0.042,
    "Berrechid":   0.033,
    "Safi":        0.030,
    "default":     0.038,   # Taux national moyen
}

def generate_forecast(prix_base: float, prix_m2_base: float, ville: str, surface: float):
    """
    Génère une projection de prix sur 5 ans à partir du prix estimé.
    Utilise un taux de croissance annuel composé par ville.
    Retourne 6 points : année 0 (actuel) + années +1 à +5.
    """
    from datetime import datetime
    annee_base = datetime.now().year
    taux = TAUX_CROISSANCE.get(ville, TAUX_CROISSANCE["default"])
    
    forecast = []
    for n in range(6):  # 0 = maintenant, 1..5 = futures années
        prix_n = prix_base * ((1 + taux) ** n)
        forecast.append({
            "year": annee_base + n,
            "prix": clean_float(prix_n),
            "prix_m2": clean_float(prix_n / surface),
        })
    
    variation_5ans = ((forecast[5]["prix"] / prix_base) - 1) * 100
    
    return {
        "points": forecast,
        "taux_annuel": clean_float(taux * 100),          # ex: 5.5
        "variation_5ans": clean_float(variation_5ans),   # ex: 30.7
    }
    
def clean_float(x):
    if pd.isna(x) or math.isinf(x): return 0.0
    return float(x)

def clean_int(x):
    if pd.isna(x) or math.isinf(x): return 0
    return int(x)

def clean_str(x):
    if pd.isna(x): return "Inconnu"
    return str(x)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

modele = None
scaler = None
moyennes_villes = None
moyennes_quartiers = None
colonnes_entrainement = None
moyenne_globale = None

print("📊 Chargement des données réelles...")
data_path = DATA_DIR / "avito_data_reel.csv"
df_stats = pd.read_csv(data_path)
df_stats.rename(columns={'Type_Bien': 'Type', 'Prix_DH': 'Prix', 'Surface_m2': 'Surface'}, inplace=True)
df_stats['Prix'] = pd.to_numeric(df_stats['Prix'], errors='coerce')
df_stats['Surface'] = pd.to_numeric(df_stats['Surface'], errors='coerce')
df_stats = df_stats.dropna(subset=['Prix', 'Surface', 'Ville', 'Quartier', 'Type'])

types_residentiels = ['Appartements', 'Maisons', 'Villas et Riads']
df_stats = df_stats[df_stats['Type'].isin(types_residentiels)]
df_stats = df_stats[(df_stats['Prix'] >= 100000) & (df_stats['Prix'] <= 15000000)]
df_stats = df_stats[(df_stats['Surface'] >= 15) & (df_stats['Surface'] <= 500)]
df_stats['Prix_m2'] = df_stats['Prix'] / df_stats['Surface']
df_stats = df_stats[(df_stats['Prix_m2'] >= 2000) & (df_stats['Prix_m2'] <= 40000)]
quartiers_poubelles = ['Toute la ville', 'Autre secteur', 'Periferie']
df_stats = df_stats[~df_stats['Quartier'].isin(quartiers_poubelles)]

app = FastAPI(title="API Immo Maroc - PFA")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_assets():
    global modele, scaler, moyennes_villes, moyennes_quartiers, colonnes_entrainement, moyenne_globale
    try:
        print("🧠 Chargement du modele ML...")
        modele = joblib.load(MODELS_DIR / "modele_champion.joblib")
        scaler = joblib.load(MODELS_DIR / "scaler.joblib")
        moyennes_villes = joblib.load(MODELS_DIR / "encodeur_villes.joblib")
        moyennes_quartiers = joblib.load(MODELS_DIR / "encodeur_quartiers.joblib")
        colonnes_entrainement = joblib.load(MODELS_DIR / "colonnes_entrainement.joblib")
        moyenne_globale = joblib.load(MODELS_DIR / "moyenne_globale.joblib")
    except Exception as exc:
        raise RuntimeError(f"Erreur de chargement des artefacts ML: {exc}")



class BienImmobilier(BaseModel):
    ville: Annotated[str, Field(min_length=1)]
    quartier: Annotated[str, Field(min_length=1)]
    type_bien: Annotated[str, Field(min_length=1)]
    surface: Annotated[float, Field(ge=15, le=500)]

    nb_chambres: Annotated[int, Field(ge=0, le=20)] = 1
    nb_salles_bain: Annotated[int, Field(ge=0, le=20)] = 1
    etage: Annotated[int, Field(ge=0, le=120)] = 1
class CompareRequest(BaseModel):
    city_a: str
    city_b: str
    type_bien: Optional[str] = None

@app.post("/api/predict")
def predire_prix(bien: BienImmobilier):
    if modele is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    
    # ── Prédiction ML (code existant — ne pas toucher) ──────────────────────
    input_data = pd.DataFrame([{
        'Surface': bien.surface, 'Type': bien.type_bien,
        'Nombre_Chambres': bien.nb_chambres,
        'Salles_de_bain': bien.nb_salles_bain, 'Etage': bien.etage
    }])
    prix_ville = moyennes_villes.get(bien.ville, moyenne_globale)
    prix_quartier = moyennes_quartiers.get(bien.quartier, prix_ville)
    input_data['Ville_Encoded'] = prix_ville
    input_data['Quartier_Encoded'] = prix_quartier
    input_encoded = pd.get_dummies(input_data)
    input_aligned = input_encoded.reindex(columns=colonnes_entrainement, fill_value=0).astype(float)
    input_scaled = scaler.transform(input_aligned)
    prix_log = modele.predict(input_scaled)[0]
    prix_estime = clean_float(max(0.0, np.expm1(prix_log)))
    prix_m2 = clean_float(prix_estime / bien.surface)
    
    # ── Nouvelle section : forecast ─────────────────────────────────────────
    forecast = generate_forecast(prix_estime, prix_m2, bien.ville, bien.surface)
    
    return {
        "estimation":      prix_estime,
        "range_min":       prix_estime * 0.85,
        "range_max":       prix_estime * 1.15,
        "prix_m2_estime":  prix_m2,
        # Nouveaux champs :
        "forecast":        forecast["points"],       # liste de 6 dicts {year, prix, prix_m2}
        "taux_annuel":     forecast["taux_annuel"],  # ex: 5.5
        "variation_5ans":  forecast["variation_5ans"] # ex: 30.7
    }

@app.get("/api/cities")
def get_cities():
    villes = df_stats['Ville'].value_counts().head(25).index.tolist()
    return [{"name": clean_str(v)} for v in sorted(villes)]

@app.get("/api/cities/{city}/quartiers")
def get_quartiers(city: str):
    quartiers = df_stats[df_stats['Ville'] == city]['Quartier'].value_counts().head(30).index.tolist()
    return [{"name": clean_str(q)} for q in sorted(quartiers)]

@app.get("/api/heatmap")
def get_heatmap():
    coords = {
        "Casablanca": (33.5731, -7.5898), "Rabat": (34.0209, -6.8416), "Marrakech": (31.6295, -7.9811),
        "Tanger": (35.7595, -5.8340), "Agadir": (30.4278, -9.5981), "Fès": (34.0331, -5.0003),
        "Meknès": (33.8935, -5.5473), "Oujda": (34.6814, -1.9086), "Kénitra": (34.2610, -6.5802),
        "Tétouan": (35.5785, -5.3684), "El Jadida": (33.2316, -8.5007), "Essaouira": (31.5085, -9.7595),
        "Al Hoceima": (35.2442, -3.9317), "Berrechid": (33.2655, -7.5875), "Bouznika": (33.7894, -7.1597),
        "Guelmim": (28.9869, -10.0574), "Jerada": (34.3117, -2.1636), "Temara": (33.9265, -6.9126),
        "Safi": (32.2994, -9.2372), "Mohammedia": (33.6858, -7.3829)
    }
    result = []
    for v in df_stats['Ville'].value_counts().head(25).index:
        sub = df_stats[df_stats['Ville'] == v]
        if v in coords and len(sub) > 0:
            result.append({
                "name": clean_str(v), "lat": coords[v][0], "lng": coords[v][1], "listing_count": len(sub),
                "avg_prix_m2": clean_float(sub['Prix_m2'].mean())
            })
    return {"cities": result}

@app.get("/api/stats/{city}")
def get_stats(city: str):
    sub = df_stats[df_stats['Ville'] == city]
    if len(sub) == 0: return {"error": "Aucune donnée"}
    
    return {
        "listing_count": clean_int(len(sub)),
        "avg_prix_m2": clean_float(sub['Prix_m2'].mean()),
        "price_range": {"min": clean_float(sub['Prix'].min()), "median": clean_float(sub['Prix'].median()), "max": clean_float(sub['Prix'].max())},
        "avg_surface": clean_float(sub['Surface'].mean()),
        "top_quartiers": [{"name": clean_str(idx), "avg_prix_m2": clean_float(val)} for idx, val in sub.groupby('Quartier')['Prix_m2'].mean().nlargest(5).items()],
        "type_distribution": [{"type": clean_str(idx), "count": clean_int(val)} for idx, val in sub['Type'].value_counts().items()]
    }

@app.post("/api/compare")
def compare_cities(payload: CompareRequest):
    def fetch_data(c):
        sub = df_stats[df_stats['Ville'] == c]
        if payload.type_bien and payload.type_bien.strip() != "":
            sub = sub[sub['Type'].str.strip() == payload.type_bien.strip()]
        if len(sub) == 0: 
            return {"city": c, "avg_prix_m2": 0, "avg_prix_total": 0, "avg_surface": 0, "listing_count": 0}
        return {"city": clean_str(c), "avg_prix_m2": clean_float(sub['Prix_m2'].mean()), "avg_prix_total": clean_float(sub['Prix'].mean()), "avg_surface": clean_float(sub['Surface'].mean()), "listing_count": len(sub)}
    return {"city_a": fetch_data(payload.city_a), "city_b": fetch_data(payload.city_b)}

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")