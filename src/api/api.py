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

# ==============================================================================
# UTILITAIRES
# ==============================================================================

def clean_float(x):
    if pd.isna(x) or math.isinf(x): return 0.0
    return float(x)

def clean_int(x):
    if pd.isna(x) or math.isinf(x): return 0
    return int(x)

def clean_str(x):
    if pd.isna(x): return "Inconnu"
    return str(x)

# ==============================================================================
# CHEMINS
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_DIR     = PROJECT_ROOT / "data"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# ==============================================================================
# CHARGEMENT DES DONNÉES STATISTIQUES (au démarrage du module, avant l'app)
# ==============================================================================

print("📊 Chargement des données réelles...")
data_path = DATA_DIR / "avito_data_reel.csv"
df_stats = pd.read_csv(data_path)
df_stats.rename(columns={'Type_Bien': 'Type', 'Prix_DH': 'Prix', 'Surface_m2': 'Surface'}, inplace=True)
df_stats['Prix']    = pd.to_numeric(df_stats['Prix'],    errors='coerce')
df_stats['Surface'] = pd.to_numeric(df_stats['Surface'], errors='coerce')
df_stats = df_stats.dropna(subset=['Prix', 'Surface', 'Ville', 'Quartier', 'Type'])

types_residentiels = ['Appartements', 'Maisons', 'Villas et Riads']
df_stats = df_stats[df_stats['Type'].isin(types_residentiels)]
df_stats = df_stats[(df_stats['Prix']    >= 100_000) & (df_stats['Prix']    <= 15_000_000)]
df_stats = df_stats[(df_stats['Surface'] >= 15)      & (df_stats['Surface'] <= 500)]
df_stats['Prix_m2'] = df_stats['Prix'] / df_stats['Surface']
df_stats = df_stats[(df_stats['Prix_m2'] >= 2_000) & (df_stats['Prix_m2'] <= 40_000)]
df_stats = df_stats[~df_stats['Quartier'].isin(['Toute la ville', 'Autre secteur', 'Periferie'])]

# ==============================================================================
# APPLICATION FASTAPI
# ==============================================================================

app = FastAPI(title="API Immo Maroc - PFA")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Variables globales — chargées une seule fois au démarrage de l'API
# ──────────────────────────────────────────────────────────────────────────────
modele                   = None
scaler                   = None
colonnes_entrainement    = None
fallbacks                = {}   # valeurs de remplacement pour zones inconnues

# Encodages villes (4 Series pandas indexées sur le nom de la ville)
enc_ville_prix_smoothe   = None   # prix moyen lissé (anti-bruit sur petits quartiers)
enc_ville_prixm2_smoothe = None   # prix/m² moyen lissé
enc_ville_prix_median    = None   # prix médian (robuste aux villas exceptionnelles)
enc_ville_prixm2_median  = None   # prix/m² médian

# Encodages quartiers (5 Series pandas indexées sur le nom du quartier)
enc_qrt_prix_smoothe     = None
enc_qrt_prixm2_smoothe   = None
enc_qrt_prix_median      = None
enc_qrt_prixm2_median    = None
enc_qrt_densite          = None   # densité normalisée : nb annonces / max (→ liquidité du marché)


@app.on_event("startup")
def load_assets():
    """
    Charge tous les artefacts ML au démarrage de l'API.
    Utilise les nouveaux encodeurs enrichis (smoothing + prix/m²).
    Compatible avec l'ancien format (fallback sur encodeur_villes.joblib)
    pour ne pas casser un environnement qui n'a pas encore relancé main.py.
    """
    global modele, scaler, colonnes_entrainement, fallbacks
    global enc_ville_prix_smoothe, enc_ville_prixm2_smoothe
    global enc_ville_prix_median,  enc_ville_prixm2_median
    global enc_qrt_prix_smoothe,   enc_qrt_prixm2_smoothe
    global enc_qrt_prix_median,    enc_qrt_prixm2_median
    global enc_qrt_densite

    print("🧠 Chargement des artefacts ML...")

    # ── Artefacts obligatoires ──────────────────────────────────────────────
    try:
        modele                = joblib.load(MODELS_DIR / "modele_champion.joblib")
        scaler                = joblib.load(MODELS_DIR / "scaler.joblib")
        colonnes_entrainement = joblib.load(MODELS_DIR / "colonnes_entrainement.joblib")
    except Exception as exc:
        raise RuntimeError(f"Artefacts core manquants : {exc}")

    # ── Encodages enrichis (nouveaux) ──────────────────────────────────────
    # On charge d'abord les fallbacks pour savoir quoi mettre si un fichier manque.
    try:
        fallbacks = joblib.load(MODELS_DIR / "encoding_fallbacks.joblib")
    except FileNotFoundError:
        # Ancien pipeline sans encoding_fallbacks → on calcule un fallback minimal
        try:
            _mg = joblib.load(MODELS_DIR / "moyenne_globale.joblib")
        except Exception:
            _mg = 1_500_000.0   # valeur raisonnable pour le marché marocain
        fallbacks = {
            'moyenne_globale'      : _mg,
            'ville_prixm2_smoothe' : 12_000.0,
            'ville_prix_median'    : _mg,
            'ville_prixm2_median'  : 12_000.0,
            'qrt_prix_smoothe'     : _mg,
            'qrt_prixm2_smoothe'   : 12_000.0,
            'qrt_prix_median'      : _mg,
            'qrt_prixm2_median'    : 12_000.0,
            'qrt_densite'          : 0.1,
            'smoothing_factor'     : 20,
        }
        print("  ⚠️  encoding_fallbacks.joblib absent — fallbacks par défaut utilisés.")

    def _load(filename, fallback_key=None):
        """Charge un .joblib ou retourne une Series vide avec fallback."""
        try:
            return joblib.load(MODELS_DIR / filename)
        except FileNotFoundError:
            val = fallbacks.get(fallback_key, 0.0) if fallback_key else 0.0
            print(f"  ⚠️  {filename} absent — fallback constant ({val:,.0f}) utilisé.")
            return pd.Series(dtype=float)   # Series vide → .get() retournera NaN → fillna(fallback)

    enc_ville_prix_smoothe   = _load("enc_ville_prix_smoothe.joblib",   "moyenne_globale")
    enc_ville_prixm2_smoothe = _load("enc_ville_prixm2_smoothe.joblib", "ville_prixm2_smoothe")
    enc_ville_prix_median    = _load("enc_ville_prix_median.joblib",    "ville_prix_median")
    enc_ville_prixm2_median  = _load("enc_ville_prixm2_median.joblib",  "ville_prixm2_median")

    enc_qrt_prix_smoothe     = _load("enc_qrt_prix_smoothe.joblib",    "qrt_prix_smoothe")
    enc_qrt_prixm2_smoothe   = _load("enc_qrt_prixm2_smoothe.joblib",  "qrt_prixm2_smoothe")
    enc_qrt_prix_median      = _load("enc_qrt_prix_median.joblib",     "qrt_prix_median")
    enc_qrt_prixm2_median    = _load("enc_qrt_prixm2_median.joblib",   "qrt_prixm2_median")
    enc_qrt_densite          = _load("enc_qrt_densite.joblib",         "qrt_densite")

    print(f"  ✅ Modèle : {type(modele).__name__}")
    print(f"  ✅ Colonnes : {len(colonnes_entrainement)} features")
    print(f"  ✅ Villes encodées : {len(enc_ville_prix_smoothe)}")
    print(f"  ✅ Quartiers encodés : {len(enc_qrt_prix_smoothe)}")


# ==============================================================================
# FONCTION D'ENCODAGE GÉOGRAPHIQUE (réutilisée par /api/predict)
# ==============================================================================

def encoder_localisation(ville: str, quartier: str) -> dict:
    """
    Applique les 9 encodages géographiques pour une ville + quartier donnés.
    Utilise les fallbacks si la zone est inconnue du modèle.

    Retourne un dict {nom_feature: valeur} prêt à être injecté dans le DataFrame.
    """
    fb = fallbacks   # raccourci

    return {
        # ── Ville (4 features) ───────────────────────────────────────────────
        'Ville_prix_smoothe'  : enc_ville_prix_smoothe.get(ville,   fb.get('moyenne_globale',      1_500_000.0)),
        'Ville_prixm2_smoothe': enc_ville_prixm2_smoothe.get(ville, fb.get('ville_prixm2_smoothe',     12_000.0)),
        'Ville_prix_median'   : enc_ville_prix_median.get(ville,    fb.get('ville_prix_median',    1_500_000.0)),
        'Ville_prixm2_median' : enc_ville_prixm2_median.get(ville,  fb.get('ville_prixm2_median',     12_000.0)),

        # ── Quartier (5 features) ─────────────────────────────────────────────
        'Qrt_prix_smoothe'    : enc_qrt_prix_smoothe.get(quartier,   fb.get('qrt_prix_smoothe',    1_500_000.0)),
        'Qrt_prixm2_smoothe'  : enc_qrt_prixm2_smoothe.get(quartier, fb.get('qrt_prixm2_smoothe',     12_000.0)),
        'Qrt_prix_median'     : enc_qrt_prix_median.get(quartier,    fb.get('qrt_prix_median',     1_500_000.0)),
        'Qrt_prixm2_median'   : enc_qrt_prixm2_median.get(quartier,  fb.get('qrt_prixm2_median',      12_000.0)),
        'Qrt_densite'         : enc_qrt_densite.get(quartier,        fb.get('qrt_densite',                0.1)),
    }


# ==============================================================================
# MODÈLES PYDANTIC
# ==============================================================================

class BienImmobilier(BaseModel):
    ville          : constr(min_length=1)
    quartier       : constr(min_length=1)
    type_bien      : Literal["Appartements", "Maisons", "Villas et Riads"]
    surface        : confloat(ge=15, le=500)
    nb_chambres    : conint(ge=0, le=20)   = 1
    nb_salles_bain : conint(ge=0, le=20)   = 1
    etage          : conint(ge=0, le=120)  = 1
    # Features NLP optionnelles (le frontend peut les envoyer ou non)
    has_piscine    : int = 0
    has_standing   : int = 0
    has_duplex     : int = 0
    has_terrasse   : int = 0
    has_centre     : int = 0

class CompareRequest(BaseModel):
    city_a    : str
    city_b    : str
    type_bien : Optional[str] = None


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.post("/api/predict")
def predire_prix(bien: BienImmobilier):
    """
    Prédit le prix d'un bien immobilier.

    Utilise les 9 encodages géographiques enrichis (smoothing + prix/m²)
    ainsi que les 5 features NLP extraites du titre.

    La transformation log1p/expm1 est appliquée automatiquement
    (le modèle prédit log(1 + Prix), on retourne expm1(prédiction)).
    """
    if modele is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé — relancer l'API")

    # ── Construction du vecteur de features ──────────────────────────────────
    geo = encoder_localisation(bien.ville, bien.quartier)

    row = {
        # Features numériques
        'Surface'         : bien.surface,
        'log_Surface'     : np.log1p(bien.surface),   # cohérent avec l'entraînement
        'Nombre_Chambres' : bien.nb_chambres,
        'Salles_de_bain'  : bien.nb_salles_bain,
        'Etage'           : bien.etage,
        # Features NLP
        'has_piscine'     : bien.has_piscine,
        'has_standing'    : bien.has_standing,
        'has_duplex'      : bien.has_duplex,
        'has_terrasse'    : bien.has_terrasse,
        'has_centre'      : bien.has_centre,
        # Feature catégorielle (sera one-hot encodée)
        'Type'            : bien.type_bien,
        # Encodages géographiques
        **geo
    }

    input_df = pd.DataFrame([row])

    # ── One-Hot Encoding (Type) ───────────────────────────────────────────────
    input_encoded = pd.get_dummies(input_df)

    # ── Alignement sur les colonnes d'entraînement ───────────────────────────
    # Colonnes manquantes → 0  |  Colonnes en trop → supprimées
    input_aligned = input_encoded.reindex(
        columns=colonnes_entrainement, fill_value=0
    ).astype(float)

    # ── Scaling + prédiction ──────────────────────────────────────────────────
    input_scaled = scaler.transform(input_aligned)
    prix_log     = modele.predict(input_scaled)[0]
    prix_estime  = clean_float(max(0.0, np.expm1(prix_log)))

    # ── Intervalle de confiance (±15% correspondant à ~MAE sur données test) ──
    return {
        "estimation"     : prix_estime,
        "range_min"      : prix_estime * 0.85,
        "range_max"      : prix_estime * 1.15,
        "prix_m2_estime" : clean_float(prix_estime / bien.surface),
        # Infos de contexte géographique (utiles pour le frontend)
        "contexte_ville" : {
            "prix_m2_moyen_ville"   : clean_float(geo['Ville_prixm2_smoothe']),
            "prix_m2_median_ville"  : clean_float(geo['Ville_prixm2_median']),
            "prix_m2_moyen_qrt"     : clean_float(geo['Qrt_prixm2_smoothe']),
        }
    }


@app.get("/api/cities")
def get_cities():
    villes = df_stats['Ville'].value_counts().head(25).index.tolist()
    return [{"name": clean_str(v)} for v in sorted(villes)]


@app.get("/api/cities/{city}/quartiers")
def get_quartiers(city: str):
    quartiers = (
        df_stats[df_stats['Ville'] == city]['Quartier']
        .value_counts().head(30).index.tolist()
    )
    return [{"name": clean_str(q)} for q in sorted(quartiers)]


@app.get("/api/heatmap")
def get_heatmap():
    coords = {
        "Casablanca": (33.5731, -7.5898), "Rabat":      (34.0209, -6.8416),
        "Marrakech":  (31.6295, -7.9811), "Tanger":     (35.7595, -5.8340),
        "Agadir":     (30.4278, -9.5981), "Fès":        (34.0331, -5.0003),
        "Meknès":     (33.8935, -5.5473), "Oujda":      (34.6814, -1.9086),
        "Kénitra":    (34.2610, -6.5802), "Tétouan":    (35.5785, -5.3684),
        "El Jadida":  (33.2316, -8.5007), "Essaouira":  (31.5085, -9.7595),
        "Al Hoceima": (35.2442, -3.9317), "Berrechid":  (33.2655, -7.5875),
        "Bouznika":   (33.7894, -7.1597), "Guelmim":    (28.9869, -10.0574),
        "Temara":     (33.9265, -6.9126), "Safi":       (32.2994, -9.2372),
        "Mohammedia": (33.6858, -7.3829),
    }
    result = []
    for v in df_stats['Ville'].value_counts().head(25).index:
        sub = df_stats[df_stats['Ville'] == v]
        if v in coords and len(sub) > 0:
            result.append({
                "name"         : clean_str(v),
                "lat"          : coords[v][0],
                "lng"          : coords[v][1],
                "listing_count": len(sub),
                "avg_prix_m2"  : clean_float(sub['Prix_m2'].mean()),
            })
    return {"cities": result}


@app.get("/api/stats/{city}")
def get_stats(city: str):
    sub = df_stats[df_stats['Ville'] == city]
    if len(sub) == 0:
        return {"error": "Aucune donnée pour cette ville"}
    return {
        "listing_count" : clean_int(len(sub)),
        "avg_prix_m2"   : clean_float(sub['Prix_m2'].mean()),
        "price_range"   : {
            "min"    : clean_float(sub['Prix'].min()),
            "median" : clean_float(sub['Prix'].median()),
            "max"    : clean_float(sub['Prix'].max()),
        },
        "avg_surface"   : clean_float(sub['Surface'].mean()),
        "top_quartiers" : [
            {"name": clean_str(idx), "avg_prix_m2": clean_float(val)}
            for idx, val in sub.groupby('Quartier')['Prix_m2'].mean().nlargest(5).items()
        ],
        "type_distribution": [
            {"type": clean_str(idx), "count": clean_int(val)}
            for idx, val in sub['Type'].value_counts().items()
        ],
    }


@app.post("/api/compare")
def compare_cities(payload: CompareRequest):
    def fetch_data(c):
        sub = df_stats[df_stats['Ville'] == c]
        if payload.type_bien and payload.type_bien.strip():
            sub = sub[sub['Type'].str.strip() == payload.type_bien.strip()]
        if len(sub) == 0:
            return {"city": c, "avg_prix_m2": 0, "avg_prix_total": 0,
                    "avg_surface": 0, "listing_count": 0}
        return {
            "city"           : clean_str(c),
            "avg_prix_m2"    : clean_float(sub['Prix_m2'].mean()),
            "avg_prix_total" : clean_float(sub['Prix'].mean()),
            "avg_surface"    : clean_float(sub['Surface'].mean()),
            "listing_count"  : len(sub),
        }
    return {"city_a": fetch_data(payload.city_a), "city_b": fetch_data(payload.city_b)}


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")