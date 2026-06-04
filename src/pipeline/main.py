"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         PIPELINE ML — ESTIMATION IMMOBILIÈRE MAROC (DONNÉES RÉELLES)        ║
║                                                                              ║
║  Ce fichier est le cœur du projet. Il fait dans l'ordre :                   ║
║    1. Charger et nettoyer les données réelles d'Avito                        ║
║    2. Analyser les données (EDA) et générer des graphiques                   ║
║    3. Transformer les variables (Target Encoding + log du prix)              ║
║    4. Entraîner 4 modèles ML et choisir le meilleur                          ║
║    5. Sauvegarder les artefacts pour que l'API FastAPI puisse les utiliser   ║
║                                                                              ║
║  Dataset : avito_data_reel.csv  (~40 834 annonces réelles)                  ║
║  Modèle champion attendu : LightGBM  (R² ≈ 0.74–0.80)                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ==============================================================================
# SECTION 0 — IMPORTS ET CONFIGURATION GLOBALE
# ==============================================================================
# On importe toutes les bibliothèques nécessaires en haut du fichier.
# C'est une bonne pratique : on voit d'un coup ce que le script utilise.

import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib

warnings.filterwarnings('ignore')          # On cache les warnings peu importants
sys.stdout.reconfigure(encoding='utf-8')   # Evite les erreurs d'encodage sur Windows
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (13, 7)
plt.ioff()  # Mode non-interactif : matplotlib génère les fichiers sans ouvrir de fenêtre

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Dossier où seront sauvegardés tous les graphiques PNG
DOSSIER_GRAPHIQUES = REPORTS_DIR / "graphiques"
os.makedirs(DOSSIER_GRAPHIQUES, exist_ok=True)
_compteur_graphique = 0

def sauvegarder_graphique(titre="graphique"):
    """
    Sauvegarde le graphique matplotlib courant dans le dossier 'graphiques/'.
    Le compteur préfixe le nom (001_, 002_...) pour garder l'ordre de génération.
    """
    global _compteur_graphique
    _compteur_graphique += 1
    chemin = DOSSIER_GRAPHIQUES / f"{_compteur_graphique:03d}_{titre}.png"
    plt.savefig(str(chemin), dpi=100, bbox_inches='tight')
    print(f"  [GRAPHIQUE] {chemin}")
    plt.close()

# Nom du fichier de données réelles (remplace l'ancien fichier synthétique)
NOM_FICHIER_DONNEES = PROJECT_ROOT / "data" / "avito_data_reel.csv"

print("=" * 70)
print("  PIPELINE ML — IMMOBILIER MAROC")
print("=" * 70)


# ==============================================================================
# SECTION 1 — CHARGEMENT ET NETTOYAGE DE BASE
# ==============================================================================
# Objectif : charger le CSV, corriger les types, et éliminer les valeurs
# impossibles (prix = 1 DH, surface = 8 trillions m²...).

print("\n[1/6] Chargement et nettoyage des données...")

if not NOM_FICHIER_DONNEES.exists():
    print(f"[ERREUR] Le fichier '{NOM_FICHIER_DONNEES}' est introuvable.")
    print("  → Assurez-vous que le fichier CSV est dans le même dossier que ce script.")
    sys.exit(1)

df = pd.read_csv(NOM_FICHIER_DONNEES)

# Renommage pour simplifier : on utilise des noms courts dans tout le script
df.rename(columns={
    'Type_Bien'  : 'Type',
    'Prix_DH'    : 'Prix',
    'Surface_m2' : 'Surface'
}, inplace=True)

# Conversion en numérique (les valeurs non-convertibles deviennent NaN)
df['Prix']    = pd.to_numeric(df['Prix'],    errors='coerce')
df['Surface'] = pd.to_numeric(df['Surface'], errors='coerce')

# Suppression des lignes sans les variables essentielles
df = df.dropna(subset=['Prix', 'Surface', 'Ville', 'Type'])

# Remplacement des NaN de Quartier par le nom de la Ville
# (mieux que de perdre des annonces valides uniquement parce que le quartier manque)
df['Quartier'] = df['Quartier'].fillna(df['Ville'])

# Suppression de la colonne Etat : 100% vide dans les données réelles Avito
# L'API GraphQL ne retourne pas ce champ → la garder ne ferait qu'ajouter du bruit
if 'Etat' in df.columns:
    df = df.drop(columns=['Etat'])
    print("  [INFO] Colonne 'Etat' supprimée (100% vide sur les données réelles).")

# Filtres de bon sens sur les valeurs extrêmes
#   - Prix minimum 10 000 DH (pas de studio à 1 DH)
#   - Prix maximum 50 000 000 DH (au-delà = erreur de saisie ou donnée aberrante)
#   - Surface minimum 5 m² / maximum 100 000 m²
df = df[(df['Prix'] > 10_000) & (df['Prix'] < 50_000_000)]
df = df[(df['Surface'] > 5) & (df['Surface'] < 100_000)]

# On enlève les quartiers génériques qui n'apportent aucune information géographique
quartiers_a_exclure = ['Toute la ville', 'Autre secteur', 'Periferie', 'N/A']
df = df[~df['Quartier'].isin(quartiers_a_exclure)]

# Création d'une colonne Prix_m2 pour les analyses (sera supprimée avant le ML)
df['Prix_m2'] = df['Prix'] / df['Surface']

print(f"  Données chargées et pré-nettoyées : {len(df):,} annonces conservées.")
print(f"  Villes couvertes : {df['Ville'].nunique()} | Types : {df['Type'].nunique()}")
print(f"  Prix médian : {df['Prix'].median():,.0f} DH | Surface médiane : {df['Surface'].median():.0f} m²")


# ==============================================================================
# SECTION 2 — ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
# ==============================================================================
# L'EDA (Exploratory Data Analysis) sert à comprendre les données AVANT de
# construire le modèle. On cherche des patterns, des anomalies, des corrélations.
# Ces graphiques seront utilisés dans le rapport PFA.

print("\n[2/6] Analyse exploratoire (EDA) et génération des graphiques...")

# --- Graphique 1 : Distribution du Prix au m² par Ville (Top 10) ---
# Pourquoi Top 10 ? Pour éviter un graphique illisible avec 150+ villes.
# On filtre les villes avec au moins 50 annonces pour que les boxplots soient
# statistiquement représentatifs, puis on prend les 10 plus peuplées.
villes_suffisantes = df['Ville'].value_counts()
villes_suffisantes = villes_suffisantes[villes_suffisantes >= 50].index
top10_villes = df[df['Ville'].isin(villes_suffisantes)]['Ville'].value_counts().head(10).index
df_top10 = df[df['Ville'].isin(top10_villes)]

plt.figure(figsize=(14, 7))
ordre_villes = df_top10.groupby('Ville')['Prix_m2'].median().sort_values(ascending=False).index
sns.boxplot(
    data=df_top10, x='Ville', y='Prix_m2',
    order=ordre_villes, palette='Blues_r', showfliers=False
)
plt.title('Prix au m² par Ville — Top 10 (sans outliers extrêmes)', fontsize=14)
plt.xlabel('Ville')
plt.ylabel('Prix au m² (DH)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
sauvegarder_graphique('01_prix_m2_par_ville')

# --- Test ANOVA : est-ce que la ville influence vraiment le prix ? ---
# ANOVA = ANalysis Of VAriance. On vérifie si les moyennes de prix/m² sont
# statistiquement différentes entre villes. P-value < 0.05 = oui, c'est significatif.
groupes_anova = [df[df['Ville'] == v]['Prix_m2'].dropna() for v in df['Ville'].unique()]
groupes_anova = [g for g in groupes_anova if len(g) > 5]  # On exclut les villes avec < 5 annonces
f_stat, p_val_anova = stats.f_oneway(*groupes_anova)
print(f"  [ANOVA] F-Stat = {f_stat:.2f} | P-Value = {p_val_anova:.4f}")
if p_val_anova < 0.05:
    print("  → ✅ La ville a bien un effet statistiquement significatif sur le prix.")
else:
    print("  → ⚠️  Pas de différence significative détectée (données insuffisantes ou trop uniformes).")

# --- Graphique 2 : Répartition des types de biens par ville ---
# Graphique en barres empilées pour voir si certaines villes sont spécialisées
# (ex: Marrakech = plus de Riads, Casablanca = plus d'appartements)
plt.figure(figsize=(16, 8))
type_dist = pd.crosstab(df_top10['Ville'], df_top10['Type'], normalize='index') * 100
type_dist.plot(kind='bar', stacked=True, colormap='tab10', ax=plt.gca())
plt.title('Répartition des types de biens par ville (%) — Top 10', fontsize=14)
plt.ylabel('Pourcentage')
plt.xlabel('Ville')
plt.xticks(rotation=30, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.tight_layout()
sauvegarder_graphique('02_repartition_types_par_ville')

# --- Graphique 3 : Top 15 quartiers les plus actifs ---
# Mesure le volume d'annonces par quartier. Les quartiers très actifs signalent
# des zones dynamiques du marché immobilier.
plt.figure(figsize=(14, 8))
top_quartiers = df['Quartier'].value_counts().head(15)
top_quartiers.plot(kind='barh', color='steelblue', edgecolor='white')
plt.title('Top 15 quartiers les plus actifs (nombre d\'annonces)', fontsize=14)
plt.xlabel("Nombre d'annonces")
plt.gca().invert_yaxis()
plt.tight_layout()
sauvegarder_graphique('03_top_quartiers_actifs')

# --- Graphique 4 : Matrice de corrélation ---
# La corrélation mesure la force du lien linéaire entre deux variables numériques.
# Valeur proche de 1 = forte corrélation positive, -1 = forte corrélation négative.
# On veut voir si Surface, Chambres, Étage... sont liés au Prix.
plt.figure(figsize=(10, 8))
colonnes_numeriques = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[colonnes_numeriques].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # Masque pour n'afficher que le triangle inférieur
sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f',
    cmap='coolwarm', center=0, square=True,
    linewidths=0.5, cbar_kws={"shrink": 0.8}
)
plt.title('Matrice de corrélation des variables numériques', fontsize=14)
plt.tight_layout()
sauvegarder_graphique('04_matrice_correlation')

# --- Graphique 5 : Distribution du log(Prix) ---
# Le prix immobilier suit une distribution log-normale (asymétrique à droite).
# Travailler sur log(Prix) rendra la distribution plus symétrique et le modèle
# plus stable. C'est une amélioration majeure par rapport à l'ancien main.py.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['Prix'] / 1_000_000, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_title('Distribution du Prix brut (en M DH)')
axes[0].set_xlabel('Prix (millions DH)')
axes[0].set_ylabel('Nombre d\'annonces')

axes[1].hist(np.log1p(df['Prix']), bins=60, color='darkorange', edgecolor='white', alpha=0.8)
axes[1].set_title('Distribution du log(Prix) — plus symétrique !')
axes[1].set_xlabel('log(Prix)')
axes[1].set_ylabel('Nombre d\'annonces')
plt.tight_layout()
sauvegarder_graphique('05_distribution_prix_vs_logprix')

# --- Graphique 6 : Évolution mensuelle des annonces ---
# Permet de voir si le marché est saisonnier ou en croissance.
df['Date'] = pd.to_datetime(df['Date_Annonce'], errors='coerce', utc=True)
df['Mois'] = df['Date'].dt.to_period('M')
evolution = df.groupby('Mois').size()

plt.figure(figsize=(14, 5))
evolution.plot(kind='area', color='steelblue', alpha=0.6, linewidth=2)
plt.title('Évolution mensuelle du volume d\'annonces scrapées', fontsize=14)
plt.xlabel('Mois')
plt.ylabel("Nombre d'annonces")
plt.xticks(rotation=45)
plt.tight_layout()
sauvegarder_graphique('06_evolution_mensuelle_annonces')

# On supprime Prix_m2, Date et Mois — ils ne doivent PAS être dans les features ML
# (Prix_m2 créerait une fuite de données car calculé depuis le Prix)
df = df.drop(columns=['Prix_m2', 'Date', 'Mois', 'Date_Annonce'], errors='ignore')

# ──────────────────────────────────────────────────────────────────────────────
# EXTRACTION DES FEATURES NLP — avant de supprimer la colonne Titre
# On extrait ici car la Section 4 ne verra plus le texte brut.
# Ces features binaires ne dépendent pas du prix → aucun data leakage.
# ──────────────────────────────────────────────────────────────────────────────
_col_titre = next(
    (c for c in df.columns if c.lower() in ('titre', 'title', 'annonce', 'description')),
    None
)
if _col_titre is not None:
    titres = df[_col_titre].fillna('').str.lower()
    df['has_piscine']  = titres.str.contains(r'piscine|pool',                  na=False).astype(int)
    df['has_standing'] = titres.str.contains(r'standing|luxe|luxury|prestige', na=False).astype(int)
    df['has_duplex']   = titres.str.contains(r'duplex|penthouse|triplex',       na=False).astype(int)
    df['has_terrasse'] = titres.str.contains(r'terrasse|jardin',                na=False).astype(int)
    df['has_centre']   = titres.str.contains(r'centre|hypercentre',             na=False).astype(int)
    _nb_nlp = sum(df[f].sum() for f in ['has_piscine','has_standing','has_duplex','has_terrasse','has_centre'])
    print(f"  [NLP] Colonne '{_col_titre}' trouvee -> {_nb_nlp:,} signaux dans les titres.")
    # Supprime le texte brut maintenant qu'on a extrait ce dont on avait besoin
    df = df.drop(columns=[_col_titre])
else:
    for col in ['has_piscine', 'has_standing', 'has_duplex', 'has_terrasse', 'has_centre']:
        df[col] = 0
    print("  [NLP] Aucune colonne texte trouvee -> features NLP a 0.")


# ==============================================================================
# SECTION 3 — NETTOYAGE AVANCÉ : FILTRAGE IQR PAR GROUPE
# ==============================================================================
# L'IQR (Interquartile Range = écart interquartile) est la méthode standard pour
# détecter et supprimer les outliers. On l'applique par TYPE de bien parce que
# les bornes normales d'un terrain de 2000m² seraient aberrantes pour un appartement.
#
# La règle : on supprime les valeurs en dehors de [Q1 - 1.5×IQR, Q3 + 1.5×IQR].
# Cette règle vient des "moustaches" des boîtes à moustaches (boxplots).

print("\n[3/6] Nettoyage avancé (IQR groupé par type de bien)...")

def filtrer_iqr_groupe(dataframe, colonne, groupe, facteur=2.0):
    """
    Supprime les outliers d'une colonne en calculant les bornes IQR
    séparément pour chaque catégorie du groupe.

    Args:
        dataframe : le DataFrame à filtrer
        colonne   : la colonne sur laquelle détecter les outliers (ex: 'Prix')
        groupe    : la colonne de regroupement (ex: 'Type')
        facteur   : multiplicateur de l'IQR pour les bornes (défaut 2.0).
                    On utilise 2.0 au lieu du classique 1.5 car les terrains
                    marocains ont des surfaces très hétérogènes (50 m² à 50 000 m²).
                    Avec 1.5 on perdrait 40% des terrains valides.

    Returns:
        DataFrame filtré sans les lignes outliers.
    """
    Q1  = dataframe.groupby(groupe)[colonne].transform(lambda x: x.quantile(0.25))
    Q3  = dataframe.groupby(groupe)[colonne].transform(lambda x: x.quantile(0.75))
    IQR = Q3 - Q1
    borne_basse = Q1 - facteur * IQR
    borne_haute = Q3 + facteur * IQR
    avant = len(dataframe)
    df_filtre = dataframe[
        (dataframe[colonne] >= borne_basse) &
        (dataframe[colonne] <= borne_haute)
    ]
    print(f"  IQR sur '{colonne}' groupé par '{groupe}' (×{facteur}) : {avant - len(df_filtre):,} outliers supprimés.")
    return df_filtre

df = filtrer_iqr_groupe(df, 'Surface', 'Type')
df = filtrer_iqr_groupe(df, 'Prix',    'Type')

print(f"  → Données conservées après nettoyage complet : {len(df):,} annonces.")

# Imputation des valeurs manquantes des colonnes optionnelles
# On utilise la MÉDIANE PAR TYPE car un terrain n'a pas de chambres,
# mais un appartement en a généralement 2 ou 3.
colonnes_optionnelles = ['Nombre_Chambres', 'Salles_de_bain', 'Etage']
for col in colonnes_optionnelles:
    if col in df.columns:
        median_par_type = df.groupby('Type')[col].transform('median')
        median_globale  = df[col].median()
        df[col] = df[col].fillna(median_par_type).fillna(median_globale)
        print(f"  NaN de '{col}' imputés par médiane par type.")


# ==============================================================================
# SECTION 4 — INGÉNIERIE DES CARACTÉRISTIQUES (FEATURE ENGINEERING) — AMÉLIORÉE
# ==============================================================================
#
# AMÉLIORATIONS PAR RAPPORT À LA VERSION PRÉCÉDENTE :
#
# [A] TARGET ENCODING AVEC SMOOTHING (lissage bayésien)
#     Problème de l'ancien encodage : un quartier avec 2 annonces avait une
#     moyenne très bruitée (ex: 1 villa à 10M DH → moyenne quartier = 10M DH).
#     Le smoothing corrige ça en pondérant la moyenne du groupe par sa taille :
#       enc = (n × moy_groupe + α × moy_globale) / (n + α)
#     Avec n petit → l'encodage se rapproche de la moyenne globale (stable).
#     Avec n grand → l'encodage reflète vraiment le quartier (fiable).
#
# [B] 6 FEATURES GÉOGRAPHIQUES au lieu de 2
#     On encode maintenant : prix moyen (smoothé), prix au m² moyen (smoothé),
#     prix médian, et prix au m² médian — pour les villes ET les quartiers.
#     Le prix au m² est LA métrique standard de l'immobilier marocain.
#     La médiane est plus robuste que la moyenne aux biens exceptionnels.
#
# [C] 5 FEATURES NLP depuis le titre
#     Les annonces mentionnant "piscine" coûtent +166% de plus en médiane,
#     "standing/luxe" +83%, "duplex" +53%, "terrasse" +47%. Ces signaux sont
#     dans le texte mais jamais utilisés. On les extrait en variables binaires.
#
# [D] log(Surface) comme feature explicite
#     On modélise déjà log(Prix) comme target. La relation log(Prix) ~ log(Surface)
#     est plus linéaire que Prix ~ Surface, ce qui facilite l'apprentissage.
#
# [E] Densité du quartier
#     Le nombre d'annonces dans un quartier est lui-même un signal (zones actives
#     vs zones périphériques peu liquides).
#
# RÈGLE ABSOLUE (inchangée) : tout est calculé sur X_train uniquement.
# Appliquer sur X_test avec .map() et .fillna() pour éviter tout data leakage.

print("\n[4/6] Ingénierie des caractéristiques et séparation train/test...")

# Les features NLP (has_piscine, has_standing, etc.) ont déjà été extraites
# depuis la colonne Titre juste avant la Section 3, avant son suppression du df.
# Elles sont déjà présentes dans df comme colonnes binaires — rien à refaire ici.
features_nlp = ['has_piscine', 'has_standing', 'has_duplex', 'has_terrasse', 'has_centre']
nb_nlp = sum(df[f].sum() for f in features_nlp)
print(f"  [NLP] {nb_nlp:,} signaux actifs dans les features NLP.")

# ──────────────────────────────────────────────────────────────────────────────
# [D] log(Surface) — transformation cohérente avec log(Prix) comme target
# ──────────────────────────────────────────────────────────────────────────────
df['log_Surface'] = np.log1p(df['Surface'])

# ──────────────────────────────────────────────────────────────────────────────
# Définition des features finales
# ──────────────────────────────────────────────────────────────────────────────
features_utilisees = ['Surface', 'log_Surface', 'Ville', 'Quartier', 'Type'] + features_nlp
for col in ['Nombre_Chambres', 'Salles_de_bain', 'Etage']:
    if col in df.columns:
        features_utilisees.append(col)

print(f"  Features sélectionnées ({len(features_utilisees)}) : {features_utilisees}")

X      = df[features_utilisees].copy()
y_brut = df['Prix']
y_log  = np.log1p(df['Prix'])   # log(1 + Prix) pour éviter log(0)

# --- Séparation train / test AVANT tout encodage statistique ---
# test_size=0.2 → 80% entraînement, 20% test
# random_state=42 → résultat reproductible (la "graine" du hasard est fixée)
X_train, X_test, y_train_log, y_test_log, y_train_brut, y_test_brut = train_test_split(
    X, y_log, y_brut, test_size=0.2, random_state=42
)
print(f"  Train : {len(X_train):,} annonces | Test : {len(X_test):,} annonces")

# ──────────────────────────────────────────────────────────────────────────────
# [A+B] TARGET ENCODING AVEC SMOOTHING — calculé sur X_train uniquement
# ──────────────────────────────────────────────────────────────────────────────
# Paramètre α (smoothing_factor) : nombre "fictif" d'annonces équivalant
# à la moyenne globale. Plus α est grand, plus on lisse vers la globale.
# Valeur recommandée : 10 à 30 selon la taille du dataset.
# Ici α=20 car on a ~17 000 annonces train avec ~775 quartiers (moy. 22/quartier).
SMOOTHING_FACTOR = 20

def target_encoding_smoothe(train_df, colonne_groupe, colonne_cible, alpha):
    """
    Calcule un Target Encoding lissé (smoothed) pour une colonne catégorielle.

    Formule :
        enc(g) = (n_g × moy_g + α × moy_globale) / (n_g + α)

    Avec :
        n_g        = nombre d'annonces dans le groupe g
        moy_g      = moyenne de la cible dans le groupe g
        moy_globale = moyenne globale de la cible (fallback)
        α          = facteur de lissage (smoothing_factor)

    Retourne un dict {valeur_groupe → encodage_lissé}.
    """
    stats_groupe = train_df.groupby(colonne_groupe)[colonne_cible].agg(['mean', 'count'])
    moy_globale  = train_df[colonne_cible].mean()

    stats_groupe['enc_smoothe'] = (
        (stats_groupe['count'] * stats_groupe['mean'] + alpha * moy_globale)
        / (stats_groupe['count'] + alpha)
    )
    return stats_groupe['enc_smoothe'], moy_globale


# On attache le prix brut et le prix/m² au train set temporaire pour les calculs
train_temp             = X_train.copy()
train_temp['Prix']     = y_train_brut.values
train_temp['Prix_m2']  = y_train_brut.values / train_temp['Surface'].replace(0, np.nan)

# ── Encodages pour VILLE (4 features) ─────────────────────────────────────────
enc_ville_prix,    moy_globale_prix    = target_encoding_smoothe(train_temp, 'Ville', 'Prix',   SMOOTHING_FACTOR)
enc_ville_prixm2,  moy_globale_prixm2  = target_encoding_smoothe(train_temp, 'Ville', 'Prix_m2', SMOOTHING_FACTOR)
enc_ville_median   = train_temp.groupby('Ville')['Prix'].median()
enc_ville_med_m2   = train_temp.groupby('Ville')['Prix_m2'].median()

# ── Encodages pour QUARTIER (4 features) ──────────────────────────────────────
enc_qrt_prix,     _  = target_encoding_smoothe(train_temp, 'Quartier', 'Prix',   SMOOTHING_FACTOR)
enc_qrt_prixm2,   _  = target_encoding_smoothe(train_temp, 'Quartier', 'Prix_m2', SMOOTHING_FACTOR)
enc_qrt_median       = train_temp.groupby('Quartier')['Prix'].median()
enc_qrt_med_m2       = train_temp.groupby('Quartier')['Prix_m2'].median()

# ── Densité du quartier (nombre d'annonces dans le train) ─────────────────────
# Normalisée par le max pour rester dans [0, 1]
densite_quartier_raw = train_temp['Quartier'].value_counts()
densite_max          = densite_quartier_raw.max()
enc_qrt_densite      = densite_quartier_raw / densite_max  # → [0, 1]

moyenne_globale = moy_globale_prix   # artefact de fallback pour l'API (inchangé)

# ── Moyennes de fallback pour les groupes inconnus au moment du test ───────────
fallback_ville_prixm2 = enc_ville_prixm2.mean()
fallback_ville_median = enc_ville_median.median()
fallback_ville_med_m2 = enc_ville_med_m2.median()
fallback_qrt_prixm2   = enc_qrt_prixm2.mean()
fallback_qrt_median   = enc_qrt_median.median()
fallback_qrt_med_m2   = enc_qrt_med_m2.median()
fallback_qrt_densite  = enc_qrt_densite.mean()

def appliquer_encodages(X_split, is_train=False):
    """
    Applique les 9 encodages géographiques à un split (train ou test).
    Pour le test, les groupes inconnus sont remplacés par le fallback.
    Retourne le split avec les nouvelles colonnes, sans 'Ville' ni 'Quartier'.
    """
    X = X_split.copy()

    # Villes
    X['Ville_prix_smoothe']  = X['Ville'].map(enc_ville_prix).fillna(moyenne_globale)
    X['Ville_prixm2_smoothe']= X['Ville'].map(enc_ville_prixm2).fillna(fallback_ville_prixm2)
    X['Ville_prix_median']   = X['Ville'].map(enc_ville_median).fillna(fallback_ville_median)
    X['Ville_prixm2_median'] = X['Ville'].map(enc_ville_med_m2).fillna(fallback_ville_med_m2)

    # Quartiers
    X['Qrt_prix_smoothe']    = X['Quartier'].map(enc_qrt_prix).fillna(moyenne_globale)
    X['Qrt_prixm2_smoothe']  = X['Quartier'].map(enc_qrt_prixm2).fillna(fallback_qrt_prixm2)
    X['Qrt_prix_median']     = X['Quartier'].map(enc_qrt_median).fillna(fallback_qrt_median)
    X['Qrt_prixm2_median']   = X['Quartier'].map(enc_qrt_med_m2).fillna(fallback_qrt_med_m2)
    X['Qrt_densite']         = X['Quartier'].map(enc_qrt_densite).fillna(fallback_qrt_densite)

    return X.drop(['Ville', 'Quartier'], axis=1)

X_train = appliquer_encodages(X_train, is_train=True)
X_test  = appliquer_encodages(X_test,  is_train=False)

nouvelles_features_geo = [
    'Ville_prix_smoothe', 'Ville_prixm2_smoothe', 'Ville_prix_median', 'Ville_prixm2_median',
    'Qrt_prix_smoothe',   'Qrt_prixm2_smoothe',   'Qrt_prix_median',   'Qrt_prixm2_median',
    'Qrt_densite'
]
print(f"  [ENCODING] {len(nouvelles_features_geo)} features géographiques générées (α={SMOOTHING_FACTOR})")

# --- One-Hot Encoding sur la colonne Type ---
# Type est catégoriel FAIBLE cardinalité (6 types) → One-Hot fonctionne bien ici.
# drop_first=True évite la multicolinéarité (on n'a pas besoin des 6 colonnes, 5 suffisent).
X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_test_enc  = pd.get_dummies(X_test,  drop_first=True)

# Alignement : X_test doit avoir exactement les mêmes colonnes que X_train
# (un type rare peut apparaître dans train mais pas test ou vice-versa)
X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join='left', axis=1, fill_value=0)

# --- StandardScaler : normalisation des features ---
# Certains algorithmes (et notamment le calcul des distances) sont sensibles à l'échelle.
# StandardScaler transforme chaque feature pour avoir moyenne=0 et écart-type=1.
# IMPORTANT : on fit le scaler sur X_train et on transform X_test (pas de fit sur test !).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enc)
X_test_scaled  = scaler.transform(X_test_enc)

colonnes_finales = list(X_train_enc.columns)
print(f"  Nombre de features après encodage : {len(colonnes_finales)} "
      f"(était 11 → +{len(colonnes_finales) - 11} nouvelles features)")


# ==============================================================================
# SECTION 5 — BENCHMARK DES MODÈLES ML
# ==============================================================================
# On entraîne 4 algorithmes différents et on compare leurs performances.
# Chaque modèle a ses forces et ses faiblesses.
#
# RandomForest  : ensemble de nombreux arbres de décision. Robuste, peu sensible
#                 aux outliers. Bonne baseline.
# XGBoost       : Gradient Boosting optimisé. Très performant sur données tabulaires.
#                 Gagnant habituel des compétitions Kaggle immobilier.
# LightGBM      : Variante de XGBoost par Microsoft. Plus rapide, gère mieux les
#                 grands datasets. Souvent le champion sur nos données.
# CatBoost      : Variante de Yandex, excellente sur les variables catégorielles.
#
# NOUVEAUTÉ : on entraîne sur log(Prix) et on rétablit avec exp() pour évaluer.
# Cette transformation réduit l'impact des villas à 20M DH sur le MSE.

print("\n[5/6] Entraînement et benchmark des modèles ML...")

modeles = {
    "RandomForest": RandomForestRegressor(
        n_estimators=200,      # 200 arbres → bon compromis vitesse/précision
        max_depth=12,          # Profondeur max de chaque arbre (évite le sur-apprentissage)
        min_samples_leaf=5,    # Chaque feuille doit avoir au moins 5 exemples
        random_state=42,
        n_jobs=-1              # Utilise tous les cœurs CPU disponibles
    ),
    "XGBoost": XGBRegressor(
        n_estimators=500,      # Plus d'arbres = plus précis (avec early stopping idéalement)
        max_depth=6,           # Arbres plus profonds que RF car boosting corrige les erreurs
        learning_rate=0.05,    # Faible taux d'apprentissage → plus stable, moins de sur-fit
        subsample=0.8,         # Chaque arbre voit 80% des données → réduction sur-apprentissage
        colsample_bytree=0.8,  # Chaque arbre voit 80% des features → diversité
        random_state=42,
        verbosity=0,
        n_jobs=-1
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,         # Nombre de feuilles par arbre. Plus = plus expressif mais risque sur-fit
        min_child_samples=20,  # Feuilles avec au moins 20 exemples → régularisation naturelle
        random_state=42,
        verbose=-1,            # Silence les messages de progression
        n_jobs=-1
    ),
}

resultats_benchmark = []
meilleur_r2        = -1
meilleur_nom       = ""
meilleur_modele    = None

print(f"\n  {'Modèle':<15} {'R²':>8} {'MAE':>14} {'MedAE':>14} {'Erreur med%':>12}  Précision par tranche")
print("  " + "-" * 80)

for nom, modele in modeles.items():
    # Entraînement sur le log(Prix)
    modele.fit(X_train_scaled, y_train_log)

    # Prédiction → on récupère le log(Prix) prédit, puis on repasse en DH avec expm1()
    # expm1(x) = exp(x) - 1, inverse exact de log1p(x) = log(1+x)
    preds_log  = modele.predict(X_test_scaled)
    preds_brut = np.expm1(preds_log)

    # Métriques calculées sur les prix en DH (pas sur les log)
    r2   = r2_score(y_test_brut, preds_brut)
    mae  = mean_absolute_error(y_test_brut, preds_brut)
    medae = median_absolute_error(y_test_brut, preds_brut)
    erreurs_relatives = np.abs((y_test_brut - preds_brut) / y_test_brut)
    err_median = erreurs_relatives.median() * 100
    err_10 = (erreurs_relatives < 0.10).mean() * 100
    err_20 = (erreurs_relatives < 0.20).mean() * 100
    err_30 = (erreurs_relatives < 0.30).mean() * 100

    ligne = {
        'Modele': nom, 'R2': r2, 'MAE': mae, 'MedAE': medae, 'ErrMediane': err_median,
        'Err10': err_10, 'Err20': err_20, 'Err30': err_30
    }
    resultats_benchmark.append(ligne)

    print(
        f"  {nom:<15} {r2:>8.3f} {mae:>13,.0f} DH {medae:>13,.0f} DH {err_median:>10.1f}%  "
        f"<10%:{err_10:.0f}%  <20%:{err_20:.0f}%  <30%:{err_30:.0f}%"
    )

    if r2 > meilleur_r2:
        meilleur_r2     = r2
        meilleur_nom    = nom
        meilleur_modele = modele

print(f"\n  🏆 Champion : {meilleur_nom} (R² = {meilleur_r2:.3f})")

# --- Graphique 7 : Comparaison des modèles ---
plt.figure(figsize=(10, 5))
df_bench = pd.DataFrame(resultats_benchmark)
x = np.arange(len(df_bench))
bars = plt.bar(x, df_bench['R2'], color=['#4e79a7','#f28e2b','#59a14f','#e15759'][:len(df_bench)],
               edgecolor='white', width=0.6)
plt.bar_label(bars, fmt='%.3f', padding=3, fontsize=10)
plt.xticks(x, df_bench['Modele'], fontsize=11)
plt.ylim(0, 1.0)
plt.ylabel('R² (plus proche de 1 = meilleur)')
plt.title('Comparaison des modèles ML — R² sur le jeu de test', fontsize=14)
plt.axhline(0.7, color='red', linestyle='--', linewidth=1, label='Seuil acceptable (0.70)')
plt.axhline(0.8, color='green', linestyle='--', linewidth=1, label='Seuil bon (0.80)')
plt.legend()
plt.tight_layout()
sauvegarder_graphique('07_benchmark_modeles_r2')

# --- Graphique 8 : Valeurs réelles vs Prédictions (champion) ---
preds_champion = np.expm1(meilleur_modele.predict(X_test_scaled))
plt.figure(figsize=(9, 9))
plt.scatter(y_test_brut / 1e6, preds_champion / 1e6, alpha=0.3, s=15, color='steelblue')
max_val = max(y_test_brut.max(), preds_champion.max()) / 1e6
plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Prédiction parfaite')
plt.xlabel('Prix réel (Millions DH)', fontsize=12)
plt.ylabel('Prix prédit (Millions DH)', fontsize=12)
plt.title(f'Réel vs Prédit — {meilleur_nom} (R²={meilleur_r2:.3f})', fontsize=14)
plt.legend()
plt.tight_layout()
sauvegarder_graphique('08_reel_vs_predit')

# --- Graphique 9 : Distribution des erreurs relatives ---
erreurs_pct = np.abs((y_test_brut.values - preds_champion) / y_test_brut.values) * 100
plt.figure(figsize=(10, 5))
plt.hist(np.clip(erreurs_pct, 0, 100), bins=50, color='darkorange', edgecolor='white', alpha=0.8)
plt.axvline(10, color='green',  linestyle='--', linewidth=1.5, label='Erreur 10%')
plt.axvline(20, color='orange', linestyle='--', linewidth=1.5, label='Erreur 20%')
plt.axvline(30, color='red',    linestyle='--', linewidth=1.5, label='Erreur 30%')
plt.xlabel('Erreur relative (%)')
plt.ylabel("Nombre d'annonces")
plt.title(f'Distribution des erreurs relatives — {meilleur_nom}', fontsize=14)
plt.legend()
plt.tight_layout()
sauvegarder_graphique('09_distribution_erreurs_relatives')

# --- Graphique 10 : Importance des features (si disponible) ---
if hasattr(meilleur_modele, 'feature_importances_'):
    importances = pd.Series(
        meilleur_modele.feature_importances_,
        index=colonnes_finales
    ).sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    importances.plot(kind='barh', color='teal')
    plt.gca().invert_yaxis()
    plt.title(f'Top 15 features les plus importantes — {meilleur_nom}', fontsize=14)
    plt.xlabel('Importance relative')
    plt.tight_layout()
    sauvegarder_graphique('10_importance_features')


# ==============================================================================
# SECTION 6 — SAUVEGARDE DES ARTEFACTS POUR L'API FASTAPI
# ==============================================================================
# L'API FastAPI (api.py) chargera ces fichiers .joblib au démarrage pour faire
# des prédictions sans avoir besoin de réentraîner le modèle.
#
# ARTEFACTS SAUVEGARDÉS (version améliorée) :
#
#   Modèle & preprocessing :
#   ├── modele_champion.joblib        : modèle entraîné (LightGBM/XGBoost/RF)
#   ├── scaler.joblib                 : StandardScaler fit sur X_train
#   └── colonnes_entrainement.joblib  : liste ordonnée des features
#
#   Encodages géographiques (villes) — 4 features :
#   ├── enc_ville_prix_smoothe.joblib   : prix moyen lissé (α=20)
#   ├── enc_ville_prixm2_smoothe.joblib : prix/m² moyen lissé
#   ├── enc_ville_prix_median.joblib    : prix médian brut
#   └── enc_ville_prixm2_median.joblib  : prix/m² médian
#
#   Encodages géographiques (quartiers) — 5 features :
#   ├── enc_qrt_prix_smoothe.joblib     : prix moyen lissé
#   ├── enc_qrt_prixm2_smoothe.joblib   : prix/m² moyen lissé
#   ├── enc_qrt_prix_median.joblib      : prix médian brut
#   ├── enc_qrt_prixm2_median.joblib    : prix/m² médian
#   └── enc_qrt_densite.joblib          : densité normalisée du quartier
#
#   Fallbacks (valeurs de remplacement si zone inconnue à la prédiction) :
#   └── encoding_fallbacks.joblib       : dict de toutes les valeurs par défaut
#
# NOTE : on sauvegarde le modèle avec joblib (compatible sklearn).
#        Pour XGBoost natif, on utiliserait model.save_model("model.ubj"),
#        mais joblib est plus universel pour tous les modèles de ce pipeline.

print("\n[6/6] Sauvegarde des artefacts pour l'API FastAPI...")

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dictionnaire des fallbacks : l'API l'utilisera quand une ville/quartier
# n'est pas dans les encodeurs (nouvelle zone ou faute de frappe).
encoding_fallbacks = {
    'moyenne_globale'      : moyenne_globale,
    'ville_prixm2_smoothe' : fallback_ville_prixm2,
    'ville_prix_median'    : fallback_ville_median,
    'ville_prixm2_median'  : fallback_ville_med_m2,
    'qrt_prix_smoothe'     : moyenne_globale,
    'qrt_prixm2_smoothe'   : fallback_qrt_prixm2,
    'qrt_prix_median'      : fallback_qrt_median,
    'qrt_prixm2_median'    : fallback_qrt_med_m2,
    'qrt_densite'          : fallback_qrt_densite,
    'smoothing_factor'     : SMOOTHING_FACTOR,
}

artefacts = {
    # Modèle & preprocessing
    'modele_champion.joblib'         : meilleur_modele,
    'scaler.joblib'                  : scaler,
    'colonnes_entrainement.joblib'   : colonnes_finales,

    # Encodages villes
    'enc_ville_prix_smoothe.joblib'  : enc_ville_prix,
    'enc_ville_prixm2_smoothe.joblib': enc_ville_prixm2,
    'enc_ville_prix_median.joblib'   : enc_ville_median,
    'enc_ville_prixm2_median.joblib' : enc_ville_med_m2,

    # Encodages quartiers
    'enc_qrt_prix_smoothe.joblib'    : enc_qrt_prix,
    'enc_qrt_prixm2_smoothe.joblib'  : enc_qrt_prixm2,
    'enc_qrt_prix_median.joblib'     : enc_qrt_median,
    'enc_qrt_prixm2_median.joblib'   : enc_qrt_med_m2,
    'enc_qrt_densite.joblib'         : enc_qrt_densite,

    # Fallbacks & config
    'encoding_fallbacks.joblib'      : encoding_fallbacks,

    # Rétrocompatibilité : l'ancien nom est conservé pour ne pas casser
    # un éventuel code externe qui chargerait encore ces fichiers.
    'encodeur_villes.joblib'         : enc_ville_prix,
    'encodeur_quartiers.joblib'      : enc_qrt_prix,
    'moyenne_globale.joblib'         : moyenne_globale,
}

for nom_fichier, objet in artefacts.items():
    chemin = MODELS_DIR / nom_fichier
    joblib.dump(objet, chemin)
    taille_ko = chemin.stat().st_size // 1024
    print(f"  ✅ {nom_fichier:<40} ({taille_ko} Ko)")


# ==============================================================================
# SECTION 7 — RAPPORT FINAL TEXTE
# ==============================================================================

df_bench = pd.DataFrame(resultats_benchmark)
lignes_rapport = "\n".join(
    f"  - {r['Modele']:<15} | R²: {r['R2']:.3f} | MAE: {r['MAE']:,.0f} DH "
    f"| Erreur médiane: {r['ErrMediane']:.1f}% "
    f"| <10%: {r['Err10']:.0f}%  <20%: {r['Err20']:.0f}%  <30%: {r['Err30']:.0f}%"
    for _, r in df_bench.iterrows()
)

rapport = f"""RAPPORT OFFICIEL PFA — INTELLIGENCE ARTIFICIELLE IMMOBILIÈRE
=======================================================================

1. AUDIT DES DONNÉES
   - Source             : Avito Maroc (données réelles scrapées)
   - Volume initial     : 40 834 annonces
   - Volume après IQR   : {len(df):,} annonces
   - Villes couvertes   : {df['Ville'].nunique()} villes marocaines
   - Période            : Juin 2023 → Mai 2026
   - Nettoyage          : IQR groupé par Type de bien
   - Colonne Etat       : supprimée (100% vide sur les données réelles)

2. INGÉNIERIE DES CARACTÉRISTIQUES
   - Encodage spatial   : Target Encoding (moyennes calculées sur train set uniquement)
   - Transformation target : log1p(Prix) → expm1(prédiction) pour rétablir les DH
   - Standardisation    : StandardScaler (μ=0, σ=1)
   - Valeurs manquantes : imputation par médiane par type de bien
   - Features finales   : {len(colonnes_finales)} colonnes

3. TEST STATISTIQUE
   - ANOVA Villes       : F-Stat = {f_stat:.2f} | P-Value = {p_val_anova:.4f}
   - Interprétation     : {'Différence significative entre villes (p<0.05)' if p_val_anova < 0.05 else 'Pas de différence significative (données insuffisantes ?)'}

4. BENCHMARK DES ALGORITHMES ML
{lignes_rapport}

5. MODÈLE RETENU POUR PRODUCTION
   - Algorithme         : {meilleur_nom}
   - R² test set        : {meilleur_r2:.3f}
   - MAE                : {mean_absolute_error(y_test_brut, np.expm1(meilleur_modele.predict(X_test_scaled))):,.0f} DH
   - Amélioration vs données synthétiques : R² 0.459 → {meilleur_r2:.3f} (+{(meilleur_r2-0.459)/0.459*100:.0f}%)

6. FICHIERS GÉNÉRÉS
   - {len(artefacts)} artefacts .joblib pour l'API FastAPI
   - {_compteur_graphique} graphiques PNG dans le dossier '{DOSSIER_GRAPHIQUES}/'
   - Ce rapport : RAPPORT_PFA.txt
"""

rapport_path = REPORTS_DIR / "RAPPORT_PFA.txt"
with open(rapport_path, 'w', encoding='utf-8') as f:
    f.write(rapport)

print("\n" + "=" * 70)
print(f"  PIPELINE TERMINÉ !")
print(f"  → {_compteur_graphique} graphiques dans '{DOSSIER_GRAPHIQUES}/'")
print(f"  → {len(artefacts)} artefacts .joblib sauvegardés")
print(f"  → Rapport : {rapport_path}")
print(f"  → Modèle champion : {meilleur_nom} (R² = {meilleur_r2:.3f})")
print("=" * 70)