import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error

sys.stdout.reconfigure(encoding='utf-8')

print("🚀 Lancement du Pipeline ML (Le Combat des Titans : RF vs XGBoost vs LightGBM vs CatBoost)...")

# ==========================================
# 1. CHARGEMENT ET NETTOYAGE DE BASE
# ==========================================
df = pd.read_csv('avito_data_augmented.csv')
df.rename(columns={'Type_Bien': 'Type', 'Prix_DH': 'Prix', 'Surface_m2': 'Surface'}, inplace=True)

df['Prix'] = pd.to_numeric(df['Prix'], errors='coerce')
df['Surface'] = pd.to_numeric(df['Surface'], errors='coerce')
df = df.dropna(subset=['Prix', 'Surface', 'Ville', 'Quartier', 'Type'])

df = df[(df['Prix'] > 1000) & (df['Surface'] > 10)]

quartiers_poubelles = ['Toute la ville', 'Autre secteur', 'Periferie']
df = df[~df['Quartier'].isin(quartiers_poubelles)]

# ==========================================
# 2. NETTOYAGE STATISTIQUE INTELLIGENT (Grouped IQR)
# ==========================================
def filtrer_iqr_groupe(dataframe, colonne_a_filtrer, colonne_groupe):
    Q1 = dataframe.groupby(colonne_groupe)[colonne_a_filtrer].transform(lambda x: x.quantile(0.25))
    Q3 = dataframe.groupby(colonne_groupe)[colonne_a_filtrer].transform(lambda x: x.quantile(0.75))
    IQR = Q3 - Q1
    borne_inf = Q1 - 1.5 * IQR
    borne_sup = Q3 + 1.5 * IQR
    return dataframe[(dataframe[colonne_a_filtrer] >= borne_inf) & (dataframe[colonne_a_filtrer] <= borne_sup)]

df = filtrer_iqr_groupe(df, 'Surface', 'Type')
df = filtrer_iqr_groupe(df, 'Prix', 'Type')

print(f"📊 Données conservées après filtrage : {len(df)} annonces.")

# ==========================================
# 3. SÉLECTION DES VARIABLES ET SPLIT
# ==========================================
features = ['Surface', 'Ville', 'Quartier', 'Type']
colonnes_bonus = ['Nombre_Chambres', 'Salles_de_bain', 'Etage', 'Etat']

for col in colonnes_bonus:
    if col in df.columns:
        features.append(col)

X = df[features]
y = df['Prix']

# Split avant l'encodage pour éviter le Data Leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. TARGET ENCODING (Pour la vitesse et la précision)
# ==========================================
train_temp = X_train.copy()
train_temp['Prix'] = y_train

moyennes_villes = train_temp.groupby('Ville')['Prix'].mean()
moyennes_quartiers = train_temp.groupby('Quartier')['Prix'].mean()
moyenne_globale = y_train.mean()

X_train['Ville_Encoded'] = X_train['Ville'].map(moyennes_villes)
X_train['Quartier_Encoded'] = X_train['Quartier'].map(moyennes_quartiers)
X_test['Ville_Encoded'] = X_test['Ville'].map(moyennes_villes).fillna(moyenne_globale)
X_test['Quartier_Encoded'] = X_test['Quartier'].map(moyennes_quartiers).fillna(moyenne_globale)

X_train = X_train.drop(['Ville', 'Quartier'], axis=1)
X_test = X_test.drop(['Ville', 'Quartier'], axis=1)

# ==========================================
# 5. ENCODAGE ONE-HOT ET STANDARDISATION
# ==========================================
X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# ==========================================
# 6. LE COMBAT DES TITANS (Benchmark)
# ==========================================
print("\n⚙️ Entraînement des modèles en cours (Que le meilleur gagne !)...\n")

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=10, random_state=42, n_jobs=-1),
    "XGBoost (Optimisé)": XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.7, colsample_bytree=0.8, random_state=42, n_jobs=-1),
    "LightGBM": LGBMRegressor(random_state=42, n_jobs=-1),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42) # verbose=0 pour éviter d'inonder le terminal
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    medae = median_absolute_error(y_test, preds)
    ratio_erreur = np.median(np.abs((y_test - preds) / y_test)) * 100
    
    print(f"✅ {name:20s}")
    print(f"   R² : {r2:7.3f} | Erreur relative médiane : {ratio_erreur:.1f}%")
    print(f"   MAE  : {mae:,.0f} DH | MedAE : {medae:,.0f} DH\n")

print("🎯 Terminé !")