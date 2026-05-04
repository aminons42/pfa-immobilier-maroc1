import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import os

# Configuration visuelle
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Mode non-interactif pour éviter les blocages
plt.ioff()  # Mode non-interactif
output_dir = "graphiques"
os.makedirs(output_dir, exist_ok=True)

counter_graphe = 0

def sauvegarder_graphique(titre="graphique"):
    global counter_graphe
    counter_graphe += 1
    nom_fichier = f"{output_dir}/{counter_graphe:03d}_{titre}.png"
    plt.savefig(nom_fichier, dpi=100, bbox_inches='tight')
    print(f"  [GRAPH] Graphique sauvegardé : {nom_fichier}")
    plt.close()

# 1. CHARGEMENT DES DONNÉES
# Charge le fichier de données disponible

# Chercher le fichier CSV
fichiers_csv = [f for f in os.listdir('.') if f.endswith('.csv')]
if not fichiers_csv:
    print("[ERROR] Aucun fichier CSV trouvé dans le répertoire!")
    exit()

nom_fichier = fichiers_csv[0]
print(f" Chargement du fichier : {nom_fichier}")
df = pd.read_csv(nom_fichier)

# ==========================================
# 2. ANALYSE DESCRIPTIVE AVANT NETTOYAGE
# ==========================================

print("\n" + "="*70)
print("[BEFORE] ANALYSE DESCRIPTIVE - DONNÉES BRUTES (AVANT NETTOYAGE)")
print("="*70)

print(f"\n[INFOS] Données brutes :")
print(f"  - Nombre de lignes : {len(df)}")
print(f"  - Nombre de colonnes : {len(df.columns)}")
print(f"  - Colonnes : {list(df.columns)}")

print(f"\n[MISSING] Données manquantes :")
print(df.isnull().sum())

print(f"\n[TYPES] Types de données :")
print(df.dtypes)

# Renommer les colonnes AVANT l'analyse pour cohérence
df.rename(columns={
    'Date_Annonce': 'Date',
    'Type_Bien': 'Type',
    'Prix_DH': 'Prix',
    'Surface_m2': 'Surface'
}, inplace=True)

print(f"\n[STATS] Statistiques des données BRUTES (avant nettoyage) :")
print(df[['Prix', 'Surface']].describe())

print(f"\n[ANOMALIES] Anomalies détectées :")
print(f"  - Prix < 10k DH : {(df['Prix'] < 10000).sum()}")
print(f"  - Prix > 100M DH : {(df['Prix'] > 100000000).sum()}")
print(f"  - Surface < 1 m² : {(df['Surface'] < 1).sum()}")
print(f"  - Surface > 5000 m² : {(df['Surface'] > 5000).sum()}")

# ==========================================
# 3. NETTOYAGE ET PRÉPRÉPARATION (DATA CLEANING)
# ==========================================

print("\n" + "="*70)
print("[CLEANING] DÉBUT DU NETTOYAGE DES DONNÉES")
print("="*70)

# Conversion de la date
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'] >= '2022-01-01']

# Conversion numérique et gestion des nuls
df['Prix'] = pd.to_numeric(df['Prix'], errors='coerce')
df['Surface'] = pd.to_numeric(df['Surface'], errors='coerce')
df = df.dropna(subset=['Prix', 'Surface', 'Ville', 'Quartier', 'Type'])

# IMPORTANT : Supprimer les données aberrantes AVANT les outliers
# Supprimer les prix et surfaces impossibles
df = df[df['Prix'] > 10000]  # Prix minimum : 10k DH
df = df[df['Surface'] > 1]   # Surface minimum : 1 m²
df = df[df['Prix'] < 100000000]  # Prix maximum : 100M DH
df = df[df['Surface'] < 5000]  # Surface maximum : 5000 m²

print(f"[INFO] Après nettoyage initial : {len(df)} lignes")

# Suppression des Outliers par la méthode IQR (Interquartile Range)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df = remove_outliers(df, 'Prix')
df = remove_outliers(df, 'Surface')

# Filtrage des quartiers avec peu de data (Représentativité)
threshold = 20
counts = df['Quartier'].value_counts()
df = df[df['Quartier'].isin(counts[counts >= threshold].index)]

# Création de variables (Feature Engineering)
df['Prix_m2'] = df['Prix'] / df['Surface']
# Remplacer les inf et NaN par la médiane
df['Prix_m2'] = df['Prix_m2'].replace([np.inf, -np.inf], np.nan)
df['Prix_m2'].fillna(df['Prix_m2'].median(), inplace=True)

df['Mois'] = df['Date'].dt.month
df['Annee'] = df['Date'].dt.year
df['Saison'] = df['Date'].dt.month.map({12:'Hiver', 1:'Hiver', 2:'Hiver', 
                                        3:'Printemps', 4:'Printemps', 5:'Printemps',
                                        6:'Ete', 7:'Ete', 8:'Ete', 
                                        9:'Automne', 10:'Automne', 11:'Automne'})

# ==========================================
# 3. ANALYSE DESCRIPTIVE ET COMPARAISON
# ==========================================

# Distribution des Prix par Ville (Boxplot)
plt.figure()
sns.boxplot(x='Ville', y='Prix_m2', data=df)
plt.title('Distribution du Prix au m² par Ville')
plt.xticks(rotation=45)
sauvegarder_graphique('01_distribution_prix_par_ville')

# Test ANOVA (Inférence Statistique)
villes = df['Ville'].unique()
groups = [df[df['Ville'] == v]['Prix_m2'] for v in villes]
f_stat, p_val = stats.f_oneway(*groups)
print(f"--- TEST ANOVA ---")
print(f"F-Statistique: {f_stat:.2f}, P-Value: {p_val:.4f}")
# Si P-Value < 0.05, la différence entre les villes est statistiquement significative.

# Pourcentage des types de biens par ville (Stacked Bar Chart)
type_distribution = pd.crosstab(df['Ville'], df['Type'], normalize='index') * 100
type_distribution.plot(kind='bar', stacked=True)
plt.title('Répartition des Types de Biens par Ville (%)')
plt.ylabel('Pourcentage')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
sauvegarder_graphique('02_repartition_types_par_ville')

# ==========================================
# 4. MODÉLISATION ET ÉVALUATION
# ==========================================

# Encodage des variables catégorielles (One-Hot)
df_model = pd.get_dummies(df[['Surface', 'Ville', 'Type', 'Saison', 'Prix']], drop_first=True)

X = df_model.drop('Prix', axis=1)
y = df_model['Prix']

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STANDARDISATION (Crucial pour la Régression et XGBoost)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Comparaison des Modèles
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = []

print(f"\n--- ÉVALUATION DES MODÈLES ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    results.append({"Modèle": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"{name} -> R2: {r2:.3f} | RMSE: {rmse:.2f} DH")

# ==========================================
# 5. TENDANCES FUTURES (FORECASTING)
# ==========================================

# Évolution de la part de marché par type de bien au fil des années
trend_data = df.groupby(['Annee', 'Type']).size().unstack(fill_value=0)
trend_pct = trend_data.divide(trend_data.sum(axis=1), axis=0) * 100

print(f"\n--- TENDANCES ET PRÉVISIONS (%) ---")
for col in trend_pct.columns:
    # Régression linéaire simple sur les pourcentages pour prédire 2027
    x_trend = np.array(trend_pct.index).reshape(-1, 1)
    y_trend = trend_pct[col].values
    line_model = LinearRegression().fit(x_trend, y_trend)
    future_2027 = line_model.predict([[2027]])[0]
    print(f"Type: {col} | Part 2026: {y_trend[-1]:.1f}% | Projection 2027: {max(0, future_2027):.1f}%")

trend_pct.plot(marker='o')
plt.title('Évolution Historique des Types de Biens')
plt.ylabel('Part de marché (%)')
sauvegarder_graphique('00_tendances_futures')

# ==========================================
# 6. ANALYSE DESCRIPTIVE APRÈS NETTOYAGE
# ==========================================

print("\n" + "="*70)
print("[AFTER] ANALYSE DESCRIPTIVE - DONNÉES NETTOYÉES (APRÈS NETTOYAGE)")
print("="*70)

print("\n--- INFORMATIONS GÉNÉRALES ---")
print(f"Nombre de lignes : {len(df)}")
print(f"Nombre de colonnes : {len(df.columns)}")
print(f"Colonnes : {list(df.columns)}")

print("\n--- STATISTIQUES DESCRIPTIVES ---")
print(df[['Prix', 'Surface', 'Prix_m2']].describe())

print("\n--- TYPES DE DONNÉES ---")
print(df.dtypes)

print("\n--- DONNÉES MANQUANTES ---")
print(df.isnull().sum())

print("\n--- IMPACT DU NETTOYAGE ---")
print(f"  [OK] Données conservées : {len(df)} lignes")
print(f"  [OK] Quartiers avec >=20 annonces : {df['Quartier'].nunique()}")

# ==========================================
# 7. ANALYSE DES QUARTIERS (REPRÉSENTATIVITÉ)
# ==========================================

print("\n" + "="*70)
print("[QUARTIERS] ANALYSE DES QUARTIERS (DONNÉES PAR QUARTIER)")
print("="*70)

quartiers_stats = df.groupby('Quartier').agg({
    'Prix': ['count', 'mean', 'min', 'max', 'std'],
    'Prix_m2': 'mean'
}).round(2)

quartiers_stats.columns = ['Nombre', 'Prix_Moyen', 'Prix_Min', 'Prix_Max', 'Std_Prix', 'Prix_m2_Moyen']
quartiers_stats = quartiers_stats.sort_values('Nombre', ascending=False)

print("\n--- TOP 10 QUARTIERS (Nombre de données) ---")
print(quartiers_stats.head(10))

# Visualiser les quartiers
plt.figure(figsize=(14, 8))
quartiers_stats['Nombre'].head(15).plot(kind='barh', color='steelblue')
plt.title('Top 15 Quartiers (Volume de Données)')
plt.xlabel('Nombre d\'annonces')
plt.gca().invert_yaxis()
plt.tight_layout()
sauvegarder_graphique('03_top_quartiers')

# ==========================================
# 8. GRAPHIQUES CIRCULAIRES (TYPES PAR VILLE)
# ==========================================

print("\n" + "="*70)
print("[TYPES] DISTRIBUTION DES TYPES D'IMMOBILIER PAR VILLE")
print("="*70)

# VILLES PRINCIPALES UNIQUEMENT
villes_principales = ['Casablanca', 'Rabat', 'Tanger', 'Agadir', 'Fes', 'Marrakech']
villes_list = [v for v in villes_principales if v in df['Ville'].values]

print(f"[INFO] Villes analysées (TOP 6) : {', '.join(villes_list)}")

for ville in sorted(villes_list):
    df_ville = df[df['Ville'] == ville]
    type_counts = df_ville['Type'].value_counts()
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(range(len(type_counts)))
    plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title(f'Types d\'Immobilier à {ville}')
    sauvegarder_graphique(f'04_pie_chart_{ville}')

# ==========================================
# 9. ANALYSE DE SAISONNALITÉ (PRIX PAR MOIS)
# ==========================================

print("\n" + "="*70)
print("[SEASONALITY] ANALYSE DE SAISONNALITÉ (ÉVOLUTION PRIX PAR MOIS)")
print("="*70)

saisonnalite = df.groupby(['Annee', 'Mois'])['Prix_m2'].agg(['mean', 'count']).reset_index()

for ville in sorted(villes_list):
    df_ville = df[df['Ville'] == ville]
    saisonnalite_ville = df_ville.groupby('Mois')['Prix_m2'].mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(saisonnalite_ville.index, saisonnalite_ville.values, marker='o', linewidth=2, markersize=8)
    plt.title(f'Saisonnalité : Évolution du Prix au m² par Mois à {ville}')
    plt.xlabel('Mois')
    plt.ylabel('Prix au m² (DH)')
    plt.xticks(range(1, 13), ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
                               'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc'])
    plt.grid(True, alpha=0.3)
    sauvegarder_graphique(f'05_saisonnalite_{ville}')

# ==========================================
# 10. SAISONNALITÉ PAR ANNÉE
# ==========================================

print("\n--- SAISONNALITÉ PAR ANNÉE ---")

for ville in sorted(villes_list):
    df_ville = df[df['Ville'] == ville]
    
    plt.figure(figsize=(12, 6))
    for annee in sorted(df_ville['Annee'].unique()):
        data = df_ville[df_ville['Annee'] == annee].groupby('Mois')['Prix_m2'].mean()
        plt.plot(data.index, data.values, marker='o', label=f'Année {int(annee)}')
    
    plt.title(f'Évolution des Prix au m² par Année à {ville}')
    plt.xlabel('Mois')
    plt.ylabel('Prix au m² (DH)')
    plt.xticks(range(1, 13), ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
                               'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    sauvegarder_graphique(f'06_saisonnalite_par_annee_{ville}')

# ==========================================
# 11. TABLEAUX SÉPARÉS PAR VILLE
# ==========================================

print("\n" + "="*70)
print("[TABLES] TABLEAUX PAR VILLE")
print("="*70)

villes_data = {}

for ville in sorted(villes_list):
    df_ville = df[df['Ville'] == ville].copy()
    villes_data[ville] = df_ville
    
    print(f"\n--- {ville.upper()} ---")
    print(f"Nombre d'annonces : {len(df_ville)}")
    print(f"Prix moyen : {df_ville['Prix'].mean():.2f} DH")
    print(f"Prix au m² moyen : {df_ville['Prix_m2'].mean():.2f} DH/m²")
    print(f"Surface moyenne : {df_ville['Surface'].mean():.2f} m²")
    print(f"\nTypes de biens : {df_ville['Type'].value_counts().to_dict()}")
    print(f"Quartiers principaux : {df_ville['Quartier'].value_counts().head(5).to_dict()}")

# ==========================================
# 12. GROUPEMENT DES TYPES (CLUSTERING)
# ==========================================

print("\n" + "="*70)
print("[CLUSTERING] ANALYSE DE SIMILARITÉS ENTRE TYPES D'IMMOBILIER")
print("="*70)

from sklearn.cluster import KMeans

# Préparation des données pour clustering
type_stats = df.groupby('Type').agg({
    'Prix': 'mean',
    'Surface': 'mean',
    'Prix_m2': 'mean'
}).reset_index()

print("\n--- STATISTIQUES PAR TYPE ---")
print(type_stats)

# Standardisation pour clustering
X_cluster = type_stats[['Prix', 'Surface', 'Prix_m2']].values
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

# KMeans clustering
kmeans = KMeans(n_clusters=min(3, len(type_stats)), random_state=42)
type_stats['Groupe'] = kmeans.fit_predict(X_cluster_scaled)

print("\n--- GROUPEMENT DES TYPES ---")
for groupe in sorted(type_stats['Groupe'].unique()):
    types_groupe = type_stats[type_stats['Groupe'] == groupe]['Type'].tolist()
    print(f"Groupe {groupe + 1}: {', '.join(types_groupe)}")

# ==========================================
# 13. EXPLICATIONS DÉTAILLÉES DES ERREURS
# ==========================================

print("\n" + "="*70)
print("[MODELS] EXPLICATIONS DÉTAILLÉES DES MODÈLES")
print("="*70)

results_df = pd.DataFrame(results).set_index('Modèle')
print("\n--- COMPARAISON COMPLÈTE DES MODÈLES ---")
print(results_df)

print("\n--- EXPLICATIONS DES MÉTRIQUES D'ERREUR ---")
print("""
1. MAE (Mean Absolute Error) - Erreur Absolue Moyenne
   - Moyenne des écarts absolus entre prédictions et valeurs réelles
   - Unité : Same as target (DH)
   - Interprétation : En moyenne, le modèle se trompe de X DH

2. RMSE (Root Mean Squared Error) - Racine Erreur Quadratique Moyenne
   - Racine carrée de la moyenne des carrés des erreurs
   - Unité : Same as target (DH)
   - Punition plus forte des grandes erreurs
   - Préféré quand les grandes erreurs sont mauvaises

3. R² (Coefficient de Détermination)
   - Pourcentage de variance expliquée par le modèle
   - Range : 0 à 1 (ou 0 à 100%)
   - Interprétation : Le modèle explique X% de la variabilité des prix
""")

# Choix du meilleur modèle
meilleur_model = results_df['R2'].idxmax()
meilleur_r2 = results_df.loc[meilleur_model, 'R2']

print(f"\n[BEST] MEILLEUR MODÈLE : {meilleur_model}")
print(f"   Raison : R² le plus élevé ({meilleur_r2:.3f})")
print(f"   Cela signifie que ce modèle explique {meilleur_r2*100:.1f}% de la variabilité des prix")

# ==========================================
# 14. RÉSUMÉ COMPLET POUR LE RAPPORT
# ==========================================

print("\n" + "="*70)
print("[REPORT] RÉSUMÉ EXÉCUTIF - RAPPORT POUR LE PROFESSEUR")
print("="*70)

rapport = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                  RAPPORT D'ANALYSE IMMOBILIÈRE                      ║
║                      Projet PFA - 2026                              ║
╚══════════════════════════════════════════════════════════════════════╝

1. DONNÉES COLLECTÉES
   ├─ Total d'annonces : {len(df):,}
   ├─ Période : {df['Date'].min().date()} à {df['Date'].max().date()}
   ├─ Villes analysées : {len(villes_list)}
   ├─ Nombre de quartiers : {df['Quartier'].nunique()}
   └─ Types de biens : {', '.join(df['Type'].unique())}

2. NETTOYAGE ET PRÉPARATION
   ├─ Méthode outliers : IQR (Interquartile Range)
   ├─ Minimum données par quartier : 20 annonces
   ├─ Données manquantes supprimées : ✅
   └─ Feature engineering : Prix/m², Mois, Année, Saison

3. ANALYSE DESCRIPTIVE
   ├─ Prix moyen : {df['Prix'].mean():.0f} DH
   ├─ Prix au m² moyen : {df['Prix_m2'].mean():.2f} DH/m²
   ├─ Surface moyenne : {df['Surface'].mean():.2f} m²
   └─ Écart-type prix : {df['Prix'].std():.0f} DH

4. QUARTIERS REPRÉSENTATIFS
   ├─ Quartier le plus actif : {quartiers_stats.index[0]}
   ├─ Nombre annonces : {int(quartiers_stats.iloc[0]['Nombre'])}
   └─ Prix moyen : {quartiers_stats.iloc[0]['Prix_Moyen']:.0f} DH

5. ANALYSE DE SAISONNALITÉ
   ├─ Variations mensuelles : ÉTUDIÉES (voir graphiques)
   ├─ Variations annuelles : ÉTUDIÉES (voir graphiques)
   └─ Tendances saisonnières : À CONSULTER DANS GRAPHIQUES

6. GROUPEMENT DES TYPES
   └─ {len(type_stats)} types d'immobilier groupés en {len(type_stats['Groupe'].unique())} catégories

7. MODÉLISATION ET PERFORMANCE
   ├─ Meilleur modèle : {meilleur_model}
   ├─ Score R² : {meilleur_r2:.3f} ({meilleur_r2*100:.1f}% variance expliquée)
   ├─ Erreur moyenne (MAE) : {results_df.loc[meilleur_model, 'MAE']:.2f} DH
   └─ Erreur RMSE : {results_df.loc[meilleur_model, 'RMSE']:.2f} DH

8. COMPARAISON DES VILLES
   ├─ Test ANOVA : P-Value = {p_val:.4f} {'< 0.05 ✅' if p_val < 0.05 else '>= 0.05'}
   └─ Conclusion : {'Différences STATISTIQUEMENT SIGNIFICATIVES' if p_val < 0.05 else 'Pas de différence significative'}

9. STANDARDISATION DES PRIX
   ├─ Méthode : StandardScaler (moyenne 0, écart-type 1)
   ├─ Bénéfices : Améliore convergence des modèles
   └─ Appliquée à : Train/Test split séparément

10. PROCHAINES ÉTAPES
    ├─ Affiner le modèle {meilleur_model}
    ├─ Analyser les résidus pour anomalies
    ├─ Prédictions futures (voir tendances)
    └─ Déploiement possible du meilleur modèle
"""

# Sauvegarder le rapport (sans l'afficher pour éviter les problèmes d'encodage)
try:
    with open('RAPPORT_ANALYSE.txt', 'w', encoding='utf-8') as f:
        f.write(rapport)
        f.write("\n\n=== RESULTATS DETAILLES DES MODELES ===\n")
        f.write(str(results_df))
        f.write("\n\n=== STATISTIQUES PAR QUARTIER ===\n")
        f.write(str(quartiers_stats.head(20)))
    print("[DONE] Rapport sauve dans 'RAPPORT_ANALYSE.txt'")
except Exception as e:
    print(f"[ERROR] Impossible de sauvegarder le rapport: {e}")

# ==========================================
# 15. MATRICE DE CORRÉLATION
# ==========================================

print("\n" + "="*70)
print("[CORRELATION] CORRÉLATIONS ENTRE VARIABLES")
print("="*70)

correlation_cols = ['Prix', 'Surface', 'Prix_m2', 'Mois']
corr_matrix = df[correlation_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matrice de Corrélation')
plt.tight_layout()
sauvegarder_graphique('07_matrice_correlation')