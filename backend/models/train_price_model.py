from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


SEASON_MAP = {
    12: "Hiver",
    1: "Hiver",
    2: "Hiver",
    3: "Printemps",
    4: "Printemps",
    5: "Printemps",
    6: "Ete",
    7: "Ete",
    8: "Ete",
    9: "Automne",
    10: "Automne",
    11: "Automne",
}


def _iqr_filter(df: pd.DataFrame, column: str) -> pd.DataFrame:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return df[
        (df[column] >= lower)
        & (df[column] <= upper)
    ]


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["Surface_m2"] = pd.to_numeric(
        df.get("Surface_m2"),
        errors="coerce",
    )

    df["Prix_DH"] = pd.to_numeric(
        df.get("Prix_DH"),
        errors="coerce",
    )

    # SUPPRESSION DOUBLONS
    df = df.drop_duplicates()

    # FILTRAGE SURFACES
    df = df[
        (df["Surface_m2"] > 15)
        & (df["Surface_m2"] < 1000)
    ]

    # FILTRAGE PRIX
    df = df[
        (df["Prix_DH"] > 50000)
        & (df["Prix_DH"] < 50000000)
    ]

    df = df.dropna(
        subset=[
            "Surface_m2",
            "Prix_DH",
            "Ville",
            "Quartier",
            "Type_Bien",
            "Date_Annonce",
        ]
    )

    # DATE
    df["Date_Annonce"] = pd.to_datetime(
        df["Date_Annonce"],
        errors="coerce",
    )

    df = df.dropna(subset=["Date_Annonce"])

    # SAISON
    df["Saison"] = (
        df["Date_Annonce"]
        .dt.month
        .map(SEASON_MAP)
    )

    # FEATURES TEMPORELLES
    df["annee"] = df["Date_Annonce"].dt.year
    df["mois"] = df["Date_Annonce"].dt.month
    df["trimestre"] = df["Date_Annonce"].dt.quarter

    return df


def _filter_top_cities_and_quartiers(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:

    top_cities = (
        df["Ville"]
        .value_counts()
        .head(15)
        .index
        .tolist()
    )

    df = df[df["Ville"].isin(top_cities)]

    quartier_counts = (
        df.groupby(["Ville", "Quartier"])
        .size()
        .reset_index(name="count")
    )

    allowed = quartier_counts[
        quartier_counts["count"] >= 20
    ]

    quartiers_per_city: Dict[str, List[str]] = (
        allowed.groupby("Ville")["Quartier"]
        .apply(list)
        .to_dict()
    )

    df = df.merge(
        allowed[["Ville", "Quartier"]],
        on=["Ville", "Quartier"],
        how="inner",
    )

    return df, top_cities, quartiers_per_city


def train_price_model(
    input_csv: Path,
    models_dir: Path,
) -> None:

    print("Chargement des données...")

    df = pd.read_csv(input_csv)

    print(f"Dataset initial : {len(df)} lignes")

    df = _prepare_data(df)

    print(f"Après nettoyage : {len(df)} lignes")

    df, top_cities, quartiers_per_city = (
        _filter_top_cities_and_quartiers(df)
    )

    print(f"Après filtre villes/quartiers : {len(df)} lignes")

    # IQR FILTER
    df = _iqr_filter(df, "Prix_DH")
    df = _iqr_filter(df, "Surface_m2")

    print(f"Après IQR : {len(df)} lignes")

    numeric_cols = [
        "Surface_m2",
        "nb_chambres",
        "nb_salles_bain",
        "etage",
        "annee",
        "mois",
        "trimestre",
    ]

    categorical_cols = [
        "Ville",
        "Quartier",
        "Type_Bien",
        "etat_bien",
        "Saison",
    ]

    df = df.dropna(
        subset=numeric_cols + categorical_cols
    )

    print("Encodage des variables catégorielles...")

    cat_df = pd.get_dummies(
        df[categorical_cols],
        prefix=categorical_cols,
    )

    num_df = df[numeric_cols].astype(float)

    scaler = StandardScaler()

    num_scaled = scaler.fit_transform(num_df)

    X = np.hstack([
        num_scaled,
        cat_df.values,
    ])

    feature_columns = (
        numeric_cols
        + cat_df.columns.tolist()
    )

    # ======================================
    # LOG(PRIX)
    # ======================================

    y = np.log1p(
        df["Prix_DH"]
        .astype(float)
        .values
    )

    print("Train/Test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("Entraînement XGBoost...")

    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    xgb_model.fit(
        X_train,
        y_train,
    )

    print("Entraînement RandomForest...")

    rf_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    rf_model.fit(
        X_train,
        y_train,
    )

    print("Évaluation...")

    preds_log = xgb_model.predict(X_test)

    # RETOUR AU VRAI PRIX
    preds_real = np.expm1(preds_log)
    y_test_real = np.expm1(y_test)

    r2 = r2_score(
        y_test_real,
        preds_real,
    )

    mae = mean_absolute_error(
        y_test_real,
        preds_real,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test_real,
            preds_real,
        )
    )

    print("\n========== RÉSULTATS ==========")
    print(f"R²   : {r2:.4f}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print("================================")

    print("Sauvegarde des modèles...")

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        xgb_model,
        models_dir / "xgb_price_model.pkl",
    )

    joblib.dump(
        rf_model,
        models_dir / "rf_price_model.pkl",
    )

    joblib.dump(
        scaler,
        models_dir / "scaler.pkl",
    )

    with open(
        models_dir / "feature_columns.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            feature_columns,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        models_dir / "top15_cities.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            top_cities,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        models_dir / "quartiers_per_city.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            quartiers_per_city,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Modèles sauvegardés avec succès.")


def main() -> None:

    base_dir = Path(__file__).resolve().parents[2]

    input_csv = (
        base_dir / "avito_data_enriched.csv"
    )

    models_dir = (
        base_dir
        / "backend"
        / "models_saved"
    )

    train_price_model(
        input_csv,
        models_dir,
    )


if __name__ == "__main__":
    main()