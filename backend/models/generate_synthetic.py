from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _base_rooms(type_bien: str, surface: float) -> int:
    if pd.isna(surface):
        return 0
    if type_bien in {"Appartements"}:
        if surface < 60:
            return 1
        if surface < 100:
            return 2
        if surface < 150:
            return 3
        return 4
    if type_bien in {"Maisons", "Villas et Riads", "Villa-Riad"}:
        if surface < 100:
            return 2
        if surface < 200:
            return 3
        if surface < 350:
            return 4
        return 5
    if type_bien in {"Bureaux", "Local", "Terrains et fermes"}:
        return 0
    return 1


def generate_enriched_csv(input_csv: Path, output_csv: Path, random_state: int = 42) -> Path:
    df = pd.read_csv(input_csv)
    df["Surface_m2"] = pd.to_numeric(df.get("Surface_m2"), errors="coerce")
    df["Type_Bien"] = df.get("Type_Bien").fillna("Autre Immobilier")

    rng = np.random.default_rng(random_state)

    base_rooms = df.apply(lambda row: _base_rooms(row["Type_Bien"], row["Surface_m2"]), axis=1)
    noise = rng.integers(-1, 2, size=len(df))
    nb_chambres = (base_rooms + noise).clip(lower=0)
    df["nb_chambres"] = nb_chambres

    bathrooms_base = np.maximum(1, nb_chambres - 1)
    bathrooms_noise = rng.integers(0, 2, size=len(df))
    df["nb_salles_bain"] = bathrooms_base + bathrooms_noise

    def etat_bien_picker(type_bien: str) -> str:
        if type_bien == "Terrains et fermes":
            return "N/A"
        if type_bien == "Appartements":
            return rng.choice(["Neuf", "Bon état", "À rénover"], p=[0.3, 0.55, 0.15])
        if type_bien in {"Villas et Riads", "Villa-Riad"}:
            return rng.choice(["Neuf", "Bon état", "À rénover"], p=[0.4, 0.5, 0.1])
        if type_bien == "Maisons":
            return rng.choice(["Neuf", "Bon état", "À rénover"], p=[0.2, 0.5, 0.3])
        if type_bien in {"Bureaux", "Local"}:
            return rng.choice(["Neuf", "Bon état", "À rénover"], p=[0.25, 0.6, 0.15])
        return rng.choice(["Neuf", "Bon état", "À rénover"], p=[0.3, 0.55, 0.15])

    df["etat_bien"] = df["Type_Bien"].apply(etat_bien_picker)

    appartement_weights = np.array([0.25, 0.2, 0.15, 0.12, 0.1, 0.07, 0.05, 0.04, 0.02])
    appartement_weights = appartement_weights / appartement_weights.sum()

    def etage_picker(type_bien: str) -> int:
        if type_bien == "Appartements":
            return int(rng.choice(np.arange(0, 9), p=appartement_weights))
        if type_bien in {"Maisons", "Villas et Riads", "Villa-Riad", "Terrains et fermes"}:
            return 0
        if type_bien == "Bureaux":
            return int(rng.integers(0, 16))
        if type_bien == "Local":
            return 0
        return 0

    df["etage"] = df["Type_Bien"].apply(etage_picker)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    input_csv = base_dir / "avito_data_2022_collection.csv"
    output_csv = base_dir / "avito_data_enriched.csv"
    generate_enriched_csv(input_csv, output_csv)


if __name__ == "__main__":
    main()
