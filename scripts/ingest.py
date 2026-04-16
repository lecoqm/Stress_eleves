from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from src.config import URL_RAW
except Exception:
    URL_RAW = "https://minio.lab.sspcloud.fr/averniere/Stress_eleves/data/stress.parquet"

DATA_DIR = ROOT_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PARQUET = DATA_DIR / "stress_latest.parquet"
OUTPUT_METADATA = DATA_DIR / "metadata.json"

REQUIRED_COLUMNS = [
    "niveau_anxiete",
    "estime_de_soi",
    "historique_sante_mentale",
    "depression",
    "maux_de_tete",
    "tension_arterielle",
    "qualite_sommeil",
    "problem_respiratoire",
    "niveau_bruit",
    "conditions_vie",
    "securite",
    "besoins_elementaires",
    "reussite_academique",
    "charge_travail",
    "relation_prof_etudiant",
    "perspective_insertion_professionnelle",
    "soutien_social",
    "pression_des_paires",
    "activites_extrascolaires",
    "harcelement",
    "niveau_stress",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def read_source(source):
    """
    Lit une source locale ou distante.
    Formats gérés :
        - .parquet
        - .csv
    """
    logger.info("Lecture de la source : %s", source)

    if source.endswith(".parquet"):
        con = duckdb.connect()
        return con.sql(f"SELECT * FROM read_parquet('{source}')").to_df()

    if source.endswith(".csv"):
        # Compatible avec votre stress.csv séparé par ';'
        return pd.read_csv(source, sep=";", quotechar='"')

    raise ValueError(
        f"Format non supporté pour la source : {source}. "
        "Utilisez un fichier .parquet ou .csv"
    )


def clean_dataframe(df):
    """
    Nettoyage minimal :
    - trim des noms de colonnes
    - suppression de la colonne d'index parasite éventuelle
    - suppression des doublons
    - conversion numérique si possible
    """
    df = df.copy()

    df.columns = [str(col).strip().replace('"', "") for col in df.columns]

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.drop_duplicates().reset_index(drop=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def validate_schema(df):
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Colonnes manquantes : {missing_columns}")


def run_quality_checks(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Le dataset est vide.")

    if len(df) < 100:
        raise ValueError(
            f"Le dataset semble anormalement petit ({len(df)} lignes)."
        )

    if "niveau_stress" not in df.columns:
        raise ValueError("La colonne cible 'niveau_stress' est absente.")

    na_ratio = df["niveau_stress"].isna().mean()
    if na_ratio > 0.05:
        raise ValueError(
            f"Trop de valeurs manquantes dans 'niveau_stress' ({na_ratio:.1%})."
        )


def save_outputs(df: pd.DataFrame, source: str) -> None:
    con = duckdb.connect()
    con.register("stress_df", df)
    con.execute(
        f"COPY stress_df TO '{OUTPUT_PARQUET.as_posix()}' (FORMAT PARQUET)"
    )

    metadata = {
        "source": source,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "output_file": str(OUTPUT_PARQUET.relative_to(ROOT_DIR)),
    }

    OUTPUT_METADATA.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Fichier écrit : %s", OUTPUT_PARQUET)
    logger.info("Métadonnées écrites : %s", OUTPUT_METADATA)


def main():
    # Ecrase la source par variable d'environnement dans GitHub Actions
    source = os.getenv("SOURCE_URL", URL_RAW)

    df = read_source(source)
    df = clean_dataframe(df)
    validate_schema(df)
    run_quality_checks(df)
    save_outputs(df, source)

    logger.info("Ingestion terminée avec succès.")


if __name__ == "__main__":
    main()