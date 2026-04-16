import logging
import duckdb
import argparse

from src.config import RAW_DATA_PATH
from src.features import prepare_datasets
from src.models import StressModels
from src.evaluation import plot_roc_curves_comparison, generate_performance_table

con = duckdb.connect(database=":memory:")

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.DEBUG,
    handlers=[logging.FileHandler("recording.log"), logging.StreamHandler()],
)

# ENVIRONMENT CONFIGURATION ------------------------------------------------------

parser = argparse.ArgumentParser(description="Variable d'intérêt")
parser.add_argument("--target_col", type=str, default="niveau_stress", help="Variable d'intérêt")
args = parser.parse_args()

target_col = args.target_col

logging.debug(f"Valeur de l'argument n_trees: {target_col}")


# FEATURES ENGINEERING -----------------------------------------------------------

if not RAW_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {RAW_DATA_PATH}. "
        "Lancez d'abord : python scripts/ingest.py"
    )
df = con.sql(f"SELECT * FROM read_parquet('{RAW_DATA_PATH.as_posix()}')").to_df()

x_train, x_test, y_train, y_test, scaler = prepare_datasets(df, target_col)

# MODELS AND OUTPUTS -------------------------------------------------------------

sm = StressModels(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
log_models = sm.train_logistic_regression()

# Visualisation ROC
plot_roc_curves_comparison(log_models, x_test, y_test)

# Entraînement des Arbres
important_vars = sm.get_top_features_from_cart()
tree_results = sm.train_tree_models()

# Tableau de résultats (Pour le site web)
df_perf = generate_performance_table(tree_results)
