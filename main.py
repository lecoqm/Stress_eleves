import argparse
import logging

from scripts.ingest import load_latest_data
from src.evaluation import (
    save_all_confusion_matrices,
    save_all_logistic_artifacts,
    save_tree_artifacts,
)
from src.features import prepare_datasets
from src.models import StressModels

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
    handlers=[logging.FileHandler("recording.log"), logging.StreamHandler()],
)

# ENVIRONMENT CONFIGURATION ------------------------------------------------------

parser = argparse.ArgumentParser(description="Variable cible à modéliser")
parser.add_argument(
    "--target_col",
    type=str,
    default="niveau_stress",
    help="Variable cible",
)
args = parser.parse_args()

target_col = args.target_col
logging.info("Variable cible utilisée : %s", target_col)

df = load_latest_data()

x_train, x_test, y_train, y_test, scaler = prepare_datasets(df, target_col)

# MODELS AND OUTPUTS -------------------------------------------------------------

stress_models = StressModels(
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
)

logging.info("Entraînement des modèles logistiques")
logistic_results = stress_models.train_logistic_regression()

logging.info("Sauvegarde des artefacts logistiques")
save_all_logistic_artifacts(
    logistic_results=logistic_results,
    x_test=x_test,
    y_test=y_test,
)

logging.info("Sauvegarde des matrices de confusion")
save_all_confusion_matrices(
    logistic_results=logistic_results,
    x_test=x_test,
    y_test=y_test,
)

logging.info("Entraînement des modèles arbres / boosting")
tree_results = stress_models.train_tree_models()

logging.info("Sauvegarde des artefacts arbres / boosting")
save_tree_artifacts(tree_results)

logging.info("Génération des artefacts terminée.")