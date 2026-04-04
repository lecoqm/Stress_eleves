import logging
import duckdb

from src.config import URL_RAW
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

# FEATURES ENGINEERING -----------------------------------------------------------

df = con.sql(f"SELECT * FROM read_parquet('{URL_RAW}')").to_df()

x_train, x_test, y_train, y_test, scaler = prepare_datasets(df)

# MODELS AND OUTPUTS -----------------------------------------------------------

sm = StressModels(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
log_models = sm.train_logistic_regression()

# Visualisation ROC
plot_roc_curves_comparison(log_models, x_test, y_test)

# Entraînement des Arbres
important_vars = sm.get_top_features_from_cart()
tree_results = sm.train_tree_models()

# Tableau de résultats (Pour le site web)
df_perf = generate_performance_table(tree_results)
