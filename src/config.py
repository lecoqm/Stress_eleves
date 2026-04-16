from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = BASE_DIR / "reports" / "tables"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "stress_latest.parquet"

for folder in [TABLES_DIR, FIGURES_DIR, RAW_DATA_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 8
C_GRID_LASSO = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

# Source distante conservée pour l'ingestion
URL_RAW = os.getenv(
    "SOURCE_URL",
    "https://minio.lab.sspcloud.fr/averniere/Stress_eleves/data/stress.parquet",
)

LOG_FORMAT = "{asctime} - {levelname} - {message}"