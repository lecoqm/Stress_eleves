import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.config import RANDOM_STATE, TEST_SIZE

logger = logging.getLogger(__name__)

def prepare_datasets(df: pd.DataFrame, target_col: str = 'niveau_stress'):
    """
    Sépare X/y, standardise et fait le train/test split stratifié.
    """

    X = df.drop([target_col], axis=1)
    y = df[target_col]
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # Données train et test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=TEST_SIZE, 
        stratify=y, 
        random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, scaler

def get_subset_data(X_train, X_test, variables: list):
    """
    Extrait un sous-ensemble de variables pour les modèles arbres.
    """
    return X_train[variables], X_test[variables]
