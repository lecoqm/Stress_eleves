import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error as rmse
from scipy.stats import randint, uniform

from src.config import RANDOM_STATE, CV_FOLDS, C_GRID_LASSO

logger = logging.getLogger(__name__)


class StressModels:
    """
    Classe présentant tous les modèles de régression et classification utilisés.
    """
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.results = {}

    def train_logistic_regression(self):
        """
        Entraîne les modèles Multiclasse, OneVsRest et Lasso CV.
        Compare les coefficients et performances.
        """
        logger.info("Entraînement des régressions logistiques...")
        models = {}
        # Multiclasse
        model_multi = LogisticRegression(l1_ratio=None, max_iter=1000, random_state=RANDOM_STATE)
        model_multi.fit(self.x_train, self.y_train)
        models['Multiclasse'] = model_multi
        # OneVsRest (Référence)
        model_ovr = OneVsRestClassifier(LogisticRegression(l1_ratio=None, max_iter=1000, random_state=RANDOM_STATE))
        model_ovr.fit(self.x_train, self.y_train)
        models['OneVsRest'] = model_ovr
        # Lasso avec Validation Croisée (Fine-tuning de C)
        model_lasso_cv = OneVsRestClassifier(
            LogisticRegressionCV(
                Cs=C_GRID_LASSO,
                l1_ratios=[1.0],
                cv=CV_FOLDS,
                solver='liblinear',
                scoring='roc_auc',
                max_iter=2000,
                random_state=RANDOM_STATE)
        )
        model_lasso_cv.fit(self.x_train, self.y_train)
        models['Lasso_CV'] = model_lasso_cv
        # Log des variables sélectionnées par Lasso
        self._log_lasso_selection(model_lasso_cv)

        return models

    def _log_lasso_selection(self, model_lasso):
        """Log les variables mises à zéro par le Lasso """
        df_coef = pd.DataFrame(model_lasso.estimators_[0].coef_, columns=self.x_train.columns)
        vars_non_zero = (df_coef != 0).any(axis=0)
        return df_coef, vars_non_zero

    def get_top_features_from_cart(self, n_top=5):
        """
        Entraîne un CART simple, extrait les importances, et retourne les noms des n_top variables.
        """
        cart = DecisionTreeRegressor(random_state=RANDOM_STATE)
        cart.fit(self.x_train, self.y_train)
        importances = cart.feature_importances_
        feature_names = self.x_train.columns
        df_imp = pd.DataFrame({'feature': feature_names,
                               'importance': importances}).sort_values(by='importance',
                                                                       ascending=False)

        top_features = df_imp['feature'].head(n_top).tolist()
        # A commenter éventuellement après
        print("--- Sélection automatique de variables (CART) ---")
        print(f"Top {n_top} variables : {top_features}")
        print(f"Importances associées : {df_imp.head(n_top)['importance'].values.round(3)}")
        return top_features

    def train_tree_models(self, feature_subset=None, n_top_auto=5):
        '''
        Args:
        feature_subset (list): Liste explicite de variables (ex: ['var1', 'var2']).
        n_top_auto (int): Si feature_subset est None, ce nombre de variables sera sélectionné
        automatiquement via CART.
        Entraîne CART, Random Forest et Gradient Boosting avec Fine-Tuning.
        '''
        results = []
        # Sélection des variables
        if feature_subset:
            selected_vars = feature_subset
        else:
            selected_vars = self.get_top_features_from_cart(n_top_auto)
        x_train_sub = self.x_train[selected_vars]
        x_test_sub = self.x_test[selected_vars]
        suffix = "_Subset" if len(selected_vars) < self.x_train.shape[1] else "_All"
        # Random Forest avec fine-tuning
        param_dist_rf = {
            'n_estimators': randint(50, 150),
            'max_depth': [5, 10, 15, None],
            'min_samples_split': randint(2, 10)
        }
        rf = RandomForestRegressor(random_state=RANDOM_STATE, oob_score=True)
        rf_search = RandomizedSearchCV(rf, param_dist_rf, n_iter=10, cv=3,
                                       scoring='neg_root_mean_squared_error', n_jobs=-1)
        rf_search.fit(x_train_sub, self.y_train)
        best_rf = rf_search.best_estimator_
        predict_train_rf = best_rf.predict(x_train_sub)
        predict_test_rf = best_rf.predict(x_test_sub)
        rmse_test_rf = rmse(self.y_test, predict_test_rf)
        rmse_train_rf = rmse(self.y_train, predict_train_rf)
        results.append({
            'model_name': f'RandomForest_Optimized{suffix}',
            'model': best_rf,
            'rmse_test': rmse_test_rf,
            'rmse_train': rmse_train_rf,
            'params': rf_search.best_params_,
            'variables_used': selected_vars
        })
        # Gradient Boosting avec fine-tuning
        param_dist_gb = {
            'n_estimators': randint(50, 200),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(3, 8)
        }
        gb = GradientBoostingRegressor(random_state=RANDOM_STATE)
        gb_search = RandomizedSearchCV(gb, param_dist_gb, n_iter=10, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
        gb_search.fit(x_train_sub, self.y_train)
        best_gb = gb_search.best_estimator_
        predict_train_gb = best_gb.predict(x_train_sub)
        predict_test_gb = best_gb.predict(x_test_sub)
        rmse_test_gb = rmse(self.y_test, predict_test_gb)
        rmse_train_gb = rmse(self.y_train, predict_train_gb)
        results.append({
            'model_name': f'GradientBoosting_Optimised{suffix}',
            'model': best_gb,
            'rmse_test': rmse_test_gb,
            'rmse_train': rmse_train_gb,
            'params': gb_search.best_params_,
            'variables_used': selected_vars
        })
        return results
