import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeRegressor

from src.config import C_GRID_LASSO, CV_FOLDS, RANDOM_STATE


class StressModels:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def train_logistic_regression(self):
        model_multinom = LogisticRegression(
            penalty=None,
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        model_multinom.fit(self.x_train, self.y_train)

        model_ovr = OneVsRestClassifier(
            LogisticRegression(
                penalty=None,
                max_iter=1000,
                random_state=RANDOM_STATE,
            )
        )
        model_ovr.fit(self.x_train, self.y_train)

        model_lasso_cv = OneVsRestClassifier(
            LogisticRegressionCV(
                Cs=C_GRID_LASSO,
                cv=CV_FOLDS,
                penalty="l1",
                solver="saga",
                max_iter=5000,
                random_state=RANDOM_STATE,
            )
        )
        model_lasso_cv.fit(self.x_train, self.y_train)

        return {
            "Multiclasse": model_multinom,
            "OneVsRest": model_ovr,
            "Lasso_CV": model_lasso_cv,
        }

    def get_top_features_from_cart(self, n_top=5):
        cart = DecisionTreeRegressor(random_state=RANDOM_STATE)
        cart.fit(self.x_train, self.y_train)

        df_imp = (
            pd.DataFrame(
                {
                    "feature": self.x_train.columns,
                    "importance": cart.feature_importances_,
                }
            )
            .sort_values(by="importance", ascending=False)
            .reset_index(drop=True)
        )

        top_features = df_imp["feature"].head(n_top).tolist()

        return {
            "feature_importances": df_imp,
            "top_features": top_features,
        }

    def train_tree_models(self):
        cart_results = self.get_top_features_from_cart(n_top=5)
        top_features = cart_results["top_features"]

        x_train_sel = self.x_train[top_features]
        x_test_sel = self.x_test[top_features]

        results = []

        rf = RandomForestRegressor(random_state=RANDOM_STATE)
        param_dist_rf = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
        }

        rf_search = RandomizedSearchCV(
            rf,
            param_distributions=param_dist_rf,
            n_iter=10,
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        rf_search.fit(x_train_sel, self.y_train)

        best_rf = rf_search.best_estimator_
        rf_pred_train = best_rf.predict(x_train_sel)
        rf_pred_test = best_rf.predict(x_test_sel)

        rf_importances = (
            pd.DataFrame(
                {
                    "feature": top_features,
                    "importance": best_rf.feature_importances_,
                }
            )
            .sort_values(by="importance", ascending=False)
            .reset_index(drop=True)
        )

        results.append(
            {
                "model_name": "RandomForest",
                "rmse_train": mean_squared_error(self.y_train, rf_pred_train) ** 0.5,
                "rmse_test": mean_squared_error(self.y_test, rf_pred_test) ** 0.5,
                "params": rf_search.best_params_,
                "variables_used": top_features,
            }
        )

        gb = GradientBoostingRegressor(random_state=RANDOM_STATE)
        param_dist_gb = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [2, 3, 4],
        }

        gb_search = RandomizedSearchCV(
            gb,
            param_distributions=param_dist_gb,
            n_iter=10,
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        gb_search.fit(x_train_sel, self.y_train)

        best_gb = gb_search.best_estimator_
        gb_pred_train = best_gb.predict(x_train_sel)
        gb_pred_test = best_gb.predict(x_test_sel)

        results.append(
            {
                "model_name": "GradientBoosting",
                "rmse_train": mean_squared_error(self.y_train, gb_pred_train) ** 0.5,
                "rmse_test": mean_squared_error(self.y_test, gb_pred_test) ** 0.5,
                "params": gb_search.best_params_,
                "variables_used": top_features,
            }
        )

        return {
            "cart_feature_importances": cart_results["feature_importances"],
            "top_features": top_features,
            "rf_feature_importances": rf_importances,
            "tree_metrics": pd.DataFrame(results),
        }