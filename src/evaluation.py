import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from config import FIGURES_DIR, TABLES_DIR


def evaluate_classifier(model, x_test, y_test, average="weighted"):
    """
    Évalue un modèle de classification sur un jeu de test.

    Parameters
    ----------
    model : estimator sklearn
        Modèle entraîné disposant d'une méthode predict (et optionnellement predict_proba).
    x_test : array-like
        Features du jeu de test.
    y_test : array-like
        Labels réels du jeu de test.
    average : str, optional
        Stratégie de moyennage pour precision et recall (default: "weighted").

    Returns
    -------
    dict
        Dictionnaire contenant y_pred, y_pred_proba, accuracy, precision et recall.
    """
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None

    return {
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
    }


def _coef_dataframe(model):
    """
    Extrait les coefficients d'un modèle logistique sous forme de DataFrame.

    Gère aussi bien les modèles OneVsRest (attribut estimators_) que les
    modèles multiclasses natifs (attribut coef_).

    Parameters
    ----------
    model : estimator sklearn
        Modèle logistique entraîné.

    Returns
    -------
    pd.DataFrame
        DataFrame (classes x features) des coefficients du modèle.
    """
    if hasattr(model, "estimators_"):
        feature_names = model.feature_names_in_
        classes = model.classes_

        data = []
        class_labels = []

        for i, estimator in enumerate(model.estimators_):
            coef = estimator.coef_[0]
            data.append(coef)
            class_labels.append(classes[i])

        return pd.DataFrame(data, columns=feature_names, index=class_labels)

    return pd.DataFrame(
        model.coef_,
        columns=model.feature_names_in_,
        index=model.classes_,
    )


def _save_table(df, filename, index=False):
    output = TABLES_DIR / filename
    df.to_csv(output, index=index)
    print(f"Tableau sauvegardé : {output}")
    return output


def save_logistic_coefficients(logistic_results):
    """
    Sauvegarde les coefficients des modèles logistiques en CSV.

    Génère les fichiers suivants dans TABLES_DIR :
    - logistic_multiclass_coefficients.csv
    - logistic_ovr_coefficients.csv
    - lasso_cv_coefficients.csv
    - lasso_selected_variables.csv  (variables avec au moins un coef non nul)
    - lasso_best_c.csv              (meilleur C par classe pour le Lasso)

    Parameters
    ----------
    logistic_results : dict
        Dictionnaire {"Multiclasse": model, "OneVsRest": model, "Lasso_CV": model}.
    """
    coef_multiclasse = _coef_dataframe(logistic_results["Multiclasse"]).T
    _save_table(coef_multiclasse, "logistic_multiclass_coefficients.csv", index=True)

    coef_ovr = _coef_dataframe(logistic_results["OneVsRest"]).T
    _save_table(coef_ovr, "logistic_ovr_coefficients.csv", index=True)

    coef_lasso = _coef_dataframe(logistic_results["Lasso_CV"]).T
    _save_table(coef_lasso, "lasso_cv_coefficients.csv", index=True)

    selected_mask = (coef_lasso != 0).any(axis=1)
    selected_variables = (
        coef_lasso[selected_mask]
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    _save_table(selected_variables, "lasso_selected_variables.csv", index=False)

    best_cs = []
    for class_label, estimator in zip(
        logistic_results["Lasso_CV"].classes_,
        logistic_results["Lasso_CV"].estimators_,
    ):
        best_c = getattr(estimator, "C_", [None])[0]
        best_cs.append({"classe": class_label, "best_c": best_c})

    best_cs_df = pd.DataFrame(best_cs)
    _save_table(best_cs_df, "lasso_best_c.csv", index=False)


def save_logistic_metrics(logistic_results, x_test, y_test):
    """
    Calcule et sauvegarde les métriques des modèles logistiques en CSV.

    Génère logistic_metrics.csv (tous les modèles) et lasso_metrics.csv
    (Lasso uniquement) dans TABLES_DIR.

    Parameters
    ----------
    logistic_results : dict
        Dictionnaire {"Multiclasse": model, "OneVsRest": model, "Lasso_CV": model}.
    x_test : array-like
        Features du jeu de test.
    y_test : array-like
        Labels réels du jeu de test.

    Returns
    -------
    pd.DataFrame
        DataFrame des métriques (accuracy, precision, recall) par modèle.
    """
    rows = []

    for model_name, model in logistic_results.items():
        metrics = evaluate_classifier(model, x_test, y_test)
        rows.append(
            {
                "modele": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )

    df_metrics = pd.DataFrame(rows)
    _save_table(df_metrics, "logistic_metrics.csv", index=False)

    lasso_metrics = df_metrics[df_metrics["modele"] == "Lasso_CV"].copy()
    lasso_metrics["modele"] = "Lasso"
    _save_table(lasso_metrics, "lasso_metrics.csv", index=False)

    return df_metrics


def plot_confusion_matrix(model, x_test, y_test, filename, title=None):
    """
    Trace et sauvegarde la matrice de confusion d'un modèle.

    Parameters
    ----------
    model : estimator sklearn
        Modèle entraîné.
    x_test : array-like
        Features du jeu de test.
    y_test : array-like
        Labels réels du jeu de test.
    filename : str
        Nom du fichier PNG de destination (dans FIGURES_DIR).
    title : str, optional
        Titre du graphique. Si None, dérivé du nom de fichier.
    """
    y_pred = model.predict(x_test)
    labels = model.classes_ if hasattr(model, "classes_") else None

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title or filename.replace("_", " ").replace(".png", ""))

    output = FIGURES_DIR / filename
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Matrice sauvegardée : {output}")


def save_all_confusion_matrices(logistic_results, x_test, y_test):
    """
    Sauvegarde les matrices de confusion des trois modèles logistiques.

    Génère dans FIGURES_DIR :
    - confusion_logistic_multiclass.png
    - confusion_logistic_ovr.png
    - confusion_lasso_cv.png

    Parameters
    ----------
    logistic_results : dict
        Dictionnaire {"Multiclasse": model, "OneVsRest": model, "Lasso_CV": model}.
    x_test : array-like
        Features du jeu de test.
    y_test : array-like
        Labels réels du jeu de test.
    """
    plot_confusion_matrix(
        logistic_results["Multiclasse"],
        x_test,
        y_test,
        "confusion_logistic_multiclass.png",
        "Matrice de confusion - Régression logistique multiclasse",
    )
    plot_confusion_matrix(
        logistic_results["OneVsRest"],
        x_test,
        y_test,
        "confusion_logistic_ovr.png",
        "Matrice de confusion - Régression logistique OneVsRest",
    )
    plot_confusion_matrix(
        logistic_results["Lasso_CV"],
        x_test,
        y_test,
        "confusion_lasso_cv.png",
        "Matrice de confusion - Régression Lasso avec CV",
    )


def _compute_roc_metrics(y_true_bin, y_pred_proba, n_classes):
    """
    Calcule les métriques ROC par classe et la courbe macro-moyenne.

    Parameters
    ----------
    y_true_bin : np.ndarray
        Labels binarisés (shape: n_samples x n_classes).
    y_pred_proba : np.ndarray
        Probabilités prédites (shape: n_samples x n_classes).
    n_classes : int
        Nombre de classes.

    Returns
    -------
    dict
        Dictionnaire indexé par classe (int) et "macro", chacun contenant
        les métriques "fpr", "tpr" et "auc".
    """
    metrics = {}
    fpr_macro = np.linspace(0, 1, 100)
    tpr_macro_accumulator = np.zeros(100)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        metrics[i] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
        tpr_macro_accumulator += np.interp(fpr_macro, fpr, tpr)

    tpr_macro = tpr_macro_accumulator / n_classes if n_classes > 0 else np.zeros(100)
    auc_macro = auc(fpr_macro, tpr_macro) if n_classes > 0 else 0.0

    metrics["macro"] = {
        "fpr": fpr_macro,
        "tpr": tpr_macro,
        "auc": auc_macro,
    }
    return metrics


def plot_roc_curves_comparison(logistic_results, x_test, y_test):
    """
    Trace les courbes ROC de tous les modèles logistiques dans une figure comparée.

    Sauvegarde roc_logistic_comparison.png dans FIGURES_DIR et
    roc_logistic_summary.csv (AUC par modèle et classe) dans TABLES_DIR.

    Parameters
    ----------
    logistic_results : dict
        Dictionnaire {nom_modele: model} des modèles à comparer.
    x_test : array-like
        Features du jeu de test.
    y_test : array-like
        Labels réels du jeu de test.
    """
    classes = np.unique(y_test)
    n_classes = len(classes)
    y_test_bin = label_binarize(y_test, classes=classes)
    n_models = len(logistic_results)

    cols = min(n_models, 3)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

    if n_models == 1:
        axes = [axes]
    else:
        axes = np.array(axes).reshape(-1)

    roc_summary_rows = []

    for idx, (name, model) in enumerate(logistic_results.items()):
        ax = axes[idx]
        y_pred_proba = model.predict_proba(x_test)
        metrics = _compute_roc_metrics(y_test_bin, y_pred_proba, n_classes)

        for i in range(n_classes):
            ax.plot(
                metrics[i]["fpr"],
                metrics[i]["tpr"],
                linewidth=1,
                alpha=0.6,
                label=f"Classe {i}",
            )
            roc_summary_rows.append(
                {
                    "modele": name,
                    "classe": int(classes[i]),
                    "auc": metrics[i]["auc"],
                }
            )

        ax.plot(
            metrics["macro"]["fpr"],
            metrics["macro"]["tpr"],
            linestyle=":",
            linewidth=2,
            label=f"Macro AUC = {metrics['macro']['auc']:.2f}",
        )
        ax.plot([0, 1], [0, 1], "k:", linewidth=1)
        ax.set_title(f"{name} - Courbes ROC")
        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Taux de vrais positifs")
        ax.legend(loc="lower right", fontsize="small")
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        roc_summary_rows.append(
            {
                "modele": name,
                "classe": "macro",
                "auc": metrics["macro"]["auc"],
            }
        )

    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    output = FIGURES_DIR / "roc_logistic_comparison.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC sauvegardée : {output}")

    roc_summary = pd.DataFrame(roc_summary_rows)
    _save_table(roc_summary, "roc_logistic_summary.csv", index=False)


def save_all_logistic_artifacts(logistic_results, x_test, y_test):
    """
    Orchestre la sauvegarde de tous les artefacts des modèles logistiques.

    Appelle successivement :
    - save_logistic_coefficients
    - save_logistic_metrics
    - plot_roc_curves_comparison

    Parameters
    ----------
    logistic_results : dict
        Dictionnaire {"Multiclasse": model, "OneVsRest": model, "Lasso_CV": model}.
    x_test : array-like
        Features du jeu de test.
    y_test : array-like
        Labels réels du jeu de test.
    """
    save_logistic_coefficients(logistic_results)
    save_logistic_metrics(logistic_results, x_test, y_test)
    plot_roc_curves_comparison(logistic_results, x_test, y_test)


def save_tree_tables(tree_results):
    """
    Sauvegarde les tableaux de résultats des modèles arborescents en CSV.

    Génère dans TABLES_DIR :
    - cart_feature_importance.csv
    - random_forest_feature_importance.csv
    - performance_trees.csv (tous les modèles)
    - performance_random_forest.csv
    - performance_boosting.csv
    - tree_model_comparison.csv (RMSE train/test par modèle)

    Parameters
    ----------
    tree_results : dict
        Dictionnaire contenant les clés "cart_feature_importances",
        "rf_feature_importances" et "tree_metrics".
    """
    cart_feature_importances = tree_results["cart_feature_importances"]
    rf_feature_importances = tree_results["rf_feature_importances"]
    tree_metrics = tree_results["tree_metrics"].copy()

    _save_table(cart_feature_importances, "cart_feature_importance.csv", index=False)
    _save_table(rf_feature_importances, "random_forest_feature_importance.csv", index=False)

    tree_metrics_to_save = tree_metrics.copy()
    tree_metrics_to_save["parametres"] = tree_metrics_to_save["params"].astype(str)
    tree_metrics_to_save["variables_utilisees"] = tree_metrics_to_save["variables_used"].apply(
        lambda x: ", ".join(x)
    )
    tree_metrics_to_save = tree_metrics_to_save.drop(columns=["params", "variables_used"])

    _save_table(tree_metrics_to_save, "performance_trees.csv", index=False)

    rf_df = tree_metrics_to_save[
        tree_metrics_to_save["model_name"].str.contains("RandomForest", case=False, na=False)
    ].copy()
    gb_df = tree_metrics_to_save[
        tree_metrics_to_save["model_name"].str.contains("GradientBoosting", case=False, na=False)
    ].copy()

    _save_table(rf_df, "performance_random_forest.csv", index=False)
    _save_table(gb_df, "performance_boosting.csv", index=False)

    comparison_df = tree_metrics_to_save[
        ["model_name", "rmse_train", "rmse_test"]
    ].rename(columns={"model_name": "modele"})
    _save_table(comparison_df, "tree_model_comparison.csv", index=False)


def plot_random_forest_feature_importance(tree_results):
    """
    Trace et sauvegarde le graphique des 10 variables les plus importantes du Random Forest.

    Sauvegarde random_forest_feature_importance.png dans FIGURES_DIR.

    Parameters
    ----------
    tree_results : dict
        Dictionnaire contenant la clé "rf_feature_importances"
        (DataFrame avec colonnes "feature" et "importance").
    """
    rf_feature_importances = tree_results["rf_feature_importances"].copy().head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(rf_feature_importances["feature"], rf_feature_importances["importance"])
    ax.invert_yaxis()
    ax.set_title("Top variables - Random Forest")
    ax.set_xlabel("Importance")

    output = FIGURES_DIR / "random_forest_feature_importance.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {output}")


def plot_tree_model_comparison(tree_results):
    """
    Trace et sauvegarde un graphique comparant les RMSE test des modèles arborescents.

    Sauvegarde tree_model_comparison.png dans FIGURES_DIR.

    Parameters
    ----------
    tree_results : dict
        Dictionnaire contenant la clé "tree_metrics"
        (DataFrame avec colonnes "model_name" et "rmse_test").
    """
    tree_metrics = tree_results["tree_metrics"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(tree_metrics["model_name"], tree_metrics["rmse_test"])
    ax.set_title("Comparaison des RMSE test")
    ax.set_ylabel("RMSE test")

    output = FIGURES_DIR / "tree_model_comparison.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {output}")


def save_tree_artifacts(tree_results):
    """
    Orchestre la sauvegarde de tous les artefacts des modèles arborescents.

    Appelle successivement :
    - save_tree_tables
    - plot_random_forest_feature_importance
    - plot_tree_model_comparison

    Parameters
    ----------
    tree_results : dict
        Dictionnaire contenant les résultats des modèles arborescents.
    """
    save_tree_tables(tree_results)
    plot_random_forest_feature_importance(tree_results)
    plot_tree_model_comparison(tree_results)