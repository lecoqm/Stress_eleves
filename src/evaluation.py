import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from src.config import FIGURES_DIR, TABLES_DIR


def evaluate_classifier(model, X_test, y_test, average="weighted"):
    """
    Évalue un modèle de classification (binaire ou multiclasses).
    Retourne un dictionnaire contenant :
    la prédiction de la classe, la prediction P(Y=1|X), l'accuracy, la précision et le recall.
    """

    y_pred = model.predict(X_test)
    y_pred_proba = None
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)

    # Métriques
    accuracy = model.score(X_test, y_test)
    precision = precision_score(y_test, y_pred, average=average)
    recall = recall_score(y_test, y_pred, average=average)

    return {
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }


def plot_confusion_matrix(model, X_test, y_test, model_name, save=True):
    """Génère et sauvegarde la matrice de confusion."""
    y_pred = model.predict(X_test)
    labels = model.classes_ if hasattr(model, 'classes_') else None

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    _, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title(f'Matrice de Confusion - {model_name}')

    if save:
        filepath = FIGURES_DIR / f"confusion_{model_name}.png"
        plt.savefig(filepath, dpi=300)
        print(f"Matrice sauvegardée : {filepath}")

    plt.show()


def _compute_roc_metrics(y_true_bin, y_pred_proba, n_classes):
    """Calcule les courbes ROC par classe et la courbe macro-moyenne."""
    metrics = {}

    fpr_macro = np.linspace(0, 1, 100)
    tpr_macro_accumulator = np.zeros(100)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        metrics[i] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        tpr_macro_accumulator += np.interp(fpr_macro, fpr, tpr)

    # Calcul de la moyenne macro
    if n_classes > 0:
        tpr_macro = tpr_macro_accumulator / n_classes
        auc_macro = auc(fpr_macro, tpr_macro)
    else:
        tpr_macro = np.zeros(100)
        auc_macro = 0.0

    metrics['macro'] = {
        'fpr': fpr_macro,
        'tpr': tpr_macro,
        'auc': auc_macro
    }
    return metrics


def plot_roc_curves_comparison(models_dict, X_test, y_test, save=True):
    """Compare les courbes ROC de plusieurs modèles (OneVsRest requis pour multiclasse)."""
    plt.figure(figsize=(10, 8))

    # Binarisation des labels pour ROC multiclasse
    classes = np.unique(y_test)
    n_classes = len(classes)
    y_test_bin = label_binarize(y_test, classes=classes)
    n_models = len(models_dict)

    if n_models <= 2:
        fig, axes = plt.subplots(1, n_models, figsize=(18, 6))
        if n_models == 1:
            axes = [axes]
    else:
        cols = min(n_models, 3)
        rows = (n_models + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(18, 6))
        axes = axes.flatten()
        for i in range(n_models, len(axes)):
            fig.delaxes(axes[i])

    for idx, (name, model) in enumerate(models_dict.items()):
        ax = axes[idx] if n_models > 1 else axes[0]
        y_pred_proba = model.predict_proba(X_test)
        metrics = _compute_roc_metrics(y_test_bin, y_pred_proba, n_classes)
        for i in range(n_classes):
            fpr = metrics[i]['fpr']
            tpr = metrics[i]['tpr']
            label = f'Classe {i}'
            ax.plot(fpr, tpr, linewidth=1, alpha=0.6, label=label)

        # Tracé Macro
        fpr_macro = metrics['macro']['fpr']
        tpr_macro = metrics['macro']['tpr']
        auc_macro = metrics['macro']['auc']
        ax.plot(fpr_macro, tpr_macro,
                label=f'Moyenne Macro (AUC = {auc_macro:.2f})',
                color='black', linestyle=':', linewidth=2)

        # Ligne diagonale de référence
        ax.plot([0, 1], [0, 1], 'k:', linewidth=1)

        # Labels et titre
        ax.set_title(f'{name} - Courbes ROC')
        ax.set_xlabel('Taux de faux positifs (FPR)')
        ax.set_ylabel('Taux de vrais positifs (TPR)')
        ax.legend(loc='lower right', fontsize='small')
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()

    if save:
        if 'FIGURES_DIR' in globals():
            filepath = FIGURES_DIR / "roc_comparison.png"
            plt.savefig(filepath, dpi=300)
            print(f"ROC sauvegardée : {filepath}")
    plt.show()


def generate_performance_table(tree_results, save=True):
    """Crée un DataFrame propre des performances des modèles arbres/boosting."""
    data = []
    for res in tree_results:
        data.append({
            'Modèle': res['model_name'],
            'RMSE Train': res['rmse_train'],
            'RMSE Test': res['rmse_test'],
            'Meilleurs Paramètres': str(res['params'])
        })
    df_res = pd.DataFrame(data)
    if save:
        filepath = TABLES_DIR / "performance_trees.csv"
        df_res.to_csv(filepath, index=False)
        print(f"Tableau sauvegardé : {filepath}")

    return df_res
