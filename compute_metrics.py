# utils/compute_metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
)


def compute_metrics(
    y_true,
    y_pred,
    y_prob=None,
    average="macro",
):
    """
    Compute classification metrics for binary and multiclass problems.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.

    y_pred : array-like
        Predicted labels.

    y_prob : array-like, optional
        Prediction probabilities.
        Binary:
            shape = (N,) or (N, 2)
        Multiclass:
            shape = (N, C)

    average : str
        Averaging method for multiclass metrics.
        Options:
            "macro"
            "weighted"
            "micro"

    Returns
    -------
    dict
        Dictionary containing all metrics.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    # ======================================================
    # Confusion Matrix
    # ======================================================
    cm = confusion_matrix(y_true, y_pred)

    # ======================================================
    # Core Metrics
    # ======================================================
    acc = accuracy_score(y_true, y_pred)

    prec = precision_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )

    rec = recall_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )

    bal_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    kappa = cohen_kappa_score(
        y_true,
        y_pred
    )

    # ======================================================
    # Sensitivity and Specificity
    # ======================================================
    n_classes = len(np.unique(y_true))

    if n_classes == 2:

        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn + 1e-12)
        specificity = tn / (tn + fp + 1e-12)

    else:
        sensitivities = []
        specificities = []

        for i in range(cm.shape[0]):

            tp = cm[i, i]

            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)

            sens = tp / (tp + fn + 1e-12)
            spec = tn / (tn + fp + 1e-12)

            sensitivities.append(sens)
            specificities.append(spec)

        sensitivity = np.mean(sensitivities)
        specificity = np.mean(specificities)

    # ======================================================
    # AUC
    # ======================================================
    auc = np.nan

    if y_prob is not None:

        try:
            y_prob = np.asarray(y_prob)

            # Binary classification
            if n_classes == 2:

                if y_prob.ndim == 2:
                    y_score = y_prob[:, 1]
                else:
                    y_score = y_prob

                auc = roc_auc_score(
                    y_true,
                    y_score
                )

            # Multiclass classification
            else:

                auc = roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class="ovr",
                    average="macro",
                )

        except Exception:
            auc = np.nan

    # ======================================================
    # Return Metrics
    # ======================================================
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall_sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "balanced_accuracy": float(bal_acc),
        "auc": float(auc) if not np.isnan(auc) else np.nan,
        "mcc": float(mcc),
        "kappa": float(kappa),
        "confusion_matrix": cm,
    }

    return metrics
