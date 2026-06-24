import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score
)


def compute_metrics(
    y_true,
    y_pred,
    y_prob=None,
    average="macro"
):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ==========================
    # Confusion Matrix
    # ==========================
    cm = confusion_matrix(y_true, y_pred)

    # ==========================
    # Core Metrics
    # ==========================
    acc = accuracy_score(y_true, y_pred)

    prec = precision_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )

    rec = recall_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )

    bal_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    # ==========================
    # AUC (robust version)
    # ==========================
    auc = None

    if y_prob is not None:
        try:
            y_prob = np.array(y_prob)

            # binary classification
            if len(np.unique(y_true)) == 2:

                if y_prob.ndim == 2:
                    y_score = y_prob[:, 1]
                else:
                    y_score = y_prob

                auc = roc_auc_score(
                    y_true,
                    y_score
                )

            # multi-class classification
            else:

                # IMPORTANT: no need for labels param
                auc = roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class="ovr",
                    average="macro"
                )

        except Exception:
            auc = None

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
        "auc": auc,
        "confusion_matrix": cm
    }
