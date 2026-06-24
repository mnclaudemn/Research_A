# experiment_logger.py
import os
from datetime import datetime
import pandas as pd

class ExperimentLogger:
"""
Log experiment configurations and results into an Excel file.
"""

```
def __init__(self, file_path="results/experiments.xlsx"):
    self.file_path = file_path

    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

def create_name(self, config):
    """
    Generate a unique experiment name.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.get("model", "model")
    return f"{model_name}_{timestamp}"

def log(
    self,
    config,
    metrics,
    loss,
    model_path,
    training_time=None,
    trainable_params=None
):
    """
    Save one experiment record.

    Parameters
    ----------
    config : dict
        Experiment configuration.

    metrics : dict
        Evaluation metrics.

    loss : float
        Validation loss.

    model_path : str
        Saved model path.

    training_time : float, optional
        Total training time in seconds.

    trainable_params : int, optional
        Number of trainable parameters.
    """

    exp_name = self.create_name(config)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        # metadata
        "timestamp": timestamp,
        "experiment_name": exp_name,

        # dataset
        "dataset": config.get("dataset_root", ""),
        "image_size": config.get("image_size", 224),

        # model
        "model": config.get("model", ""),
        "n_unfreeze": config.get("n_unfreeze", 0),
        "trainable_params": trainable_params,

        # reproducibility
        "seed": config.get("seed", 42),

        # training
        "batch_size": config.get("batch_size", 16),
        "epochs": config.get("epochs", 10),
        "lr": config.get("lr", 1e-4),
        "optimizer": config.get("optimizer", "adam"),
        "scheduler": config.get("scheduler", "none"),
        "training_time_sec": training_time,

        # evaluation metrics
        "accuracy": metrics.get("accuracy", 0.0),
        "f1_score": metrics.get(
            "f1_score",
            metrics.get("f1", 0.0)
        ),
        "recall": metrics.get(
            "recall_sensitivity",
            metrics.get("recall", 0.0)
        ),
        "specificity": metrics.get("specificity", 0.0),
        "auc": metrics.get("auc", 0.0),
        "val_loss": loss if loss is not None else 0.0,

        # saved model
        "model_path": model_path
    }

    # Load previous experiments
    if os.path.exists(self.file_path):
        try:
            df = pd.read_excel(
                self.file_path,
                engine="openpyxl"
            )
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    # Append new experiment
    df = pd.concat(
        [df, pd.DataFrame([row])],
        ignore_index=True
    )

    # Save
    df.to_excel(
        self.file_path,
        index=False,
        engine="openpyxl"
    )

    print(f"[LOGGED] {exp_name}")

    return exp_name
```
