import random
import numpy as np
import torch
import os
import yaml
import matplotlib.pyplot as plt

def set_seed(seed=42):
    """
    Set random seeds for reproducible experiments.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def load_config(path="configs/config.yaml"):
    """
    Load YAML configuration file.

    Parameters
    ----------
    path : str
        Path to config file.

    Returns
    -------
    dict
        Configuration dictionary.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

import os
import pandas as pd


def load_results(csv_path="results.csv"):
    """
    Load experiment results CSV file.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found!")

    return pd.read_csv(csv_path)


def save_result(path, result):
    """
    Append one experiment result to CSV.

    Parameters
    ----------
    path : str
        CSV file path.

    result : dict
        Dictionary containing experiment metrics.
    """

    file_exists = os.path.isfile(path)

    df = pd.DataFrame([result])

    df.to_csv(
        path,
        mode="a",
        header=not file_exists,
        index=False
    )

def plot_accuracy(csv_path="results.csv"):
    """
    Plot accuracy versus number of unfrozen blocks.
    """

    df = load_results(csv_path)

    plt.figure(figsize=(8, 5))

    for model in df["model"].unique():
        sub = df[df["model"] == model]
        sub = sub.sort_values("n_unfreeze")

        plt.plot(
            sub["n_unfreeze"],
            sub["accuracy"],
            marker="o",
            label=model
        )

    plt.title("Model Accuracy vs Fine-Tuning Depth")
    plt.xlabel("Number of Unfrozen Blocks")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_loss(csv_path="results.csv"):
    """
    Plot validation loss versus number of unfrozen blocks.
    """

    df = load_results(csv_path)

    plt.figure(figsize=(8, 5))

    for model in df["model"].unique():
        sub = df[df["model"] == model]
        sub = sub.sort_values("n_unfreeze")

        plt.plot(
            sub["n_unfreeze"],
            sub["val_loss"],
            marker="o",
            label=model
        )

    plt.title("Validation Loss vs Fine-Tuning Depth")
    plt.xlabel("Number of Unfrozen Blocks")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def best_model(csv_path="results.csv"):
    """
    Return experiment with highest accuracy.
    """

    df = load_results(csv_path)

    best = df.loc[df["accuracy"].idxmax()]

    print("\n========== BEST MODEL ==========\n")
    print(best)

    return best

def print_trainable_layers(model):
    """
    Print trainable layers and parameter count.
    """

    total = 0

    print("\nTrainable parameters:\n")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            total += param.numel()

    print(f"\nTotal trainable parameters: {total:,}")

