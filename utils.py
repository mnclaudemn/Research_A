#This is mandatory for publishable research.
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import yaml
import os

def load_config(path="configs/config.yaml"):
    """
    Load YAML configuration file and return it as a dictionary.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config
    
# ==========================
# Load experiment results
# ==========================
def load_results(csv_path="results.csv"):

    if not os.path.exists(csv_path):
        raise ValueError("results.csv not found!")

    return pd.read_csv(csv_path)


# ==========================
# 1. Accuracy comparison
# ==========================
def plot_accuracy(csv_path="results.csv"):

    df = load_results(csv_path)

    plt.figure(figsize=(8, 5))

    for model in df["model"].unique():
        sub = df[df["model"] == model]
        plt.plot(sub["n_unfreeze"], sub["accuracy"], marker="o", label=model)

    plt.title("Model Accuracy vs Fine-Tuning Depth")
    plt.xlabel("Number of Unfrozen Blocks (n_unfreeze)")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()

    plt.show()


# ==========================
# 2. Loss comparison
# ==========================
def plot_loss(csv_path="results.csv"):

    df = load_results(csv_path)

    plt.figure(figsize=(8, 5))

    for model in df["model"].unique():
        sub = df[df["model"] == model]
        plt.plot(sub["n_unfreeze"], sub["val_loss"], marker="o", label=model)

    plt.title("Validation Loss vs Fine-Tuning Depth")
    plt.xlabel("Number of Unfrozen Blocks (n_unfreeze)")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid()

    plt.show()


# ==========================
# 3. Best model summary
# ==========================
def best_model(csv_path="results.csv"):

    df = load_results(csv_path)

    best = df.loc[df["accuracy"].idxmax()]

    print("\n================ BEST MODEL ================\n")
    print(best)


def print_trainable_layers(model):

    print("\nTrainable parameters:\n")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


  def save_result(path, result):

    file_exists = os.path.isfile(path)

    df = pd.DataFrame([result])

    df.to_csv(path, mode="a", header=not file_exists, index=False)
