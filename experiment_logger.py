import pandas as pd
import os
from datetime import datetime


class ExperimentLogger:

    def __init__(self, file_path="experiments.xlsx"):

        self.file_path = file_path

        if os.path.exists(file_path):
            self.df = pd.read_excel(file_path)
        else:
            self.df = pd.DataFrame()

    def create_name(self, config):
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{config['model']}_{time}"

    # ==========================
    # MAIN LOG FUNCTION (UPDATED)
    # ==========================
    def log(self, config, metrics, loss, model_path):

        exp_name = self.create_name(config)

        row = {
            "experiment_name": exp_name,

            # dataset
            "dataset": config["dataset_root"],
            "image_size": config["image_size"],

            # model
            "model": config["model"],
            "n_unfreeze": config["n_unfreeze"],

            # training
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "lr": config["lr"],
            "optimizer": config["optimizer"],

            # results (NEW)
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "recall": metrics["recall_sensitivity"],
            "specificity": metrics["specificity"],
            "auc": metrics["auc"],
            "val_loss": loss,

            # output
            "model_path": model_path
        }

        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

        self.save()

        print(f"[LOGGED] {exp_name}")

        return exp_name

    def save(self):
        self.df.to_excel(self.file_path, index=False)
