import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


class callbacks:

    def __init__(
        self,
        monitor="val_f1",
        mode="max",
        patience=4,
        min_delta=1e-3,
        best_model_path="best_model.pth",
        last_checkpoint_path="last_checkpoint.pth",
        log_path="training_log.csv",
        scheduler=None
    ):

        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self.best_model_path = best_model_path
        self.last_checkpoint_path = last_checkpoint_path
        self.log_path = log_path

        self.scheduler = scheduler

        self.best_metric = (
            -float("inf")
            if mode == "max"
            else float("inf")
        )

        self.best_epoch = -1
        self.counter = 0
        self.stop_training = False
        self.start_time = None

        self.history = []

    # ==========================
    # Properties
    # ==========================
    @property
    def best_score(self):
        return self.best_metric

    # ==========================
    # Timer
    # ==========================
    def on_epoch_begin(self):
        self.start_time = time.time()

    # ==========================
    # Epoch End
    # ==========================
    def on_epoch_end(
        self,
        epoch,
        model,
        optimizer,
        train_loss,
        val_loss,
        train_metrics,
        val_metrics
    ):

        # --------------------------
        # Safe metric extraction
        # --------------------------
        def get_metric(metrics, name):
            return metrics.get(name, None)

        if self.monitor == "val_loss":
            current_metric = val_loss

        elif self.monitor == "train_loss":
            current_metric = train_loss

        else:
            metric_name = (
                self.monitor
                .replace("val_", "")
                .replace("train_", "")
            )

            if "val_" in self.monitor:
                current_metric = get_metric(
                    val_metrics,
                    metric_name
                )
            else:
                current_metric = get_metric(
                    train_metrics,
                    metric_name
                )

        if current_metric is None:
            raise ValueError(
                f"Metric '{self.monitor}' not found "
                f"in provided metrics."
            )

        # --------------------------
        # Early Stopping Logic
        # --------------------------
        improved = (
            current_metric > self.best_metric + self.min_delta
            if self.mode == "max"
            else current_metric < self.best_metric - self.min_delta
        )

        if improved:

            self.best_metric = current_metric
            self.best_epoch = epoch
            self.counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "metric": current_metric,
                    "model_state_dict": model.state_dict()
                },
                self.best_model_path
            )

            print(
                f"[BEST] epoch={epoch+1} "
                f"{self.monitor}={current_metric:.4f}"
            )

        else:

            self.counter += 1

            print(
                f"[NO IMPROVEMENT] "
                f"{self.counter}/{self.patience}"
            )

            if self.counter >= self.patience:
                self.stop_training = True
                print("Early stopping triggered.")

        # --------------------------
        # Save checkpoint
        # --------------------------
        torch.save(
            {
                "epoch": epoch,
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
                "counter": self.counter,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            },
            self.last_checkpoint_path
        )

        # --------------------------
        # Scheduler
        # --------------------------
        if self.scheduler is not None:
            if isinstance(
                self.scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step(current_metric)
            else:
                self.scheduler.step()

        # --------------------------
        # Logging
        # --------------------------
        epoch_time = time.time() - self.start_time
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,

            "train_acc": train_metrics["acc"],
            "val_acc": val_metrics["acc"],

            "train_precision": train_metrics["precision"],
            "val_precision": val_metrics["precision"],

            "train_recall": train_metrics["recall"],
            "val_recall": val_metrics["recall"],

            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],

            "train_bal_acc": train_metrics["balanced_accuracy"],
            "val_bal_acc": val_metrics["balanced_accuracy"],

            "lr": lr,
            "time_sec": epoch_time
        }

        self.history.append(row)

        pd.DataFrame(self.history).to_csv(self.log_path, index=False)

        print(
            f"Epoch {epoch+1} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"F1 {val_metrics['f1']:.4f} | "
            f"LR {lr:.2e} | "
            f"{epoch_time:.1f}s"
        )

    # ==========================
    # Resume
    # ==========================
    def resume(self, model, optimizer):

        if not os.path.exists(self.last_checkpoint_path):
            return 0

        checkpoint = torch.load(self.last_checkpoint_path)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.best_metric = checkpoint.get("best_metric", self.best_metric)
        self.best_epoch = checkpoint.get("best_epoch", -1)
        self.counter = checkpoint.get("counter", 0)

        if os.path.exists(self.log_path):
            try:
                self.history = pd.read_csv(self.log_path).to_dict("records")
            except Exception:
                self.history = []

        print(f"Resuming from epoch {checkpoint['epoch'] + 1}")
        return checkpoint["epoch"] + 1

    # ==========================
    # Plot
    # ==========================
    def plot_curves(self):

        if len(self.history) == 0:
            return

        df = pd.DataFrame(self.history)

        for metric in ["loss", "acc", "f1"]:

            plt.figure(figsize=(8, 5))
            plt.plot(df[f"train_{metric}"], label="Train")
            plt.plot(df[f"val_{metric}"], label="Val")

            plt.title(f"{metric} curves")
            plt.xlabel("Epoch")
            plt.legend()

            plt.tight_layout()
            plt.savefig(f"{metric}_curve.png")
            plt.close()

    # ==========================
    # Confusion Matrix
    # ==========================
    def save_confusion_matrix(self, y_true, y_pred, class_names=None):

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 8))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
        )

        disp.plot(ax=ax, cmap="Blues", values_format="d")

        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()

        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names
        ) if class_names else classification_report(y_true, y_pred)

        with open("classification_report.txt", "w") as f:
            f.write(report)
