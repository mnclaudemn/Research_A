import torch
import torch.nn as nn

from configs.load_config import load_config
from utils.seed import set_seed
from datasets.dataset_report import analyze_dataset, show_samples
from datasets.medical_dataset import get_loaders
from models.backbones import get_model
from utils.fine_tuning import unfreeze_last_n
from training.engine import train_one_epoch, evaluate
from utils.experiment_logger import ExperimentLogger
from utils.callbacks import callbacks


def main():

    # ==========================
    # Load Config
    # ==========================
    config = load_config()

    dataset_root = config["dataset_root"]
    model_name = config["model"]["name"]
    n_unfreeze = config["model"]["n_unfreeze"]
    lr = config["model"]["lr"]
    batch_size = config["model"]["batch_size"]
    epochs = config["model"]["epochs"]
    image_size = config["image_size"]
    seed = config["seed"]

    # ==========================
    # Setup
    # ==========================
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ==========================
    # Dataset analysis
    # ==========================
    splits, train_ds = analyze_dataset(dataset_root)

    show_samples(train_ds, num_classes=5)

    # ==========================
    # Data loaders
    # ==========================
    train_loader, val_loader, test_loader, num_classes = get_loaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        image_size=image_size
    )

    # ==========================
    # Model
    # ==========================
    model = get_model(model_name, num_classes).to(device)

    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda"
        else None
    )

    trainable_blocks = unfreeze_last_n(
        model=model,
        model_name=model_name,
        n=n_unfreeze
    )

    print(f"\nModel: {model_name}")
    print(f"Trainable blocks: {trainable_blocks}\n")

    # ==========================
    # Loss / Optimizer / Scheduler
    # ==========================
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=2,
        min_lr=1e-7
    )

    # ==========================
    # Callbacks
    # ==========================
    cb = callbacks(
        monitor="val_f1",
        mode="max",
        patience=4,
        min_delta=1e-3,
        average="weighted",
        scheduler=scheduler
    )

    start_epoch = cb.resume(model, optimizer)

    # ==========================
    # Logger
    # ==========================
    logger = ExperimentLogger()

    # ==========================
    # Training Loop
    # ==========================
    for epoch in range(start_epoch, epochs):

        cb.on_epoch_begin()

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler
        )

        eval_loader = val_loader if val_loader is not None else test_loader

        val_loss, val_metrics = evaluate(
            model,
            eval_loader,
            criterion,
            device
        )

        cb.on_epoch_end(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_loss=train_loss,
            val_loss=val_loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics
        )

        if cb.stop_training:
            break

    # ==========================
    # Load Best Model
    # ==========================
    checkpoint = torch.load(cb.best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # ==========================
    # Save Curves
    # ==========================
    cb.plot_curves()

    # ==========================
    # Logging Experiment
    # ==========================
    logger.log(
        config={
            "dataset_root": dataset_root,
            "model": model_name,
            "n_unfreeze": n_unfreeze,
            "image_size": image_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "optimizer": "adam"
        },
        metrics={
            "best_metric": cb.best_score,
            "best_epoch": cb.best_epoch
        },
        loss=None,
        model_path=cb.best_model_path
    )


if __name__ == "__main__":
    main()
