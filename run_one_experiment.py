import time
import torch
import torch.nn as nn

from models.backbones import get_model
from utils.fine_tuning import unfreeze_last_n
from engineA import train_one_epoch, evaluate
from callbacks import callbacks
from compute_metrics import compute_metrics
from utils import count_parameters

def run_one_experiment(
    model_name,
    n_unfreeze,
    train_loader,
    val_loader,
    test_loader,
    num_classes,
    config,
    device,
    logger
):

    # Dynamic extraction of config values with multi-layer fallback settings
    lr = config.get("model", {}).get("lr", config.get("lr", 1e-4))
    epochs = config.get("model", {}).get("epochs", config.get("epochs", 10))
    batch_size = config.get("model", {}).get("batch_size", config.get("batch_size", 16))

    print("\n" + "=" * 60)
    print(f"Model: {model_name}")
    print(f"n_unfreeze: {n_unfreeze}")
    print("=" * 60)

    # --------------------------------------------------
    # Model Setup
    # --------------------------------------------------
    model = get_model(model_name, num_classes).to(device)

    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda"
        else None
    )

    # Configure exact network unfreezing depth
    trainable_blocks = unfreeze_last_n(model, model_name, n_unfreeze)
    trainable_params, _ = count_parameters(model)

    print(f"Trainable blocks: {trainable_blocks}")

    # --------------------------------------------------
    # Training Objects
    # --------------------------------------------------
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

    cb = callbacks(
        monitor="val_f1",
        mode="max",
        patience=4,
        min_delta=1e-3,
        average="weighted",
        scheduler=scheduler
    )

    start_epoch = cb.resume(model, optimizer)

    # --------------------------------------------------
    # Core Training Loop
    # --------------------------------------------------
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        cb.on_epoch_begin()

        train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler
        )

        eval_loader = val_loader if val_loader is not None else test_loader

        val_loss, val_metrics = evaluate(
            model=model,
            loader=eval_loader,
            criterion=criterion,
            device=device
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

    training_time = time.time() - start_time

    # --------------------------------------------------
    # Load Best Weights for Post-Hoc Evaluation
    # --------------------------------------------------
    checkpoint = torch.load(cb.best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # --------------------------------------------------
    # Final Evaluation Summary
    # --------------------------------------------------
    final_loader = test_loader if test_loader is not None else val_loader

    final_loss, final_metrics = evaluate(
        model=model,
        loader=final_loader,
        criterion=criterion,
        device=device
    )

    # Compute descriptive performance indicators for paper metrics tables
    metrics = compute_metrics(
        y_true=final_metrics["y_true"],
        y_pred=final_metrics["y_pred"],
        y_prob=final_metrics["y_prob"]
    )

    # Save training metrics vector curves
    cb.plot_curves()

    # Pass configuration and results structured cleanly to your Excel logger engine
    logger.log(
        config={
            "dataset_root": config.get("dataset_root", ""),
            "model": model_name,
            "n_unfreeze": n_unfreeze,
            "image_size": config.get("image_size", 224),
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "optimizer": "adam",
            "scheduler": "ReduceLROnPlateau",
            "seed": config.get("seed", 42),
            "trainable_parameters": trainable_params,
            "execution_duration_sec": round(training_time, 2)
        },
        metrics=metrics,
        loss=final_loss,
        model_path=cb.best_model_path
    )
