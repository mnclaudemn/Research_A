# run_one_experiment.py

import time
import torch
import torch.nn as nn

from model import *
from compute_metrics import *
from dataset import *
from engineA import *
from experiment_logger import *
from callbacks import callbacks


def run_one_experiment(
    train_loader,
    val_loader,
    test_loader,
    num_classes,
    config,
    device,
    logger
):
    """
    Run one complete experiment:
    - Build model
    - Fine-tune selected layers
    - Train
    - Evaluate
    - Save best model
    - Log experiment
    """

    # ==================================================
    # Configuration
    # ==================================================
    config = load_config("config.yaml")
    expr_name = config["expr_name"]
    model_name = config["model_name"]
    n_unfreeze = config["n_unfreeze"]

    lr = config.get("lr", 1e-4)
    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 16)

    print("\n" + "=" * 60)
    print(f"Experiment : {expr_name}")
    print(f"Model      : {model_name}")
    print(f"n_unfreeze : {n_unfreeze}")
    print("=" * 60)

    # ==================================================
    # Build Model
    # ==================================================
    model = get_model(
        model_name=model_name,
        num_classes=num_classes
    ).to(device)

    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda")
    )

    trainable_blocks = unfreeze_last_n(
        model=model,
        model_name=model_name,
        n=n_unfreeze
    )

    trainable_params, total_params = count_parameters(model)

    print(f"Trainable blocks : {trainable_blocks}")
    print(f"Trainable params : {trainable_params:,}")
    print(f"Total params     : {total_params:,}")

    # ==================================================
    # Loss / Optimizer / Scheduler
    # ==================================================
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=lr
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=2,
        min_lr=1e-7
    )

    # ==================================================
    # Callbacks
    # ==================================================
    cb = callbacks(
        monitor="val_f1",
        mode="max",
        patience=4,
        min_delta=1e-3,
        average="weighted",
        scheduler=scheduler
    )

    start_epoch = cb.resume(
        model,
        optimizer
    )

    # ==================================================
    # Validation Loader
    # ==================================================
    eval_loader = (
        val_loader
        if val_loader is not None
        else test_loader
    )

    if eval_loader is None:
        raise ValueError(
            "No validation or test loader found."
        )

    # ==================================================
    # Training Loop
    # ==================================================
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
            print(
                f"\nEarly stopping at epoch {epoch + 1}"
            )
            break

    training_time = time.time() - start_time

    # ==================================================
    # Load Best Model
    # ==================================================
    if cb.best_model_path is not None:

        checkpoint = torch.load(
            cb.best_model_path,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    # ==================================================
    # Final Evaluation
    # ==================================================
    final_loader = (
        test_loader
        if test_loader is not None
        else val_loader
    )

    if final_loader is None:
        raise ValueError(
            "No evaluation loader found."
        )

    final_loss, final_metrics = evaluate(
        model=model,
        loader=final_loader,
        criterion=criterion,
        device=device
    )

    metrics = compute_metrics(
        y_true=final_metrics["y_true"],
        y_pred=final_metrics["y_pred"],
        y_prob=final_metrics["y_prob"]
    )

    # ==================================================
    # Save Curves
    # ==================================================
    cb.plot_curves()

    # ==================================================
    # Log Experiment
    # ==================================================
    logger.log(
        config={
            "expr_name": expr_name,
            "dataset_root": config.get(
                "dataset_root",
                ""
            ),
            "model_name": model_name,
            "n_unfreeze": n_unfreeze,
            "image_size": config.get(
                "image_size",
                224
            ),
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "seed": config.get(
                "seed",
                42
            )
        },
        metrics=metrics,
        loss=final_loss,
        model_path=cb.best_model_path,
        training_time=training_time,
        trainable_params=trainable_params
    )

    # ==================================================
    # Print Results
    # ==================================================
    print("\nFinal Results")
    print("-" * 40)

    for key, value in metrics.items():
        if isinstance(value, (float, int)):
            print(f"{key}: {value:.4f}")

    print(
        f"\nTraining time: "
        f"{training_time:.2f} sec"
    )

    # ==================================================
    # Free GPU Memory
    # ==================================================
    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ==================================================
    # Return Results
    # ==================================================
    return {
        "expr_name": expr_name,
        "model_name": model_name,
        "n_unfreeze": n_unfreeze,
        "metrics": metrics,
        "loss": final_loss,
        "training_time": training_time,
        "model_path": cb.best_model_path,
        "trainable_params": trainable_params,
        "total_params": total_params
    }
