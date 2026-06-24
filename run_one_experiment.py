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
import time
import torch
import torch.nn as nn

```
from models.backbones import get_model
from utils.fine_tuning import unfreeze_last_n
from training.engine import (
    train_one_epoch,
    evaluate
)
from utils.callbacks import callbacks
from utils.metrics import compute_metrics
from utils.model_utils import count_parameters

lr = config["model"]["lr"]
epochs = config["model"]["epochs"]

print("\n" + "=" * 60)
print(f"Model: {model_name}")
print(f"n_unfreeze: {n_unfreeze}")
print("=" * 60)

# --------------------------------------------------
# Model
# --------------------------------------------------
model = get_model(
    model_name,
    num_classes
).to(device)

scaler = (
    torch.amp.GradScaler("cuda")
    if device.type == "cuda"
    else None
)

trainable_blocks = unfreeze_last_n(
    model,
    model_name,
    n_unfreeze
)

trainable_params, _ = count_parameters(
    model
)

print(
    f"Trainable blocks: "
    f"{trainable_blocks}"
)

# --------------------------------------------------
# Training objects
# --------------------------------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=lr
)

scheduler = (
    torch.optim.lr_scheduler
    .ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=2,
        min_lr=1e-7
    )
)

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

# --------------------------------------------------
# Training loop
# --------------------------------------------------
start_time = time.time()

for epoch in range(
    start_epoch,
    epochs
):

    cb.on_epoch_begin()

    train_loss, train_metrics = (
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler
        )
    )

    eval_loader = (
        val_loader
        if val_loader is not None
        else test_loader
    )

    val_loss, val_metrics = (
        evaluate(
            model,
            eval_loader,
            criterion,
            device
        )
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

training_time = (
    time.time() - start_time
)

# --------------------------------------------------
# Load best model
# --------------------------------------------------
checkpoint = torch.load(
    cb.best_model_path,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

# --------------------------------------------------
# Final evaluation
# --------------------------------------------------
final_loader = (
    test_loader
    if test_loader is not None
    else val_loader
)

final_loss, final_metrics = (
    evaluate(
        model,
        final_loader,
        criterion,
        device
    )
)

metrics = compute_metrics(
    final_metrics["y_true"],
    final_metrics["y_pred"],
    final_metrics["y_prob"]
)

cb.plot_curves()

logger.log(
    config={
        "dataset_root":
            config["dataset_root"],
        "model":
            model_name,
        "n_unfreeze":
            n_unfreeze,
        "image_size":
            config["image_size"],
        "batch_size":
            config["model"]["batch_size"],
        "epochs":
            epochs,
        "lr":
            lr,
        "optimizer":
            "adam",
        "scheduler":
            "ReduceLROnPlateau",
        "seed":
            config["seed"]
    },
    metrics=metrics,
    loss=final_loss,
    model_path=cb.best_model_path,
    training_time=training_time,
    trainable_params=trainable_params
)
```
