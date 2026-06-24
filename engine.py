import torch
import numpy as np


def _is_cuda(device):
    return isinstance(device, torch.device) and device.type == "cuda"


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler=None
):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    is_cuda = _is_cuda(device)

    for x, y in loader:

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=is_cuda
            ):
                outputs = model(x)
                loss = criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        preds = outputs.detach().argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())

    metrics = {
        "y_true": np.array(all_labels),
        "y_pred": np.array(all_preds)
    }

    return total_loss / max(1, len(loader)), metrics


def evaluate(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    is_cuda = _is_cuda(device)

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=is_cuda
            ):
                outputs = model(x)
                loss = criterion(outputs, y)

            total_loss += loss.item()

            preds = outputs.detach().argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())

    metrics = {
        "y_true": np.array(all_labels),
        "y_pred": np.array(all_preds)
    }

    return total_loss / max(1, len(loader)), metrics
