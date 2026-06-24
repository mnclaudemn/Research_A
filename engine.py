import torch
import numpy as np

def _is_cuda(device):
"""
Check whether device is CUDA.
"""
return (
isinstance(device, torch.device)
and device.type == "cuda"
)

def train_one_epoch(
model,
loader,
optimizer,
criterion,
device,
scaler=None
):
"""
Train model for one epoch.

```
Returns
-------
epoch_loss : float
metrics : dict
    {
        "y_true": ndarray,
        "y_pred": ndarray,
        "y_prob": ndarray
    }
"""

model.train()

total_loss = 0.0
total_samples = 0

all_preds = []
all_labels = []
all_probs = []

is_cuda = _is_cuda(device)

for x, y in loader:

    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True).long()

    batch_size = x.size(0)

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

    probs = torch.softmax(outputs, dim=1)
    preds = probs.argmax(dim=1)

    total_loss += loss.item() * batch_size
    total_samples += batch_size

    all_preds.extend(
        preds.detach().cpu().numpy()
    )

    all_labels.extend(
        y.detach().cpu().numpy()
    )

    all_probs.extend(
        probs.detach().cpu().numpy()
    )

metrics = {
    "y_true": np.array(all_labels),
    "y_pred": np.array(all_preds),
    "y_prob": np.array(all_probs)
}

epoch_loss = total_loss / max(1, total_samples)

return epoch_loss, metrics
```

def evaluate(
model,
loader,
criterion,
device
):
"""
Evaluate model.

```
Returns
-------
epoch_loss : float
metrics : dict
    {
        "y_true": ndarray,
        "y_pred": ndarray,
        "y_prob": ndarray
    }
"""

if loader is None:
    return None, None

model.eval()

total_loss = 0.0
total_samples = 0

all_preds = []
all_labels = []
all_probs = []

is_cuda = _is_cuda(device)

with torch.no_grad():

    for x, y in loader:

        x = x.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        ).long()

        batch_size = x.size(0)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=is_cuda
        ):
            outputs = model(x)
            loss = criterion(outputs, y)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        preds = probs.argmax(dim=1)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        all_preds.extend(
            preds.detach().cpu().numpy()
        )

        all_labels.extend(
            y.detach().cpu().numpy()
        )

        all_probs.extend(
            probs.detach().cpu().numpy()
        )

metrics = {
    "y_true": np.array(all_labels),
    "y_pred": np.array(all_preds),
    "y_prob": np.array(all_probs)
}

epoch_loss = total_loss / max(
    1,
    total_samples
)

return epoch_loss, metrics
```
