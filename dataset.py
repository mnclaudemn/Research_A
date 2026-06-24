import os
from collections import Counter

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================================

# Transforms

# ==========================================================

def get_transforms(image_size=224):
"""
Create training and evaluation transforms.
"""

```
train_tf = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_tf = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

return train_tf, test_tf
```

# ==========================================================

# Folder Detection

# ==========================================================

def find_folder(dataset_path, name_options):
"""
Automatically find train/val/test folders.
"""

```
for d in os.listdir(dataset_path):
    if d.lower() in name_options:
        return os.path.join(dataset_path, d)

return None
```

def detect_splits(root):
"""
Detect dataset splits automatically.
"""

```
splits = {}

for d in os.listdir(root):
    path = os.path.join(root, d)

    if not os.path.isdir(path):
        continue

    name = d.lower()

    if name in ["train", "training"]:
        splits["train"] = path

    elif name in ["val", "valid", "validation"]:
        splits["val"] = path

    elif name in ["test", "testing"]:
        splits["test"] = path

return splits
```

# ==========================================================

# Data Loaders

# ==========================================================

def get_loaders(
dataset_path,
batch_size=16,
image_size=224,
num_workers=2
):
"""
Create train, validation and test dataloaders.
"""

```
train_tf, test_tf = get_transforms(image_size)

train_dir = find_folder(
    dataset_path,
    ["train", "training"]
)

val_dir = find_folder(
    dataset_path,
    ["val", "valid", "validation"]
)

test_dir = find_folder(
    dataset_path,
    ["test", "testing"]
)

if train_dir is None:
    raise ValueError("Train folder not found!")

train_ds = datasets.ImageFolder(
    train_dir,
    transform=train_tf
)

val_ds = (
    datasets.ImageFolder(val_dir, transform=test_tf)
    if val_dir else None
)

test_ds = (
    datasets.ImageFolder(test_dir, transform=test_tf)
    if test_dir else None
)

pin_memory = torch.cuda.is_available()
persistent = num_workers > 0

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent
)

val_loader = (
    DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent
    )
    if val_ds else None
)

test_loader = (
    DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent
    )
    if test_ds else None
)

return (
    train_loader,
    val_loader,
    test_loader,
    len(train_ds.classes)
)
```

# ==========================================================

# Dataset Analysis

# ==========================================================

def analyze_dataset(dataset_root):
"""
Print dataset statistics.
"""

```
splits = detect_splits(dataset_root)

if "train" not in splits:
    raise ValueError("Train folder not found!")

train_ds = datasets.ImageFolder(
    splits["train"]
)

class_counts = Counter(train_ds.targets)
class_names = train_ds.classes

print("\n========== DATASET REPORT ==========\n")

print(f"Train path: {splits['train']}")

if "val" in splits:
    print(f"Val path: {splits['val']}")

if "test" in splits:
    print(f"Test path: {splits['test']}")

print(f"\nNumber of classes: {len(class_names)}")
print(f"Total training images: {len(train_ds)}\n")

print("Class distribution:")

for i, count in class_counts.items():
    print(f"{class_names[i]}: {count}")

max_count = max(class_counts.values())
min_count = min(class_counts.values())

imbalance_ratio = max_count / min_count

print(f"\nImbalance ratio: {imbalance_ratio:.2f}")

if imbalance_ratio > 3:
    print("WARNING: Dataset is highly imbalanced!")

return splits, train_ds
```

# ==========================================================

# Visualization

# ==========================================================

def show_samples(train_ds, num_classes=5):
"""
Show one sample image from each class.
"""

```
num_classes = min(
    num_classes,
    len(train_ds.classes)
)

fig, axes = plt.subplots(
    1,
    num_classes,
    figsize=(4 * num_classes, 4)
)

if num_classes == 1:
    axes = [axes]

class_indices = {}

for idx, label in enumerate(train_ds.targets):
    if label not in class_indices:
        class_indices[label] = idx

    if len(class_indices) >= num_classes:
        break

for i, (label, idx) in enumerate(
    list(class_indices.items())
):
    img, _ = train_ds[idx]

    if torch.is_tensor(img):
        img = img.permute(1, 2, 0).numpy()

    axes[i].imshow(img)
    axes[i].set_title(train_ds.classes[label])
    axes[i].axis("off")

plt.tight_layout()
plt.show()
plt.close()
```
