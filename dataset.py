import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import Counter
from torchvision import datasets

def get_transforms(image_size=224):

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


def find_folder(dataset_path, name_options):
    """
    Find folder like train/test/val automatically
    """
    for d in os.listdir(dataset_path):
        if d.lower() in name_options:
            return os.path.join(dataset_path, d)
    return None


def get_loaders(dataset_path, batch_size=16, image_size=224):

    train_tf, test_tf = get_transforms(image_size)

    #  Auto-detect folders
    train_dir = find_folder(dataset_path, ["train", "training"])
    test_dir = find_folder(dataset_path, ["test", "testing"])
    val_dir = find_folder(dataset_path, ["val", "valid", "validation"])

    if train_dir is None:
        raise ValueError("Train folder not found!")

    #  datasets
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)

    val_ds = datasets.ImageFolder(val_dir, transform=test_tf) if val_dir else None
    test_ds = datasets.ImageFolder(test_dir, transform=test_tf) if test_dir else None

    # loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0) if test_ds else None

    return train_loader, val_loader, test_loader, len(train_ds.classes)


def detect_splits(root):

    splits = {}
    for d in os.listdir(root):
        path = os.path.join(root, d)

        if not os.path.isdir(path):
            continue

        name = d.lower()

        if name in ["train", "training"]:
            splits["train"] = path

        elif name in ["test", "testing"]:
            splits["test"] = path

        elif name in ["val", "valid", "validation"]:
            splits["val"] = path

    return splits


def analyze_dataset(dataset_root):

    splits = detect_splits(dataset_root)

    if "train" not in splits:
        raise ValueError("Train folder not found!")

    train_ds = datasets.ImageFolder(splits["train"])

    class_counts = Counter(train_ds.targets)
    class_names = train_ds.classes

    print("\n================ DATASET REPORT ================\n")

    print(f"Train path: {splits['train']}")
    if "val" in splits:
        print(f"Val path: {splits['val']}")
    if "test" in splits:
        print(f"Test path: {splits['test']}")

    print(f"\nNumber of classes: {len(class_names)}")
    print(f"Total training images: {len(train_ds)}\n")

    print("Class distribution:")
    for i, count in class_counts.items():
        print(f"  {class_names[i]}: {count}")

    # imbalance check
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())

    imbalance_ratio = max_count / min_count

    print(f"\nImbalance ratio: {imbalance_ratio:.2f}")

    if imbalance_ratio > 3:
        print("⚠ WARNING: Dataset is highly imbalanced!")

    return splits, train_ds


def show_samples(train_ds, num_classes=5):

    import numpy as np

    fig, axes = plt.subplots(1, num_classes, figsize=(15, 5))

    class_indices = {}

    for idx, label in enumerate(train_ds.targets):
        if label not in class_indices:
            class_indices[label] = idx

        if len(class_indices) >= num_classes:
            break

    for i, (label, idx) in enumerate(list(class_indices.items())[:num_classes]):

        img, _ = train_ds[idx]

        img = img.permute(1, 2, 0).numpy()

        axes[i].imshow(img)
        axes[i].set_title(train_ds.classes[label])
        axes[i].axis("off")

    plt.show()

      
