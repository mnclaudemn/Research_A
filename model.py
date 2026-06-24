# models/backbones.py

import torch
import torch.nn as nn
import torchvision.models as models
import timm


# ==========================================================
# Fine-tuning block maps
# ==========================================================
BLOCKS = {
    "resnet50": [
        "layer1",
        "layer2",
        "layer3",
        "layer4",
        "fc",
    ],

    "densenet121": [
        "features.denseblock1",
        "features.denseblock2",
        "features.denseblock3",
        "features.denseblock4",
        "classifier",
    ],

    "vit": [
        "blocks.0",
        "blocks.1",
        "blocks.2",
        "blocks.3",
        "blocks.4",
        "blocks.5",
        "blocks.6",
        "blocks.7",
        "blocks.8",
        "blocks.9",
        "blocks.10",
        "blocks.11",
        "norm",
        "head",
    ],

    "dino": [
        "backbone.blocks.0",
        "backbone.blocks.1",
        "backbone.blocks.2",
        "backbone.blocks.3",
        "backbone.blocks.4",
        "backbone.blocks.5",
        "backbone.blocks.6",
        "backbone.blocks.7",
        "backbone.blocks.8",
        "backbone.blocks.9",
        "backbone.blocks.10",
        "backbone.blocks.11",
        "head",
    ],
}


# ==========================================================
# Utilities
# ==========================================================
def count_trainable_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ==========================================================
# Fine-tuning
# ==========================================================
def unfreeze_last_n(model, model_name, n):
    """
    Unfreeze the last n blocks.

    Examples
    --------
    ResNet50:
        n=0 -> classifier only
        n=1 -> fc
        n=2 -> layer4 + fc
        n=3 -> layer3 + layer4 + fc

    DenseNet121:
        n=0 -> classifier only
        n=1 -> classifier
        n=2 -> denseblock4 + classifier
        n=3 -> denseblock3 + denseblock4 + classifier
    """

    model_name = model_name.lower()

    if model_name not in BLOCKS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list(BLOCKS.keys())}"
        )

    blocks = BLOCKS[model_name]

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Classifier-only baseline
    if n == 0:

        if model_name == "resnet50":
            for p in model.fc.parameters():
                p.requires_grad = True

        elif model_name == "densenet121":
            for p in model.classifier.parameters():
                p.requires_grad = True

        elif model_name == "vit":
            for p in model.head.parameters():
                p.requires_grad = True

        elif model_name == "dino":
            for p in model.head.parameters():
                p.requires_grad = True

        trainable = count_trainable_parameters(model)

        print("\nClassifier-only training")
        print(f"Trainable parameters: {trainable:,}\n")

        return []

    if n < 0 or n > len(blocks):
        raise ValueError(
            f"n must be between 0 and {len(blocks)}"
        )

    blocks_to_train = blocks[-n:]

    for name, module in model.named_modules():

        if any(
            name == b or name.startswith(b + ".")
            for b in blocks_to_train
        ):
            for p in module.parameters():
                p.requires_grad = True

    # Optional:
    # Keep BatchNorm statistics frozen
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    trainable = count_trainable_parameters(model)

    print(f"\nTrainable blocks: {blocks_to_train}")
    print(f"Trainable parameters: {trainable:,}\n")

    return blocks_to_train


# ==========================================================
# Model builders
# ==========================================================
def resnet50(num_classes, pretrained=True):

    weights = (
        models.ResNet50_Weights.IMAGENET1K_V2
        if pretrained
        else None
    )

    model = models.resnet50(weights=weights)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


def densenet121(num_classes, pretrained=True):

    weights = (
        models.DenseNet121_Weights.IMAGENET1K_V1
        if pretrained
        else None
    )

    model = models.densenet121(weights=weights)

    model.classifier = nn.Linear(
        model.classifier.in_features,
        num_classes
    )

    return model


def vit_base(num_classes, pretrained=True):

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )

    return model


# ==========================================================
# DINO
# ==========================================================
class DinoClassifier(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.backbone = timm.create_model(
            "vit_small_patch16_224.dino",
            pretrained=True,
            num_classes=0,
        )

        self.head = nn.Linear(
            self.backbone.num_features,
            num_classes,
        )

    def forward(self, x):

        features = self.backbone(x)

        # Safety for different DINO variants
        if features.ndim == 3:
            features = features[:, 0]

        logits = self.head(features)

        return logits


def dino_vit(num_classes):
    return DinoClassifier(num_classes)


# ==========================================================
# Factory
# ==========================================================
def get_model(name, num_classes):

    name = name.lower()

    if name == "resnet50":
        return resnet50(num_classes)

    elif name == "densenet121":
        return densenet121(num_classes)

    elif name == "vit":
        return vit_base(num_classes)

    elif name == "dino":
        return dino_vit(num_classes)

    raise ValueError(
        f"Unknown model '{name}'. "
        f"Available models: {list(BLOCKS.keys())}"
    )
