# utils/fine_tuning.py
import torchvision.models as models
import torch.nn as nn
import timm
from models.dino import dino_vit
import torch
import torch.nn as nn

BLOCKS = {
    "resnet50": [ "layer1","layer2","layer3","layer4", "fc"],

    "densenet121": ["features.denseblock1", "features.denseblock2","features.denseblock3", "features.denseblock4", "classifier" ]
    BLOCKS["dino"] = [ "backbone.blocks.0", "backbone.blocks.1","backbone.blocks.2","backbone.blocks.3",
    "backbone.blocks.4", "backbone.blocks.5", "backbone.blocks.6","backbone.blocks.7","backbone.blocks.8", "backbone.blocks.9",
    "backbone.blocks.10","backbone.blocks.11", "head" ]
}


def unfreeze_last_n(model, model_name, n):
    """
    Unfreeze the last n blocks.

    Examples
    --------
    ResNet50:
        n=1 -> fc
        n=2 -> layer4 + fc
        n=3 -> layer3 + layer4 + fc

    DenseNet121:
        n=1 -> classifier
        n=2 -> denseblock4 + classifier
        n=3 -> denseblock3 + denseblock4 + classifier
    """

    blocks = BLOCKS[model_name]

    if n < 1 or n > len(blocks):
        raise ValueError(
            f"n must be between 1 and {len(blocks)}"
        )

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Select last n blocks
    blocks_to_train = blocks[-n:]

    # Unfreeze selected blocks
    for name, module in model.named_modules():
        if name in blocks_to_train:
            for p in module.parameters():
                p.requires_grad = True

    return blocks_to_train

def get_model(name, num_classes):

    name = name.lower()

    if name == "resnet50":
        return resnet50(num_classes)

    if name == "densenet121":
        return densenet121(num_classes)

    if name == "vit":
        return vit_base(num_classes)

    if name == "dino":
        return dino_vit(num_classes)

    raise ValueError("Unknown model")

def densenet121(num_classes, pretrained=True):

    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None

    model = models.densenet121(weights=weights)

    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return model



class DinoClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = torch.hub.load(
            'facebookresearch/dino:main',
            'dino_vits16'
        )

        self.head = nn.Linear(
            self.backbone.embed_dim,
            num_classes
        )

    def forward(self, x):

        features = self.backbone(x)
def resnet50(num_classes, pretrained=True):

    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None

    model = models.resnet50(weights=weights)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

  def vit_base(num_classes, pretrained=True):

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes
    )

    return model
        logits = self.head(features)

        return logits


def dino_vit(num_classes):
    return DinoClassifier(num_classes
