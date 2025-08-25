"""
Model configurations for different architectures
"""

MODEL_CONFIGS = {
    "DeepLabV3Plus_resnet": {
        "architecture": "DeepLabV3Plus",
        "encoder_name": "resnet50",
        "encoder_weights": None,
        "in_channels": 3,
        "classes": 5,
        "activation": None,
        "checkpoint_path": "checkpoints/fullsize/DeepLabV3Plus_resnet/DeepLabV3Plus_resnet.pt",
    },
    "DeepLabV3Plus_efficientnet": {
        "architecture": "DeepLabV3Plus",
        "encoder_name": "timm-efficientnet-b3",
        "encoder_weights": None,
        "in_channels": 3,
        "classes": 5,
        "activation": None,
        "checkpoint_path": "checkpoints/fullsize/DeepLabV3Plus_efficientnet/DeepLabV3Plus_efficientnet.pt",
    },
    "UNET_resnet": {
        "architecture": "Unet",
        "encoder_name": "resnet50",
        "encoder_weights": None,
        "in_channels": 3,
        "classes": 5,
        "activation": None,
        "checkpoint_path": "checkpoints/fullsize/UNET_resnet/UNET_resnet.pt",
    },
    "UNET_efficientnet": {
        "architecture": "Unet",
        "encoder_name": "timm-efficientnet-b3",
        "encoder_weights": None,
        "in_channels": 3,
        "classes": 5,
        "activation": None,
        "checkpoint_path": "checkpoints/fullsize/UNET_efficientnet/UNET_efficientnet.pt",
    },
    "FPN_resnet": {
        "architecture": "FPN",
        "encoder_name": "resnet50",
        "encoder_weights": None,
        "in_channels": 3,
        "classes": 5,
        "activation": None,
        "checkpoint_path": "checkpoints/fullsize/FPN_resnet/FPN_resnet.pt",
    },
    "FPN_efficientnet": {
        "architecture": "FPN",
        "encoder_name": "timm-efficientnet-b3",
        "encoder_weights": None,
        "in_channels": 3,
        "classes": 5,
        "activation": None,
        "checkpoint_path": "checkpoints/fullsize/FPN_efficientnet/FPN_efficientnet.pt",
    },
}

AVAILABLE_STRATEGIES = [
    "magnitude",
    "random",
    "taylor",
    "lamp",
    "taylor_weight",
    "magnitude_taylor",
]
