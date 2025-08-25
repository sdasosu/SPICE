"""
Model factory for creating segmentation models
"""

import segmentation_models_pytorch as smp

from .config import MODEL_CONFIGS


def create_model_from_config(config: dict):
    """
    Create a model instance from configuration dictionary

    Args:
        config: Model configuration dictionary

    Returns:
        Model instance
    """
    architecture = config["architecture"]
    params = {
        k: v for k, v in config.items() if k not in ["architecture", "checkpoint_path"]
    }

    if architecture == "DeepLabV3Plus":
        return smp.DeepLabV3Plus(**params)
    elif architecture == "Unet":
        return smp.Unet(**params)
    elif architecture == "FPN":
        return smp.FPN(**params)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def get_model_by_name(model_name: str):
    """
    Get model instance and checkpoint path by name

    Args:
        model_name: Name of the model

    Returns:
        tuple: (model_instance, checkpoint_path)
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

    config = MODEL_CONFIGS[model_name]
    model = create_model_from_config(config)
    checkpoint_path = config["checkpoint_path"]

    return model, checkpoint_path
