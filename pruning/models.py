"""
Model factory for structured pruning
"""

import logging
from pathlib import Path
from typing import List

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from .config import PruningConfig
from .utils import calculate_model_size_mb, count_model_parameters, set_random_seeds

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating and managing models for pruning"""

    @staticmethod
    def create_model(config: PruningConfig) -> nn.Module:
        set_random_seeds(config.seed)

        model_config = config.model_config
        architecture = model_config["architecture"]

        model_params = {
            k: v
            for k, v in model_config.items()
            if k not in ["architecture", "checkpoint_path"]
        }
        if architecture == "DeepLabV3Plus":
            model = smp.DeepLabV3Plus(**model_params)
        elif architecture == "Unet":
            model = smp.Unet(**model_params)
        elif architecture == "FPN":
            model = smp.FPN(**model_params)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        return model

    @staticmethod
    def load_pretrained_model(config: PruningConfig) -> nn.Module:
        model = ModelFactory.create_model(config)

        checkpoint_path = config.checkpoint_path
        if not Path(checkpoint_path).exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            logger.info("Using model with random initialization")
            return model

        try:
            state_dict = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(state_dict)
            logger.info(f"Loaded pretrained weights from {checkpoint_path}")

            sample_param = next(model.parameters()).detach().cpu()
            logger.debug(
                f"Sample parameter sum after loading: {sample_param.sum().item():.6f}"
            )
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise RuntimeError(
                "Checkpoint loading failed - cannot proceed with random weights"
            )

        return model

    @staticmethod
    def get_ignored_layers(model: nn.Module, config: PruningConfig) -> List[nn.Module]:
        """Get layers to ignore during pruning (typically final classification layers)"""
        model_config = config.model_config
        architecture = model_config["architecture"]

        ignored_layers = []

        try:
            if architecture == "DeepLabV3Plus":
                if hasattr(model, "segmentation_head"):
                    for module in model.segmentation_head.modules():
                        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                            ignored_layers.append(module)
                elif hasattr(model, "classifier"):
                    ignored_layers.append(model.classifier[-1])

            elif architecture in ["Unet", "FPN"]:
                if hasattr(model, "segmentation_head"):
                    for module in model.segmentation_head.modules():
                        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                            ignored_layers.append(module)

        except Exception as e:
            logger.warning(f"Could not identify output layers for {architecture}: {e}")

        logger.info(f"Ignored layers for {architecture}: {len(ignored_layers)}")
        return ignored_layers

    @staticmethod
    def get_model_info(model: nn.Module) -> dict:
        total_params = count_model_parameters(model)
        trainable_params = count_model_parameters(model, only_trainable=True)
        model_size_mb = calculate_model_size_mb(model)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_size_mb": model_size_mb,
        }

    @staticmethod
    def save_model(model: nn.Module, path: str, config: PruningConfig):
        try:
            model.zero_grad()
            torch.save(model, path)
            logger.info(f"Pruned model saved to {path}")

            from pathlib import Path

            model_path = Path(path)
            info_path = model_path.parent / "model_info.txt"
            model_info = ModelFactory.get_model_info(model)

            with open(str(info_path), "w") as f:
                f.write(f"Model: {config.model_name}\n")
                f.write(f"Pruning Strategy: {config.pruning_strategy}\n")
                f.write(f"Pruning Ratio: {config.pruning_ratio}\n")
                f.write(f"Total Parameters: {model_info['total_params']:,}\n")
                f.write(f"Trainable Parameters: {model_info['trainable_params']:,}\n")
                f.write(f"Model Size: {model_info['model_size_mb']:.2f} MB\n")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
