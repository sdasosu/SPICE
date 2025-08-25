"""
Common utilities for the pruning package
"""

import hashlib
import logging
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility across all modules

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Get torch device with automatic detection

    Args:
        device_str: Device string ('cuda', 'cpu', 'auto', or None)

    Returns:
        torch.device object
    """
    if device_str is None or device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    return torch.device(device_str)


def generate_config_hash(config_str: str) -> str:
    """
    Generate deterministic hash from configuration string

    Args:
        config_str: Configuration string

    Returns:
        8-character hash string
    """
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def count_model_parameters(model: torch.nn.Module, only_trainable: bool = False) -> int:
    """
    Count model parameters

    Args:
        model: PyTorch model
        only_trainable: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: PyTorch model

    Returns:
        Model size in megabytes
    """
    total_params = count_model_parameters(model)
    return total_params * 4 / (1024 * 1024)


def setup_logging(level: int = logging.INFO) -> None:
    """
    Setup consistent logging format across modules

    Args:
        level: Logging level
    """
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
