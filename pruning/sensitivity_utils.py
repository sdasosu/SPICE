"""
Utility functions for sensitivity analysis
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_taylor_scores(
    model: nn.Module, dataloader, device: str, max_batches: int = 100
) -> Dict[nn.Module, torch.Tensor]:
    """
    Compute Taylor weight-based sensitivity scores.

    Args:
        model: Model to analyze
        dataloader: Calibration data loader
        device: Device for computation
        max_batches: Maximum calibration batches

    Returns:
        Dictionary mapping modules to their per-channel importance scores
    """
    was_training = model.training
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
    scores = {}
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            scores[module] = torch.zeros(module.out_channels, device=device)
    seen_batches = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        if isinstance(batch, (list, tuple)):
            images = batch[0].to(device)
            targets = batch[1].to(device) if len(batch) > 1 else None
        else:
            images = batch.to(device)
            targets = None
        model.zero_grad(set_to_none=True)
        outputs = model(images)
        if isinstance(outputs, dict):
            outputs = outputs.get("out", list(outputs.values())[0])
        if targets is not None:
            if len(targets.shape) == 3:
                loss = F.cross_entropy(outputs, targets.long())
            else:
                loss = F.cross_entropy(outputs, targets)
        else:
            loss = outputs.abs().mean()
        loss.backward()
        for module in scores:
            if module.weight.grad is not None:
                grad = module.weight.grad.detach().abs()
                weight = module.weight.detach().abs()
                sensitivity = (grad * weight).sum(dim=(1, 2, 3))
                scores[module] += sensitivity

        seen_batches += 1
    for module in scores:
        scores[module] = scores[module] / max(seen_batches, 1)

    model.train(was_training)
    return scores


def get_sensitivity_report(scores: Dict[nn.Module, torch.Tensor]) -> Dict:
    """
    Generate sensitivity analysis report for logging/debugging.

    Args:
        scores: Sensitivity scores from compute_taylor_scores

    Returns:
        Dictionary with sensitivity statistics
    """
    report = {"num_layers": len(scores), "layer_stats": []}

    for module in scores:
        if isinstance(module, nn.Conv2d):
            module_scores = scores[module]
            stats = {
                "out_channels": module.out_channels,
                "mean_sensitivity": module_scores.mean().item(),
                "std_sensitivity": module_scores.std().item(),
                "min_sensitivity": module_scores.min().item(),
                "max_sensitivity": module_scores.max().item(),
            }
            report["layer_stats"].append(stats)

    return report


def get_layer_sensitivities(
    scores: Dict[nn.Module, torch.Tensor],
) -> Dict[nn.Module, float]:
    """
    Get mean sensitivity score for each layer.

    Args:
        scores: Sensitivity scores from compute_taylor_scores

    Returns:
        Dictionary mapping modules to their mean sensitivity scores
    """
    layer_sensitivities = {}
    for module, module_scores in scores.items():
        if isinstance(module, nn.Conv2d):
            layer_sensitivities[module] = module_scores.mean().item()
    return layer_sensitivities


def identify_skip_connection_layers(
    model: nn.Module, model_name: str
) -> List[nn.Module]:
    """
    Identify layers that are part of skip connections.

    Args:
        model: Model to analyze
        model_name: Name of the model (used for architecture-specific logic)

    Returns:
        List of modules that are part of skip connections
    """
    skip_layers = []

    # Architecture-specific logic for identifying skip connections
    if any(arch in model_name for arch in ["resnet", "FPN", "UNET"]):
        # For ResNet-based models, typically all conv layers are part of skip connections
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                skip_layers.append(module)

    return skip_layers
