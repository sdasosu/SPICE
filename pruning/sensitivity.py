"""
Taylor-based sensitivity analysis for structured pruning
Implementation based on Molchanov et al. weight-based Taylor approximation
"""

import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp

from .sensitivity_utils import (
    compute_taylor_scores,
    get_layer_sensitivities,
    get_sensitivity_report,
)

logger = logging.getLogger(__name__)


class TaylorWeightImportance(tp.importance.Importance):
    """
    Weight-based Taylor sensitivity importance for structured pruning.
    Computes importance as |gradient * weight| following first-order Taylor approximation.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader,
        device: str,
        max_batches: int = 100,
        min_channels: int = 8,
    ):
        """
        Initialize Taylor weight importance calculator.

        Args:
            model: Model to analyze
            dataloader: Calibration data loader
            device: Device for computation
            max_batches: Maximum calibration batches
            min_channels: Minimum channels to keep per layer
        """
        self.model = model
        self.device = device
        self.min_channels = min_channels

        # Precompute sensitivity scores for all layers
        logger.info("Computing Taylor weight sensitivity scores...")
        self.sensitivity_scores = self._compute_sensitivity(
            model, dataloader, device, max_batches
        )

        logger.info(f"Computed sensitivity for {len(self.sensitivity_scores)} layers")

    def _compute_sensitivity(
        self, model: nn.Module, dataloader, device: str, max_batches: int
    ) -> Dict[nn.Module, torch.Tensor]:
        """
        Compute weight-based Taylor sensitivity scores.

        Returns:
            Dictionary mapping modules to their per-channel importance scores
        """
        return compute_taylor_scores(model, dataloader, device, max_batches)

    def __call__(self, group: tp.dependency.Group) -> torch.Tensor:
        """
        Compute importance scores for a pruning group.

        Args:
            group: Pruning group from DepGraph

        Returns:
            1-D tensor of importance scores (lower = less important)
        """
        if not group:
            return torch.tensor([])

        # Find the primary module in the group (usually the first Conv2d)
        primary_module = None
        for dep, idxs in group:
            if isinstance(dep.target.module, nn.Conv2d):
                primary_module = dep.target.module
                break

        if primary_module is None:
            # If no Conv2d found, return uniform importance
            _, idxs = group[0]
            return torch.ones(len(idxs))

        # Get precomputed sensitivity scores
        if primary_module in self.sensitivity_scores:
            scores = self.sensitivity_scores[primary_module]

            # Handle index mapping if needed
            _, idxs = group[0]
            if len(idxs) < len(scores):
                scores = scores[torch.tensor(idxs)]

            # IMPORTANT: torch-pruning expects:
            # Low score = low importance = prune first
            # High score = high importance = keep
            # So we return scores directly (high sensitivity = keep)
            return scores
        else:
            # Fallback to uniform importance
            _, idxs = group[0]
            return torch.ones(len(idxs))

    def get_sensitivity_report(self) -> Dict:
        """
        Generate sensitivity analysis report for logging/debugging.

        Returns:
            Dictionary with sensitivity statistics
        """
        return get_sensitivity_report(self.sensitivity_scores)

    def get_layer_sensitivities(self) -> Dict[nn.Module, float]:
        """
        Get mean sensitivity score for each layer.

        Returns:
            Dictionary mapping modules to their mean sensitivity scores
        """
        return get_layer_sensitivities(self.sensitivity_scores)


class GroupSyncTaylorImportance(TaylorWeightImportance):
    """
    Enhanced Taylor importance with group synchronization for skip connections.
    Ensures consistent pruning across residual branches and concatenations.
    """

    def __call__(self, group: tp.dependency.Group) -> torch.Tensor:
        """
        Compute group-synchronized importance scores.

        Aggregates scores across all modules in the group to ensure
        consistent pruning decisions for skip connections.
        """
        if not group:
            return torch.tensor([])

        # Collect scores from all modules in the group
        group_scores = []
        num_channels = None

        for dep, idxs in group:
            module = dep.target.module

            if num_channels is None:
                num_channels = len(idxs)
            if module in self.sensitivity_scores:
                scores = self.sensitivity_scores[module]

                if len(scores) >= num_channels:
                    scores = scores[:num_channels]
                else:
                    scores = F.pad(scores, (0, num_channels - len(scores)), value=1.0)
                group_scores.append(scores)

        if not group_scores:
            return torch.ones(num_channels)
        group_scores = torch.stack(group_scores, dim=0)
        aggregated_scores = group_scores.mean(dim=0)
        return aggregated_scores

    def get_layer_sensitivities(self) -> Dict[nn.Module, float]:
        """
        Get mean sensitivity score for each layer with group synchronization.

        Returns:
            Dictionary mapping modules to their mean sensitivity scores
        """
        layer_sensitivities = super().get_layer_sensitivities()
        return layer_sensitivities


def create_taylor_importance(
    model: nn.Module, dataloader, config
) -> TaylorWeightImportance:
    """
    Factory function to create Taylor importance calculator.

    Args:
        model: Model to analyze
        dataloader: Calibration dataloader
        config: Pruning configuration

    Returns:
        Configured TaylorWeightImportance instance
    """
    # Use group sync version for architectures with skip connections
    if any(arch in config.model_name for arch in ["resnet", "FPN", "UNET"]):
        logger.info("Using group-synchronized Taylor importance")
        return GroupSyncTaylorImportance(
            model=model,
            dataloader=dataloader,
            device=config.device,
            max_batches=getattr(config, "calibration_batches", 100),
            min_channels=getattr(config, "min_out_channels", 8),
        )
    else:
        return TaylorWeightImportance(
            model=model,
            dataloader=dataloader,
            device=config.device,
            max_batches=getattr(config, "calibration_batches", 100),
            min_channels=getattr(config, "min_out_channels", 8),
        )
