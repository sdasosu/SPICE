"""
Learning rate schedulers for pruning fine-tuning.
Uses PyTorch's built-in schedulers with SequentialLR for warmup+cosine combination.
"""

import logging

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)

logger = logging.getLogger(__name__)


def create_scheduler(config, optimizer, num_epochs):
    """
    Factory function to create appropriate scheduler based on config.
    Uses PyTorch's built-in schedulers including SequentialLR for combinations.

    Args:
        config: PruningConfig object
        optimizer: The optimizer to schedule
        num_epochs: Total number of training epochs

    Returns:
        Learning rate scheduler instance
    """
    if config.use_cosine_annealing:
        min_lr = config.fine_tune_lr * config.lr_min_factor

        if config.lr_warmup_epochs > 0 and num_epochs > config.lr_warmup_epochs:
            warmup_epochs = min(config.lr_warmup_epochs, num_epochs // 4)

            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer, T_max=num_epochs - warmup_epochs, eta_min=min_lr
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )

            logger.info(
                f"Using PyTorch SequentialLR: LinearLR warmup ({warmup_epochs} epochs) "
                f"+ CosineAnnealingLR (min_lr={min_lr:.6f})"
            )
            scheduler.warmup_epochs = warmup_epochs

        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)
            logger.info(
                f"Using PyTorch CosineAnnealingLR (no warmup) with min_lr={min_lr:.6f}"
            )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=config.fine_tune_lr * config.lr_min_factor,
        )
        logger.info("Using PyTorch ReduceLROnPlateau scheduler")

    return scheduler
