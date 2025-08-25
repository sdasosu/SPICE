"""
Loss functions for knowledge distillation and training
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PixelKDLoss(nn.Module):
    """Pixel-level knowledge distillation loss using KL divergence"""

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate pixel-level KD loss

        Args:
            student_logits: Student model output (B, C, H, W)
            teacher_logits: Teacher model output (B, C, H, W)
            mask: Optional mask for valid pixels (B, H, W)

        Returns:
            KD loss value
        """
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        kl_div = F.kl_div(student_soft, teacher_soft, reduction="none")
        kl_div = kl_div.sum(dim=1)

        if mask is not None:
            kl_div = kl_div * mask.float()
            n_valid = mask.sum()
            if n_valid > 0:
                loss = kl_div.sum() / n_valid
            else:
                loss = kl_div.sum() * 0
        else:
            loss = kl_div.mean()
        loss = loss * (self.temperature**2)

        return loss


class SegmentationLoss(nn.Module):
    """Standard segmentation loss with class weighting"""

    def __init__(self, num_classes: int = 5, ignore_index: int = 255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        if num_classes == 5:
            class_weights = torch.tensor([0.5, 1.0, 1.1, 1.5, 2.0], dtype=torch.float)
            self.ce_loss = nn.CrossEntropyLoss(
                weight=class_weights, ignore_index=ignore_index
            )
        else:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate cross-entropy loss

        Args:
            predictions: Model predictions (B, C, H, W)
            targets: Ground truth targets (B, H, W)

        Returns:
            Cross-entropy loss
        """
        return self.ce_loss(predictions, targets)


class KDLiteLoss(nn.Module):
    """Combined loss for KD-Lite: CE + KD"""

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        ignore_index: int = 255,
        boundary_weight: float = 1.0,
        confidence_weight: bool = False,
        num_classes: int = 5,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.boundary_weight = boundary_weight
        self.confidence_weight = confidence_weight

        self.ce_loss = SegmentationLoss(num_classes, ignore_index)
        self.kd_loss = PixelKDLoss(temperature)

    def _get_boundary_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """Extract boundary pixels using morphological operations"""
        kernel_size = 3
        padding = kernel_size // 2

        labels_float = labels.float().unsqueeze(1)
        dilated = F.max_pool2d(labels_float, kernel_size, stride=1, padding=padding)
        eroded = -F.max_pool2d(-labels_float, kernel_size, stride=1, padding=padding)
        boundary = (dilated != eroded).squeeze(1).float()

        return boundary

    def _get_confidence_weights(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Get confidence-based weights from teacher predictions"""
        teacher_probs = F.softmax(teacher_logits, dim=1)
        confidence, _ = teacher_probs.max(dim=1)
        return confidence

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate combined KD-Lite loss

        Args:
            student_logits: Student model output (B, C, H, W)
            teacher_logits: Teacher model output (B, C, H, W)
            labels: Ground truth labels (B, H, W)

        Returns:
            Dictionary with total loss and components
        """
        ce_loss = self.ce_loss(student_logits, labels)
        valid_mask = labels != self.ignore_index
        kd_loss = self.kd_loss(student_logits, teacher_logits, valid_mask)
        if self.boundary_weight > 1.0:
            boundary_mask = self._get_boundary_mask(labels)
            boundary_mask = boundary_mask * valid_mask.float()
            boundary_kd = self.kd_loss(student_logits, teacher_logits, boundary_mask)
            kd_loss = kd_loss + (self.boundary_weight - 1.0) * boundary_kd

        if self.confidence_weight:
            confidence_weights = self._get_confidence_weights(teacher_logits)
            confidence_weights = confidence_weights * valid_mask.float()
            weighted_kd = self.kd_loss(
                student_logits, teacher_logits, confidence_weights
            )
            kd_loss = 0.5 * kd_loss + 0.5 * weighted_kd
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss

        return {"total": total_loss, "ce": ce_loss, "kd": kd_loss}
