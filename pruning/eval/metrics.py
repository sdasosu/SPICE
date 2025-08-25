"""
Metrics calculation for segmentation evaluation using torchmetrics
"""

from typing import Any, Dict

import torch
import torchmetrics


class SegmentationMetrics:
    """Calculate segmentation metrics using torchmetrics"""

    def __init__(self, num_classes: int = 5, device: str = "cpu"):
        self.num_classes = num_classes
        self.device = device

        # Use torchmetrics built-in metrics
        self.mean_iou = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device)

        self.per_class_iou = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=num_classes, average="none"
        ).to(device)

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device)

        self.per_class_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        ).to(device)

        self.confusion_matrix = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        ).to(device)

    def reset(self):
        """Reset all metrics"""
        self.mean_iou.reset()
        self.per_class_iou.reset()
        self.accuracy.reset()
        self.per_class_accuracy.reset()
        self.confusion_matrix.reset()

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with batch predictions

        Args:
            predictions: Predicted class indices [B, H, W]
            targets: Ground truth class indices [B, H, W]
        """
        # Flatten spatial dimensions
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Filter out invalid labels if needed
        mask = (targets >= 0) & (targets < self.num_classes)
        if not mask.all():
            predictions = predictions[mask]
            targets = targets[mask]

        # Update all metrics
        self.mean_iou.update(predictions, targets)
        self.per_class_iou.update(predictions, targets)
        self.accuracy.update(predictions, targets)
        self.per_class_accuracy.update(predictions, targets)
        self.confusion_matrix.update(predictions, targets)

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute all metrics

        Returns:
            Dictionary containing:
            - miou: Mean IoU
            - mean_acc: Mean accuracy
            - per_class_iou: IoU for each class
            - per_class_acc: Accuracy for each class
            - confusion_matrix: Confusion matrix
        """
        # Compute metrics using torchmetrics
        miou = self.mean_iou.compute()
        per_class_iou = self.per_class_iou.compute()
        mean_acc = self.accuracy.compute()
        per_class_acc = self.per_class_accuracy.compute()
        conf_matrix = self.confusion_matrix.compute()

        return {
            "miou": miou.item() if miou.numel() == 1 else miou.cpu().numpy(),
            "mean_acc": mean_acc.item()
            if mean_acc.numel() == 1
            else mean_acc.cpu().numpy(),
            "per_class_iou": per_class_iou.cpu().numpy().tolist(),
            "per_class_acc": per_class_acc.cpu().numpy().tolist(),
            "confusion_matrix": conf_matrix.cpu().numpy(),
        }

    def get_class_names(self) -> list:
        """Get class names for visualization"""
        return [f"Class_{i}" for i in range(self.num_classes)]


def compute_metrics(confusion_matrix: torch.Tensor) -> Dict[str, Any]:
    """
    Compute metrics from confusion matrix

    Args:
        confusion_matrix: Confusion matrix [num_classes, num_classes]

    Returns:
        Dictionary with metrics
    """
    if not isinstance(confusion_matrix, torch.Tensor):
        confusion_matrix = torch.tensor(confusion_matrix)

    num_classes = confusion_matrix.shape[0]
    per_class_iou = []
    per_class_acc = []

    for c in range(num_classes):
        intersection = confusion_matrix[c, c].float()
        gt_sum = confusion_matrix[c, :].sum().float()
        pred_sum = confusion_matrix[:, c].sum().float()
        union = gt_sum + pred_sum - intersection

        # Calculate IoU
        iou = intersection / (union + 1e-10)
        per_class_iou.append(iou.item())

        # Calculate accuracy
        acc = intersection / (gt_sum + 1e-10)
        per_class_acc.append(acc.item())

    import numpy as np

    return {
        "miou": np.mean(per_class_iou),
        "mean_acc": np.mean(per_class_acc),
        "per_class_iou": per_class_iou,
        "per_class_acc": per_class_acc,
    }
