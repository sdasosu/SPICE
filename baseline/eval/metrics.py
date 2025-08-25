"""
Metrics computation utilities for segmentation evaluation
"""

import torch


def update_confusion_matrix(
    conf: torch.Tensor, preds: torch.Tensor, target: torch.Tensor, num_classes: int
) -> None:
    """Update confusion matrix with predictions and targets"""
    p = preds.view(-1)
    t = target.view(-1)
    valid = (t >= 0) & (t < num_classes)
    p = p[valid]
    t = t[valid]
    idx = t * num_classes + p
    binc = torch.bincount(idx, minlength=num_classes * num_classes)
    conf += binc.view(num_classes, num_classes)


def compute_metrics_from_confusion(conf: torch.Tensor):
    """
    Compute IoU and accuracy metrics from confusion matrix

    Returns:
        tuple: (per_class_iou, per_class_acc, mean_iou)
    """
    conf = conf.float()
    diag = torch.diag(conf)
    row_sum = conf.sum(dim=1)
    col_sum = conf.sum(dim=0)
    union = row_sum + col_sum - diag
    iou = diag / (union + 1e-10)
    acc = diag / (row_sum + 1e-10)
    return iou.cpu().numpy(), acc.cpu().numpy(), float(iou.mean().item())
