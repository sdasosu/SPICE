"""
Evaluation module for baseline segmentation models
"""

from .inference import evaluate_model, inference
from .metrics import compute_metrics_from_confusion, update_confusion_matrix

__all__ = [
    "compute_metrics_from_confusion",
    "update_confusion_matrix",
    "inference",
    "evaluate_model",
]
