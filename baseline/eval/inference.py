"""
Inference utilities for model evaluation
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_metrics_from_confusion, update_confusion_matrix


@torch.no_grad()
def inference(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int = 5,
):
    """
    Run inference on test dataset and compute metrics
    Compatible with baseline/run_quant.py usage
    """
    if hasattr(model, "eval"):
        model.eval()
    model = model.to(device)

    conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device, non_blocking=True)
                masks = batch[1].to(device, non_blocking=True).long()
            else:
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True).long()

            outputs = model(images)

            if isinstance(outputs, dict):
                outputs = outputs.get("out", outputs.get("logits", outputs))

            preds = outputs.argmax(dim=1)
            update_confusion_matrix(conf, preds, masks, num_classes)

    iou, acc, miou = compute_metrics_from_confusion(conf)

    return {
        "miou": miou,
        "mean_acc": float(np.mean(acc)),
        "per_class_iou": iou.tolist(),
        "per_class_acc": acc.tolist(),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    model_path: str,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int = 5,
):
    """Load model weights and evaluate"""
    if Path(model_path).exists():
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        return None

    return inference(model, data_loader, device, num_classes)
