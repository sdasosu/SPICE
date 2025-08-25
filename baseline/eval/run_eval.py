#!/usr/bin/env python3
"""
Script to evaluate baseline models (.pt files) and compute mIOU
Usage: python baseline/eval/run_eval.py
"""

import sys
from pathlib import Path

from tqdm import tqdm

import wandb

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch

from baseline.eval import evaluate_model
from baseline.eval.config import (
    BATCH_SIZE,
    IMG_SIZE,
    MODEL_CONFIGS,
    NUM_CLASSES,
    NUM_WORKERS,
)
from baseline.eval.model_factory import get_model_by_name
from data.data import get_loaders


def main():
    wandb.init(project="epic-v2", name="baseline-evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, test_loader = get_loaders(
        data_root=str(ROOT / "data"),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_size=IMG_SIZE,
    )

    eval_loader = test_loader if test_loader is not None else val_loader

    models_to_eval = list(MODEL_CONFIGS.keys())

    results = []

    for model_name in tqdm(models_to_eval, desc="Models"):
        model, checkpoint_path = get_model_by_name(model_name)
        full_path = ROOT / checkpoint_path

        if full_path.exists():
            metrics = evaluate_model(
                model, str(full_path), eval_loader, device, NUM_CLASSES
            )
            if metrics:
                results.append({"model": model_name, **metrics})

                wandb.log(
                    {
                        f"{model_name}/mIoU": metrics["miou"],
                        f"{model_name}/mean_acc": metrics["mean_acc"],
                    }
                )

                for i, (iou, acc) in enumerate(
                    zip(metrics["per_class_iou"], metrics["per_class_acc"])
                ):
                    wandb.log(
                        {
                            f"{model_name}/class_{i}_iou": iou,
                            f"{model_name}/class_{i}_acc": acc,
                        }
                    )

    table_data = []
    for r in results:
        table_data.append([r["model"], r["miou"], r["mean_acc"]])

    table = wandb.Table(columns=["Model", "mIoU", "Mean Acc"], data=table_data)
    wandb.log({"evaluation_summary": table})

    wandb.log(
        {
            "miou_comparison": wandb.plot.bar(
                wandb.Table(
                    data=[[r["model"], r["miou"]] for r in results],
                    columns=["Model", "mIoU"],
                ),
                "Model",
                "mIoU",
                title="mIoU Comparison",
            ),
            "accuracy_comparison": wandb.plot.bar(
                wandb.Table(
                    data=[[r["model"], r["mean_acc"]] for r in results],
                    columns=["Model", "Mean Accuracy"],
                ),
                "Model",
                "Mean Accuracy",
                title="Mean Accuracy Comparison",
            ),
        }
    )

    print("\n" + "=" * 50)
    print(f"{'Model':<30} {'mIoU':<10} {'Mean Acc':<10}")
    print("-" * 50)

    for r in results:
        print(f"{r['model']:<30} {r['miou']:<10.4f} {r['mean_acc']:<10.4f}")
    print("=" * 50)

    wandb.finish()


if __name__ == "__main__":
    main()
