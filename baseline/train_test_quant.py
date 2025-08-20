# quantization/train_test_quant.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional

import time
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

def _update_confusion_matrix(conf: torch.Tensor, preds: torch.Tensor, target: torch.Tensor, num_classes: int) -> None:
    p = preds.view(-1); t = target.view(-1)
    valid = (t >= 0) & (t < num_classes)
    p = p[valid]; t = t[valid]
    idx = t * num_classes + p
    binc = torch.bincount(idx, minlength=num_classes * num_classes)
    conf += binc.view(num_classes, num_classes)

def _metrics_from_confusion(conf: torch.Tensor):
    conf = conf.float()
    diag = torch.diag(conf)
    row_sum = conf.sum(dim=1)
    col_sum = conf.sum(dim=0)
    union = row_sum + col_sum - diag
    iou = diag / (union + 1e-10)
    acc = diag / (row_sum + 1e-10)
    return iou.cpu().numpy(), acc.cpu().numpy(), float(iou.mean().item())

class SegLightningModuleTimed(pl.LightningModule):
    def __init__(self, model: nn.Module, criterian: nn.Module, num_classes: int, model_name: str,
                 save_dir: Path, external_optimizer, external_scheduler=None, tag: str = "RUN"):
        super().__init__()
        self.model = model; self.criterian = criterian; self.num_classes = num_classes
        self.model_name = model_name; self.save_dir = Path(save_dir)
        self.external_optimizer = external_optimizer; self.external_scheduler = external_scheduler
        self.tag = tag
        self.register_buffer("train_conf", torch.zeros(num_classes, num_classes, dtype=torch.long))
        self.register_buffer("val_conf",   torch.zeros(num_classes, num_classes, dtype=torch.long))
        self.best_val_miou = -1.0; self.best_epoch = -1; self.best_epoch_acc = 0.0
        self._epoch_start_time = 0.0

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            out = out.get("out", out.get("logits", None))
            if out is None: raise RuntimeError("Model forward returned dict without 'out'/'logits'.")
        return out

    def configure_optimizers(self):
        if self.external_scheduler is None: return self.external_optimizer
        return {"optimizer": self.external_optimizer,
                "lr_scheduler": {"scheduler": self.external_scheduler, "interval": "epoch"}}

    def on_train_epoch_start(self):
        self._epoch_start_time = time.time()
        print(f"\n--- [TRAIN START] Epoch {self.current_epoch} | tag={self.tag} ---\n")

    def training_step(self, batch, batch_idx):
        images, target = (batch[0], batch[1]) if isinstance(batch,(list,tuple)) else (batch["image"], batch["mask"])
        logits = self.forward(images); loss = self.criterian(logits, target)
        with torch.no_grad():
            preds = logits.argmax(dim=1); _update_confusion_matrix(self.train_conf, preds, target, self.num_classes)
        self.log("train_loss", loss, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        conf_cpu = self.train_conf.detach().cpu()
        iou, acc, miou = _metrics_from_confusion(conf_cpu)
        self.train_conf.zero_()
        dt = time.time() - self._epoch_start_time
        print(f"\n[Epoch {self.current_epoch}] TRAIN mIoU={miou:.4f} | AvgAcc={float(np.mean(acc)):.4f} | time={dt:.2f}s")
        print(f"  per-class IoU: {np.array2string(iou, precision=4)}")
        print(f"  per-class Acc: {np.array2string(acc, precision=4)}\n")

    def validation_step(self, batch, batch_idx):
        images, target = (batch[0], batch[1]) if isinstance(batch,(list,tuple)) else (batch["image"], batch["mask"])
        logits = self.forward(images); loss = self.criterian(logits, target)
        preds = logits.argmax(dim=1); _update_confusion_matrix(self.val_conf, preds, target, self.num_classes)
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        print(f"\n--- [VAL END] Epoch {self.current_epoch} | tag={self.tag} ---")
        conf_cpu = self.val_conf.detach().cpu()
        iou, acc, miou = _metrics_from_confusion(conf_cpu)
        avg_acc = float(np.mean(acc)); self.val_conf.zero_()
        self.log("val_mIoU", miou, prog_bar=False)
        print(f"[Epoch {self.current_epoch}] VAL  mIoU={miou:.4f} | AvgAcc={avg_acc:.4f}")
        print(f"  per-class IoU: {np.array2string(iou, precision=4)}")
        print(f"  per-class Acc: {np.array2string(acc, precision=4)}\n")
        if miou > self.best_val_miou:
            self.best_val_miou = miou; self.best_epoch = int(self.current_epoch); self.best_epoch_acc = avg_acc
            out_dir = self.save_dir / "checkpoints" / "fullsize" / self.model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            target_path = out_dir / f"{self.model_name}.pt"
            torch.save(self.model.state_dict(), target_path)
            print(f"[INFO] Saved BEST (mIoU={miou:.4f}) to {target_path}\n")

def train_validation_timed(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterian: nn.Module,
                           optimizer, scheduler=None, num_epoch: int=10, device: torch.device=torch.device("cuda"),
                           seed: int=42, num_classes: int=5, model_name: str="model", tag: str="RUN"):
    pl.seed_everything(seed, workers=True)
    save_root = Path(__file__).resolve().parent.parent
    lit_module = SegLightningModuleTimed(model, criterian, num_classes, model_name, save_dir=save_root,
                                         external_optimizer=optimizer, external_scheduler=scheduler, tag=tag)
    ckpt_cb = ModelCheckpoint(dirpath=str(save_root / "checkpoints" / "fullsize" / model_name),
                              filename=f"{model_name}_lightning", monitor="val_mIoU", mode="max",
                              save_top_k=1, save_weights_only=True)
    print(f"\n=== [TRAINER START] tag={tag} | epochs={num_epoch} ===\n")
    trainer = pl.Trainer(max_epochs=num_epoch, accelerator="gpu" if device.type=="cuda" else "cpu",
                         devices=1, precision=32, log_every_n_steps=50, enable_progress_bar=False,
                         callbacks=[ckpt_cb])
    trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"\n=== [TRAINER END] tag={tag} ===\n")
    return {"best_miou": float(lit_module.best_val_miou),
            "best_acc": float(lit_module.best_epoch_acc),
            "best_epoch": int(lit_module.best_epoch)}

@torch.no_grad()
def evaluate_one_epoch_val(model: nn.Module, val_loader: DataLoader, device: torch.device, num_classes: int = 5):
    print("\n--- [EVAL] PTQ int8 validation (1 epoch aggregate) ---\n")
    model.eval().to(device)
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    for batch in val_loader:
        images, target = (batch[0].to(device), batch[1].to(device).long()) if isinstance(batch,(list,tuple)) \
                         else (batch["image"].to(device), batch["mask"].to(device).long())
        out = model(images)
        if isinstance(out, dict):
            out = out.get("out", out.get("logits", None))
            if out is None: raise RuntimeError("Model forward returned dict without 'out'/'logits'.")
        preds = out.argmax(dim=1)
        _update_confusion_matrix(conf, preds, target, num_classes)
    iou, acc, miou = _metrics_from_confusion(conf)
    print(f"[EVAL RESULT] mIoU={miou:.4f} | AvgAcc={float(np.mean(acc)):.4f}\n")
    return float(miou), float(np.mean(acc))
