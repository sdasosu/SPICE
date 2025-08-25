"""
Training module for fine-tuning during pruning
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .config import PruningConfig
from .dataset import PruningDataHandler
from .kd_lite import KnowledgeDistiller
from .losses import SegmentationLoss
from .lr_scheduler import create_scheduler
from .utils import generate_config_hash, set_random_seeds

logger = logging.getLogger(__name__)


class PruningTrainer:
    """Handles fine-tuning during pruning process"""

    def __init__(self, config: PruningConfig):
        self.config = config
        self.data_handler = PruningDataHandler(
            data_root=config.data_root,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=config.seed,
        )
        self._setup_seeds()
        config_str = f"{config.model_name}_{config.pruning_strategy}_{config.pruning_ratio}_{config.seed}"
        config_hash = generate_config_hash(config_str)
        pid = os.getpid()
        self.temp_dir = (
            Path(config.output_dir)
            / "temp"
            / f"{config.model_name}_{config_hash}_{pid}"
        )
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.distiller = None
        self.kd_loss_fn = None
        self.wandb_tracker = None
        self.last_train_loss = 0.0
        self.last_val_loss = 0.0

    def _setup_seeds(self):
        set_random_seeds(self.config.seed)

    def set_wandb_tracker(self, tracker):
        self.wandb_tracker = tracker

    def _get_dataloaders(
        self, use_subset: bool = False
    ) -> Tuple[DataLoader, DataLoader]:
        train_loader, val_loader = self.data_handler.get_train_val_loaders()
        if use_subset and self.config.enable_kd_lite:
            train_loader = self._create_kd_subset_loader(train_loader)

        return train_loader, val_loader

    def _create_kd_subset_loader(self, train_loader: DataLoader) -> DataLoader:
        import random

        train_dataset = train_loader.dataset
        total_samples = len(train_dataset)
        subset_size = int(total_samples * self.config.kd_data_ratio)
        rng = random.Random(self.config.seed)
        indices = rng.sample(range(total_samples), subset_size)
        subset = Subset(train_dataset, indices)
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)

        subset_loader = DataLoader(
            subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            generator=generator,
        )

        logger.info(
            f"Using {subset_size}/{total_samples} samples ({self.config.kd_data_ratio * 100:.0f}%) for KD-Lite training"
        )

        return subset_loader

    def _setup_criterion(self) -> nn.Module:
        criterion = SegmentationLoss(self.config.num_classes)
        criterion.to(self.config.device)
        return criterion

    def _setup_optimizer(self, model: nn.Module) -> optim.Optimizer:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        return optim.Adam(
            trainable_params,
            lr=self.config.fine_tune_lr,
            weight_decay=self.config.weight_decay,
        )

    def _setup_scheduler(self, optimizer: optim.Optimizer, num_epochs: int):
        return create_scheduler(self.config, optimizer, num_epochs)

    def setup_kd_training(
        self,
        student_model: nn.Module,
        teacher_model: Optional[nn.Module] = None,
        teacher_path: Optional[str] = None,
    ):
        if not self.config.enable_kd_lite:
            logger.warning("KD-Lite is not enabled in config. Skipping KD setup.")
            return student_model

        logger.info("Setting up KD-Lite training")
        from .models import ModelFactory

        model_factory = ModelFactory()
        self.distiller = KnowledgeDistiller(self.config, model_factory)

        # Setup teacher model
        if teacher_model is not None:
            self.distiller.teacher_model = teacher_model
            self.distiller.teacher_model.to(self.config.device)
            self.distiller.teacher_model.eval()
            for param in self.distiller.teacher_model.parameters():
                param.requires_grad = False
        else:
            self.distiller.setup_teacher(teacher_path)

        # Setup student model
        student_model = self.distiller.setup_student(student_model)

        # Store KD loss function
        self.kd_loss_fn = self.distiller.loss_fn

        return student_model

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> float:
        model.train()
        if self.config.enable_kd_lite and self.config.freeze_backbone:
            if hasattr(model, "encoder"):
                model.encoder.eval()

        total_loss = 0.0
        num_batches = 0
        kd_loss_sum = 0.0
        ce_loss_sum = 0.0
        batch_log_freq = (
            self.config.wandb_log_frequency
            if hasattr(self.config, "wandb_log_frequency")
            else 10
        )
        accumulation_steps = max(1, self.config.batch_size // 32)

        for batch_idx, (images, masks) in enumerate(
            tqdm(train_loader, desc="Training", leave=False)
        ):
            images = images.to(self.config.device)
            masks = masks.to(self.config.device)
            if self.config.enable_kd_lite and self.distiller is not None:
                optimizer.zero_grad()
                student_outputs = model(images)
                with torch.no_grad():
                    teacher_outputs = self.distiller.teacher_model(images)
                loss_dict = self.kd_loss_fn(student_outputs, teacher_outputs, masks)
                loss = loss_dict["total"]

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                kd_loss_sum += loss_dict["kd"].item()
                ce_loss_sum += loss_dict["ce"].item()
                if self.wandb_tracker and batch_idx % batch_log_freq == 0:
                    self.wandb_tracker.log_batch_metrics(
                        batch_idx=batch_idx,
                        loss=loss.item(),
                        kd_loss=loss_dict["kd"].item(),
                        ce_loss=loss_dict["ce"].item(),
                    )

            else:
                if accumulation_steps > 1:
                    chunk_size = images.size(0) // accumulation_steps
                    for i in range(accumulation_steps):
                        start_idx = i * chunk_size
                        end_idx = (
                            start_idx + chunk_size
                            if i < accumulation_steps - 1
                            else images.size(0)
                        )

                        images_chunk = images[start_idx:end_idx]
                        masks_chunk = masks[start_idx:end_idx]

                        outputs = model(images_chunk)
                        loss = criterion(outputs, masks_chunk) / accumulation_steps
                        loss.backward()

                        total_loss += loss.item() * accumulation_steps
                        del outputs, loss
                        torch.cuda.empty_cache()
                else:
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    total_loss += loss.item()

                optimizer.step()
                optimizer.zero_grad()
                if self.wandb_tracker and batch_idx % batch_log_freq == 0:
                    self.wandb_tracker.log_batch_metrics(
                        batch_idx=batch_idx,
                        loss=total_loss / (num_batches + 1),
                    )

            num_batches += 1

        # Log KD-specific losses if applicable
        if self.config.enable_kd_lite and num_batches > 0:
            avg_kd_loss = kd_loss_sum / num_batches
            avg_ce_loss = ce_loss_sum / num_batches
            logger.debug(f"KD Loss: {avg_kd_loss:.4f}, CE Loss: {avg_ce_loss:.4f}")

            # Log KD metrics to WandB
            if self.wandb_tracker:
                self.wandb_tracker.log_kd_metrics(
                    kd_loss=avg_kd_loss,
                    ce_loss=avg_ce_loss,
                    temperature=self.config.kd_temperature,
                    alpha=self.config.kd_alpha,
                    data_ratio=self.config.kd_data_ratio,
                )

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate_epoch(
        self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module
    ) -> float:
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation", leave=False):
                images = images.to(self.config.device)
                masks = masks.to(self.config.device)

                outputs = model(images)
                if self.config.enable_kd_lite and self.distiller is not None:
                    teacher_outputs = self.distiller.teacher_model(images)
                    loss_dict = self.kd_loss_fn(outputs, teacher_outputs, masks)
                    loss = loss_dict["ce"]
                else:
                    loss = criterion(outputs, masks)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def fine_tune(
        self,
        model: nn.Module,
        num_epochs: int,
        step_info: Optional[str] = None,
        teacher_model: Optional[nn.Module] = None,
    ) -> nn.Module:
        self._setup_seeds()

        if self.config.enable_kd_lite:
            logger.info(
                f"Starting KD-Lite fine-tuning for {num_epochs} epochs"
                + (f" ({step_info})" if step_info else "")
            )
            if self.distiller is None:
                model = self.setup_kd_training(model, teacher_model)
        else:
            logger.info(
                f"Starting fine-tuning for {num_epochs} epochs"
                + (f" ({step_info})" if step_info else "")
            )

        model.to(self.config.device)
        model.train()
        use_subset = self.config.enable_kd_lite
        train_loader, val_loader = self._get_dataloaders(use_subset=use_subset)
        criterion = self._setup_criterion()
        optimizer = self._setup_optimizer(model)
        scheduler = self._setup_scheduler(optimizer, num_epochs)
        best_val_loss = float("inf")
        early_stop_counter = 0
        best_model_path = self.temp_dir / f"best_model_{self.config.model_name}.pt"
        for epoch in range(num_epochs):
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss = self._validate_epoch(model, val_loader, criterion)
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            self.last_train_loss = train_loss
            self.last_val_loss = val_loss
            if self.wandb_tracker:
                self.wandb_tracker.log_training_epoch(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=current_lr,
                    step_info=step_info,
                )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), best_model_path)
                logger.debug(
                    f"Saved best model at epoch {epoch + 1} (val_loss: {val_loss:.4f})"
                )

            else:
                early_stop_counter += 1
                if early_stop_counter >= self.config.early_stop_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        if best_model_path.exists():
            model.load_state_dict(
                torch.load(best_model_path, map_location=self.config.device)
            )
            logger.info(f"Restored best model with val_loss: {best_val_loss:.4f}")

        return model

    def cleanup(self):
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.debug("Cleaned up temporary files")


class PruningMetrics:
    """Track metrics during pruning process"""

    def __init__(self):
        self.history = []

    def add_step(
        self, step: int, train_loss: float, val_loss: float, params: int, macs: int
    ):
        """Add metrics for a pruning step"""
        self.history.append(
            {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "params": params,
                "macs": macs,
                "params_million": params / 1e6,
                "macs_million": macs / 1e6,
            }
        )

    def save_history(self, filepath: str):
        """Save metrics history to file"""
        import json

        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved metrics history to {filepath}")

    def get_final_metrics(self) -> dict:
        """Get final pruning metrics"""
        if not self.history:
            return {}

        final = self.history[-1]
        initial = self.history[0] if len(self.history) > 1 else final

        return {
            "initial_params": initial["params"],
            "final_params": final["params"],
            "params_reduction": (initial["params"] - final["params"])
            / initial["params"],
            "initial_macs": initial["macs"],
            "final_macs": final["macs"],
            "macs_reduction": (initial["macs"] - final["macs"]) / initial["macs"]
            if initial["macs"] > 0
            else 0,
            "final_val_loss": final["val_loss"],
        }
