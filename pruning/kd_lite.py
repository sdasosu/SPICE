"""
Knowledge Distillation Lite (KD-Lite) module for pruning
Based on pixel-level distillation for semantic segmentation
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from .losses import KDLiteLoss

logger = logging.getLogger(__name__)


class KnowledgeDistiller:
    """Manager for knowledge distillation training"""

    def __init__(self, config, model_factory):
        self.config = config
        self.model_factory = model_factory
        self.teacher_model = None
        self.loss_fn = None

    def setup_teacher(self, teacher_path: Optional[str] = None):
        logger.info("Loading teacher model for knowledge distillation")

        if teacher_path:
            self.teacher_model = torch.load(
                teacher_path, map_location=self.config.device
            )
        else:
            self.teacher_model = self.model_factory.load_pretrained_model(self.config)

        self.teacher_model.to(self.config.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        logger.info("Teacher model loaded and frozen")

    def setup_student(self, student_model: nn.Module) -> nn.Module:
        logger.info("Configuring student model for KD-Lite")

        student_model.to(self.config.device)

        if self.config.freeze_backbone:
            self._freeze_backbone(student_model)
            logger.info("Backbone frozen, decoder and head unfrozen")
        else:
            logger.info("All layers trainable (no freezing)")

        self.loss_fn = KDLiteLoss(
            temperature=self.config.kd_temperature,
            alpha=self.config.kd_alpha,
            boundary_weight=self.config.boundary_weight,
            confidence_weight=self.config.confidence_weight,
            num_classes=self.config.num_classes,
        ).to(self.config.device)

        return student_model

    def _freeze_backbone(self, model: nn.Module):
        if hasattr(model, "encoder"):
            for param in model.encoder.parameters():
                param.requires_grad = False

            for m in model.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

    def compute_loss(
        self, student_outputs: torch.Tensor, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if self.teacher_model is None:
            raise RuntimeError(
                "Teacher model not initialized. Call setup_teacher() first."
            )

        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
        loss_dict = self.loss_fn(student_outputs, teacher_outputs, labels)

        return loss_dict

    def get_teacher_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.teacher_model is None:
            raise RuntimeError(
                "Teacher model not initialized. Call setup_teacher() first."
            )

        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        return teacher_outputs
