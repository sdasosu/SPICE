"""WandB tracking module for structured pruning"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

import wandb

from .wandb_config import ErrorHandler, WandBConfigManager
from .wandb_constants import WandBConstants
from .wandb_metrics import MetricLogger, SensitivityLogger
from .wandb_results import ArtifactManager, ComparisonLogger, FinalResultLogger

logger = logging.getLogger(__name__)


class WandBTracker:
    def __init__(
        self,
        config: Any,
        project: str = WandBConstants.DEFAULT_PROJECT,
        entity: Optional[str] = WandBConstants.DEFAULT_ENTITY,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        enabled: bool = True,
    ):
        try:
            self.config_manager = WandBConfigManager(
                config=config,
                project=project,
                entity=entity,
                name=name,
                tags=tags,
                enabled=enabled,
            )
            self.enabled = self.config_manager.enabled
        except Exception as e:
            logger.error(f"Failed to initialize configuration: {e}")
            self.enabled = False
            return

        self.run = None
        self.metric_logger = MetricLogger(enabled=self.enabled)
        self.sensitivity_logger = SensitivityLogger(enabled=self.enabled)

        if not self.enabled:
            logger.info(WandBConstants.WARNING_WANDB_DISABLED)
            return

        self._initialize_wandb()

    def _initialize_wandb(self) -> None:
        try:
            init_params = self.config_manager.get_init_params()
            self.run = wandb.init(**init_params)
            ErrorHandler.log_success(
                WandBConstants.INFO_WANDB_INITIALIZED, self.run.url
            )
        except Exception as e:
            self.enabled = ErrorHandler.handle_init_error(e)

    def log_initial_model_info(self, model_info: Dict[str, Any]) -> None:
        self.metric_logger.log_initial_model_info(model_info)

    def log_pruning_step(
        self,
        step: int,
        model_stats: Dict[str, Any],
        train_loss: float = 0.0,
        val_loss: float = 0.0,
    ) -> None:
        self.metric_logger.log_pruning_step(
            step=step,
            model_stats=model_stats,
            iterative_steps=self.config_manager.config.iterative_steps,
            train_loss=train_loss,
            val_loss=val_loss,
        )

    def log_training_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        step_info: Optional[str] = None,
    ) -> None:
        self.metric_logger.log_training_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=learning_rate,
            step_info=step_info,
        )

    def log_batch_metrics(
        self,
        batch_idx: int,
        loss: float,
        kd_loss: Optional[float] = None,
        ce_loss: Optional[float] = None,
    ) -> None:
        self.metric_logger.log_batch_metrics(
            batch_idx=batch_idx, loss=loss, kd_loss=kd_loss, ce_loss=ce_loss
        )

    def log_sensitivity_analysis(
        self, sensitivity_report: Dict[str, Any], layer_quotas: Optional[Dict] = None
    ) -> None:
        self.sensitivity_logger.log_sensitivity_analysis(
            sensitivity_report=sensitivity_report,
            layer_quotas=layer_quotas,
            global_step=self.metric_logger.global_step,
        )

    def log_layer_pruning_details(self, layer_details: List[Dict]) -> None:
        self.sensitivity_logger.log_layer_pruning_details(layer_details)

    def log_kd_metrics(
        self,
        kd_loss: float,
        ce_loss: float,
        temperature: float,
        alpha: float,
        data_ratio: float,
    ) -> None:
        self.metric_logger.log_kd_metrics(
            kd_loss=kd_loss,
            ce_loss=ce_loss,
            temperature=temperature,
            alpha=alpha,
            data_ratio=data_ratio,
        )

    def log_evaluation_metrics(
        self, metrics: Dict[str, Any], phase: str = "validation"
    ) -> None:
        self.metric_logger.log_evaluation_metrics(metrics=metrics, phase=phase)

    def log_final_results(self, results: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        final_logger = FinalResultLogger(enabled=self.enabled)
        final_logger.log_final_results(
            results=results,
            config=self.config_manager.config,
            global_step=self.metric_logger.increment_global_step(),
        )

    def save_model_checkpoint(
        self, model_path: str, aliases: Optional[List[str]] = None
    ) -> None:
        if not self.enabled:
            return

        artifact_manager = ArtifactManager(enabled=self.enabled)
        artifact_manager.save_model_checkpoint(
            model_path=model_path,
            model_name=self.config_manager.config.model_name,
            pruning_strategy=self.config_manager.config.pruning_strategy,
            run=self.run,
            aliases=aliases,
        )

    def log_model_graph(self, model: nn.Module, example_inputs: torch.Tensor) -> None:
        if not self.enabled:
            return

        try:
            # Avoid pickling issues in test mode
            if (
                not hasattr(self.config_manager.config, "wandb_run_name")
                or "test" not in self.config_manager.config.wandb_run_name
            ):
                wandb.watch(model, log="all", log_freq=100)
                ErrorHandler.log_success(WandBConstants.INFO_MODEL_GRAPH_LOGGED)
        except Exception as e:
            ErrorHandler.handle_graph_log_error(e)

    def create_comparison_chart(
        self,
        before_stats: Dict,
        after_stats: Dict,
        title: str = "Before vs After Pruning",
    ) -> None:
        if not self.enabled:
            return

        comparison_logger = ComparisonLogger(enabled=self.enabled)
        comparison_logger.create_comparison_chart(
            before_stats=before_stats,
            after_stats=after_stats,
            title=title,
            global_step=self.metric_logger.increment_global_step(),
            iterative_steps=self.config_manager.config.iterative_steps,
        )

    def finish(self) -> None:
        if self.enabled and self.run:
            wandb.finish()
            ErrorHandler.log_success(WandBConstants.INFO_WANDB_FINISHED)
