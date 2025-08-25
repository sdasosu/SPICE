"""Metric logging utilities for WandB tracking"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

import wandb

from .wandb_constants import MetricCalculator, WandBConstants

logger = logging.getLogger(__name__)


class MetricLogger:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.global_step = 0
        self.pruning_step = 0

        self.initial_params = 0
        self.initial_params_million = 0
        self.initial_size_mb = 0
        self.initial_macs = 0
        self.initial_macs_million = 0
        self.initial_layers = 0

    def increment_global_step(self) -> int:
        self.global_step += 1
        return self.global_step

    def set_pruning_step(self, step: int) -> None:
        self.pruning_step = step

    def log_initial_model_info(self, model_info: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        self.initial_params = model_info["total_params"]
        self.initial_params_million = MetricCalculator.params_to_million(
            self.initial_params
        )
        self.initial_size_mb = model_info["model_size_mb"]
        self.initial_macs = model_info.get("macs", 0)
        self.initial_macs_million = MetricCalculator.macs_to_million(self.initial_macs)
        self.initial_layers = model_info.get("total_layers", 0)

        initial_metrics = self._create_model_metrics(
            params=self.initial_params,
            macs=self.initial_macs,
            layers=self.initial_layers,
            size_mb=self.initial_size_mb,
            compression_ratio=1.0,
            step=0,
            progress=0,
        )

        wandb.log(initial_metrics, step=0)
        self._save_initial_summary()

    def _create_model_metrics(
        self,
        params: int,
        macs: int,
        layers: int,
        size_mb: float,
        compression_ratio: float,
        step: int,
        progress: float,
    ) -> Dict[str, Any]:
        return {
            f"{WandBConstants.PREFIX_MODEL}/{WandBConstants.METRIC_PARAMS}": params,
            f"{WandBConstants.PREFIX_MODEL}/{WandBConstants.METRIC_PARAMS_MILLION}": MetricCalculator.params_to_million(
                params
            ),
            f"{WandBConstants.PREFIX_MODEL}/{WandBConstants.METRIC_SIZE_MB}": size_mb,
            f"{WandBConstants.PREFIX_MODEL}/{WandBConstants.METRIC_MACS}": macs,
            f"{WandBConstants.PREFIX_MODEL}/{WandBConstants.METRIC_MACS_MILLION}": MetricCalculator.macs_to_million(
                macs
            ),
            f"{WandBConstants.PREFIX_MODEL}/{WandBConstants.METRIC_LAYERS}": layers,
            f"{WandBConstants.PREFIX_MODEL}/{WandBConstants.METRIC_COMPRESSION_RATIO}": compression_ratio,
            f"{WandBConstants.PREFIX_PRUNING}/{WandBConstants.METRIC_STEP}": step,
            f"{WandBConstants.PREFIX_PRUNING}/{WandBConstants.METRIC_PROGRESS}": progress,
        }

    def _save_initial_summary(self) -> None:
        wandb.run.summary[
            f"{WandBConstants.PREFIX_MODEL}/initial_{WandBConstants.METRIC_PARAMS}"
        ] = self.initial_params
        wandb.run.summary[
            f"{WandBConstants.PREFIX_MODEL}/initial_{WandBConstants.METRIC_PARAMS_MILLION}"
        ] = self.initial_params_million
        wandb.run.summary[
            f"{WandBConstants.PREFIX_MODEL}/initial_{WandBConstants.METRIC_SIZE_MB}"
        ] = self.initial_size_mb
        wandb.run.summary[
            f"{WandBConstants.PREFIX_MODEL}/initial_{WandBConstants.METRIC_MACS}"
        ] = self.initial_macs
        wandb.run.summary[
            f"{WandBConstants.PREFIX_MODEL}/initial_{WandBConstants.METRIC_MACS_MILLION}"
        ] = self.initial_macs_million
        wandb.run.summary[
            f"{WandBConstants.PREFIX_MODEL}/initial_{WandBConstants.METRIC_LAYERS}"
        ] = self.initial_layers

    def log_pruning_step(
        self,
        step: int,
        model_stats: Dict[str, Any],
        iterative_steps: int,
        train_loss: float = 0.0,
        val_loss: float = 0.0,
    ) -> None:
        if not self.enabled:
            return

        self.set_pruning_step(step)

        metrics = self._create_model_metrics(
            params=model_stats["params"],
            macs=model_stats["macs"],
            layers=model_stats.get("layers", self.initial_layers),
            size_mb=MetricCalculator.params_to_mb(model_stats["params"]),
            compression_ratio=model_stats.get("compression_ratio", 1.0),
            step=step,
            progress=step / iterative_steps,
        )

        if train_loss > 0:
            metrics[f"{WandBConstants.PREFIX_LOSS}/train"] = train_loss
        if val_loss > 0:
            metrics[f"{WandBConstants.PREFIX_LOSS}/validation"] = val_loss

        wandb.log(metrics, step=self.increment_global_step())

    def log_training_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        step_info: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return

        prefix = f"{step_info}/" if step_info else f"{WandBConstants.PREFIX_TRAINING}/"

        metrics = {
            f"{prefix}epoch": epoch,
            f"{prefix}train_loss": train_loss,
            f"{prefix}val_loss": val_loss,
            f"{prefix}learning_rate": learning_rate,
            f"{WandBConstants.PREFIX_PRUNING}/current_iteration": self.pruning_step,
        }

        wandb.log(metrics, step=self.increment_global_step())

    def log_batch_metrics(
        self,
        batch_idx: int,
        loss: float,
        kd_loss: Optional[float] = None,
        ce_loss: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return

        metrics = {
            f"{WandBConstants.PREFIX_BATCH}/loss": loss,
            f"{WandBConstants.PREFIX_BATCH}/idx": batch_idx,
        }

        if kd_loss is not None:
            metrics[f"{WandBConstants.PREFIX_BATCH}/kd_loss"] = kd_loss
        if ce_loss is not None:
            metrics[f"{WandBConstants.PREFIX_BATCH}/ce_loss"] = ce_loss

        wandb.log(metrics, commit=False)

    def log_kd_metrics(
        self,
        kd_loss: float,
        ce_loss: float,
        temperature: float,
        alpha: float,
        data_ratio: float,
    ) -> None:
        if not self.enabled:
            return

        metrics = {
            f"{WandBConstants.PREFIX_KD}/loss": kd_loss,
            f"{WandBConstants.PREFIX_KD}/ce_loss": ce_loss,
            f"{WandBConstants.PREFIX_KD}/temperature": temperature,
            f"{WandBConstants.PREFIX_KD}/alpha": alpha,
            f"{WandBConstants.PREFIX_KD}/data_ratio": data_ratio,
            f"{WandBConstants.PREFIX_KD}/loss_ratio": kd_loss / (ce_loss + 1e-8),
        }

        wandb.log(metrics, step=self.global_step, commit=False)

    def log_evaluation_metrics(
        self, metrics: Dict[str, Any], phase: str = "validation"
    ) -> None:
        if not self.enabled:
            return

        log_dict = {
            f"{phase}/{WandBConstants.METRIC_MIOU}": metrics.get("miou", 0),
            f"{phase}/{WandBConstants.METRIC_MEAN_ACC}": metrics.get("mean_acc", 0),
        }

        if "per_class_iou" in metrics:
            for i, iou in enumerate(metrics["per_class_iou"]):
                log_dict[f"{phase}/class_{i}_iou"] = iou

        if "per_class_acc" in metrics:
            for i, acc in enumerate(metrics["per_class_acc"]):
                log_dict[f"{phase}/class_{i}_acc"] = acc

        wandb.log(log_dict, commit=False)


class SensitivityLogger:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def log_sensitivity_analysis(
        self,
        sensitivity_report: Dict[str, Any],
        layer_quotas: Optional[Dict] = None,
        global_step: int = 0,
    ) -> None:
        if not self.enabled:
            return

        self._log_layer_sensitivities(sensitivity_report, global_step)

        if layer_quotas:
            self._log_quota_allocation(layer_quotas, global_step)

    def _log_layer_sensitivities(
        self, sensitivity_report: Dict[str, Any], global_step: int
    ) -> None:
        if "layer_sensitivities" not in sensitivity_report:
            return

        sensitivities = sensitivity_report["layer_sensitivities"]
        sensitivity_values = [s["sensitivity"] for s in sensitivities]

        sensitivity_metrics = {
            f"{WandBConstants.PREFIX_SENSITIVITY}/mean": np.mean(sensitivity_values),
            f"{WandBConstants.PREFIX_SENSITIVITY}/std": np.std(sensitivity_values),
            f"{WandBConstants.PREFIX_SENSITIVITY}/min": np.min(sensitivity_values),
            f"{WandBConstants.PREFIX_SENSITIVITY}/max": np.max(sensitivity_values),
            f"{WandBConstants.PREFIX_SENSITIVITY}/histogram": wandb.Histogram(
                sensitivity_values
            ),
        }

        wandb.log(sensitivity_metrics, step=global_step, commit=False)

        self._create_sensitivity_table(sensitivities)

    def _create_sensitivity_table(self, sensitivities: List[Dict]) -> None:
        top_layers = sensitivities[: WandBConstants.MAX_SENSITIVITY_LAYERS]

        table_data = []
        for layer in top_layers:
            table_data.append(
                [
                    layer["layer_name"],
                    layer["sensitivity"],
                    layer.get("channels", 0),
                ]
            )

        table = wandb.Table(
            columns=["Layer", "Sensitivity", "Channels"], data=table_data
        )
        wandb.log({f"{WandBConstants.PREFIX_SENSITIVITY}/layer_table": table})

    def _log_quota_allocation(self, layer_quotas: Dict, global_step: int) -> None:
        quota_values = []
        for v in layer_quotas.values():
            if isinstance(v, (int, float)):
                quota_values.append(float(v))
            elif hasattr(v, "item"):
                quota_values.append(float(v.item()))
            else:
                try:
                    quota_values.append(float(v))
                except (TypeError, ValueError):
                    continue

        if not quota_values:
            return

        quota_metrics = {
            f"{WandBConstants.PREFIX_QUOTAS}/mean": np.mean(quota_values),
            f"{WandBConstants.PREFIX_QUOTAS}/std": np.std(quota_values),
            f"{WandBConstants.PREFIX_QUOTAS}/histogram": wandb.Histogram(quota_values),
        }

        wandb.log(quota_metrics, step=global_step, commit=False)

    def log_layer_pruning_details(self, layer_details: List[Dict]) -> None:
        if not self.enabled:
            return

        table_data = []
        for detail in layer_details:
            table_data.append(
                [
                    detail["layer_type"],
                    detail["out_channels"],
                    detail["sensitivity"],
                    detail["pruning_quota"],
                    detail["channels_to_prune"],
                ]
            )

        table = wandb.Table(
            columns=[
                "Layer Type",
                "Original Channels",
                "Sensitivity",
                "Pruning Quota",
                "Channels to Prune",
            ],
            data=table_data,
        )
        wandb.log(
            {f"{WandBConstants.PREFIX_PRUNING}/layer_details": table}, commit=False
        )
