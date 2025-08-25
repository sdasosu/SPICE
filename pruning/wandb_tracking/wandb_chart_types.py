"""Specific chart implementations for pruning visualizations"""

import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn

from .wandb_chart_base import BaseChart, DataProcessor, MultiPanelChart, ValidationUtils
from .wandb_constants import ChartConfig, WandBConstants

logger = logging.getLogger(__name__)


class PruningProgressChart(MultiPanelChart):
    def create_chart(self, pruning_history: List[Dict[str, Any]]) -> None:
        if not self.enabled or not ValidationUtils.validate_data_not_empty(
            pruning_history, "pruning_history"
        ):
            return

        fig, axes = self._create_figure(2, 2, WandBConstants.LARGE_FIGURE_SIZE)

        steps = DataProcessor.extract_numeric_values(pruning_history, "step")
        params = DataProcessor.extract_numeric_values(pruning_history, "params_million")
        macs = DataProcessor.extract_numeric_values(pruning_history, "macs_million")
        train_losses = DataProcessor.extract_numeric_values(
            pruning_history, "train_loss"
        )
        val_losses = DataProcessor.extract_numeric_values(pruning_history, "val_loss")

        self._create_parameter_panel(axes[0], steps, params)

        self._create_macs_panel(axes[1], steps, macs)

        self._create_loss_panel(axes[2], steps, train_losses, val_losses)

        self._create_compression_panel(axes[3], steps, params)

        self._finalize_chart(fig, "pruning_progress")

    def _create_parameter_panel(
        self, ax, steps: List[float], params: List[float]
    ) -> None:
        ax.plot(
            steps,
            params,
            marker="o",
            linewidth=2,
            markersize=8,
            color=WandBConstants.COLOR_PRIMARY,
        )
        ax.set_xlabel("Pruning Step")
        ax.set_ylabel("Parameters (M)")
        ax.set_title("Model Parameters Reduction")
        ax.grid(True, alpha=WandBConstants.GRID_ALPHA)

    def _create_macs_panel(self, ax, steps: List[float], macs: List[float]) -> None:
        ax.plot(
            steps,
            macs,
            marker="s",
            linewidth=2,
            markersize=8,
            color=WandBConstants.COLOR_SECONDARY,
        )
        ax.set_xlabel("Pruning Step")
        ax.set_ylabel("MACs (M)")
        ax.set_title("Computational Complexity Reduction")
        ax.grid(True, alpha=WandBConstants.GRID_ALPHA)

    def _create_loss_panel(
        self, ax, steps: List[float], train_losses: List[float], val_losses: List[float]
    ) -> None:
        if any(train_losses) or any(val_losses):
            ax.plot(steps, train_losses, marker="^", label="Train", linewidth=2)
            ax.plot(steps, val_losses, marker="v", label="Val", linewidth=2)
            ax.set_xlabel("Pruning Step")
            ax.set_ylabel("Loss")
            ax.set_title("Loss Evolution")
            ax.legend()
            ax.grid(True, alpha=WandBConstants.GRID_ALPHA)
        else:
            ax.axis("off")

    def _create_compression_panel(
        self, ax, steps: List[float], params: List[float]
    ) -> None:
        if params:
            compression_ratios = DataProcessor.calculate_compression_ratios(
                params[0], params
            )
            ax.bar(
                steps,
                compression_ratios,
                color=WandBConstants.COLOR_SUCCESS,
                alpha=WandBConstants.BAR_ALPHA,
            )
            ax.set_xlabel("Pruning Step")
            ax.set_ylabel("Compression Ratio")
            ax.set_title("Cumulative Compression")
            ax.grid(True, alpha=WandBConstants.GRID_ALPHA)


class SensitivityHeatmap(BaseChart):
    def create_chart(
        self,
        layer_sensitivities: List[Dict[str, Any]],
        top_n: int = WandBConstants.DEFAULT_TOP_N_LAYERS,
    ) -> None:
        if not self.enabled or not ValidationUtils.validate_data_not_empty(
            layer_sensitivities, "layer_sensitivities"
        ):
            return

        sorted_layers = DataProcessor.filter_top_n(
            layer_sensitivities, "sensitivity", top_n
        )

        layer_names = [l["layer_name"] for l in sorted_layers]
        sensitivities = [l["sensitivity"] for l in sorted_layers]

        fig, ax = plt.subplots(figsize=(12, max(6, len(layer_names) * 0.3)))

        # 1 row per layer for visualization
        data = np.array(sensitivities).reshape(-1, 1)

        sns.heatmap(
            data,
            yticklabels=layer_names,
            xticklabels=["Sensitivity"],
            ax=ax,
            **ChartConfig.HEATMAP_CONFIG,
        )

        ax.set_title(f"Layer Sensitivity Analysis (Top {top_n})")
        self._log_chart(fig, "sensitivity_heatmap")


class QuotaDistributionChart(MultiPanelChart):
    def create_chart(
        self, layer_quotas: Dict[nn.Module, float], layer_names: Optional[Dict] = None
    ) -> None:
        if not self.enabled or not layer_quotas:
            return

        quota_values = ValidationUtils.sanitize_numeric_list(
            list(layer_quotas.values())
        )
        if not quota_values:
            logger.warning("No valid quota values found")
            return

        labels = self._generate_labels(layer_quotas, layer_names)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=WandBConstants.LARGE_FIGURE_SIZE)

        self._create_quota_histogram(ax1, quota_values)

        self._create_top_quotas_chart(ax2, quota_values, labels)

        self._finalize_chart(fig, "quota_distribution")

    def _generate_labels(
        self, layer_quotas: Dict, layer_names: Optional[Dict]
    ) -> List[str]:
        if layer_names:
            return [
                layer_names.get(m, f"Layer_{i}")
                for i, m in enumerate(layer_quotas.keys())
            ]
        else:
            return [f"Layer_{i}" for i in range(len(layer_quotas))]

    def _create_quota_histogram(self, ax, quota_values: List[float]) -> None:
        ax.hist(quota_values, **ChartConfig.HISTOGRAM_CONFIG)
        ax.set_xlabel("Pruning Quota")
        ax.set_ylabel("Number of Layers")
        ax.set_title("Distribution of Layer Pruning Quotas")

        mean_quota = np.mean(quota_values)
        ax.axvline(
            mean_quota,
            color=WandBConstants.COLOR_WARNING,
            linestyle="--",
            label=f"Mean: {mean_quota:.3f}",
        )
        ax.legend()

    def _create_top_quotas_chart(
        self, ax, quota_values: List[float], labels: List[str]
    ) -> None:
        sorted_indices = np.argsort(quota_values)[::-1][
            : WandBConstants.MAX_QUOTA_LAYERS
        ]
        top_quotas = [quota_values[i] for i in sorted_indices]
        top_labels = [labels[i] for i in sorted_indices]

        ax.barh(range(len(top_quotas)), top_quotas, color=WandBConstants.COLOR_PRIMARY)
        ax.set_yticks(range(len(top_quotas)))
        ax.set_yticklabels(top_labels, fontsize=8)
        ax.set_xlabel("Pruning Quota")
        ax.set_title("Top 20 Layers by Pruning Quota")
        ax.invert_yaxis()


class KnowledgeDistillationChart(MultiPanelChart):
    def create_chart(self, kd_history: List[Dict[str, float]]) -> None:
        if not self.enabled or not ValidationUtils.validate_data_not_empty(
            kd_history, "kd_history"
        ):
            return

        required_keys = ["kd_loss", "ce_loss", "total_loss"]
        if not ValidationUtils.validate_keys_exist(
            kd_history, required_keys, "kd_history"
        ):
            return

        fig, axes = self._create_figure(2, 2, (12, 10))

        epochs = list(range(len(kd_history)))
        kd_losses = DataProcessor.extract_numeric_values(kd_history, "kd_loss")
        ce_losses = DataProcessor.extract_numeric_values(kd_history, "ce_loss")
        total_losses = DataProcessor.extract_numeric_values(kd_history, "total_loss")
        loss_ratios = [
            DataProcessor.safe_division(kd, ce) for kd, ce in zip(kd_losses, ce_losses)
        ]

        self._create_loss_comparison_panel(axes[0], epochs, kd_losses, ce_losses)

        self._create_total_loss_panel(axes[1], epochs, total_losses)

        self._create_loss_ratio_panel(axes[2], epochs, loss_ratios)

        self._create_loss_composition_panel(axes[3], epochs, kd_losses, ce_losses)

        self._finalize_chart(fig, "kd_analysis")

    def _create_loss_comparison_panel(
        self, ax, epochs: List[int], kd_losses: List[float], ce_losses: List[float]
    ) -> None:
        ax.plot(epochs, kd_losses, label="KD Loss", marker="o")
        ax.plot(epochs, ce_losses, label="CE Loss", marker="s")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("KD vs CE Loss Components")
        ax.legend()
        ax.grid(True, alpha=WandBConstants.GRID_ALPHA)

    def _create_total_loss_panel(
        self, ax, epochs: List[int], total_losses: List[float]
    ) -> None:
        ax.plot(
            epochs,
            total_losses,
            color=WandBConstants.COLOR_SUCCESS,
            marker="^",
            linewidth=2,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss")
        ax.set_title("Total Training Loss")
        ax.grid(True, alpha=WandBConstants.GRID_ALPHA)

    def _create_loss_ratio_panel(
        self, ax, epochs: List[int], loss_ratios: List[float]
    ) -> None:
        ax.plot(epochs, loss_ratios, color=WandBConstants.COLOR_PURPLE, marker="d")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("KD/CE Ratio")
        ax.set_title("Knowledge Distillation Influence")
        ax.grid(True, alpha=WandBConstants.GRID_ALPHA)

    def _create_loss_composition_panel(
        self, ax, epochs: List[int], kd_losses: List[float], ce_losses: List[float]
    ) -> None:
        ax.stackplot(
            epochs,
            kd_losses,
            ce_losses,
            labels=["KD Loss", "CE Loss"],
            alpha=0.7,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Composition Over Time")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=WandBConstants.GRID_ALPHA)


class ModelComparisonRadar(BaseChart):
    def create_chart(
        self,
        original_metrics: Dict[str, float],
        pruned_metrics: Dict[str, float],
    ) -> None:
        if not self.enabled:
            return

        # Normalized to 0-1 scale for radar chart
        categories = ["Parameters", "MACs", "Size (MB)", "mIoU", "Accuracy"]

        original_values = self._normalize_original_metrics(original_metrics)
        pruned_values = self._normalize_pruned_metrics(pruned_metrics)

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        original_values += original_values[:1]
        pruned_values += pruned_values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(
            figsize=WandBConstants.SMALL_FIGURE_SIZE,
            subplot_kw=dict(projection="polar"),
        )

        ax.plot(
            angles,
            original_values,
            "o-",
            linewidth=2,
            label="Original",
            color=WandBConstants.COLOR_INFO,
        )
        ax.fill(
            angles,
            original_values,
            alpha=WandBConstants.FILL_ALPHA,
            color=WandBConstants.COLOR_INFO,
        )

        ax.plot(
            angles,
            pruned_values,
            "o-",
            linewidth=2,
            label="Pruned",
            color=WandBConstants.COLOR_WARNING,
        )
        ax.fill(
            angles,
            pruned_values,
            alpha=WandBConstants.FILL_ALPHA,
            color=WandBConstants.COLOR_WARNING,
        )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title("Model Performance Comparison", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        self._log_chart(fig, "model_comparison_radar")

    def _normalize_original_metrics(
        self, original_metrics: Dict[str, float]
    ) -> List[float]:
        return [
            1.0,
            1.0,
            1.0,
            original_metrics.get("miou", 0.8),
            original_metrics.get("accuracy", 0.85),
        ]

    def _normalize_pruned_metrics(
        self, pruned_metrics: Dict[str, float]
    ) -> List[float]:
        return [
            pruned_metrics.get("param_ratio", 0.5),
            pruned_metrics.get("macs_ratio", 0.5),
            pruned_metrics.get("size_ratio", 0.5),
            pruned_metrics.get("miou", 0.75),
            pruned_metrics.get("accuracy", 0.82),
        ]


class PruningEfficiencyChart(BaseChart):
    def create_chart(self, pruning_steps: List[Dict]) -> None:
        if not self.enabled or not ValidationUtils.validate_data_not_empty(
            pruning_steps, "pruning_steps"
        ):
            return

        fig, ax = plt.subplots(figsize=WandBConstants.DEFAULT_FIGURE_SIZE)

        compression_ratios = DataProcessor.extract_numeric_values(
            pruning_steps, "compression_ratio", 1.0
        )
        performance_scores = DataProcessor.extract_numeric_values(
            pruning_steps, "performance", 1.0
        )
        labels = [f"Step {step.get('step', i)}" for i, step in enumerate(pruning_steps)]

        scatter = ax.scatter(
            compression_ratios,
            performance_scores,
            c=range(len(compression_ratios)),
            s=100,
            alpha=0.6,
            edgecolors="black",
            **ChartConfig.SCATTER_CONFIG,
        )

        self._add_step_labels(ax, compression_ratios, performance_scores, labels)

        self._add_ideal_line(ax, compression_ratios)

        ax.set_xlabel("Compression Ratio")
        ax.set_ylabel("Performance Score")
        ax.set_title("Pruning Efficiency: Performance vs Compression Trade-off")
        ax.legend()
        ax.grid(True, alpha=WandBConstants.GRID_ALPHA)

        plt.colorbar(scatter, ax=ax, label="Pruning Step")

        self._log_chart(fig, "pruning_efficiency")

    def _add_step_labels(
        self,
        ax,
        compression_ratios: List[float],
        performance_scores: List[float],
        labels: List[str],
    ) -> None:
        for i, label in enumerate(labels):
            if i == 0 or i == len(labels) - 1 or i % 2 == 0:
                ax.annotate(
                    label,
                    (compression_ratios[i], performance_scores[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

    def _add_ideal_line(self, ax, compression_ratios: List[float]) -> None:
        max_compression = max(compression_ratios) if compression_ratios else 1
        ax.plot(
            [1, max_compression],
            [1, 1],
            "r--",
            alpha=0.5,
            label="Ideal (no performance drop)",
        )
