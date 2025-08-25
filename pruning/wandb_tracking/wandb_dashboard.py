"""Final summary dashboard creation for WandB tracking"""

import logging
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from .wandb_chart_base import BaseChart
from .wandb_constants import WandBConstants

logger = logging.getLogger(__name__)


class FinalSummaryDashboard(BaseChart):
    def create_chart(self, results: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        fig = plt.figure(figsize=WandBConstants.DASHBOARD_FIGURE_SIZE)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        self._create_metric_cards(fig, gs, results)
        self._create_comparison_bars(fig, gs, results)
        self._create_strategy_info(fig, gs, results)
        self._create_performance_timeline(fig, gs, results)

        plt.suptitle(
            f"Pruning Summary: {results.get('model_name', 'Unknown Model')}",
            fontsize=16,
            fontweight="bold",
        )

        self._log_chart(fig, "final_summary_dashboard")
        logger.info(WandBConstants.INFO_DASHBOARD_CREATED)

    def _create_metric_cards(self, fig, gs, results: Dict[str, Any]) -> None:
        metrics = [
            ("Parameters", f"{results.get('params_reduction', 0):.1%}", "reduction"),
            ("Model Size", f"{results.get('size_reduction', 0):.1%}", "reduction"),
            ("MACs", f"{results.get('macs_reduction', 0):.1%}", "reduction"),
            ("Final mIoU", f"{results.get('final_miou', 0):.3f}", "score"),
        ]

        for i, (name, value, type_) in enumerate(metrics[:4]):
            if i >= 3:
                break

            ax = fig.add_subplot(gs[0, i])
            ax.text(0.5, 0.7, name, ha="center", fontsize=14, fontweight="bold")

            color = (
                WandBConstants.COLOR_SUCCESS
                if type_ == "reduction"
                else WandBConstants.COLOR_INFO
            )
            ax.text(0.5, 0.3, value, ha="center", fontsize=20, color=color)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

    def _create_comparison_bars(self, fig, gs, results: Dict[str, Any]) -> None:
        ax1 = fig.add_subplot(gs[1, :2])

        categories = ["Parameters (M)", "Size (MB)", "MACs (M)"]
        before = [
            results.get("initial_params", 0) / WandBConstants.MILLION_DIVISOR,
            results.get("initial_size_mb", 0),
            results.get("initial_macs", 0) / WandBConstants.MILLION_DIVISOR,
        ]
        after = [
            results.get("final_params", 0) / WandBConstants.MILLION_DIVISOR,
            results.get("final_size_mb", 0),
            results.get("final_macs", 0) / WandBConstants.MILLION_DIVISOR,
        ]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax1.bar(
            x - width / 2,
            before,
            width,
            label="Before",
            color=WandBConstants.COLOR_PRIMARY,
        )
        bars2 = ax1.bar(
            x + width / 2,
            after,
            width,
            label="After",
            color=WandBConstants.COLOR_SECONDARY,
        )

        ax1.set_xlabel("Metrics")
        ax1.set_ylabel("Value")
        ax1.set_title("Before vs After Pruning")
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()

        self._add_bar_labels(ax1, bars1)
        self._add_bar_labels(ax1, bars2)

    def _add_bar_labels(self, ax, bars) -> None:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    def _create_strategy_info(self, fig, gs, results: Dict[str, Any]) -> None:
        ax2 = fig.add_subplot(gs[1, 2])

        strategy_text = (
            f"Strategy: {results.get('pruning_strategy', 'N/A')}\n"
            f"Ratio: {results.get('pruning_ratio', 0):.2f}\n"
            f"Steps: {results.get('iterative_steps', 'N/A')}\n"
        )

        if results.get("kd_lite_enabled"):
            strategy_text += (
                f"\nKD-Lite: Enabled\n"
                f"Temperature: {results.get('kd_temperature', 'N/A')}\n"
                f"Alpha: {results.get('kd_alpha', 'N/A')}"
            )

        ax2.text(0.1, 0.5, strategy_text, fontsize=11, verticalalignment="center")
        ax2.set_title("Pruning Configuration")
        ax2.axis("off")

    def _create_performance_timeline(self, fig, gs, results: Dict[str, Any]) -> None:
        if "performance_history" not in results:
            return

        ax3 = fig.add_subplot(gs[2, :])
        history = results["performance_history"]

        if not history:
            return

        steps = list(range(len(history)))
        performance = [h.get("miou", 0) for h in history]

        ax3.plot(
            steps,
            performance,
            marker="o",
            linewidth=2,
            markersize=8,
            color=WandBConstants.COLOR_PRIMARY,
        )
        ax3.set_xlabel("Pruning Step")
        ax3.set_ylabel("mIoU")
        ax3.set_title("Performance Evolution During Pruning")
        ax3.grid(True, alpha=WandBConstants.GRID_ALPHA)
        ax3.fill_between(
            steps,
            performance,
            alpha=WandBConstants.FILL_ALPHA,
            color=WandBConstants.COLOR_PRIMARY,
        )
