"""
Standard plot generators for evaluation visualization
"""

import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .base import BaseVisualizer
from .utils import VisualizationUtils

logger = logging.getLogger(__name__)


class StandardPlotGenerator(BaseVisualizer):
    """Generate standard evaluation visualizations"""

    def generate_mean_iou_curves(self, df: pd.DataFrame) -> None:
        """Generate mean IoU curves grouped by model type"""
        if not self._validate_data_columns(df, ["model_type", "miou"]):
            return

        fig, ax = plt.subplots(figsize=self.config.DEFAULT_FIGSIZE)

        line_config = self.config.PLOT_CONFIGS["line"]

        for model_type in df["model_type"].unique():
            model_data = df[df["model_type"] == model_type]
            if "pruning_ratio" in model_data.columns:
                grouped = (
                    model_data.groupby("pruning_ratio")["miou"].mean().reset_index()
                )
                ax.plot(
                    grouped["pruning_ratio"],
                    grouped["miou"],
                    marker="o",
                    label=model_type,
                    color=self._get_color_for_model(model_type),
                    **line_config,
                )

        ax.set_xlabel("Pruning Ratio", fontsize=14)
        ax.set_ylabel("Mean IoU", fontsize=14)
        ax.set_title("Mean IoU vs Pruning Ratio", fontsize=16)
        ax.legend(title="Model Type", frameon=True, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.45, 0.95])

        plt.tight_layout()
        self._save_figure(fig, "mean_iou_curves")
        plt.close()

    def generate_per_class_iou_curves(
        self, df: pd.DataFrame, class_names: List[str]
    ) -> None:
        """Generate per-class IoU curves"""
        if not class_names:
            logger.warning("No class names provided, skipping per-class IoU curves")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, class_name in enumerate(class_names):
            ax = axes[idx]
            col_name = f"iou_{class_name}"

            if col_name in df.columns:
                for model_type in df["model_type"].unique():
                    model_data = df[df["model_type"] == model_type]
                    if "pruning_ratio" in model_data.columns:
                        grouped = (
                            model_data.groupby("pruning_ratio")[col_name]
                            .mean()
                            .reset_index()
                        )
                        ax.plot(
                            grouped["pruning_ratio"],
                            grouped[col_name],
                            marker="o",
                            label=model_type,
                            color=self._get_color_for_model(model_type),
                            alpha=0.7,
                        )

                ax.set_xlabel("Pruning Ratio")
                ax.set_ylabel("IoU")
                ax.set_title(f"Class: {class_name}")
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0.45, 0.95])

        for idx in range(len(class_names), len(axes)):
            axes[idx].set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            frameon=True,
            bbox_to_anchor=(0.5, -0.05),
        )

        plt.suptitle("Per-Class IoU vs Pruning Ratio", fontsize=18, y=1.02)
        plt.tight_layout()
        self._save_figure(fig, "per_class_iou_curves")
        plt.close()

    def generate_model_comparison_curves(self, df: pd.DataFrame) -> None:
        """Generate model comparison curves for IoU and accuracy"""
        if not self._validate_data_columns(df, ["model_type", "miou"]):
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        for model_type in df["model_type"].unique():
            model_data = df[df["model_type"] == model_type]
            if "pruning_ratio" in model_data.columns:
                grouped = (
                    model_data.groupby("pruning_ratio")["miou"].mean().reset_index()
                )
                ax1.plot(
                    grouped["pruning_ratio"],
                    grouped["miou"],
                    marker="o",
                    label=model_type,
                    color=self._get_color_for_model(model_type),
                )

        ax1.set_xlabel("Pruning Ratio")
        ax1.set_ylabel("Mean IoU")
        ax1.set_title("Model Comparison: Mean IoU")
        ax1.legend(fontsize=10, loc="best")
        ax1.grid(True, alpha=0.3)

        if "mean_acc" in df.columns:
            for model_type in df["model_type"].unique():
                model_data = df[df["model_type"] == model_type]
                if "pruning_ratio" in model_data.columns:
                    grouped = (
                        model_data.groupby("pruning_ratio")["mean_acc"]
                        .mean()
                        .reset_index()
                    )
                    ax2.plot(
                        grouped["pruning_ratio"],
                        grouped["mean_acc"],
                        marker="s",
                        label=model_type,
                        color=self._get_color_for_model(model_type),
                    )

            ax2.set_xlabel("Pruning Ratio")
            ax2.set_ylabel("Mean Accuracy")
            ax2.set_title("Model Comparison: Mean Accuracy")
            ax2.legend(fontsize=10, loc="best")
            ax2.grid(True, alpha=0.3)

        plt.suptitle("Model Performance Comparison", fontsize=18)
        plt.tight_layout()
        self._save_figure(fig, "model_comparison_curves")
        plt.close()

    def generate_compression_efficiency_plot(self, df: pd.DataFrame) -> None:
        """Generate compression efficiency plot (performance vs model size)"""
        if not self._validate_data_columns(
            df, ["model_type", "size_percentage", "miou"]
        ):
            return

        fig, ax = plt.subplots(figsize=self.config.DEFAULT_FIGSIZE)

        markers = ["o", "s", "^", "D", "v", "p"]
        for idx, model_type in enumerate(df["model_type"].unique()):
            model_data = df[df["model_type"] == model_type]
            ax.scatter(
                model_data["size_percentage"],
                model_data["miou"],
                label=model_type,
                marker=markers[idx % len(markers)],
                s=100,
                alpha=0.7,
                color=self._get_color_for_model(model_type),
            )

            z = np.polyfit(model_data["size_percentage"], model_data["miou"], 2)
            p = np.poly1d(z)
            x_trend = np.linspace(
                model_data["size_percentage"].min(),
                model_data["size_percentage"].max(),
                100,
            )
            ax.plot(
                x_trend,
                p(x_trend),
                "--",
                alpha=0.5,
                color=self._get_color_for_model(model_type),
            )

        ax.set_xlabel("Model Size (% of Original)", fontsize=14)
        ax.set_ylabel("Mean IoU", fontsize=14)
        ax.set_title("Compression Efficiency: Performance vs Model Size", fontsize=16)
        ax.legend(title="Model Type", frameon=True, loc="best")
        ax.grid(True, alpha=0.3)

        ax.axhline(y=df["miou"].max(), color="gray", linestyle=":", alpha=0.5)
        ax.axvline(
            x=df["size_percentage"].min(), color="gray", linestyle=":", alpha=0.5
        )

        plt.tight_layout()
        self._save_figure(fig, "compression_efficiency_plot")
        plt.close()

    def generate_parameter_reduction_heatmap(self, df: pd.DataFrame) -> None:
        """Generate heatmap showing parameter reduction across models and pruning ratios"""
        required_cols = ["params_percentage", "pruning_ratio", "model_type"]
        if not self._validate_data_columns(df, required_cols):
            return

        pivot_data = df.pivot_table(
            values="params_percentage",
            index="model_type",
            columns="pruning_ratio",
            aggfunc="mean",
        )

        fig, ax = plt.subplots(figsize=(12, 8))

        heatmap_config = self.config.PLOT_CONFIGS["heatmap"]

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd_r",
            cbar_kws={"label": "Parameters (% of Original)"},
            ax=ax,
            **heatmap_config,
        )

        ax.set_xlabel("Pruning Ratio", fontsize=14)
        ax.set_ylabel("Model Type", fontsize=14)
        ax.set_title("Parameter Reduction Heatmap", fontsize=16)

        plt.tight_layout()
        self._save_figure(fig, "parameter_reduction_heatmap")
        plt.close()

    def generate_per_model_analysis(self, df: pd.DataFrame) -> None:
        """Generate detailed analysis for each model type"""
        if not self._validate_data_columns(df, ["model_type"]):
            return

        unique_models = df["model_type"].unique()
        n_models = len(unique_models)

        if n_models == 0:
            return

        fig, axes = plt.subplots(n_models, 3, figsize=(15, 5 * n_models), squeeze=False)

        for idx, model_type in enumerate(unique_models):
            model_data = df[df["model_type"] == model_type]

            ax1 = axes[idx, 0]
            if "pruning_ratio" in model_data.columns and "miou" in model_data.columns:
                ax1.plot(
                    model_data["pruning_ratio"],
                    model_data["miou"],
                    marker="o",
                    label="mIoU",
                    linewidth=2,
                )
                if "mean_acc" in model_data.columns:
                    ax1_twin = ax1.twinx()
                    ax1_twin.plot(
                        model_data["pruning_ratio"],
                        model_data["mean_acc"],
                        marker="s",
                        label="Accuracy",
                        color="orange",
                        linewidth=2,
                    )
                    ax1_twin.set_ylabel("Accuracy", color="orange")
                    ax1_twin.tick_params(axis="y", labelcolor="orange")

            ax1.set_xlabel("Pruning Ratio")
            ax1.set_ylabel("mIoU", color="blue")
            ax1.tick_params(axis="y", labelcolor="blue")
            ax1.set_title(f"{model_type}: Performance Metrics")
            ax1.grid(True, alpha=0.3)

            ax2 = axes[idx, 1]
            if (
                "size_percentage" in model_data.columns
                and "pruning_ratio" in model_data.columns
            ):
                ax2.bar(
                    model_data["pruning_ratio"],
                    model_data["size_percentage"],
                    alpha=0.7,
                    color="green",
                )
                ax2.set_xlabel("Pruning Ratio")
                ax2.set_ylabel("Size (% of Original)")
                ax2.set_title(f"{model_type}: Model Size")
                ax2.grid(True, alpha=0.3, axis="y")

            ax3 = axes[idx, 2]
            if "size_percentage" in model_data.columns and "miou" in model_data.columns:
                scatter = ax3.scatter(
                    model_data["size_percentage"],
                    model_data["miou"],
                    c=model_data.get("pruning_ratio", [0.5] * len(model_data)),
                    cmap="viridis",
                    s=100,
                    alpha=0.7,
                )
                plt.colorbar(scatter, ax=ax3, label="Pruning Ratio")
                ax3.set_xlabel("Size (% of Original)")
                ax3.set_ylabel("mIoU")
                ax3.set_title(f"{model_type}: Efficiency Trade-off")
                ax3.grid(True, alpha=0.3)

        plt.suptitle("Per-Model Detailed Analysis", fontsize=18, y=1.02)
        plt.tight_layout()
        self._save_figure(fig, "per_model_analysis")
        plt.close()

    def generate_grouped_boxplot(
        self,
        df: pd.DataFrame,
        metric: str = "miou",
        group_by: str = "model_name",
        hue_by: str = "strategy",
    ) -> None:
        """Create grouped boxplot for comparing distributions"""
        if not self._validate_data_columns(df, [metric, group_by]):
            return

        hue_by = VisualizationUtils.validate_hue_column(df, hue_by)

        fig, ax = plt.subplots(figsize=(12, 6))

        if hue_by:
            sns.boxplot(data=df, x=group_by, y=metric, hue=hue_by, ax=ax)
            sns.stripplot(
                data=df,
                x=group_by,
                y=metric,
                hue=hue_by,
                dodge=True,
                alpha=0.3,
                size=4,
                ax=ax,
                legend=False,
            )
        else:
            sns.boxplot(data=df, x=group_by, y=metric, ax=ax)
            sns.stripplot(data=df, x=group_by, y=metric, alpha=0.3, size=4, ax=ax)

        ax.set_xlabel(self._format_column_name(group_by), fontsize=14)
        ax.set_ylabel(metric.upper(), fontsize=14)
        title = f"{metric.upper()} Distribution by {self._format_column_name(group_by)}"
        if hue_by:
            title += f" and {self._format_column_name(hue_by)}"
        ax.set_title(title, fontsize=16)

        if len(df[group_by].unique()) > 4:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        self._save_figure(fig, f"boxplot_{metric}_by_{group_by}")
        plt.close()

    def generate_fine_tune_method_comparison(self, df: pd.DataFrame) -> None:
        """Compare different fine-tuning methods"""
        if "fine_tune_method" not in df.columns:
            logger.warning("fine_tune_method column not found, skipping comparison")
            return

        valid_methods = ["Standard", "KD-Lite", "Standard+KD"]
        df_filtered = df[df["fine_tune_method"].isin(valid_methods)]

        if df_filtered.empty:
            logger.warning(
                "No valid fine_tune_method values found, skipping comparison"
            )
            return

        fine_tune_methods = df_filtered["fine_tune_method"].unique()

        if len(fine_tune_methods) <= 1:
            logger.info(
                f"Only {len(fine_tune_methods)} fine-tune method(s) found, skipping comparison chart"
            )
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        for method in fine_tune_methods:
            method_data = df_filtered[df_filtered["fine_tune_method"] == method]
            if "pruning_ratio" in method_data.columns:
                grouped = (
                    method_data.groupby("pruning_ratio")["miou"].mean().reset_index()
                )
                ax1.plot(
                    grouped["pruning_ratio"],
                    grouped["miou"],
                    marker="o",
                    label=method,
                    linewidth=2,
                )

        ax1.set_xlabel("Pruning Ratio")
        ax1.set_ylabel("Mean IoU")
        ax1.set_title("Fine-tuning Method Comparison: mIoU")
        ax1.legend(title="Fine-tune Method")
        ax1.grid(True, alpha=0.3)

        ax2.boxplot(
            [
                df_filtered[df_filtered["fine_tune_method"] == method]["miou"].values
                for method in fine_tune_methods
            ],
            labels=fine_tune_methods,
        )
        ax2.set_xlabel("Fine-tune Method")
        ax2.set_ylabel("mIoU Distribution")
        ax2.set_title("Fine-tuning Method Distribution")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.suptitle("Fine-tuning Method Analysis", fontsize=18)
        plt.tight_layout()
        self._save_figure(fig, "fine_tune_method_comparison")
        plt.close()

    def generate_strategy_comparison(self, df: pd.DataFrame) -> None:
        """Compare different pruning strategies"""
        if "strategy" not in df.columns:
            logger.warning("strategy column not found")
            return

        strategies = df["strategy"].unique()
        if len(strategies) <= 1:
            logger.info("Only one strategy found, skipping strategy comparison")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        ax1 = axes[0, 0]
        for strategy in strategies:
            strategy_data = df[df["strategy"] == strategy]
            if "pruning_ratio" in strategy_data.columns:
                grouped = (
                    strategy_data.groupby("pruning_ratio")["miou"].mean().reset_index()
                )
                ax1.plot(
                    grouped["pruning_ratio"],
                    grouped["miou"],
                    marker="o",
                    label=strategy,
                    linewidth=2,
                )

        ax1.set_xlabel("Pruning Ratio")
        ax1.set_ylabel("Mean IoU")
        ax1.set_title("Pruning Strategy Comparison")
        ax1.legend(title="Strategy")
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        if "size_percentage" in df.columns:
            for strategy in strategies:
                strategy_data = df[df["strategy"] == strategy]
                ax2.scatter(
                    strategy_data["size_percentage"],
                    strategy_data["miou"],
                    label=strategy,
                    alpha=0.7,
                    s=80,
                )

            ax2.set_xlabel("Model Size (% of Original)")
            ax2.set_ylabel("Mean IoU")
            ax2.set_title("Strategy Efficiency")
            ax2.legend(title="Strategy")
            ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        strategy_stats = df.groupby("strategy")["miou"].describe()
        x_pos = np.arange(len(strategies))
        means = [strategy_stats.loc[s, "mean"] for s in strategies]
        stds = [strategy_stats.loc[s, "std"] for s in strategies]

        ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(strategies)
        ax3.set_xlabel("Strategy")
        ax3.set_ylabel("Mean IoU")
        ax3.set_title("Strategy Performance Distribution")
        ax3.grid(True, alpha=0.3, axis="y")

        ax4 = axes[1, 1]
        data_for_violin = [df[df["strategy"] == s]["miou"].values for s in strategies]
        parts = ax4.violinplot(data_for_violin, positions=x_pos, showmeans=True)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(strategies)
        ax4.set_xlabel("Strategy")
        ax4.set_ylabel("mIoU Distribution")
        ax4.set_title("Strategy Performance Violin Plot")
        ax4.grid(True, alpha=0.3, axis="y")

        plt.suptitle("Pruning Strategy Analysis", fontsize=18)
        plt.tight_layout()
        self._save_figure(fig, "strategy_comparison")
        plt.close()
