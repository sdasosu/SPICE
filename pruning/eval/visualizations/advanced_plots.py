"""
Advanced plot generators for pruning evaluation
"""

import logging
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .base import BaseVisualizer
from .utils import VisualizationUtils

logger = logging.getLogger(__name__)


class AdvancedPlotGenerator(BaseVisualizer):
    def create_3d_scatter_plot(
        self,
        df: pd.DataFrame,
        x_col: str = "size_percentage",
        y_col: str = "params_percentage",
        z_col: str = "miou",
        color_col: str = "model_type",
    ) -> None:
        if not self._validate_data_columns(df, [x_col, y_col, z_col]):
            return

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        categories = df[color_col].unique() if color_col in df.columns else ["all"]
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

        plot_config = self.config.PLOT_CONFIGS["3d"].copy()
        elev = plot_config.pop("elev", 20)
        azim = plot_config.pop("azim", 45)

        for idx, category in enumerate(categories):
            if color_col in df.columns:
                mask = df[color_col] == category
                data = df[mask]
            else:
                data = df

            ax.scatter(
                data[x_col],
                data[y_col],
                data[z_col],
                c=[colors[idx]] * len(data),
                label=category,
                **plot_config,
            )

        ax.set_xlabel(self._format_column_name(x_col), fontsize=12)
        ax.set_ylabel(self._format_column_name(y_col), fontsize=12)
        ax.set_zlabel(self._format_column_name(z_col), fontsize=12)
        ax.set_title("3D Model Efficiency Analysis", fontsize=16)
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

        ax.grid(True, alpha=0.3)
        ax.view_init(elev=elev, azim=azim)

        plt.tight_layout()
        self._save_figure(fig, "3d_efficiency_scatter")
        plt.close()

    def create_pareto_frontier_plot(
        self,
        df: pd.DataFrame,
        x_col: str = "size_percentage",
        y_col: str = "miou",
    ) -> None:
        if not self._validate_data_columns(df, [x_col, y_col]):
            return

        fig, ax = plt.subplots(figsize=self.config.DEFAULT_FIGSIZE)

        scatter_config = self.config.PLOT_CONFIGS["scatter"]

        scatter = ax.scatter(
            df[x_col],
            df[y_col],
            c=df.get("pruning_ratio", [0.5] * len(df)),
            cmap="viridis",
            **scatter_config,
        )

        pareto_points = VisualizationUtils.find_pareto_frontier(
            df[x_col].values, df[y_col].values, minimize_x=True, maximize_y=True
        )

        if pareto_points:
            pareto_df = df.iloc[pareto_points]
            pareto_df_sorted = pareto_df.sort_values(x_col)

            line_config = self.config.PLOT_CONFIGS["line"]
            ax.plot(
                pareto_df_sorted[x_col],
                pareto_df_sorted[y_col],
                "r-",
                label="Pareto Frontier",
                **line_config,
            )
            ax.scatter(
                pareto_df[x_col],
                pareto_df[y_col],
                s=200,
                marker="*",
                c="red",
                edgecolors="darkred",
                linewidth=1,
                zorder=5,
                label="Pareto Optimal",
            )

            for _, row in pareto_df.iterrows():
                if "model_name" in row:
                    label = row["model_name"].split("_")[0]
                    ax.annotate(
                        label,
                        xy=(row[x_col], row[y_col]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.7,
                    )

        ax.set_xlabel(self._format_column_name(x_col), fontsize=14)
        ax.set_ylabel(self._format_column_name(y_col), fontsize=14)
        ax.set_title("Pareto Frontier: Optimal Trade-offs", fontsize=16)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        if "pruning_ratio" in df.columns:
            plt.colorbar(scatter, ax=ax, label="Pruning Ratio")

        plt.tight_layout()
        self._save_figure(fig, "pareto_frontier")
        plt.close()

    def create_confusion_matrix_heatmap(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        normalize: bool = True,
    ) -> None:
        fig, ax = plt.subplots(figsize=self.config.DEFAULT_FIGSIZE)

        if normalize:
            cm_normalized = (
                confusion_matrix.astype("float")
                / confusion_matrix.sum(axis=1)[:, np.newaxis]
            )
            cm_normalized = np.nan_to_num(cm_normalized)
        else:
            cm_normalized = confusion_matrix

        heatmap_config = self.config.PLOT_CONFIGS["heatmap"]

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="YlOrRd",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Proportion" if normalize else "Count"},
            ax=ax,
            **heatmap_config,
        )

        ax.set_xlabel("Predicted Class", fontsize=14)
        ax.set_ylabel("True Class", fontsize=14)
        ax.set_title(
            f"Confusion Matrix {'(Normalized)' if normalize else '(Raw Counts)'}",
            fontsize=16,
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()
        self._save_figure(
            fig, f"confusion_matrix_{'normalized' if normalize else 'raw'}"
        )
        plt.close()

    def create_density_contour_plot(
        self,
        df: pd.DataFrame,
        x_col: str = "size_percentage",
        y_col: str = "miou",
    ) -> None:
        if not self._validate_data_columns(df, [x_col, y_col]):
            return

        z = VisualizationUtils.calculate_density(df[x_col].values, df[y_col].values)
        if z is None:
            return

        fig, ax = plt.subplots(figsize=self.config.DEFAULT_FIGSIZE)

        scatter_config = self.config.PLOT_CONFIGS["scatter"]

        scatter = ax.scatter(
            df[x_col],
            df[y_col],
            c=z,
            cmap="viridis",
            **scatter_config,
        )

        try:
            from scipy.stats import gaussian_kde

            xi = np.linspace(df[x_col].min(), df[x_col].max(), 100)
            yi = np.linspace(df[y_col].min(), df[y_col].max(), 100)
            Xi, Yi = np.meshgrid(xi, yi)

            positions = np.vstack([Xi.ravel(), Yi.ravel()])
            kernel = gaussian_kde(np.vstack([df[x_col], df[y_col]]))
            Zi = np.reshape(kernel(positions).T, Xi.shape)

            contour = ax.contour(Xi, Yi, Zi, colors="black", alpha=0.3, linewidths=1)
            ax.clabel(contour, inline=True, fontsize=8)
        except ImportError:
            pass

        plt.colorbar(scatter, ax=ax, label="Density")

        ax.set_xlabel(self._format_column_name(x_col), fontsize=14)
        ax.set_ylabel(self._format_column_name(y_col), fontsize=14)
        ax.set_title("Model Distribution Density", fontsize=16)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, "density_contour_plot")
        plt.close()

    def create_performance_trajectory(
        self,
        df: pd.DataFrame,
        model_types: Optional[List[str]] = None,
    ) -> None:
        if not self._validate_data_columns(df, ["pruning_ratio", "miou"]):
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        if model_types is None and "model_type" in df.columns:
            model_types = df["model_type"].unique()
        elif model_types is None:
            model_types = ["all"]

        for model_type in model_types:
            if "model_type" in df.columns and model_type != "all":
                model_data = df[df["model_type"] == model_type]
            else:
                model_data = df

            if len(model_data) == 0:
                continue

            grouped = (
                model_data.groupby("pruning_ratio")
                .agg({"miou": ["mean", "std", "count"]})
                .reset_index()
            )
            grouped.columns = ["pruning_ratio", "mean", "std", "count"]

            ax.errorbar(
                grouped["pruning_ratio"],
                grouped["mean"],
                yerr=grouped["std"],
                marker="o",
                markersize=8,
                linewidth=2,
                capsize=5,
                capthick=2,
                label=model_type,
                alpha=0.8,
            )

            curve_params = VisualizationUtils.fit_exponential_curve(
                grouped["pruning_ratio"].values, grouped["mean"].values
            )
            if curve_params is not None:
                x_fit = np.linspace(
                    grouped["pruning_ratio"].min(),
                    grouped["pruning_ratio"].max(),
                    100,
                )
                y_fit = VisualizationUtils.exponential_decay(x_fit, *curve_params)

                ax.plot(
                    x_fit,
                    y_fit,
                    "--",
                    alpha=0.5,
                    linewidth=1,
                    label=f"{model_type} (fitted)",
                )

        ax.set_xlabel("Pruning Ratio", fontsize=14)
        ax.set_ylabel("Mean IoU", fontsize=14)
        ax.set_title("Performance Degradation Trajectory", fontsize=16)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        ax.axhline(
            y=0.5, color="red", linestyle=":", alpha=0.5, label="0.5 mIoU threshold"
        )

        plt.tight_layout()
        self._save_figure(fig, "performance_trajectory")
        plt.close()

    def create_correlation_matrix(self, df: pd.DataFrame) -> None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        exclude_cols = ["Unnamed: 0", "index"]
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

        if len(numerical_cols) < 2:
            logger.warning("Not enough numerical columns for correlation matrix")
            return

        corr_matrix = df[numerical_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        heatmap_config = self.config.PLOT_CONFIGS["heatmap"]

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={"label": "Correlation", "shrink": 0.8},
            ax=ax,
            **heatmap_config,
        )

        ax.set_title("Feature Correlation Matrix", fontsize=16)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()
        self._save_figure(fig, "correlation_matrix")
        plt.close()

    def create_pruning_impact_analysis(
        self,
        df: pd.DataFrame,
        reference_ratio: float = 0.0,
    ) -> None:
        if not self._validate_data_columns(df, ["pruning_ratio"]):
            return

        metrics = ["miou", "mean_acc", "params_percentage", "size_percentage"]
        available_metrics = [m for m in metrics if m in df.columns]

        if len(available_metrics) < 2:
            logger.warning("Not enough metrics for impact analysis")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, metric in enumerate(available_metrics[:4]):
            ax = axes[idx]

            grouped = (
                df.groupby("pruning_ratio")[metric].agg(["mean", "std"]).reset_index()
            )

            if reference_ratio in grouped["pruning_ratio"].values:
                reference_value = grouped[grouped["pruning_ratio"] == reference_ratio][
                    "mean"
                ].values[0]
            else:
                reference_value = grouped["mean"].max()

            grouped["relative_change"] = (
                (grouped["mean"] - reference_value) / reference_value * 100
            )

            bars = ax.bar(
                grouped["pruning_ratio"],
                grouped["relative_change"],
                yerr=grouped["std"] / reference_value * 100,
                capsize=5,
                alpha=0.7,
                color=[
                    "green" if x >= 0 else "red" for x in grouped["relative_change"]
                ],
            )

            ax.set_xlabel("Pruning Ratio", fontsize=12)
            ax.set_ylabel("Relative Change (%)", fontsize=12)
            ax.set_title(f"{self._format_column_name(metric)} Impact", fontsize=14)
            ax.grid(True, alpha=0.3, axis="y")
            ax.axhline(y=0, color="black", linewidth=1)

            for bar, val in zip(bars, grouped["relative_change"]):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8,
                )

        for idx in range(len(available_metrics), 4):
            axes[idx].set_visible(False)

        plt.suptitle("Pruning Impact Analysis", fontsize=18)
        plt.tight_layout()
        self._save_figure(fig, "pruning_impact_analysis")
        plt.close()
