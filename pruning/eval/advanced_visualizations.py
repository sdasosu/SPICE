"""
Advanced visualization module for pruned model evaluation
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from .visualizations import AdvancedPlotGenerator, VisualizationConfig
except ImportError:
    from visualizations import AdvancedPlotGenerator

logger = logging.getLogger(__name__)


class AdvancedVisualizer:
    """Create advanced visualizations for pruning evaluation"""

    def __init__(
        self,
        save_dir: str,  # Required, no default
        dpi: int = 600,
        style: str = "seaborn-v0_8-whitegrid",
    ):
        """
        Initialize advanced visualizer

        Args:
            save_dir: Directory to save visualizations
            dpi: DPI for saved figures
            style: Matplotlib style
        """
        self.plot_generator = AdvancedPlotGenerator(
            save_dir=save_dir,
            dpi=dpi,
            style=style,
        )

    def create_3d_scatter_plot(
        self,
        df: pd.DataFrame,
        x_col: str = "size_percentage",
        y_col: str = "params_percentage",
        z_col: str = "miou",
        color_col: str = "model_type",
    ) -> None:
        """Create 3D scatter plot showing relationship between size, parameters, and performance"""
        self.plot_generator.create_3d_scatter_plot(df, x_col, y_col, z_col, color_col)

    def create_pareto_frontier_plot(
        self,
        df: pd.DataFrame,
        x_col: str = "size_percentage",
        y_col: str = "miou",
    ) -> None:
        """Create Pareto frontier plot showing optimal trade-offs"""
        self.plot_generator.create_pareto_frontier_plot(df, x_col, y_col)

    def create_confusion_matrix_heatmap(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        normalize: bool = True,
    ) -> None:
        """Create confusion matrix heatmap"""
        self.plot_generator.create_confusion_matrix_heatmap(
            confusion_matrix, class_names, normalize
        )

    def create_grouped_boxplot(
        self,
        df: pd.DataFrame,
        metric: str = "miou",
        group_by: str = "model_name",
        hue_by: str = "strategy",
    ) -> None:
        """Create grouped boxplot for comparing distributions"""
        # Use standard plot generator for boxplots
        try:
            from .visualizations import StandardPlotGenerator
        except ImportError:
            from visualizations import StandardPlotGenerator
        std_generator = StandardPlotGenerator(
            save_dir=self.plot_generator.save_dir,
            dpi=self.plot_generator.dpi,
        )
        std_generator.generate_grouped_boxplot(df, metric, group_by, hue_by)

    def create_density_contour_plot(
        self,
        df: pd.DataFrame,
        x_col: str = "size_percentage",
        y_col: str = "miou",
    ) -> None:
        """Create density contour plot showing concentration of models"""
        self.plot_generator.create_density_contour_plot(df, x_col, y_col)

    def create_performance_trajectory(
        self,
        df: pd.DataFrame,
        model_types: Optional[List[str]] = None,
    ) -> None:
        """Create performance trajectory showing how models degrade with pruning"""
        self.plot_generator.create_performance_trajectory(df, model_types)

    def create_correlation_matrix(self, df: pd.DataFrame) -> None:
        """Create correlation matrix heatmap for numerical columns"""
        self.plot_generator.create_correlation_matrix(df)

    def create_pruning_impact_analysis(
        self,
        df: pd.DataFrame,
        reference_ratio: float = 0.0,
    ) -> None:
        """Create analysis showing impact of pruning on different metrics"""
        self.plot_generator.create_pruning_impact_analysis(df, reference_ratio)
