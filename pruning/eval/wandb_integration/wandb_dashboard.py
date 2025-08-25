"""
Dashboard creation and management for WandB evaluation tracking
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .wandb_artifacts import VisualizationManager
from .wandb_metrics import (
    BestModelsLogger,
    ComparisonTableLogger,
    SummaryStatisticsLogger,
)
from .wandb_plots import PlotManager

try:
    from ...wandb_tracking.wandb_visualizations import PruningVisualizer
except ImportError:
    import sys
    from pathlib import Path as PathModule

    project_root = PathModule(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    from pruning.wandb_tracking.wandb_visualizations import PruningVisualizer

logger = logging.getLogger(__name__)


class EvaluationDashboardComponents:
    """Components for creating evaluation dashboard"""

    def __init__(self, enabled: bool = True):
        """Initialize dashboard components"""
        self.enabled = enabled
        self.comparison_logger = ComparisonTableLogger(enabled)
        self.best_models_logger = BestModelsLogger(enabled)
        self.summary_logger = SummaryStatisticsLogger(enabled)
        self.plot_manager = PlotManager(enabled)
        self.visualization_manager = VisualizationManager(enabled)

    def log_comparison_data(self, df: pd.DataFrame) -> None:
        """Log comparison table and best models"""
        if not self.enabled:
            return

        self.comparison_logger.log_comparison_table(df)
        self.best_models_logger.log_best_models(df)

    def create_plots(self, df: pd.DataFrame) -> None:
        """Create interactive plots"""
        if not self.enabled:
            return

        self.plot_manager.create_interactive_plots(df)

    def log_visualizations_and_artifacts(
        self, visualizations_dir: Path, results_dir: Optional[Path] = None, run=None
    ) -> None:
        """Log visualizations and artifacts"""
        if not self.enabled:
            return

        self.visualization_manager.log_all_visualizations(
            visualizations_dir, results_dir, run
        )

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Log summary statistics"""
        if not self.enabled:
            return

        self.summary_logger.log_summary_statistics(summary)


class AdvancedVisualizationHandler:
    """Handles advanced visualization using existing infrastructure"""

    def __init__(self, tracker, enabled: bool = True):
        """Initialize advanced visualization handler"""
        self.enabled = enabled and tracker and tracker.enabled
        self.tracker = tracker

        if self.enabled and hasattr(tracker, "_tracker") and tracker._tracker:
            self.visualizer = PruningVisualizer(tracker._tracker)
        else:
            self.visualizer = None

    def create_advanced_visualizations(
        self, df: pd.DataFrame, summary: Dict[str, Any]
    ) -> None:
        """Create advanced visualizations using existing infrastructure"""
        if not self.enabled or not self.visualizer:
            return

        try:
            results_for_dashboard = {
                "final_results": summary,
                "evaluation_complete": True,
                "total_models_evaluated": len(df) if df is not None else 0,
            }

            self.visualizer.create_final_summary_dashboard(results_for_dashboard)
            logger.info("Created advanced visualizations using existing infrastructure")

        except Exception as viz_error:
            logger.warning(f"Could not create advanced visualizations: {viz_error}")


class EvaluationDashboard:
    """
    Create comprehensive evaluation dashboard extending existing visualizations

    This class orchestrates the creation of a complete dashboard by coordinating
    multiple specialized components, each with a single responsibility.
    """

    def __init__(self, tracker):
        """
        Initialize dashboard creator with access to existing visualization tools

        Args:
            tracker: WandB evaluation tracker instance
        """
        self.tracker = tracker
        self.enabled = tracker and tracker.enabled

        # Initialize specialized components
        self.components = EvaluationDashboardComponents(self.enabled)
        self.advanced_viz = AdvancedVisualizationHandler(tracker, self.enabled)

    def create_basic_dashboard(self, df: pd.DataFrame, summary: Dict[str, Any]) -> None:
        """Create basic dashboard components"""
        if not self.enabled:
            return

        try:
            self.components.log_comparison_data(df)

            self.components.create_plots(df)

            self.components.log_summary(summary)

            logger.info("Created basic dashboard components")

        except Exception as e:
            logger.error(f"Failed to create basic dashboard: {e}")

    def create_visualization_components(
        self, visualizations_dir: Path, results_dir: Optional[Path] = None
    ) -> None:
        """Create visualization components"""
        if not self.enabled:
            return

        try:
            run = self.tracker.run if hasattr(self.tracker, "run") else None
            self.components.log_visualizations_and_artifacts(
                visualizations_dir, results_dir, run
            )

            logger.info("Created visualization components")

        except Exception as e:
            logger.error(f"Failed to create visualization components: {e}")

    def create_comprehensive_dashboard(
        self,
        df: pd.DataFrame,
        visualizations_dir: Path,
        summary: Dict[str, Any],
        results_dir: Optional[Path] = None,
    ) -> None:
        """
        Create comprehensive dashboard leveraging existing visualization capabilities

        Args:
            df: DataFrame with evaluation results
            visualizations_dir: Directory containing visualizations
            summary: Summary statistics
            results_dir: Optional directory containing additional results
        """
        if not self.enabled:
            return

        try:
            self.create_basic_dashboard(df, summary)

            results_dir_path = results_dir or visualizations_dir.parent
            self.create_visualization_components(visualizations_dir, results_dir_path)

            self.advanced_viz.create_advanced_visualizations(df, summary)

            logger.info("Created comprehensive WandB evaluation dashboard")

        except Exception as e:
            logger.error(f"Failed to create evaluation dashboard: {e}")

    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration"""
        if not self.enabled:
            return {}

        return self.components.plot_manager.get_dashboard_config()

    def log_dashboard_info(self) -> None:
        """Log dashboard configuration and information"""
        if not self.enabled:
            return

        try:
            self.components.plot_manager.log_dashboard_config()
            logger.info("Logged dashboard configuration")

        except Exception as e:
            logger.error(f"Failed to log dashboard info: {e}")


class DashboardFactory:
    """Factory for creating evaluation dashboards"""

    @staticmethod
    def create_dashboard(tracker) -> EvaluationDashboard:
        """Create evaluation dashboard with appropriate configuration"""
        return EvaluationDashboard(tracker)

    @staticmethod
    def create_components_only(enabled: bool = True) -> EvaluationDashboardComponents:
        """Create dashboard components without tracker dependency"""
        return EvaluationDashboardComponents(enabled)
