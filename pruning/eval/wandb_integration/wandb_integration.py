"""
Evaluation-specific WandB integration using existing wandb_tracking module
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .wandb_artifacts import VisualizationManager
from .wandb_config import WandBInitializer
from .wandb_metrics import (
    BestModelsLogger,
    ComparisonTableLogger,
    ModelEvaluationLogger,
    SummaryStatisticsLogger,
)
from .wandb_plots import PlotManager

try:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 200000000
except ImportError:
    pass

try:
    from ...wandb_tracking.wandb_tracker import WandBTracker
except ImportError:
    import sys
    from pathlib import Path as PathModule

    project_root = PathModule(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    from pruning.wandb_tracking.wandb_tracker import WandBTracker

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not installed. Install with: pip install wandb")

logger = logging.getLogger(__name__)


class WandBEvaluationTracker:
    """
    Evaluation-specific WandB tracker extending existing wandb_tracking module

    This class serves as the main orchestrator for evaluation tracking,
    delegating specific responsibilities to specialized components.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        """
        Initialize evaluation tracker using existing WandBTracker

        Args:
            project: WandB project name
            entity: WandB entity (team/user)
            name: Run name
            tags: List of tags for the run
            config: Configuration dictionary
            enabled: Whether to enable wandb tracking
        """
        self.enabled = enabled and WANDB_AVAILABLE

        if not self.enabled:
            if not WANDB_AVAILABLE:
                logger.warning("WandB not available. Tracking disabled.")
            self.run = None
            self._tracker = None
            self._setup_disabled_components()
            return

        try:
            init_params = WandBInitializer.prepare_init_params(
                project, entity, name, tags, config, enabled
            )

            self._tracker = WandBTracker(
                config=init_params["config"],
                project=init_params["project"],
                entity=init_params["entity"],
                name=init_params["name"],
                tags=init_params["tags"],
                enabled=init_params["enabled"],
            )

            self.run = self._tracker.run
            self.enabled = self._tracker.enabled

            self._setup_enabled_components()

            if self.enabled:
                logger.info(f"WandB evaluation tracker initialized: {self.run.url}")

        except Exception as e:
            logger.error(f"Failed to initialize WandB evaluation tracker: {e}")
            self.enabled = False
            self.run = None
            self._tracker = None
            self._setup_disabled_components()

    def _setup_enabled_components(self) -> None:
        """Setup components when WandB is enabled"""
        self._model_logger = ModelEvaluationLogger(self)
        self._comparison_logger = ComparisonTableLogger(self.enabled)
        self._best_models_logger = BestModelsLogger(self.enabled)
        self._summary_logger = SummaryStatisticsLogger(self.enabled)
        self._plot_manager = PlotManager(self.enabled)
        self._visualization_manager = VisualizationManager(self.enabled)

    def _setup_disabled_components(self) -> None:
        """Setup components when WandB is disabled"""
        self._model_logger = None
        self._comparison_logger = None
        self._best_models_logger = None
        self._summary_logger = None
        self._plot_manager = None
        self._visualization_manager = None

    def log_model_evaluation(
        self,
        model_name: str,
        results: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """
        Log evaluation results for a single model using specialized logger

        Args:
            model_name: Name of the model
            results: Evaluation results dictionary
            step: Optional step number
        """
        if self._model_logger:
            self._model_logger.log_model_evaluation(model_name, results, step)

    def log_comparison_table(self, df: pd.DataFrame) -> None:
        """
        Log comparison table of all evaluated models using specialized logger

        Args:
            df: DataFrame with evaluation results
        """
        if self._comparison_logger:
            self._comparison_logger.log_comparison_table(df)

    def log_visualizations(self, visualizations_dir: Path) -> None:
        """
        Upload visualization images to WandB using specialized manager

        Args:
            visualizations_dir: Directory containing visualization images
        """
        if self._visualization_manager:
            self._visualization_manager.log_all_visualizations(
                visualizations_dir, None, self.run
            )

    def create_interactive_plots(self, df: pd.DataFrame) -> None:
        """
        Create interactive plots in WandB using specialized plot manager

        Args:
            df: DataFrame with evaluation results
        """
        if self._plot_manager:
            self._plot_manager.create_interactive_plots(df)

    def log_best_models(self, df: pd.DataFrame, top_k: int = 5) -> None:
        """
        Log the best performing models using specialized logger

        Args:
            df: DataFrame with evaluation results
            top_k: Number of top models to log
        """
        if self._best_models_logger:
            self._best_models_logger.log_best_models(df, top_k)

    def log_per_class_analysis(self, df: pd.DataFrame, class_names: List[str]) -> None:
        """
        Log per-class IoU analysis using specialized analyzer

        Args:
            df: DataFrame with evaluation results
            class_names: List of class names
        """
        if self._visualization_manager:
            self._visualization_manager.analyze_per_class_performance(df, class_names)

    def create_dashboard_config(self) -> Dict[str, Any]:
        """Create WandB dashboard configuration using plot manager"""
        if self._plot_manager:
            return self._plot_manager.get_dashboard_config()
        return {}

    def log_artifacts(self, results_dir: Path) -> None:
        """
        Save evaluation results as WandB artifacts using specialized manager

        Args:
            results_dir: Directory containing evaluation results
        """
        if self._visualization_manager:
            self._visualization_manager.log_all_visualizations(
                results_dir, results_dir, self.run
            )

    def log_summary_statistics(self, summary: Dict[str, Any]) -> None:
        """
        Log summary statistics using specialized logger

        Args:
            summary: Summary statistics dictionary
        """
        if self._summary_logger:
            self._summary_logger.log_summary_statistics(summary)

    def finish(self) -> None:
        """Finish WandB run using base tracker"""
        if self.enabled and self._tracker:
            try:
                if WANDB_AVAILABLE:
                    wandb.summary["evaluation_status"] = "completed"

                self._tracker.finish()
                logger.info("WandB evaluation run finished successfully")

            except Exception as e:
                logger.error(f"Failed to finish WandB evaluation run: {e}")
