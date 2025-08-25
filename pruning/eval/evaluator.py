"""
Main evaluator class for pruned models
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from torch.utils.data import DataLoader

try:
    # Try relative imports first (when used as a module)
    from .advanced_visualizations import AdvancedVisualizer
    from .components import CacheManager, ConfigManager, CoreEvaluator, ResultProcessor
    from .config import EvaluationConfig
    from .loader import PrunedModelLoader
    from .metrics import SegmentationMetrics
    from .visualizer import EvaluationVisualizer
    from .wandb_integration import EvaluationDashboard, WandBEvaluationTracker
except ImportError:
    # Fall back to absolute imports (when run directly)
    from advanced_visualizations import AdvancedVisualizer
    from components import CacheManager, ConfigManager, CoreEvaluator, ResultProcessor
    from config import EvaluationConfig
    from loader import PrunedModelLoader
    from metrics import SegmentationMetrics
    from visualizer import EvaluationVisualizer
    from wandb_integration.wandb_dashboard import EvaluationDashboard
    from wandb_integration.wandb_integration import WandBEvaluationTracker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from data.data import get_loaders

logger = logging.getLogger(__name__)


class PrunedModelEvaluator:
    """Evaluator for pruned segmentation models"""

    def __init__(self, config: EvaluationConfig):
        self.config = config

        # Initialize configuration manager
        self.config_manager = ConfigManager(config)
        self.device = self.config_manager.device

        # Create output directories
        self.config_manager.create_output_directories()

        # Initialize core components
        self.loader = PrunedModelLoader()
        self.cache_manager = CacheManager(
            cache_dir=config.output_dir, enabled=self.config_manager.cache_enabled
        )
        self.core_evaluator = CoreEvaluator(
            device=self.device, num_classes=config.num_classes
        )
        self.result_processor = ResultProcessor(config.output_dir)

        # Initialize visualization components if enabled
        self.visualizer = None
        self.advanced_visualizer = None
        if self.config_manager.visualization_enabled:
            self._initialize_visualizers()

        # Initialize WandB tracking if enabled
        self.wandb_tracker = None
        self.wandb_dashboard = None
        if self.config_manager.should_use_wandb():
            self._initialize_wandb()

    def _initialize_visualizers(self):
        """Initialize visualization components"""
        self.visualizer = EvaluationVisualizer(
            save_dir=self.config.visualization_dir,
            dpi=self.config.figure_dpi,
            save_formats=self.config.figure_formats,
        )
        self.advanced_visualizer = AdvancedVisualizer(
            save_dir=str(Path(self.config.visualization_dir) / "advanced"),
            dpi=self.config.figure_dpi,
        )
        logger.info("Visualization components initialized")

    def _initialize_wandb(self):
        """Initialize WandB tracking"""
        wandb_config = self.config_manager.get_wandb_config()

        self.wandb_tracker = WandBEvaluationTracker(
            project=wandb_config["project"],
            entity=wandb_config["entity"],
            name=wandb_config["name"],
            tags=wandb_config["tags"],
            config=wandb_config["config"],
            enabled=True,
        )
        self.wandb_dashboard = EvaluationDashboard(self.wandb_tracker)
        logger.info("WandB tracking initialized")

    def evaluate_single_model(
        self,
        model_path: str,
        data_loader: DataLoader,
        model_info: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single pruned model

        Args:
            model_path: Path to the model file
            data_loader: DataLoader for evaluation
            model_info: Optional model information

        Returns:
            Dictionary with evaluation results
        """
        # Load model
        model = self.loader.load_pruned_model(model_path, self.device)

        # Initialize metrics
        metrics = SegmentationMetrics(self.config.num_classes, device=self.device.type)

        # Use core evaluator to perform evaluation
        return self.core_evaluator.evaluate_single_model(
            model=model,
            data_loader=data_loader,
            metrics_calculator=metrics,
            model_info=model_info,
        )

    def evaluate_all_models(self):
        """Evaluate all pruned models in the configured directory"""
        # Load cache
        cache = self.cache_manager.load_cache()

        # Get data loader
        _, val_loader, test_loader = get_loaders(
            data_root=self.config.data_root,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            img_size=self.config.img_size,
        )

        # Use test loader if available, otherwise validation
        eval_loader = test_loader if test_loader is not None else val_loader

        # Find all pruned models
        pruned_models = self.loader.find_pruned_models(self.config.pruned_models_dir)

        if not pruned_models:
            logger.warning(f"No pruned models found in {self.config.pruned_models_dir}")
            return

        # Check which models need evaluation
        models_to_eval = []
        for model_info in pruned_models:
            cached_result = self.cache_manager.get_cached_result(model_info, cache)
            if cached_result:
                self.result_processor.add_result(cached_result)
            else:
                models_to_eval.append(model_info)

        logger.info(
            f"Found {len(pruned_models)} models, {len(models_to_eval)} need evaluation"
        )

        # Evaluate remaining models
        for idx, model_info in enumerate(models_to_eval, 1):
            try:
                print(
                    f"\nEvaluating [{idx}/{len(models_to_eval)}]: {model_info['dir_name']}"
                )

                results = self.evaluate_single_model(
                    model_info["path"], eval_loader, model_info
                )

                # Process results (print, save, cache)
                self.result_processor.process_single_result(
                    results, model_info["dir_name"]
                )
                self.cache_manager.store_result(model_info, results, cache)

            except Exception as e:
                logger.error(f"Failed to evaluate {model_info['dir_name']}: {e}")
                continue

        # Generate summary and post-processing
        self._finalize_evaluation()

    def _finalize_evaluation(self):
        """Finalize evaluation with summary and optional visualizations/WandB"""
        # Generate summary
        csv_file = self.result_processor.generate_summary(save_csv=self.config.save_csv)

        # Generate visualizations if enabled
        if self.config_manager.visualization_enabled:
            self._generate_visualizations(csv_file)

        # Log to WandB if enabled
        if self.config_manager.should_use_wandb() and self.wandb_tracker:
            self._log_to_wandb(csv_file)

    def _generate_visualizations(self, csv_file: str):
        """Generate all visualizations from evaluation results"""
        logger.info("Generating visualizations...")

        if not self.visualizer:
            logger.warning("Visualizer not initialized")
            return

        if not csv_file or not Path(csv_file).exists():
            logger.warning(f"CSV file not found: {csv_file}")
            return

        # Generate basic visualizations
        if self.visualizer.load_data(csv_file):
            self.visualizer.generate_all_visualizations()

            # Generate summary report
            summary = self.visualizer.generate_summary_report()

            # Save summary as JSON
            summary_file = Path(self.config.visualization_dir) / "summary_report.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Saved summary report to {summary_file}")

        # Generate advanced visualizations if available
        if self.advanced_visualizer:
            self._generate_advanced_visualizations(csv_file)

    def _generate_advanced_visualizations(self, csv_file: str):
        """Generate advanced visualizations"""
        df = pd.read_csv(csv_file)

        # Parse per_class_iou if stored as string
        if "per_class_iou" in df.columns:
            import ast

            df["per_class_iou"] = df["per_class_iou"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

        # Generate advanced charts
        self.advanced_visualizer.create_3d_scatter_plot(df)
        self.advanced_visualizer.create_pareto_frontier_plot(df)
        self.advanced_visualizer.create_grouped_boxplot(df)
        self.advanced_visualizer.create_density_contour_plot(df)
        self.advanced_visualizer.create_performance_trajectory(df)
        self.advanced_visualizer.create_correlation_matrix(df)
        self.advanced_visualizer.create_pruning_impact_analysis(df)

        logger.info("Advanced visualizations generated")

    def _log_to_wandb(self, csv_file: str):
        """Log results to WandB"""
        logger.info("Logging results to WandB...")

        if not self.wandb_tracker or not self.wandb_tracker.enabled:
            logger.warning("WandB tracker not enabled")
            return

        if not csv_file or not Path(csv_file).exists():
            logger.warning(f"CSV file not found: {csv_file}")
            return

        df = pd.read_csv(csv_file)

        # Parse per_class_iou if stored as string
        if "per_class_iou" in df.columns:
            import ast

            df["per_class_iou"] = df["per_class_iou"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

        # Log individual model results
        for idx, (_, row) in enumerate(df.iterrows()):
            self.wandb_tracker.log_model_evaluation(
                model_name=row.get("model_name", f"model_{idx}"),
                results=row.to_dict(),
                step=idx,
            )

        # Create comprehensive dashboard if enabled
        self._create_wandb_dashboard(df)

        # Log comparison data
        self.wandb_tracker.log_comparison_table(df)
        self.wandb_tracker.log_best_models(df)

        # Log per-class analysis
        class_names = ["Background", "Adult", "Egg masses", "Instar 1-3", "Instar 4"]
        self.wandb_tracker.log_per_class_analysis(df, class_names)

        # Create interactive plots
        self.wandb_tracker.create_interactive_plots(df)

        logger.info("Results logged to WandB")

        # Finish WandB run
        self.wandb_tracker.finish()

    def _create_wandb_dashboard(self, df: pd.DataFrame):
        """Create WandB dashboard with visualizations"""
        if not self.wandb_dashboard or not self.config.visualization_dir:
            return

        vis_dir = Path(self.config.visualization_dir)

        if not vis_dir.exists():
            logger.warning(f"Visualization directory does not exist: {vis_dir}")
            return

        png_files = list(vis_dir.rglob("*.png"))
        logger.info(f"Found {len(png_files)} PNG files in {vis_dir}")

        # Load summary if exists
        summary_file = vis_dir / "summary_report.json"
        summary = {}
        if summary_file.exists():
            with open(summary_file, "r") as f:
                summary = json.load(f)

        # Create dashboard
        self.wandb_dashboard.create_comprehensive_dashboard(
            df=df,
            visualizations_dir=vis_dir,
            summary=summary,
        )
