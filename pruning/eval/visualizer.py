"""
Visualization module for pruned model evaluation results
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .visualizations import (
        ReportGenerator,
        StandardPlotGenerator,
        VisualizationConfig,
        VisualizationDataProcessor,
    )
except ImportError:
    from visualizations import (
        ReportGenerator,
        StandardPlotGenerator,
        VisualizationConfig,
        VisualizationDataProcessor,
    )

logger = logging.getLogger(__name__)


class EvaluationVisualizer:
    """Handles generation of visualizations from evaluation results"""

    def __init__(
        self,
        save_dir: str,  # Required, no default
        csv_file: Optional[str] = None,
        dpi: int = 600,
        save_formats: List[str] = ["png", "pdf"],
    ):
        """
        Initialize visualization generator

        Args:
            csv_file: Path to CSV file with evaluation results
            save_dir: Directory to save visualization results
            dpi: DPI for saved figures
            save_formats: List of formats to save figures in
        """
        self.csv_file = csv_file
        self.save_dir = Path(save_dir)
        self.dpi = dpi
        self.save_formats = save_formats

        # Initialize components
        self.data_processor = VisualizationDataProcessor()
        self.plot_generator = StandardPlotGenerator(
            save_dir=str(save_dir),
            dpi=dpi,
            save_formats=save_formats,
        )
        self.report_generator = ReportGenerator(save_dir=str(save_dir))

        self.df = None
        self.class_names = VisualizationConfig.DEFAULT_CLASS_NAMES

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, csv_file: Optional[str] = None) -> bool:
        """Load data from CSV file"""
        csv_path = csv_file or self.csv_file
        if not csv_path:
            logger.error("No CSV file specified")
            return False

        success = self.data_processor.load_data(csv_path)
        if success:
            self.df = self.data_processor.get_data()

        return success

    @staticmethod
    def _extract_model_type(model_name: str) -> str:
        """Extract model type from model name"""
        from .visualizations.utils import VisualizationUtils

        return VisualizationUtils.extract_model_type(model_name)

    def generate_all_visualizations(self) -> bool:
        """Generate all visualizations from loaded data"""
        if self.df is None or self.df.empty:
            logger.error("No data loaded. Call load_data() first.")
            return False

        try:
            # Generate different visualizations using the plot generator
            self.plot_generator.generate_mean_iou_curves(self.df)
            self.plot_generator.generate_per_class_iou_curves(self.df, self.class_names)
            self.plot_generator.generate_model_comparison_curves(self.df)
            self.plot_generator.generate_compression_efficiency_plot(self.df)
            self.plot_generator.generate_mac_efficiency_plot(self.df)
            self.plot_generator.generate_parameter_reduction_heatmap(self.df)
            self.plot_generator.generate_per_model_analysis(self.df)
            self.plot_generator.generate_fine_tune_method_comparison(self.df)
            self.plot_generator.generate_strategy_comparison(self.df)

            logger.info(f"All visualizations saved to {self.save_dir}")
            return True

        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            return False

    def generate_mean_iou_curves(self) -> None:
        """Generate mean IoU curves grouped by model type"""
        if self.df is not None:
            self.plot_generator.generate_mean_iou_curves(self.df)

    def generate_per_class_iou_curves(self) -> None:
        """Generate per-class IoU curves"""
        if self.df is not None:
            self.plot_generator.generate_per_class_iou_curves(self.df, self.class_names)

    def generate_model_comparison_curves(self) -> None:
        """Generate model comparison curves for IoU and accuracy"""
        if self.df is not None:
            self.plot_generator.generate_model_comparison_curves(self.df)

    def generate_compression_efficiency_plot(self) -> None:
        """Generate compression efficiency plot (performance vs model size)"""
        if self.df is not None:
            self.plot_generator.generate_compression_efficiency_plot(self.df)

    def generate_mac_efficiency_plot(self) -> None:
        """Generate MAC efficiency plot (performance vs MAC reduction)"""
        if self.df is not None:
            self.plot_generator.generate_mac_efficiency_plot(self.df)

    def generate_parameter_reduction_heatmap(self) -> None:
        """Generate heatmap showing parameter reduction across models and pruning ratios"""
        if self.df is not None:
            self.plot_generator.generate_parameter_reduction_heatmap(self.df)

    def generate_per_model_analysis(self) -> None:
        """Generate detailed analysis for each model type"""
        if self.df is not None:
            self.plot_generator.generate_per_model_analysis(self.df)

    def generate_fine_tune_method_comparison(self) -> None:
        """Compare different fine-tuning methods"""
        if self.df is not None:
            self.plot_generator.generate_fine_tune_method_comparison(self.df)

    def generate_strategy_comparison(self) -> None:
        """Compare different pruning strategies"""
        if self.df is not None:
            self.plot_generator.generate_strategy_comparison(self.df)

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary statistics report"""
        if self.df is None or self.df.empty:
            return {}
        return self.report_generator.generate_summary_report(self.df)
