"""
Plot and interactive visualization functionality for WandB evaluation tracking
"""

import logging
from typing import Dict, List

import pandas as pd

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not installed. Install with: pip install wandb")

logger = logging.getLogger(__name__)


class ScatterPlotCreator:
    """Creates scatter plots for evaluation data"""

    def __init__(self, enabled: bool = True):
        """Initialize scatter plot creator"""
        self.enabled = enabled and WANDB_AVAILABLE

    def prepare_scatter_data(self, df: pd.DataFrame) -> List[Dict[str, any]]:
        """Prepare data for scatter plot"""
        scatter_data = []
        for _, row in df.iterrows():
            data_point = {
                "Model": row.get("model_name", "Unknown"),
                "Model Type": row.get("model_type", "Unknown"),
                "Size (%)": row.get("size_percentage", 100),
                "mIoU": row.get("miou", 0),
                "Pruning Ratio": row.get("pruning_ratio", 0),
            }
            if "gmacs" in row:
                data_point["GMACs"] = row["gmacs"]
            if "mac_reduction" in row:
                data_point["MAC Reduction (%)"] = row["mac_reduction"]
            scatter_data.append(data_point)
        return scatter_data

    def create_efficiency_scatter(self, df: pd.DataFrame) -> None:
        """Create scatter plot for model efficiency: Size vs Performance"""
        if not self.enabled or df is None or df.empty:
            return

        if "size_percentage" not in df.columns or "miou" not in df.columns:
            logger.warning("Required columns for scatter plot not found")
            return

        try:
            scatter_data = self.prepare_scatter_data(df)
            table = wandb.Table(dataframe=pd.DataFrame(scatter_data))

            wandb.log(
                {
                    "evaluation/efficiency_scatter": wandb.plot.scatter(
                        table,
                        "Size (%)",
                        "mIoU",
                        title="Model Efficiency: Size vs Performance",
                    )
                }
            )

            logger.info("Created efficiency scatter plot")

        except Exception as e:
            logger.error(f"Failed to create efficiency scatter plot: {e}")

    def create_mac_efficiency_scatter(self, df: pd.DataFrame) -> None:
        """Create scatter plot for MAC efficiency: MAC Reduction vs Performance"""
        if not self.enabled or df is None or df.empty:
            return

        if "mac_reduction" not in df.columns:
            logger.warning("MAC reduction data not available")
            return

        try:
            scatter_data = self.prepare_scatter_data(df)
            table = wandb.Table(dataframe=pd.DataFrame(scatter_data))

            wandb.log(
                {
                    "evaluation/mac_efficiency_scatter": wandb.plot.scatter(
                        table,
                        "MAC Reduction (%)",
                        "mIoU",
                        title="MAC Efficiency: Computation Reduction vs Performance",
                    )
                }
            )

            logger.info("Created MAC efficiency scatter plot")

        except Exception as e:
            logger.error(f"Failed to create MAC efficiency scatter plot: {e}")


class LinePlotCreator:
    """Creates line plots for evaluation data"""

    def __init__(self, enabled: bool = True):
        """Initialize line plot creator"""
        self.enabled = enabled and WANDB_AVAILABLE

    def prepare_line_data(self, model_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for line plot by grouping by pruning ratio"""
        return model_data.groupby("pruning_ratio")["miou"].mean().reset_index()

    def create_performance_curves(self, df: pd.DataFrame) -> None:
        """Create line plot for performance curves by model type"""
        if not self.enabled or df is None or df.empty:
            return

        if "model_type" not in df.columns and "model_name" in df.columns:
            df["model_type"] = df["model_name"].apply(
                lambda x: x.split("_")[0] if x else "Unknown"
            )

        if "pruning_ratio" not in df.columns or "model_type" not in df.columns:
            logger.warning("Required columns for line plot not found")
            return

        try:
            for model_type in df["model_type"].unique():
                model_data = df[df["model_type"] == model_type]
                line_data = self.prepare_line_data(model_data)

                table = wandb.Table(dataframe=line_data)
                wandb.log(
                    {
                        f"evaluation/performance_curve_{model_type}": wandb.plot.line(
                            table,
                            "pruning_ratio",
                            "miou",
                            title=f"{model_type} Performance Curve",
                        )
                    }
                )

            logger.info("Created performance curve plots")

        except Exception as e:
            logger.error(f"Failed to create performance curves: {e}")


class InteractivePlotManager:
    """Manages creation of interactive plots in WandB"""

    def __init__(self, enabled: bool = True):
        """Initialize interactive plot manager"""
        self.enabled = enabled and WANDB_AVAILABLE
        self.scatter_creator = ScatterPlotCreator(enabled)
        self.line_creator = LinePlotCreator(enabled)

    def create_all_interactive_plots(self, df: pd.DataFrame) -> None:
        """Create all interactive plots"""
        if not self.enabled or df is None or df.empty:
            return

        try:
            self.scatter_creator.create_efficiency_scatter(df)

            if "mac_reduction" in df.columns:
                self.scatter_creator.create_mac_efficiency_scatter(df)

            self.line_creator.create_performance_curves(df)

            logger.info("Created all interactive plots in WandB")

        except Exception as e:
            logger.error(f"Failed to create interactive plots: {e}")


class DashboardConfigGenerator:
    """Generates dashboard configuration for WandB"""

    @staticmethod
    def create_dashboard_config() -> Dict[str, any]:
        """Create WandB dashboard configuration for evaluation"""
        dashboard_config = {
            "panels": [
                {
                    "name": "Evaluation Performance Overview",
                    "type": "line",
                    "metrics": ["evaluation/miou", "evaluation/mean_acc"],
                    "x_axis": "pruning_ratio",
                },
                {
                    "name": "Evaluation Compression Efficiency",
                    "type": "scatter",
                    "x_metric": "size_percentage",
                    "y_metric": "miou",
                    "color_by": "model_type",
                },
                {
                    "name": "Best Evaluated Models",
                    "type": "table",
                    "sort_by": "miou",
                    "sort_order": "desc",
                },
                {
                    "name": "Parameter Reduction Analysis",
                    "type": "bar",
                    "metric": "params_percentage",
                    "group_by": "model_type",
                },
            ]
        }

        return dashboard_config

    @staticmethod
    def get_dashboard_url_template() -> str:
        """Get template URL for dashboard access"""
        return "https://wandb.ai/{entity}/{project}/workspace"


class PlotManager:
    """Main plot management class combining all plot functionality"""

    def __init__(self, enabled: bool = True):
        """Initialize plot manager"""
        self.enabled = enabled and WANDB_AVAILABLE
        self.interactive_manager = InteractivePlotManager(enabled)
        self.config_generator = DashboardConfigGenerator()

    def create_interactive_plots(self, df: pd.DataFrame) -> None:
        """Create interactive plots in WandB with evaluation prefix"""
        if not self.enabled:
            return

        self.interactive_manager.create_all_interactive_plots(df)

    def get_dashboard_config(self) -> Dict[str, any]:
        """Get dashboard configuration"""
        return self.config_generator.create_dashboard_config()

    def log_dashboard_config(self) -> None:
        """Log dashboard configuration to WandB"""
        if not self.enabled:
            return

        try:
            config = self.get_dashboard_config()
            wandb.log({"evaluation/dashboard_config": config})
            logger.info("Logged dashboard configuration")

        except Exception as e:
            logger.error(f"Failed to log dashboard configuration: {e}")
