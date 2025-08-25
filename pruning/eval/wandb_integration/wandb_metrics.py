"""
Metrics logging functionality for WandB evaluation tracking
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not installed. Install with: pip install wandb")

logger = logging.getLogger(__name__)


class MetricsProcessor:
    """Processes and formats metrics for logging"""

    @staticmethod
    def process_evaluation_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
        """Process evaluation results into standardized metrics"""
        eval_metrics = {
            "miou": results.get("miou", 0),
            "mean_acc": results.get("mean_acc", 0),
            "per_class_iou": results.get("per_class_iou", []),
            "per_class_acc": results.get("per_class_acc", []),
        }
        return eval_metrics

    @staticmethod
    def extract_additional_metrics(
        results: Dict[str, Any], model_name: str
    ) -> Dict[str, Any]:
        """Extract additional metrics like params_percentage and size_percentage"""
        additional_metrics = {}

        if "params_percentage" in results:
            additional_metrics[f"{model_name}/params_percentage"] = results[
                "params_percentage"
            ]

        if "size_percentage" in results:
            additional_metrics[f"{model_name}/size_percentage"] = results[
                "size_percentage"
            ]

        return additional_metrics

    @staticmethod
    def prepare_table_columns(df: pd.DataFrame) -> List[str]:
        """Prepare table columns for comparison table"""
        default_columns = [
            "model_name",
            "model_type",
            "strategy",
            "pruning_ratio",
            "miou",
            "mean_acc",
            "params_percentage",
            "size_percentage",
        ]

        return [col for col in default_columns if col in df.columns]

    @staticmethod
    def flatten_summary_dict(
        summary: Dict[str, Any], prefix: str = "evaluation"
    ) -> Dict[str, Any]:
        """Flatten nested dictionary with evaluation prefix"""
        flat_summary = {}

        def flatten_dict(d: Dict[str, Any], current_prefix: str):
            for k, v in d.items():
                key = f"{current_prefix}/{k}" if current_prefix else k
                if isinstance(v, dict):
                    flatten_dict(v, key)
                else:
                    flat_summary[key] = v

        flatten_dict(summary, prefix)
        return flat_summary


class ModelEvaluationLogger:
    """Handles logging of model evaluation results"""

    def __init__(self, tracker):
        """Initialize with tracker instance"""
        self.tracker = tracker
        self.enabled = tracker and tracker.enabled

    def log_model_evaluation(
        self,
        model_name: str,
        results: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """Log evaluation results for a single model"""
        if not self.enabled or not self.tracker._tracker:
            return

        try:
            eval_metrics = MetricsProcessor.process_evaluation_metrics(results)

            self.tracker._tracker.log_evaluation_metrics(eval_metrics, phase=model_name)

            additional_metrics = MetricsProcessor.extract_additional_metrics(
                results, model_name
            )
            if additional_metrics:
                if step is not None:
                    wandb.log(additional_metrics, step=step)
                else:
                    wandb.log(additional_metrics)

            logger.info(f"Logged evaluation metrics for {model_name}")

        except Exception as e:
            logger.error(f"Failed to log model evaluation: {e}")


class ComparisonTableLogger:
    """Handles logging of comparison tables"""

    def __init__(self, enabled: bool = True):
        """Initialize comparison table logger"""
        self.enabled = enabled and WANDB_AVAILABLE

    def log_comparison_table(self, df: pd.DataFrame) -> None:
        """Log comparison table of all evaluated models"""
        if not self.enabled or df is None or df.empty:
            return

        try:
            if "model_type" not in df.columns and "model_name" in df.columns:
                df["model_type"] = df["model_name"].apply(
                    lambda x: x.split("_")[0] if x else "Unknown"
                )

            existing_columns = MetricsProcessor.prepare_table_columns(df)

            table_data = df[existing_columns].values.tolist()

            table = wandb.Table(columns=existing_columns, data=table_data)

            wandb.log({"evaluation/comparison_table": table})

            logger.info("Logged comparison table to WandB")

        except Exception as e:
            logger.error(f"Failed to log comparison table: {e}")


class BestModelsLogger:
    """Handles logging of best performing models"""

    def __init__(self, enabled: bool = True):
        """Initialize best models logger"""
        self.enabled = enabled and WANDB_AVAILABLE

    def log_best_models(self, df: pd.DataFrame, top_k: int = 5) -> None:
        """Log the best performing models"""
        if not self.enabled or df is None or df.empty:
            return

        try:
            top_models = df.nlargest(top_k, "miou")

            summary = {
                "evaluation/best_models": {
                    f"rank_{i + 1}": {
                        "model_name": row.get("model_name", ""),
                        "miou": row.get("miou", 0),
                        "pruning_ratio": row.get("pruning_ratio", 0),
                        "size_percentage": row.get("size_percentage", 100),
                    }
                    for i, (_, row) in enumerate(top_models.iterrows())
                }
            }

            wandb.summary.update(summary)

            table = wandb.Table(dataframe=top_models)
            wandb.log({"evaluation/best_models_table": table})

            logger.info(f"Logged top {top_k} models to WandB")

        except Exception as e:
            logger.error(f"Failed to log best models: {e}")


class SummaryStatisticsLogger:
    """Handles logging of summary statistics"""

    def __init__(self, enabled: bool = True):
        """Initialize summary statistics logger"""
        self.enabled = enabled and WANDB_AVAILABLE

    def log_summary_statistics(self, summary: Dict[str, Any]) -> None:
        """Log summary statistics"""
        if not self.enabled:
            return

        try:
            flat_summary = MetricsProcessor.flatten_summary_dict(summary)

            wandb.summary.update(flat_summary)

            logger.info("Logged summary statistics to WandB")

        except Exception as e:
            logger.error(f"Failed to log summary statistics: {e}")
