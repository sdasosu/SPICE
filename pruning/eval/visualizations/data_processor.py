"""
Data processing module for visualization
"""

import logging
import os
from typing import List

import pandas as pd

from .config import VisualizationConfig
from .utils import VisualizationUtils

logger = logging.getLogger(__name__)


class VisualizationDataProcessor:
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or VisualizationConfig.DEFAULT_CLASS_NAMES
        self.df = None

    def load_data(self, csv_file: str) -> bool:
        if not os.path.exists(csv_file):
            logger.error(f"CSV file not found: {csv_file}")
            return False

        try:
            self.df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(self.df)} evaluation results from {csv_file}")

            self._process_per_class_iou()
            self._add_model_type_column()
            self._sort_data()

            return True

        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            return False

    def _process_per_class_iou(self) -> None:
        if "per_class_iou" not in self.df.columns:
            return

        self.df = VisualizationUtils.process_per_class_iou(self.df, self.class_names)

    def _add_model_type_column(self) -> None:
        if "model_type" not in self.df.columns and "model_name" in self.df.columns:
            self.df["model_type"] = self.df["model_name"].apply(
                VisualizationUtils.extract_model_type
            )

    def _sort_data(self) -> None:
        sort_columns = []
        if "model_type" in self.df.columns:
            sort_columns.append("model_type")
        if "pruning_ratio" in self.df.columns:
            sort_columns.append("pruning_ratio")

        if sort_columns:
            self.df = self.df.sort_values(sort_columns)

    def get_data(self) -> pd.DataFrame:
        if self.df is None:
            logger.warning("No data loaded. Call load_data() first.")
            return pd.DataFrame()
        return self.df

    def validate_columns(self, required_columns: List[str]) -> bool:
        if self.df is None:
            return False

        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
        return True

    def filter_data(self, filters: dict) -> pd.DataFrame:
        if self.df is None:
            return pd.DataFrame()

        filtered_df = self.df.copy()

        for column, value in filters.items():
            if column in filtered_df.columns:
                if isinstance(value, list):
                    filtered_df = filtered_df[filtered_df[column].isin(value)]
                else:
                    filtered_df = filtered_df[filtered_df[column] == value]
            else:
                logger.warning(f"Filter column '{column}' not found in data")

        return filtered_df

    def get_summary_stats(self) -> dict:
        if self.df is None or self.df.empty:
            return {}

        summary = {
            "total_models_evaluated": len(self.df),
            "unique_model_types": list(self.df["model_type"].unique())
            if "model_type" in self.df.columns
            else [],
            "pruning_ratios": sorted(self.df["pruning_ratio"].unique().tolist())
            if "pruning_ratio" in self.df.columns
            else [],
            "metrics_available": [
                col
                for col in self.df.columns
                if col in ["miou", "mean_acc", "params_percentage", "size_percentage"]
            ],
        }

        if "miou" in self.df.columns:
            best_idx = self.df["miou"].idxmax()
            best_row = self.df.loc[best_idx]
            summary["best_model"] = {
                "model_name": best_row.get("model_name", ""),
                "model_type": best_row.get("model_type", ""),
                "pruning_ratio": best_row.get("pruning_ratio", 0),
                "miou": best_row.get("miou", 0),
                "mean_acc": best_row.get("mean_acc", 0),
                "size_percentage": best_row.get("size_percentage", 100),
            }

        return summary
