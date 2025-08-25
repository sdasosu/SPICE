import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ResultFormatter:
    @staticmethod
    def print_single_result(results: Dict[str, Any]):
        print(f"\n{'=' * 50}")

        if "model_name" in results:
            model_display = results["model_name"]
            if "strategy" in results:
                model_display += f" ({results['strategy']})"
            print(f"Model: {model_display}")

        if "fine_tune_method" in results:
            print(f"Fine-tune Method: {results['fine_tune_method']}")

        if "pruning_ratio" in results:
            print(f"Pruning Ratio: {results['pruning_ratio']:.2f}")

        print(f"mIoU: {results['miou']:.4f}")
        print(f"Mean Accuracy: {results['mean_acc']:.4f}")

        params_str = f"Parameters: {results.get('total_params', 0):,}"
        if "params_percentage" in results:
            params_str += f" ({results['params_percentage']:.1f}% of original)"
        print(params_str)

        size_str = f"Model Size: {results.get('model_size_mb', 0):.2f} MB"
        if "size_percentage" in results:
            size_str += f" ({results['size_percentage']:.1f}% of original)"
        print(size_str)

        if "per_class_iou" in results:
            print("\nPer-class IoU:")
            for i, iou in enumerate(results["per_class_iou"]):
                print(f"  Class {i}: {iou:.4f}")

    @staticmethod
    def format_summary_table(df: pd.DataFrame) -> pd.DataFrame:
        display_columns = [
            "model_name",
            "strategy",
            "fine_tune_method",
            "pruning_ratio",
            "miou",
            "mean_acc",
            "total_params",
            "params_percentage",
            "model_size_mb",
            "size_percentage",
        ]

        display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[display_columns].copy()

        for col in ["params_percentage", "size_percentage"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%")

        column_renames = {"params_percentage": "params_%", "size_percentage": "size_%"}
        display_df = display_df.rename(columns=column_renames)

        return display_df

    @staticmethod
    def print_summary_table(df: pd.DataFrame):
        print("\n" + "=" * 140)
        print("EVALUATION SUMMARY")
        print("=" * 140)

        display_df = ResultFormatter.format_summary_table(df)
        print(display_df.to_string(index=False))
        print("=" * 140)


class ResultSaver:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_single_model_results(self, results: Dict[str, Any], model_name: str):
        model_dir = self.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        metrics_file = model_dir / "metrics.json"

        json_results = self._serialize_for_json(results)

        with open(metrics_file, "w") as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"Saved results to {metrics_file}")

    def save_summary_csv(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""

        df = pd.DataFrame(results)

        if "model_name" in df.columns and "pruning_ratio" in df.columns:
            df = df.sort_values(["model_name", "pruning_ratio"])

        csv_file = self.output_dir / "evaluation_summary.csv"

        df_csv = df.copy()
        complex_columns = ["per_class_iou", "per_class_acc", "confusion_matrix"]

        for col in complex_columns:
            if col in df_csv.columns:
                df_csv[col] = df_csv[col].apply(self._serialize_complex_field)

        df_csv.to_csv(csv_file, index=False)
        logger.info(f"Saved summary to {csv_file}")

        return str(csv_file)

    def _serialize_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        json_data = {}
        for k, v in data.items():
            if hasattr(v, "tolist"):
                json_data[k] = v.tolist()
            else:
                json_data[k] = v
        return json_data

    def _serialize_complex_field(self, value):
        if isinstance(value, (list, np.ndarray)):
            return json.dumps(
                value.tolist() if isinstance(value, np.ndarray) else value
            )
        return value


class ResultProcessor:
    def __init__(self, output_dir: str):
        self.formatter = ResultFormatter()
        self.saver = ResultSaver(output_dir)
        self.results = []

    def add_result(self, result: Dict[str, Any]):
        self.results.append(result)

    def process_single_result(self, result: Dict[str, Any], model_name: str):
        self.formatter.print_single_result(result)

        self.saver.save_single_model_results(result, model_name)

        self.add_result(result)

    def generate_summary(self, save_csv: bool = True) -> str:
        if not self.results:
            logger.warning("No results to summarize")
            return ""

        df = pd.DataFrame(self.results)
        self.formatter.print_summary_table(df)

        csv_file = ""
        if save_csv:
            csv_file = self.saver.save_summary_csv(self.results)

        return csv_file

    def get_results_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results)
