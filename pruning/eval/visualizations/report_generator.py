"""
Report generation module for evaluation results
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def generate_summary_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df is None or df.empty:
            return {}

        summary = {
            "dataset_info": self._get_dataset_info(df),
            "model_performance": self._get_model_performance(df),
            "pruning_analysis": self._get_pruning_analysis(df),
            "efficiency_metrics": self._get_efficiency_metrics(df),
            "best_models": self._get_best_models(df),
            "statistical_summary": self._get_statistical_summary(df),
        }

        report_file = self.save_dir / "summary_report.json"
        with open(report_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary report saved to {report_file}")

        return summary

    def _get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "total_models_evaluated": len(df),
            "unique_model_types": list(df["model_type"].unique())
            if "model_type" in df.columns
            else [],
            "pruning_ratios": sorted(df["pruning_ratio"].unique().tolist())
            if "pruning_ratio" in df.columns
            else [],
            "strategies": list(df["strategy"].unique())
            if "strategy" in df.columns
            else [],
            "fine_tune_methods": list(df["fine_tune_method"].unique())
            if "fine_tune_method" in df.columns
            else [],
        }

    def _get_model_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        performance = {}

        if "miou" in df.columns:
            performance["miou"] = {
                "overall_mean": float(df["miou"].mean()),
                "overall_std": float(df["miou"].std()),
                "overall_min": float(df["miou"].min()),
                "overall_max": float(df["miou"].max()),
            }

        if "mean_acc" in df.columns:
            performance["mean_accuracy"] = {
                "overall_mean": float(df["mean_acc"].mean()),
                "overall_std": float(df["mean_acc"].std()),
                "overall_min": float(df["mean_acc"].min()),
                "overall_max": float(df["mean_acc"].max()),
            }

        return performance

    def _get_pruning_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        analysis = {}

        if "pruning_ratio" in df.columns and "miou" in df.columns:
            pruning_stats = (
                df.groupby("pruning_ratio")["miou"]
                .agg(["mean", "std", "count"])
                .to_dict("index")
            )
            analysis["performance_by_pruning_ratio"] = pruning_stats

        if "strategy" in df.columns and "miou" in df.columns:
            strategy_stats = (
                df.groupby("strategy")["miou"]
                .agg(["mean", "std", "count"])
                .to_dict("index")
            )
            analysis["performance_by_strategy"] = strategy_stats

        return analysis

    def _get_efficiency_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        efficiency = {}

        if "params_percentage" in df.columns:
            efficiency["parameter_reduction"] = {
                "mean_params_remaining": float(df["params_percentage"].mean()),
                "best_compression": float(df["params_percentage"].min()),
                "compression_range": [
                    float(df["params_percentage"].min()),
                    float(df["params_percentage"].max()),
                ],
            }

        if "size_percentage" in df.columns:
            efficiency["size_reduction"] = {
                "mean_size_remaining": float(df["size_percentage"].mean()),
                "best_compression": float(df["size_percentage"].min()),
                "compression_range": [
                    float(df["size_percentage"].min()),
                    float(df["size_percentage"].max()),
                ],
            }

        if "mac_reduction" in df.columns:
            efficiency["mac_reduction"] = {
                "mean_reduction": float(df["mac_reduction"].mean()),
                "best_reduction": float(df["mac_reduction"].max()),
                "reduction_range": [
                    float(df["mac_reduction"].min()),
                    float(df["mac_reduction"].max()),
                ],
            }

        if "gmacs" in df.columns:
            efficiency["gmacs"] = {
                "mean": float(df["gmacs"].mean()),
                "min": float(df["gmacs"].min()),
                "max": float(df["gmacs"].max()),
            }

        if "miou" in df.columns and "params_percentage" in df.columns:
            efficiency_ratio = df["miou"] / (df["params_percentage"] / 100)
            efficiency["performance_per_parameter"] = {
                "mean": float(efficiency_ratio.mean()),
                "std": float(efficiency_ratio.std()),
                "max": float(efficiency_ratio.max()),
            }

        if "miou" in df.columns and "mac_reduction" in df.columns:
            mac_efficiency = df["miou"] / (100 - df["mac_reduction"])
            efficiency["performance_per_mac"] = {
                "mean": float(mac_efficiency.mean()),
                "std": float(mac_efficiency.std()),
                "max": float(mac_efficiency.max()),
            }

        return efficiency

    def _get_best_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        best_models = {}

        if "miou" in df.columns:
            best_idx = df["miou"].idxmax()
            best_row = df.loc[best_idx]
            best_models["best_performance"] = self._model_row_to_dict(best_row)

            if "params_percentage" in df.columns:
                efficiency_ratio = df["miou"] / (df["params_percentage"] / 100)
                best_efficiency_idx = efficiency_ratio.idxmax()
                best_efficiency_row = df.loc[best_efficiency_idx]
                best_models["best_efficiency"] = self._model_row_to_dict(
                    best_efficiency_row
                )

            # > 80% of max mIoU threshold for good performance
            if "size_percentage" in df.columns:
                performance_threshold = df["miou"].max() * 0.8
                good_performance_df = df[df["miou"] >= performance_threshold]
                if not good_performance_df.empty:
                    most_compressed_idx = good_performance_df[
                        "size_percentage"
                    ].idxmin()
                    most_compressed_row = df.loc[most_compressed_idx]
                    best_models["most_compressed_good_performance"] = (
                        self._model_row_to_dict(most_compressed_row)
                    )

        return best_models

    def _get_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        summary = {}

        if "model_type" in df.columns:
            for model_type in df["model_type"].unique():
                model_data = df[df["model_type"] == model_type]
                model_stats = {}

                for metric in [
                    "miou",
                    "mean_acc",
                    "params_percentage",
                    "size_percentage",
                    "gmacs",
                    "mac_reduction",
                ]:
                    if metric in model_data.columns:
                        model_stats[metric] = {
                            "mean": float(model_data[metric].mean()),
                            "std": float(model_data[metric].std()),
                            "min": float(model_data[metric].min()),
                            "max": float(model_data[metric].max()),
                            "count": int(model_data[metric].count()),
                        }

                summary[model_type] = model_stats

        return summary

    def _model_row_to_dict(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "model_name": row.get("model_name", ""),
            "model_type": row.get("model_type", ""),
            "pruning_ratio": float(row.get("pruning_ratio", 0)),
            "miou": float(row.get("miou", 0)),
            "mean_acc": float(row.get("mean_acc", 0)) if "mean_acc" in row else None,
            "params_percentage": float(row.get("params_percentage", 100))
            if "params_percentage" in row
            else None,
            "size_percentage": float(row.get("size_percentage", 100))
            if "size_percentage" in row
            else None,
            "strategy": row.get("strategy", ""),
            "fine_tune_method": row.get("fine_tune_method", ""),
        }

    def generate_comparison_report(
        self,
        df: pd.DataFrame,
        comparison_column: str = "model_type",
        metric: str = "miou",
    ) -> Dict[str, Any]:
        if (
            df is None
            or df.empty
            or comparison_column not in df.columns
            or metric not in df.columns
        ):
            return {}

        comparison_report = {
            "comparison_type": comparison_column,
            "metric": metric,
            "groups": {},
            "ranking": [],
        }

        groups = df.groupby(comparison_column)
        for group_name, group_data in groups:
            group_stats = {
                "count": len(group_data),
                "mean": float(group_data[metric].mean()),
                "std": float(group_data[metric].std()),
                "min": float(group_data[metric].min()),
                "max": float(group_data[metric].max()),
                "median": float(group_data[metric].median()),
            }
            comparison_report["groups"][group_name] = group_stats
            comparison_report["ranking"].append(
                {"name": group_name, "mean_score": group_stats["mean"]}
            )

        comparison_report["ranking"].sort(key=lambda x: x["mean_score"], reverse=True)

        report_file = self.save_dir / f"{comparison_column}_{metric}_comparison.json"
        with open(report_file, "w") as f:
            json.dump(comparison_report, f, indent=2, default=str)
        logger.info(f"Comparison report saved to {report_file}")

        return comparison_report
