"""Final results and comparison logging for WandB tracking"""

import logging
from typing import Any, Dict, List, Optional

import wandb

from .wandb_constants import MetricCalculator, WandBConstants

logger = logging.getLogger(__name__)


class FinalResultLogger:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def log_final_results(
        self, results: Dict[str, Any], config: Any, global_step: int
    ) -> None:
        if not self.enabled:
            return

        summary = {
            f"{WandBConstants.PREFIX_FINAL}/params_reduction": results[
                "params_reduction"
            ],
            f"{WandBConstants.PREFIX_FINAL}/size_reduction": results["size_reduction"],
            f"{WandBConstants.PREFIX_FINAL}/initial_params": results["initial_params"],
            f"{WandBConstants.PREFIX_FINAL}/final_params": results["final_params"],
            f"{WandBConstants.PREFIX_FINAL}/initial_size_mb": results[
                "initial_size_mb"
            ],
            f"{WandBConstants.PREFIX_FINAL}/final_size_mb": results["final_size_mb"],
            f"{WandBConstants.PREFIX_FINAL}/compression_ratio": results["final_params"]
            / results["initial_params"],
        }

        final_step = config.iterative_steps
        wandb.log(summary, step=final_step)

        self._create_summary_table(results, global_step)

    def _create_summary_table(self, results: Dict[str, Any], global_step: int) -> None:
        table_data = [
            ["Parameter Reduction", f"{results['params_reduction']:.2%}"],
            ["Size Reduction", f"{results['size_reduction']:.2%}"],
            [
                "Parameters",
                f"{results['initial_params']:,} → {results['final_params']:,}",
            ],
            [
                "Model Size",
                f"{results['initial_size_mb']:.2f} MB → {results['final_size_mb']:.2f} MB",
            ],
        ]

        if results.get("kd_lite_enabled"):
            table_data.extend(
                [
                    ["KD Temperature", str(results.get("kd_temperature", "N/A"))],
                    ["KD Alpha", str(results.get("kd_alpha", "N/A"))],
                    ["KD Data Ratio", str(results.get("kd_data_ratio", "N/A"))],
                ]
            )

        table = wandb.Table(columns=["Metric", "Value"], data=table_data)
        wandb.log(
            {f"{WandBConstants.PREFIX_FINAL}/summary_table": table}, step=global_step
        )


class ComparisonLogger:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def create_comparison_chart(
        self,
        before_stats: Dict,
        after_stats: Dict,
        title: str,
        global_step: int,
        iterative_steps: int,
    ) -> None:
        if not self.enabled:
            return

        params_before = MetricCalculator.params_to_million(before_stats["params"])
        params_after = MetricCalculator.params_to_million(after_stats["params"])
        params_reduction = MetricCalculator.calculate_reduction_percentage(
            params_before, params_after
        )

        macs_before = MetricCalculator.macs_to_million(before_stats.get("macs", 0))
        macs_after = MetricCalculator.macs_to_million(after_stats.get("macs", 0))
        macs_reduction = MetricCalculator.calculate_reduction_percentage(
            macs_before, macs_after
        )

        self._log_summary_metrics(
            params_before,
            params_after,
            params_reduction,
            macs_before,
            macs_after,
            macs_reduction,
        )

        self._create_comparison_table(
            params_before,
            params_after,
            params_reduction,
            macs_before,
            macs_after,
            macs_reduction,
            global_step,
        )

        self._create_bar_charts(
            params_before, params_after, macs_before, macs_after, iterative_steps
        )

    def _log_summary_metrics(
        self,
        params_before: float,
        params_after: float,
        params_reduction: float,
        macs_before: float,
        macs_after: float,
        macs_reduction: float,
    ) -> None:
        wandb.run.summary[f"{WandBConstants.PREFIX_COMPARISON}/params_before_M"] = (
            params_before
        )
        wandb.run.summary[f"{WandBConstants.PREFIX_COMPARISON}/params_after_M"] = (
            params_after
        )
        wandb.run.summary[f"{WandBConstants.PREFIX_COMPARISON}/params_reduction_%"] = (
            params_reduction
        )
        wandb.run.summary[f"{WandBConstants.PREFIX_COMPARISON}/macs_before_M"] = (
            macs_before
        )
        wandb.run.summary[f"{WandBConstants.PREFIX_COMPARISON}/macs_after_M"] = (
            macs_after
        )
        wandb.run.summary[f"{WandBConstants.PREFIX_COMPARISON}/macs_reduction_%"] = (
            macs_reduction
        )

    def _create_comparison_table(
        self,
        params_before: float,
        params_after: float,
        params_reduction: float,
        macs_before: float,
        macs_after: float,
        macs_reduction: float,
        global_step: int,
    ) -> None:
        comparison_table = wandb.Table(
            columns=["Metric", "Before", "After", "Reduction (%)"],
            data=[
                [
                    "Parameters (M)",
                    f"{params_before:.2f}",
                    f"{params_after:.2f}",
                    f"{params_reduction:.1f}%",
                ],
                [
                    "MACs (M)",
                    f"{macs_before:.1f}",
                    f"{macs_after:.1f}",
                    f"{macs_reduction:.1f}%",
                ],
                [
                    "Compression Ratio",
                    "100%",
                    f"{(params_after / params_before * 100):.1f}%",
                    f"{params_reduction:.1f}%",
                ],
            ],
        )

        wandb.log(
            {f"{WandBConstants.PREFIX_COMPARISON}/summary_table": comparison_table},
            step=global_step,
        )

    def _create_bar_charts(
        self,
        params_before: float,
        params_after: float,
        macs_before: float,
        macs_after: float,
        iterative_steps: int,
    ) -> None:
        params_chart = wandb.Table(
            columns=["Stage", "Parameters (M)"],
            data=[
                ["Before", params_before],
                ["After", params_after],
            ],
        )

        macs_chart = wandb.Table(
            columns=["Stage", "MACs (M)"],
            data=[
                ["Before", macs_before],
                ["After", macs_after],
            ],
        )

        wandb.log(
            {
                f"{WandBConstants.PREFIX_COMPARISON}/parameters_chart": wandb.plot.bar(
                    params_chart,
                    "Stage",
                    "Parameters (M)",
                    title="Model Parameters - Before vs After",
                ),
                f"{WandBConstants.PREFIX_COMPARISON}/macs_chart": wandb.plot.bar(
                    macs_chart,
                    "Stage",
                    "MACs (M)",
                    title="Computational Complexity - Before vs After",
                ),
            },
            step=iterative_steps,
        )


class ArtifactManager:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def save_model_checkpoint(
        self,
        model_path: str,
        model_name: str,
        pruning_strategy: str,
        run: Any,
        aliases: Optional[List[str]] = None,
    ) -> None:
        if not self.enabled:
            return

        try:
            artifact = wandb.Artifact(
                name=f"pruned_model_{model_name}",
                type=WandBConstants.ARTIFACT_TYPE_MODEL,
                description=f"Pruned {model_name} with {pruning_strategy}",
            )
            artifact.add_file(model_path)

            if aliases:
                run.log_artifact(artifact, aliases=aliases)
            else:
                run.log_artifact(artifact)

            logger.info(WandBConstants.INFO_MODEL_SAVED.format(model_path))

        except Exception as e:
            logger.warning(WandBConstants.WARNING_MODEL_SAVE_FAILED.format(str(e)))
