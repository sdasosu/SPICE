"""
Core pruning functionality separated from orchestration
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch_pruning as tp

from .config import PruningConfig
from .dataset import PruningDataHandler
from .sensitivity import create_taylor_importance

logger = logging.getLogger(__name__)


class PruningOrchestrator:
    """
    Orchestrates the pruning process with different strategies
    """

    def __init__(self, config: PruningConfig):
        self.config = config
        self.wandb_tracker = None

    def set_wandb_tracker(self, tracker):
        self.wandb_tracker = tracker

    def setup_importance_and_pruner(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor,
        ignored_layers: List[nn.Module],
    ) -> Tuple[tp.pruner.MagnitudePruner, Dict]:
        """
        Setup importance criterion and pruner based on strategy

        Args:
            model: Model to prune
            example_inputs: Example inputs for dependency graph
            ignored_layers: Layers to ignore during pruning

        Returns:
            Tuple of (pruner, additional_info_dict)
        """
        strategy = self.config.pruning_strategy
        additional_info = {}

        if strategy == "taylor_weight":
            return self._setup_taylor_weight_pruning(
                model, example_inputs, ignored_layers, additional_info
            )
        elif strategy == "magnitude_taylor":
            return self._setup_hybrid_pruning(
                model, example_inputs, ignored_layers, additional_info
            )
        else:
            return self._setup_standard_pruning(
                model, example_inputs, ignored_layers, strategy
            ), additional_info

    def _setup_taylor_weight_pruning(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor,
        ignored_layers: List[nn.Module],
        additional_info: Dict,
    ) -> Tuple[tp.pruner.MagnitudePruner, Dict]:
        logger.info(
            "Using Taylor weight-based sensitivity analysis (pure Taylor mode)..."
        )

        data_handler = PruningDataHandler(
            data_root=self.config.data_root,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            seed=self.config.seed,
        )
        calibration_loader = data_handler.get_calibration_dataloader()
        importance = create_taylor_importance(model, calibration_loader, self.config)
        sensitivity_report = importance.get_sensitivity_report()
        additional_info["sensitivity_report"] = sensitivity_report
        pruner = tp.pruner.MagnitudePruner(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            pruning_ratio=self.config.pruning_ratio,
            iterative_steps=self.config.iterative_steps,
            ignored_layers=ignored_layers,
            global_pruning=True,
        )

        return pruner, additional_info

    def _setup_hybrid_pruning(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor,
        ignored_layers: List[nn.Module],
        additional_info: Dict,
    ) -> Tuple[tp.pruner.MagnitudePruner, Dict]:
        from .pruner import create_hybrid_pruner

        logger.info(
            "Using hybrid Magnitude+Taylor pruning (Taylor for quotas, Magnitude for selection)..."
        )

        data_handler = PruningDataHandler(
            data_root=self.config.data_root,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            seed=self.config.seed,
        )
        calibration_loader = data_handler.get_calibration_dataloader()
        taylor_importance = create_taylor_importance(
            model, calibration_loader, self.config
        )
        pruner, summary = create_hybrid_pruner(
            model=model,
            config=self.config,
            taylor_importance=taylor_importance,
            example_inputs=example_inputs,
            ignored_layers=ignored_layers,
        )
        sensitivity_report = taylor_importance.get_sensitivity_report()
        sensitivity_report["quota_allocation"] = summary
        additional_info["hybrid_report"] = sensitivity_report

        return pruner, additional_info

    def _setup_standard_pruning(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor,
        ignored_layers: List[nn.Module],
        strategy: str,
    ) -> tp.pruner.MagnitudePruner:
        if strategy == "magnitude":
            importance = tp.importance.MagnitudeImportance(
                p=self.config.importance_norm
            )
        elif strategy == "random":
            importance = tp.importance.RandomImportance()
        elif strategy == "taylor":
            importance = tp.importance.TaylorImportance()
        elif strategy == "lamp":
            importance = tp.importance.LAMPImportance()
        else:
            raise ValueError(f"Unknown pruning strategy: {strategy}")
        return tp.pruner.MagnitudePruner(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            pruning_ratio=self.config.pruning_ratio,
            iterative_steps=self.config.iterative_steps,
            ignored_layers=ignored_layers,
            global_pruning=True,
        )

    def save_reports(self, pruning_dir: Path, additional_info: Dict) -> None:
        if "sensitivity_report" in additional_info:
            report_path = pruning_dir / "sensitivity_report.json"
            with open(report_path, "w") as f:
                json.dump(additional_info["sensitivity_report"], f, indent=2)
            logger.info(f"Saved sensitivity report to {report_path}")

        if "hybrid_report" in additional_info:
            report_path = pruning_dir / "hybrid_pruning_report.json"
            with open(report_path, "w") as f:
                json.dump(additional_info["hybrid_report"], f, indent=2)
            logger.info(f"Saved hybrid pruning report to {report_path}")


def compute_model_statistics(
    model: nn.Module, example_inputs: torch.Tensor
) -> Dict[str, Any]:
    """
    Compute model statistics (parameters, MACs)

    Args:
        model: Model to analyze
        example_inputs: Example inputs for analysis

    Returns:
        Dictionary with model statistics
    """
    try:
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    except Exception as e:
        logger.warning(f"Could not calculate MACs: {e}")
        macs = 0
        nparams = sum(p.numel() for p in model.parameters())

    return {
        "params": nparams,
        "params_million": nparams / 1e6,
        "macs": macs,
        "macs_million": macs / 1e6,
    }
