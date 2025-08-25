"""
Pruning strategies for structured pruning
"""

import logging
from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn
import torch_pruning as tp

from .config import PruningConfig

logger = logging.getLogger(__name__)


class BasePruningStrategy(ABC):
    """Base class for pruning strategies"""

    def __init__(self, config: PruningConfig):
        self.config = config

    @abstractmethod
    def create_importance(self) -> tp.importance.Importance:
        pass

    def create_pruner(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor,
        ignored_layers: List[nn.Module],
    ) -> tp.pruner.MagnitudePruner:
        importance = self.create_importance()

        pruner = tp.pruner.MagnitudePruner(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            pruning_ratio=self.config.pruning_ratio,
            iterative_steps=self.config.iterative_steps,
            ignored_layers=ignored_layers,
            global_pruning=True,
        )

        return pruner


class MagnitudePruningStrategy(BasePruningStrategy):
    """Magnitude-based structured pruning"""

    def create_importance(self) -> tp.importance.Importance:
        return tp.importance.MagnitudeImportance(p=self.config.importance_norm)


class RandomPruningStrategy(BasePruningStrategy):
    """Random structured pruning"""

    def create_importance(self) -> tp.importance.Importance:
        return tp.importance.RandomImportance()


class TaylorPruningStrategy(BasePruningStrategy):
    """Taylor importance-based structured pruning"""

    def create_importance(self) -> tp.importance.Importance:
        return tp.importance.TaylorImportance()


class LAMPPruningStrategy(BasePruningStrategy):
    """LAMP (Layer-adaptive Magnitude-based Pruning) strategy"""

    def create_importance(self) -> tp.importance.Importance:
        return tp.importance.LAMPImportance()


class PruningStrategyFactory:
    """Factory for creating pruning strategies"""

    STRATEGIES = {
        "magnitude": MagnitudePruningStrategy,
        "random": RandomPruningStrategy,
        "taylor": TaylorPruningStrategy,
        "lamp": LAMPPruningStrategy,
    }

    @classmethod
    def create_strategy(cls, config: PruningConfig) -> BasePruningStrategy:
        strategy_name = config.pruning_strategy.lower()

        if strategy_name not in cls.STRATEGIES:
            raise ValueError(f"Unknown pruning strategy: {strategy_name}")

        strategy_class = cls.STRATEGIES[strategy_name]
        return strategy_class(config)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """List available pruning strategies"""
        return list(cls.STRATEGIES.keys())


class ProgressivePruner:
    """Handles progressive pruning execution"""

    def __init__(self, strategy: BasePruningStrategy, config: PruningConfig):
        self.strategy = strategy
        self.config = config
        self.pruner = None

    def setup_pruner(
        self, model: nn.Module, ignored_layers: List[nn.Module]
    ) -> tp.pruner.MagnitudePruner:
        """Setup pruner for the model"""
        # Use fixed seed for consistent example inputs
        torch.manual_seed(self.config.seed)
        example_inputs = torch.randn(*self.config.example_input_size).to(
            self.config.device
        )
        self.pruner = self.strategy.create_pruner(model, example_inputs, ignored_layers)
        return self.pruner

    def prune_step(self) -> dict:
        """Execute one pruning step"""
        if self.pruner is None:
            raise RuntimeError("Pruner not setup. Call setup_pruner first.")

        # Execute pruning step
        self.pruner.step()

        # Get model statistics
        model = self.pruner.model

        # Calculate statistics using torch_pruning utilities
        try:
            # Use fixed seed for consistent statistics calculation
            torch.manual_seed(self.config.seed)
            example_inputs = torch.randn(*self.config.example_input_size).to(
                self.config.device
            )
            macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        except Exception as e:
            logger.warning(f"Could not calculate MACs: {e}")
            macs = 0
            nparams = sum(p.numel() for p in model.parameters())

        stats = {
            "params": nparams,
            "params_million": nparams / 1e6,
            "macs": macs,
            "macs_million": macs / 1e6,
        }

        logger.info(
            f"After pruning step: {stats['params_million']:.2f}M params, "
            f"{stats['macs_million']:.2f}M MACs"
        )

        return stats
