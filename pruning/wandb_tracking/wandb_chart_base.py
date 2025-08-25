"""Base classes and utilities for chart creation"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns

import wandb

from .wandb_constants import ChartConfig, WandBConstants

logger = logging.getLogger(__name__)


class ChartStyleManager:
    @staticmethod
    def configure_matplotlib():
        for key, value in ChartConfig.STYLE_SETTINGS.items():
            plt.rcParams[key] = value

    @staticmethod
    def configure_seaborn():
        sns.set_style("whitegrid")
        sns.set_palette(ChartConfig.COLOR_PALETTE)

    @staticmethod
    def apply_grid(ax, alpha: float = WandBConstants.GRID_ALPHA):
        ax.grid(True, alpha=alpha)


class BaseChart(ABC):
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if self.enabled:
            ChartStyleManager.configure_matplotlib()
            ChartStyleManager.configure_seaborn()

    @abstractmethod
    def create_chart(self, *args, **kwargs) -> None:
        pass

    def _log_chart(self, fig, chart_name: str) -> None:
        if not self.enabled:
            return

        try:
            wandb.log(
                {f"{WandBConstants.PREFIX_CHARTS}/{chart_name}": wandb.Image(fig)}
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to log chart {chart_name}: {e}")
            plt.close(fig)


class MultiPanelChart(BaseChart):
    def _create_figure(self, nrows: int, ncols: int, figsize: tuple) -> tuple:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1 and ncols == 1:
            axes = [axes]
        elif nrows == 1 or ncols == 1:
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        return fig, axes

    def _finalize_chart(self, fig, chart_name: str) -> None:
        plt.tight_layout()
        self._log_chart(fig, chart_name)


class DataProcessor:
    @staticmethod
    def extract_numeric_values(
        data: List[Dict], key: str, default: float = 0.0
    ) -> List[float]:
        return [item.get(key, default) for item in data]

    @staticmethod
    def filter_top_n(
        data: List[Dict], key: str, n: int, reverse: bool = True
    ) -> List[Dict]:
        return sorted(data, key=lambda x: x.get(key, 0), reverse=reverse)[:n]

    @staticmethod
    def calculate_compression_ratios(
        initial_value: float, values: List[float]
    ) -> List[float]:
        return [initial_value / v if v > 0 else 1.0 for v in values]

    @staticmethod
    def safe_division(
        numerator: float, denominator: float, default: float = 0.0
    ) -> float:
        return numerator / denominator if denominator != 0 else default


class ColorManager:
    @staticmethod
    def get_chart_colors(n_colors: int) -> List[str]:
        colors = [
            WandBConstants.COLOR_PRIMARY,
            WandBConstants.COLOR_SECONDARY,
            WandBConstants.COLOR_SUCCESS,
            WandBConstants.COLOR_WARNING,
            WandBConstants.COLOR_INFO,
            WandBConstants.COLOR_PURPLE,
        ]

        while len(colors) < n_colors:
            colors.extend(colors)

        return colors[:n_colors]

    @staticmethod
    def get_gradient_colors(n_colors: int, cmap: str = "viridis") -> List[str]:
        cmap = plt.cm.get_cmap(cmap)
        return [cmap(i / (n_colors - 1)) for i in range(n_colors)]


class ValidationUtils:
    @staticmethod
    def validate_data_not_empty(data: List, data_name: str) -> bool:
        if not data:
            logger.warning(f"Empty data provided for {data_name}")
            return False
        return True

    @staticmethod
    def validate_keys_exist(
        data: List[Dict], required_keys: List[str], data_name: str
    ) -> bool:
        if not data:
            return False

        for item in data[:1]:
            for key in required_keys:
                if key not in item:
                    logger.warning(f"Missing key '{key}' in {data_name}")
                    return False
        return True

    @staticmethod
    def sanitize_numeric_list(values: List[Any]) -> List[float]:
        numeric_values = []
        for v in values:
            if isinstance(v, (int, float)):
                numeric_values.append(float(v))
            elif hasattr(v, "item"):
                try:
                    numeric_values.append(float(v.item()))
                except (TypeError, ValueError):
                    continue
            else:
                try:
                    numeric_values.append(float(v))
                except (TypeError, ValueError):
                    continue
        return numeric_values
