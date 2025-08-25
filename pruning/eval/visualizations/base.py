"""
Base visualization class with common functionality
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from .config import VisualizationConfig

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


class BaseVisualizer:
    def __init__(
        self,
        save_dir: str,  # Required, no default
        dpi: int = None,
        save_formats: List[str] = None,
        style: str = None,
    ):
        self.save_dir = Path(save_dir)
        self.dpi = dpi or VisualizationConfig.DEFAULT_DPI
        self.save_formats = save_formats or VisualizationConfig.DEFAULT_FORMATS
        self.config = VisualizationConfig()

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._configure_plotting(style)

    def _configure_plotting(self, style: Optional[str] = None) -> None:
        plt.rcParams.update(self.config.STYLE_CONFIG)
        sns.set_style("whitegrid")
        sns.set_context("paper")
        sns.set_palette("husl")

        if style and style in plt.style.available:
            plt.style.use(style)
        elif self.config.DEFAULT_STYLE in plt.style.available:
            plt.style.use(self.config.DEFAULT_STYLE)

    def _save_figure(self, fig, name: str) -> None:
        for fmt in self.save_formats:
            filepath = self.save_dir / f"{name}.{fmt}"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight", format=fmt)
            logger.info(f"Saved {filepath}")

    def _get_color_for_model(self, model_type: str) -> Optional[str]:
        return self.config.COLOR_PALETTE.get(model_type)

    def _format_column_name(self, col_name: str) -> str:
        return col_name.replace("_", " ").title()

    def _validate_data_columns(self, df, required_columns: List[str]) -> bool:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
        return True
