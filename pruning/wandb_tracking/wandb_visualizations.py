"""Enhanced visualizations for WandB pruning tracking"""

import logging
from typing import Any, Dict, List, Optional

import torch.nn as nn

from .wandb_chart_base import ChartStyleManager
from .wandb_chart_types import (
    KnowledgeDistillationChart,
    ModelComparisonRadar,
    PruningEfficiencyChart,
    PruningProgressChart,
    QuotaDistributionChart,
    SensitivityHeatmap,
)
from .wandb_constants import WandBConstants
from .wandb_dashboard import FinalSummaryDashboard

logger = logging.getLogger(__name__)


class PruningVisualizer:
    def __init__(self, tracker):
        self.tracker = tracker
        self.enabled = tracker.enabled if tracker else False

        if self.enabled:
            ChartStyleManager.configure_matplotlib()
            ChartStyleManager.configure_seaborn()

        self._initialize_chart_creators()

    def _initialize_chart_creators(self) -> None:
        self.progress_chart = PruningProgressChart(enabled=self.enabled)
        self.sensitivity_heatmap = SensitivityHeatmap(enabled=self.enabled)
        self.quota_distribution = QuotaDistributionChart(enabled=self.enabled)
        self.kd_chart = KnowledgeDistillationChart(enabled=self.enabled)
        self.radar_chart = ModelComparisonRadar(enabled=self.enabled)
        self.efficiency_chart = PruningEfficiencyChart(enabled=self.enabled)
        self.dashboard_creator = FinalSummaryDashboard(enabled=self.enabled)

    def create_pruning_progress_chart(
        self, pruning_history: List[Dict[str, Any]]
    ) -> None:
        self.progress_chart.create_chart(pruning_history)

    def create_sensitivity_heatmap(
        self,
        layer_sensitivities: List[Dict[str, Any]],
        top_n: int = WandBConstants.DEFAULT_TOP_N_LAYERS,
    ) -> None:
        self.sensitivity_heatmap.create_chart(layer_sensitivities, top_n)

    def create_layer_quota_distribution(
        self, layer_quotas: Dict[nn.Module, float], layer_names: Optional[Dict] = None
    ) -> None:
        self.quota_distribution.create_chart(layer_quotas, layer_names)

    def create_kd_analysis_charts(self, kd_history: List[Dict[str, float]]) -> None:
        self.kd_chart.create_chart(kd_history)

    def create_model_comparison_radar(
        self,
        original_metrics: Dict[str, float],
        pruned_metrics: Dict[str, float],
    ) -> None:
        self.radar_chart.create_chart(original_metrics, pruned_metrics)

    def create_pruning_efficiency_plot(self, pruning_steps: List[Dict]) -> None:
        self.efficiency_chart.create_chart(pruning_steps)

    def create_final_summary_dashboard(self, results: Dict[str, Any]) -> None:
        self.dashboard_creator.create_chart(results)
