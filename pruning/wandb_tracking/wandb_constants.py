"""Constants and configuration values for WandB tracking"""

from typing import List


class WandBConstants:
    DEFAULT_PROJECT = "epic-v2"
    DEFAULT_ENTITY = None

    PREFIX_MODEL = "model"
    PREFIX_PRUNING = "pruning"
    PREFIX_LOSS = "loss"
    PREFIX_TRAINING = "training"
    PREFIX_BATCH = "batch"
    PREFIX_KD = "kd"
    PREFIX_SENSITIVITY = "sensitivity"
    PREFIX_QUOTAS = "quotas"
    PREFIX_COMPARISON = "comparison"
    PREFIX_FINAL = "final"
    PREFIX_CHARTS = "charts"

    METRIC_PARAMS = "params"
    METRIC_PARAMS_MILLION = "params_million"
    METRIC_SIZE_MB = "size_mb"
    METRIC_MACS = "macs"
    METRIC_MACS_MILLION = "macs_million"
    METRIC_LAYERS = "layers"
    METRIC_COMPRESSION_RATIO = "compression_ratio"
    METRIC_STEP = "step"
    METRIC_PROGRESS = "progress"
    METRIC_MIOU = "miou"
    METRIC_MEAN_ACC = "mean_acc"

    BYTES_PER_FLOAT32 = 4
    MB_DIVISOR = 1024 * 1024
    MILLION_DIVISOR = 1e6

    DEFAULT_FIGURE_SIZE = (10, 6)
    LARGE_FIGURE_SIZE = (15, 10)
    SMALL_FIGURE_SIZE = (8, 6)
    DASHBOARD_FIGURE_SIZE = (16, 10)

    MAX_SENSITIVITY_LAYERS = 20
    MAX_QUOTA_LAYERS = 20
    DEFAULT_TOP_N_LAYERS = 30

    COLOR_PRIMARY = "steelblue"
    COLOR_SECONDARY = "orange"
    COLOR_SUCCESS = "green"
    COLOR_WARNING = "red"
    COLOR_INFO = "blue"
    COLOR_PURPLE = "purple"

    GRID_ALPHA = 0.3
    FILL_ALPHA = 0.25
    BAR_ALPHA = 0.7

    ARTIFACT_TYPE_MODEL = "model"

    ERROR_WANDB_INIT_FAILED = "Failed to initialize WandB: {}"
    ERROR_MODEL_SAVE_FAILED = "Failed to save model to WandB: {}"
    ERROR_GRAPH_LOG_FAILED = "Failed to log model graph: {}"

    WARNING_WANDB_DISABLED = "WandB tracking disabled"
    WARNING_WANDB_INIT_FAILED = "Failed to initialize WandB: {}"
    WARNING_MODEL_SAVE_FAILED = "Failed to save model to WandB: {}"
    WARNING_GRAPH_LOG_FAILED = "Failed to log model graph: {}"

    INFO_WANDB_INITIALIZED = "WandB tracking initialized: {}"
    INFO_MODEL_SAVED = "Model checkpoint saved to WandB: {}"
    INFO_MODEL_GRAPH_LOGGED = "Model architecture logged to WandB"
    INFO_WANDB_FINISHED = "WandB run finished"
    INFO_DASHBOARD_CREATED = "Created final summary dashboard for WandB"


class ChartConfig:
    STYLE_SETTINGS = {
        "figure.figsize": WandBConstants.DEFAULT_FIGURE_SIZE,
        "axes.grid": True,
        "grid.alpha": WandBConstants.GRID_ALPHA,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }

    COLOR_PALETTE = "tab10"

    HEATMAP_CONFIG = {
        "annot": True,
        "fmt": ".4f",
        "cmap": "YlOrRd",
        "cbar_kws": {"label": "Taylor Sensitivity Score"},
    }

    HISTOGRAM_CONFIG = {
        "bins": 20,
        "edgecolor": "black",
        "alpha": WandBConstants.BAR_ALPHA,
    }

    SCATTER_CONFIG = {"cmap": "viridis", "s": 100, "alpha": 0.6, "edgecolors": "black"}


class MetricCalculator:
    @staticmethod
    def calculate_compression_ratio(
        initial_params: float, current_params: float
    ) -> float:
        if initial_params <= 0:
            return 1.0
        return initial_params / current_params

    @staticmethod
    def calculate_reduction_percentage(initial: float, current: float) -> float:
        if initial <= 0:
            return 0.0
        return (1 - current / initial) * 100

    @staticmethod
    def params_to_million(params: int) -> float:
        return params / WandBConstants.MILLION_DIVISOR

    @staticmethod
    def params_to_mb(params: int) -> float:
        return (params * WandBConstants.BYTES_PER_FLOAT32) / WandBConstants.MB_DIVISOR

    @staticmethod
    def macs_to_million(macs: int) -> float:
        return macs / WandBConstants.MILLION_DIVISOR


class TagGenerator:
    @staticmethod
    def generate_basic_tags(config) -> List[str]:
        tags = [
            config.model_name,
            config.pruning_strategy,
            f"ratio_{config.pruning_ratio}",
            f"steps_{config.iterative_steps}",
        ]
        return tags

    @staticmethod
    def add_kd_tags(tags: List[str], config) -> List[str]:
        if config.enable_kd_lite:
            tags.append("kd_lite")
            tags.append(f"kd_{getattr(config, 'kd_mode', 'replace')}")
        return tags


class NameGenerator:
    @staticmethod
    def generate_run_name(config) -> str:
        name_parts = [
            config.model_name,
            config.pruning_strategy,
            f"ratio_{config.pruning_ratio:.2f}",
        ]

        if config.enable_kd_lite:
            name_parts.append("kd")
            if getattr(config, "kd_mode", "") == "refine":
                name_parts.append("refine")

        return "_".join(name_parts)
