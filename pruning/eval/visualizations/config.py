"""
Visualization configuration module
"""


class VisualizationConfig:
    OUTPUT_DIR = "outputs"

    STYLE_CONFIG = {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
        "lines.linewidth": 2,
        "lines.markersize": 8,
    }

    COLOR_PALETTE = {
        "UNET_ResNet": "#1f77b4",
        "UNET_EfficientNet": "#ff7f0e",
        "DeepLabV3Plus_ResNet": "#2ca02c",
        "DeepLabV3Plus_EfficientNet": "#d62728",
        "FPN_ResNet": "#9467bd",
        "FPN_EfficientNet": "#8c564b",
    }

    DEFAULT_DPI = 600
    DEFAULT_FORMATS = ["png", "pdf"]
    DEFAULT_FIGSIZE = (10, 8)
    DEFAULT_STYLE = "seaborn-v0_8-whitegrid"

    PLOT_CONFIGS = {
        "scatter": {
            "alpha": 0.7,
            "s": 100,
            "edgecolors": "black",
            "linewidth": 0.5,
        },
        "line": {
            "linewidth": 2,
            "markersize": 8,
            "alpha": 0.8,
        },
        "heatmap": {
            "linewidths": 0.5,
            "linecolor": "gray",
        },
        "3d": {
            "alpha": 0.7,
            "s": 100,
            "edgecolors": "black",
            "linewidth": 0.5,
            "elev": 20,
            "azim": 45,
        },
    }

    DEFAULT_CLASS_NAMES = [
        "Background",
        "Adult",
        "Egg masses",
        "Instar 1-3",
        "Instar 4",
    ]
