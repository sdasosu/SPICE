"""
Visualization utilities and helper functions
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from scipy.optimize import curve_fit
    from scipy.stats import gaussian_kde

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class VisualizationUtils:
    @staticmethod
    def extract_model_type(model_name: str) -> str:
        model_name_lower = model_name.lower()

        if "unet" in model_name_lower:
            if "efficientnet" in model_name_lower:
                return "UNET_EfficientNet"
            elif "resnet" in model_name_lower:
                return "UNET_ResNet"
        elif "deeplabv3plus" in model_name_lower:
            if "efficientnet" in model_name_lower:
                return "DeepLabV3Plus_EfficientNet"
            elif "resnet" in model_name_lower:
                return "DeepLabV3Plus_ResNet"
        elif "fpn" in model_name_lower:
            if "efficientnet" in model_name_lower:
                return "FPN_EfficientNet"
            elif "resnet" in model_name_lower:
                return "FPN_ResNet"

        return model_name

    @staticmethod
    def find_pareto_frontier(
        x: np.ndarray,
        y: np.ndarray,
        minimize_x: bool = True,
        maximize_y: bool = True,
    ) -> List[int]:
        points = np.column_stack((x, y))
        pareto_points = []

        for i, point in enumerate(points):
            is_pareto = True
            for j, other in enumerate(points):
                if i == j:
                    continue

                if minimize_x and maximize_y:
                    if other[0] <= point[0] and other[1] >= point[1]:
                        if other[0] < point[0] or other[1] > point[1]:
                            is_pareto = False
                            break
                elif minimize_x and not maximize_y:
                    if other[0] <= point[0] and other[1] <= point[1]:
                        if other[0] < point[0] or other[1] < point[1]:
                            is_pareto = False
                            break

            if is_pareto:
                pareto_points.append(i)

        return pareto_points

    @staticmethod
    def exponential_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    @staticmethod
    def calculate_density(
        x_data: np.ndarray, y_data: np.ndarray
    ) -> Optional[np.ndarray]:
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available for density calculation")
            return None

        try:
            xy = np.vstack([x_data, y_data])
            z = gaussian_kde(xy)(xy)
            return z
        except Exception as e:
            logger.warning(f"Failed to calculate density: {e}")
            return None

    @staticmethod
    def fit_exponential_curve(
        x_data: np.ndarray, y_data: np.ndarray
    ) -> Optional[tuple]:
        if not SCIPY_AVAILABLE:
            return None

        try:
            popt, _ = curve_fit(
                VisualizationUtils.exponential_decay,
                x_data,
                y_data,
                p0=[0.5, 1.0, 0.1],
            )
            return popt
        except Exception:
            return None

    @staticmethod
    def validate_hue_column(df: pd.DataFrame, hue_col: str) -> Optional[str]:
        if hue_col not in df.columns:
            return None

        unique_values = df[hue_col].nunique()
        if unique_values <= 1:
            logger.info(
                f"Only {unique_values} unique value(s) in {hue_col}, not using for hue"
            )
            return None

        return hue_col

    @staticmethod
    def process_per_class_iou(df: pd.DataFrame, class_names: List[str]) -> pd.DataFrame:
        if "per_class_iou" not in df.columns:
            return df

        import ast

        df = df.copy()
        df["per_class_iou"] = df["per_class_iou"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        for i, name in enumerate(class_names):
            df[f"iou_{name}"] = df["per_class_iou"].apply(
                lambda x: x[i] if isinstance(x, list) and len(x) > i else 0
            )

        return df
