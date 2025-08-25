"""
Artifact and visualization management for WandB evaluation tracking
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

try:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 200000000
except ImportError:
    pass

try:
    from ...wandb_tracking.wandb_results import ArtifactManager
except ImportError:
    import sys
    from pathlib import Path as PathModule

    project_root = PathModule(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not installed. Install with: pip install wandb")

logger = logging.getLogger(__name__)


class VisualizationUploader:
    """Handles uploading of visualization images to WandB"""

    def __init__(self, enabled: bool = True):
        """Initialize visualization uploader"""
        self.enabled = enabled and WANDB_AVAILABLE

    def find_image_files(self, visualizations_dir: Path) -> List[Path]:
        """Find all image files in visualization directory"""
        if not visualizations_dir.exists() or not visualizations_dir.is_dir():
            logger.warning(f"Visualizations directory not found: {visualizations_dir}")
            return []

        image_files = list(visualizations_dir.rglob("*.png"))
        return image_files

    def create_image_key(self, img_path: Path, visualizations_dir: Path) -> str:
        """Create a descriptive key for the image based on its path"""
        relative_path = img_path.relative_to(visualizations_dir)

        if relative_path.parent == Path("."):
            return f"evaluation/visualizations/{img_path.stem}"
        else:
            return f"evaluation/visualizations/{relative_path.parent}/{img_path.stem}"

    def prepare_images_dict(
        self, image_files: List[Path], visualizations_dir: Path
    ) -> Dict[str, any]:
        """Prepare dictionary of images for uploading"""
        images_dict = {}

        for img_path in image_files:
            key = self.create_image_key(img_path, visualizations_dir)
            images_dict[key] = wandb.Image(str(img_path), caption=img_path.stem)

        return images_dict

    def log_visualizations(self, visualizations_dir: Path) -> None:
        """Upload visualization images to WandB"""
        if not self.enabled:
            return

        try:
            image_files = self.find_image_files(visualizations_dir)

            if not image_files:
                logger.warning(f"No PNG files found in {visualizations_dir}")
                return

            images_dict = self.prepare_images_dict(image_files, visualizations_dir)

            if images_dict:
                wandb.log(images_dict)
                logger.info(f"Uploaded {len(image_files)} visualizations to WandB")

        except Exception as e:
            logger.error(f"Failed to log visualizations: {e}")


class ArtifactLogger:
    """Handles logging of evaluation artifacts"""

    def __init__(self, enabled: bool = True):
        """Initialize artifact logger"""
        self.enabled = enabled and WANDB_AVAILABLE

    def create_evaluation_artifact(self, results_dir: Path) -> Optional[any]:
        """Create evaluation artifact with all result files"""
        if not self.enabled:
            return None

        try:
            artifact = wandb.Artifact(
                name="evaluation-results",
                type="evaluation",
                description="Pruned model evaluation results",
            )

            if results_dir.exists() and results_dir.is_dir():
                for file_path in results_dir.iterdir():
                    if file_path.is_file():
                        artifact.add_file(str(file_path))

            return artifact

        except Exception as e:
            logger.error(f"Failed to create evaluation artifact: {e}")
            return None

    def log_artifacts(self, results_dir: Path, run) -> None:
        """Save evaluation results as WandB artifacts"""
        if not self.enabled or not run:
            return

        try:
            artifact = self.create_evaluation_artifact(results_dir)

            if artifact:
                run.log_artifact(artifact)
                logger.info(f"Logged evaluation artifacts from {results_dir}")

        except Exception as e:
            logger.error(f"Failed to log artifacts: {e}")


class PerClassAnalyzer:
    """Handles per-class analysis and visualization"""

    def __init__(self, enabled: bool = True):
        """Initialize per-class analyzer"""
        self.enabled = enabled and WANDB_AVAILABLE

    def create_per_class_columns(self, df, class_names: List[str]) -> None:
        """Create per-class IoU columns in dataframe"""
        for i, class_name in enumerate(class_names):
            col_name = f"iou_{class_name}"
            if col_name not in df.columns and "per_class_iou" in df.columns:
                df[col_name] = df["per_class_iou"].apply(
                    lambda x: x[i] if isinstance(x, list) and len(x) > i else 0
                )

    def calculate_average_class_iou(
        self, model_data, class_names: List[str]
    ) -> List[float]:
        """Calculate average per-class IoU for model type"""
        class_ious = []
        for class_name in class_names:
            col_name = f"iou_{class_name}"
            if col_name in model_data.columns:
                class_ious.append(model_data[col_name].mean())
            else:
                class_ious.append(0)
        return class_ious

    def create_radar_chart(
        self, class_ious: List[float], class_names: List[str], model_type: str
    ):
        """Create radar chart for per-class IoU analysis"""
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

        angles = [n / len(class_names) * 2 * 3.14159 for n in range(len(class_names))]
        angles += angles[:1]
        class_ious += class_ious[:1]

        ax.plot(angles, class_ious, "o-", linewidth=2)
        ax.fill(angles, class_ious, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(class_names)
        ax.set_ylim(0, 1)
        ax.set_title(f"{model_type} - Per-Class IoU", fontsize=16)
        ax.grid(True)

        plt.tight_layout()
        return fig

    def log_per_class_analysis(self, df, class_names: List[str]) -> None:
        """Log per-class IoU analysis"""
        if not self.enabled or df is None or df.empty:
            return

        try:
            self.create_per_class_columns(df, class_names)

            for model_type in df["model_type"].unique():
                model_data = df[df["model_type"] == model_type]

                class_ious = self.calculate_average_class_iou(model_data, class_names)

                fig = self.create_radar_chart(class_ious, class_names, model_type)

                wandb.log(
                    {f"evaluation/per_class_radar_{model_type}": wandb.Image(fig)}
                )
                plt.close(fig)

            logger.info("Logged per-class analysis to WandB")

        except Exception as e:
            logger.error(f"Failed to log per-class analysis: {e}")


class VisualizationManager:
    """Manages all visualization-related functionality"""

    def __init__(self, enabled: bool = True):
        """Initialize visualization manager"""
        self.enabled = enabled and WANDB_AVAILABLE
        self.uploader = VisualizationUploader(enabled)
        self.artifact_logger = ArtifactLogger(enabled)
        self.per_class_analyzer = PerClassAnalyzer(enabled)

    def log_all_visualizations(
        self, visualizations_dir: Path, results_dir: Optional[Path] = None, run=None
    ) -> None:
        """Log all visualizations and artifacts"""
        if not self.enabled:
            return

        self.uploader.log_visualizations(visualizations_dir)

        if results_dir and run:
            self.artifact_logger.log_artifacts(results_dir, run)

    def analyze_per_class_performance(self, df, class_names: List[str]) -> None:
        """Analyze and log per-class performance"""
        if not self.enabled:
            return

        self.per_class_analyzer.log_per_class_analysis(df, class_names)
