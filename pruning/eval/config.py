"""
Configuration for pruned model evaluation
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class EvaluationConfig:
    """Configuration for evaluating pruned models"""

    # Paths
    pruned_models_dir: str = "pruned_models"
    data_root: str = "data"
    output_dir: str = "pruning/eval/results"
    visualization_dir: str = "pruning/eval/outputs"

    # Data settings
    batch_size: int = 32
    num_workers: int = 4
    img_size: int = 576
    num_classes: int = 5

    # Device settings
    device: str = "cuda"  # cuda, cpu, or auto

    # Evaluation settings
    save_predictions: bool = False  # Save prediction visualizations
    max_vis_samples: int = 10  # Maximum samples to visualize
    use_cache: bool = True  # Cache evaluation results to avoid re-evaluation

    # Reporting
    save_csv: bool = True
    save_plots: bool = True

    # Visualization settings
    generate_visualizations: bool = True
    figure_dpi: int = 300  # Reduced from 600 to avoid oversized images
    figure_formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    generate_advanced_plots: bool = True

    # WandB settings
    use_wandb: bool = True
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=lambda: ["evaluation", "pruning"])

    # Advanced visualization settings
    create_3d_plots: bool = True
    create_pareto_frontier: bool = True
    create_correlation_matrix: bool = True
    create_interactive_plots: bool = True

    def __post_init__(self):
        """Validate and setup configuration"""
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Create visualization directory if needed
        if self.generate_visualizations:
            Path(self.visualization_dir).mkdir(parents=True, exist_ok=True)
            Path(self.visualization_dir) / "advanced"
            (Path(self.visualization_dir) / "advanced").mkdir(
                parents=True, exist_ok=True
            )

        # Auto-detect device if specified
        if self.device == "auto":
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
