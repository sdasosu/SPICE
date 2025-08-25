"""
Configuration module for structured pruning
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .model_configs import MODEL_CONFIGS


@dataclass
class PruningConfig:
    """Configuration class for structured pruning parameters"""

    model_name: str
    output_dir: str = "pruned_models"
    data_root: str = "data"

    num_classes: int = 5
    input_size: Tuple[int, int] = (576, 576)
    example_input_size: Tuple[int, ...] = (1, 3, 576, 576)

    pruning_ratio: float = 0.5
    iterative_steps: int = 5
    pruning_strategy: str = "magnitude"
    importance_norm: int = 2

    device: str = "cuda"
    batch_size: int = 8
    fine_tune_epochs: int = 15
    final_fine_tune_epochs: int = 30
    fine_tune_lr: float = 1e-4
    weight_decay: float = 1e-4

    use_cosine_annealing: bool = True
    lr_warmup_epochs: int = 2
    lr_min_factor: float = 0.01

    early_stop_patience: int = 5

    seed: int = 42

    num_workers: int = 4

    use_timestamp: bool = False

    calibration_batches: int = 100
    min_out_channels: int = 8
    flops_alpha: float = 0.7

    use_taylor_sensitivity: bool = True
    sensitivity_alpha: float = 0.7
    enable_group_sync: bool = True

    enable_kd_lite: bool = False
    kd_mode: str = "refine"
    kd_temperature: float = 3.0
    kd_alpha: float = 0.5
    kd_data_ratio: float = 0.3
    kd_refine_epochs: int = 5
    freeze_backbone: bool = False
    boundary_weight: float = 1.0
    confidence_weight: bool = False

    use_wandb: bool = True
    wandb_project: str = "epic-v2"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_log_frequency: int = 10
    wandb_save_model: bool = True

    def __post_init__(self):
        if self.model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {self.model_name}")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @property
    def model_config(self) -> Dict[str, Any]:
        return MODEL_CONFIGS[self.model_name]

    @property
    def checkpoint_path(self) -> str:
        return self.model_config["checkpoint_path"]

    @property
    def output_path(self) -> str:
        filename = (
            f"pruned_{self.model_name}_{self.pruning_strategy}_{self.pruning_ratio:.2f}"
        )
        if self.enable_kd_lite:
            filename += "_kd"
        filename += ".pt"
        return str(Path(self.output_dir) / filename)
