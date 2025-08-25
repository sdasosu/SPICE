import logging
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from config import EvaluationConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device)

        self._setup_logging()

        self._set_seeds()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _set_seeds(self, seed: int = 42):
        try:
            import sys
            from pathlib import Path

            sys.path.append(str(Path(__file__).parent.parent.parent))
            from pruning.utils import set_random_seeds

            set_random_seeds(seed)
        except ImportError:
            logger.warning("Could not import set_random_seeds function")

    def create_output_directories(self):
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        if self.config.generate_visualizations:
            Path(self.config.visualization_dir).mkdir(parents=True, exist_ok=True)
            advanced_viz_dir = Path(self.config.visualization_dir) / "advanced"
            advanced_viz_dir.mkdir(parents=True, exist_ok=True)

    def should_use_wandb(self) -> bool:
        return self.config.use_wandb

    def get_wandb_config(self) -> dict:
        return {
            "project": self.config.wandb_project,
            "entity": self.config.wandb_entity,
            "name": self.config.wandb_run_name or "pruning_evaluation",
            "tags": self.config.wandb_tags,
            "config": self.config.__dict__,
        }

    @property
    def visualization_enabled(self) -> bool:
        return self.config.generate_visualizations

    @property
    def cache_enabled(self) -> bool:
        return getattr(self.config, "use_cache", True)
