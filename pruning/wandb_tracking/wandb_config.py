"""Configuration management for WandB tracking"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .wandb_constants import NameGenerator, TagGenerator, WandBConstants

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    pass


class WandBConfigManager:
    def __init__(
        self,
        config: Any,
        project: str = WandBConstants.DEFAULT_PROJECT,
        entity: Optional[str] = WandBConstants.DEFAULT_ENTITY,
        name: Optional[str] = None,
        tags: Optional[list] = None,
        enabled: bool = True,
    ):
        self.config = config
        self.project = project
        self.entity = entity
        self.name = name or NameGenerator.generate_run_name(config)
        self.tags = tags or self._generate_tags()
        self.enabled = enabled and getattr(config, "use_wandb", True)

        self._validate_config()

    def _validate_config(self):
        required_attrs = [
            "model_name",
            "pruning_strategy",
            "pruning_ratio",
            "iterative_steps",
        ]

        for attr in required_attrs:
            if not hasattr(self.config, attr):
                raise ConfigurationError(f"Missing required config attribute: {attr}")

        if not isinstance(self.project, str) or not self.project.strip():
            raise ConfigurationError("Project name must be a non-empty string")

    def _generate_tags(self) -> list:
        tags = TagGenerator.generate_basic_tags(self.config)
        tags = TagGenerator.add_kd_tags(tags, self.config)
        return tags

    def to_dict(self) -> Dict[str, Any]:
        config_dict = {}
        for key, value in vars(self.config).items():
            if not key.startswith("_"):
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                elif hasattr(value, "__dict__"):
                    config_dict[key] = str(value)
                else:
                    config_dict[key] = value
        return config_dict

    def get_init_params(self) -> Dict[str, Any]:
        return {
            "project": self.project,
            "entity": self.entity,
            "name": self.name,
            "tags": self.tags,
            "config": self.to_dict(),
        }


class ErrorHandler:
    @staticmethod
    def handle_init_error(error: Exception) -> bool:
        logger.warning(WandBConstants.WARNING_WANDB_INIT_FAILED.format(str(error)))
        return False

    @staticmethod
    def handle_model_save_error(error: Exception, model_path: str) -> None:
        logger.warning(WandBConstants.WARNING_MODEL_SAVE_FAILED.format(str(error)))

    @staticmethod
    def handle_graph_log_error(error: Exception) -> None:
        logger.warning(WandBConstants.WARNING_GRAPH_LOG_FAILED.format(str(error)))

    @staticmethod
    def log_success(message: str, *args) -> None:
        logger.info(message.format(*args) if args else message)
