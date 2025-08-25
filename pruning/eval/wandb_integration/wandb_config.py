"""
Configuration management for WandB evaluation tracking
"""

import logging
from typing import Any, Dict, List, Optional

try:
    from ...wandb_tracking.wandb_constants import WandBConstants
except ImportError:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    from pruning.wandb_tracking.wandb_constants import WandBConstants

logger = logging.getLogger(__name__)


class EvaluationConfig:
    """Configuration class for evaluation tracking"""

    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "evaluation")
        self.pruning_strategy = kwargs.get("pruning_strategy", "evaluation")
        self.pruning_ratio = kwargs.get("pruning_ratio", 0.0)
        self.iterative_steps = kwargs.get("iterative_steps", 1)
        self.use_wandb = kwargs.get("enabled", True)
        self.enable_kd_lite = kwargs.get("enable_kd_lite", False)

        self._original_config = kwargs


class WandBConfigValidator:
    """Validates WandB configuration parameters"""

    @staticmethod
    def validate_project_name(project: Optional[str]) -> str:
        """Validate and return project name"""
        if not project:
            return WandBConstants.DEFAULT_PROJECT

        if not isinstance(project, str) or not project.strip():
            logger.warning("Invalid project name, using default")
            return WandBConstants.DEFAULT_PROJECT

        return project.strip()

    @staticmethod
    def validate_entity(entity: Optional[str]) -> Optional[str]:
        """Validate and return entity name"""
        if entity and isinstance(entity, str) and entity.strip():
            return entity.strip()
        return WandBConstants.DEFAULT_ENTITY

    @staticmethod
    def validate_tags(tags: Optional[List[str]]) -> List[str]:
        """Validate and return tags list"""
        if not tags:
            return ["evaluation", "pruning"]

        if not isinstance(tags, list):
            logger.warning("Tags must be a list, using default tags")
            return ["evaluation", "pruning"]

        valid_tags = [tag for tag in tags if isinstance(tag, str) and tag.strip()]

        default_tags = ["evaluation", "pruning"]
        for tag in default_tags:
            if tag not in valid_tags:
                valid_tags.append(tag)

        return valid_tags

    @staticmethod
    def validate_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate configuration dictionary"""
        if not config:
            return {"model_name": "evaluation"}

        if not isinstance(config, dict):
            logger.warning("Config must be a dictionary, using default")
            return {"model_name": "evaluation"}

        validated_config = config.copy()
        if "model_name" not in validated_config:
            validated_config["model_name"] = "evaluation"

        return validated_config


class WandBInitializer:
    """Handles WandB initialization and setup"""

    @staticmethod
    def create_evaluation_config(
        config: Optional[Dict[str, Any]], enabled: bool = True
    ) -> EvaluationConfig:
        """Create evaluation config for WandBTracker"""
        validated_config = WandBConfigValidator.validate_config(config)

        eval_config_params = validated_config.copy()
        model_name = eval_config_params.pop("model_name", "evaluation")

        return EvaluationConfig(
            enabled=enabled, model_name=model_name, **eval_config_params
        )

    @staticmethod
    def get_run_name(name: Optional[str]) -> str:
        """Generate run name"""
        return name or "evaluation-run"

    @staticmethod
    def prepare_init_params(
        project: Optional[str],
        entity: Optional[str],
        name: Optional[str],
        tags: Optional[List[str]],
        config: Optional[Dict[str, Any]],
        enabled: bool = True,
    ) -> Dict[str, Any]:
        """Prepare initialization parameters"""
        return {
            "project": WandBConfigValidator.validate_project_name(project),
            "entity": WandBConfigValidator.validate_entity(entity),
            "name": WandBInitializer.get_run_name(name),
            "tags": WandBConfigValidator.validate_tags(tags),
            "config": WandBInitializer.create_evaluation_config(config, enabled),
            "enabled": enabled,
        }
