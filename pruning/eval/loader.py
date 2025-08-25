"""
Model loader for pruned models
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PrunedModelLoader:
    """Loader for pruned models saved in different formats"""

    @staticmethod
    def load_pruned_model(
        model_path: str, device: torch.device = torch.device("cpu")
    ) -> nn.Module:
        """
        Load a pruned model from file

        Args:
            model_path: Path to the model file (.pth)
            device: Device to load the model on

        Returns:
            Loaded model
        """
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            model = torch.load(model_path, map_location=device, weights_only=False)
            logger.info(f"Successfully loaded pruned model from {model_path}")

            # Move model to device
            model.to(device)
            model.eval()

            return model

        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    @staticmethod
    def find_pruned_models(pruned_dir: str) -> list:
        """
        Find all pruned models in a directory

        Args:
            pruned_dir: Root directory containing pruned models

        Returns:
            List of (model_path, model_info) tuples
        """
        pruned_dir = Path(pruned_dir)

        if not pruned_dir.exists():
            logger.warning(f"Pruned models directory not found: {pruned_dir}")
            return []

        models = []

        # Look for organized structure: model_strategy_ratio/final_model.pth
        for model_dir in pruned_dir.iterdir():
            if model_dir.is_dir():
                final_model_path = model_dir / "final_model.pth"
                if final_model_path.exists():
                    dir_name = model_dir.name
                    parts = dir_name.split("_")

                    if len(parts) > 2 and parts[0].isdigit() and parts[1].isdigit():
                        parts = parts[2:]

                    uses_kd = False
                    kd_mode = None
                    if parts[-1] == "kd":
                        uses_kd = True
                        kd_mode = "replace"
                        parts = parts[:-1]
                    elif (
                        len(parts) >= 2 and parts[-2] == "kd" and parts[-1] == "refine"
                    ):
                        uses_kd = True
                        kd_mode = "refine"
                        parts = parts[:-2]

                    try:
                        ratio = float(parts[-1])
                        has_ratio = True
                    except ValueError:
                        ratio = 0.0
                        has_ratio = False

                    if has_ratio and len(parts) >= 3:
                        if (
                            len(parts) >= 4
                            and parts[-3] == "taylor"
                            and parts[-2] == "weight"
                        ):
                            model_name = "_".join(parts[:-3])
                            strategy = "taylor_weight"
                        elif (
                            len(parts) >= 4
                            and parts[-3] == "magnitude"
                            and parts[-2] == "taylor"
                        ):
                            model_name = "_".join(parts[:-3])
                            strategy = "magnitude_taylor"
                        else:
                            model_name = "_".join(parts[:-2])
                            strategy = parts[-2]

                        if uses_kd:
                            if kd_mode == "refine":
                                fine_tune_method = "Standard+KD"
                            else:
                                fine_tune_method = "KD-Lite"
                        else:
                            fine_tune_method = "Standard"

                        model_info = {
                            "path": str(final_model_path),
                            "model_name": model_name,
                            "strategy": strategy,
                            "pruning_ratio": ratio,
                            "dir_name": model_dir.name,
                            "fine_tune_method": fine_tune_method,
                        }
                        models.append(model_info)
                        logger.info(
                            f"Found pruned model: {model_dir.name} (model={model_name}, strategy={strategy}, fine-tune={fine_tune_method})"
                        )

        models.sort(key=lambda x: (x["model_name"], x["pruning_ratio"]))

        return models

    @staticmethod
    def load_intermediate_models(model_dir: str) -> list:
        """
        Load all intermediate models from a pruning job

        Args:
            model_dir: Directory containing the pruning job

        Returns:
            List of loaded models with step information
        """
        model_dir = Path(model_dir)
        intermediate_dir = model_dir / "intermediate"

        if not intermediate_dir.exists():
            logger.warning(f"No intermediate models found in {model_dir}")
            return []

        models = []

        for model_file in sorted(intermediate_dir.glob("*.pth")):
            filename = model_file.stem
            if "step" in filename:
                parts = filename.split("_step_")
                if len(parts) == 2:
                    step_info = parts[1].split("_of_")
                    if len(step_info) == 2:
                        current_step = int(step_info[0])
                        total_steps = int(step_info[1])

                        model_info = {
                            "path": str(model_file),
                            "step": current_step,
                            "total_steps": total_steps,
                        }
                        models.append(model_info)

        return models
