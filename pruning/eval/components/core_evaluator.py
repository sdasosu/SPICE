import logging
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelInfoCalculator:
    def __init__(self):
        self._original_model_cache = {}

    def calculate_model_info(self, model: torch.nn.Module) -> Dict[str, float]:
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = total_params * 4 / (1024 * 1024)

        return {
            "total_params": total_params,
            "model_size_mb": model_size_mb,
        }

    def get_original_model_info(self, model_name: str) -> Dict[str, float]:
        if model_name in self._original_model_cache:
            return self._original_model_cache[model_name]

        try:
            import sys
            from pathlib import Path

            sys.path.append(str(Path(__file__).parent.parent.parent))
            from pruning.model_configs import MODEL_CONFIGS

            if model_name in MODEL_CONFIGS:
                import segmentation_models_pytorch as smp

                config = MODEL_CONFIGS[model_name]
                model_class = getattr(smp, config["architecture"])
                original_model = model_class(
                    encoder_name=config["encoder_name"],
                    encoder_weights=None,
                    in_channels=config["in_channels"],
                    classes=config["classes"],
                    activation=config["activation"],
                )

                info = self.calculate_model_info(original_model)
                info_with_prefix = {
                    "original_params": info["total_params"],
                    "original_size_mb": info["model_size_mb"],
                }

                self._original_model_cache[model_name] = info_with_prefix

                del original_model
                torch.cuda.empty_cache()

                return info_with_prefix
        except Exception as e:
            logger.warning(f"Could not get original model info for {model_name}: {e}")

        return {"original_params": 0, "original_size_mb": 0}

    def calculate_percentages(
        self, current_info: Dict, original_info: Dict
    ) -> Dict[str, float]:
        percentages = {}

        if original_info["original_params"] > 0:
            percentages["params_percentage"] = (
                current_info["total_params"] / original_info["original_params"]
            ) * 100
            percentages["size_percentage"] = (
                current_info["model_size_mb"] / original_info["original_size_mb"]
            ) * 100
        else:
            percentages["params_percentage"] = 100.0
            percentages["size_percentage"] = 100.0

        return percentages


class CoreEvaluator:
    def __init__(self, device: torch.device, num_classes: int):
        self.device = device
        self.num_classes = num_classes
        self.model_info_calc = ModelInfoCalculator()

    def evaluate_single_model(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        metrics_calculator,
        model_info: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        model.eval()
        with torch.no_grad():
            with tqdm(data_loader, desc="Evaluating") as pbar:
                for batch in pbar:
                    images, masks = self._extract_batch_data(batch)

                    outputs = self._forward_pass(model, images)

                    predictions = outputs.argmax(dim=1)

                    metrics_calculator.update(predictions, masks)

        results = metrics_calculator.compute_metrics()

        if model_info:
            results.update(model_info)

        current_model_info = self.model_info_calc.calculate_model_info(model)
        results.update(current_model_info)

        if model_info and "model_name" in model_info:
            original_info = self.model_info_calc.get_original_model_info(
                model_info["model_name"]
            )
            percentages = self.model_info_calc.calculate_percentages(
                current_model_info, original_info
            )
            results.update(percentages)

        return results

    def _extract_batch_data(self, batch):
        if isinstance(batch, (list, tuple)):
            images = batch[0].to(self.device)
            masks = batch[1].to(self.device)
        else:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
        return images, masks

    def _forward_pass(
        self, model: torch.nn.Module, images: torch.Tensor
    ) -> torch.Tensor:
        with torch.amp.autocast(self.device.type, enabled=(self.device.type == "cuda")):
            outputs = model(images)

        if isinstance(outputs, dict):
            outputs = outputs.get("out", outputs.get("logits", outputs))

        return outputs
