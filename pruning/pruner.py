"""
Main pruning orchestrator with improved modularity and separation of concerns
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch_pruning as tp
from tqdm import tqdm

from .config import PruningConfig
from .models import ModelFactory
from .pruning_core import PruningOrchestrator, compute_model_statistics
from .sensitivity import TaylorWeightImportance
from .trainer import PruningMetrics, PruningTrainer
from .utils import set_random_seeds
from .wandb_tracking import WandBTracker

logger = logging.getLogger(__name__)


class LayerQuotaCalculator:
    """Calculates layer-wise pruning quotas based on sensitivity analysis"""

    @staticmethod
    def compute_layer_quotas(
        model: nn.Module,
        taylor_importance: TaylorWeightImportance,
        global_ratio: float,
        alpha: float = 0.7,
        min_channels: int = 8,
    ) -> Tuple[Dict[nn.Module, float], Dict[nn.Module, float]]:
        """Compute per-layer pruning quotas based on Taylor sensitivity.

        Returns:
            Tuple of (layer_quotas, layer_sensitivities)
        """
        sensitivity_scores = taylor_importance.sensitivity_scores
        prunable_layers_data = LayerQuotaCalculator._collect_prunable_layers(
            model, sensitivity_scores, min_channels
        )

        if not prunable_layers_data:
            logger.warning("No prunable layers found")
            return {}, {}

        weights = LayerQuotaCalculator._compute_pruning_weights(
            prunable_layers_data, alpha
        )

        return LayerQuotaCalculator._allocate_layer_quotas(
            prunable_layers_data, weights, global_ratio, min_channels
        )

    @staticmethod
    def _collect_prunable_layers(
        model: nn.Module, sensitivity_scores: Dict, min_channels: int
    ) -> List[Dict]:
        """Collect information about prunable layers"""
        prunable_layers_data = []

        for module in model.modules():
            if (
                isinstance(module, nn.Conv2d)
                and module in sensitivity_scores
                and module.out_channels > min_channels
            ):
                scores = sensitivity_scores[module]
                layer_sensitivity = scores.mean().item()
                flops_weight = LayerQuotaCalculator._estimate_flops_contribution(module)

                prunable_layers_data.append(
                    {
                        "module": module,
                        "sensitivity": layer_sensitivity,
                        "flops": flops_weight,
                        "channels": module.out_channels,
                    }
                )

        return prunable_layers_data

    @staticmethod
    def _estimate_flops_contribution(module: nn.Conv2d) -> int:
        """Estimate FLOPs contribution for a convolution layer"""
        return (
            module.out_channels
            * module.in_channels
            * module.kernel_size[0]
            * module.kernel_size[1]
        )

    @staticmethod
    def _compute_pruning_weights(
        prunable_layers_data: List[Dict], alpha: float
    ) -> torch.Tensor:
        """Compute pruning weights combining sensitivity and FLOPs"""
        sensitivities = torch.tensor(
            [data["sensitivity"] for data in prunable_layers_data]
        )
        sensitivities = sensitivities / (sensitivities.mean() + 1e-8)
        inv_sensitivities = 1.0 / (sensitivities + 1e-6)

        if alpha > 0:
            flops = torch.tensor(
                [data["flops"] for data in prunable_layers_data], dtype=torch.float32
            )
            flops = flops / flops.max()
            weights = inv_sensitivities * (flops**alpha)
        else:
            weights = inv_sensitivities

        return weights / weights.sum()

    @staticmethod
    def _allocate_layer_quotas(
        prunable_layers_data: List[Dict],
        weights: torch.Tensor,
        global_ratio: float,
        min_channels: int,
    ) -> Tuple[Dict[nn.Module, float], Dict[nn.Module, float]]:
        """Allocate pruning quotas to individual layers with redistribution"""
        total_channels = sum(data["channels"] for data in prunable_layers_data)
        target_prune_channels = int(total_channels * global_ratio)

        layer_quotas = {}
        layer_sensitivities = {}

        # Initialize allocation tracking
        allocated_channels = [0] * len(prunable_layers_data)
        max_prunable_per_layer = [
            data["channels"] - min_channels for data in prunable_layers_data
        ]

        # Iterative allocation with redistribution
        remaining_to_prune = target_prune_channels
        allocated_channels = [0] * len(prunable_layers_data)
        max_iterations = 10

        for iteration in range(max_iterations):
            if remaining_to_prune <= 0:
                break

            # Find layers that can still be pruned
            active_indices = []
            active_weights = []
            for i, data in enumerate(prunable_layers_data):
                max_prunable = data["channels"] - min_channels
                if allocated_channels[i] < max_prunable:
                    active_indices.append(i)
                    active_weights.append(weights[i].item())

            if not active_indices:
                break

            # Normalize active weights
            total_active_weight = sum(active_weights)
            if total_active_weight == 0:
                # Distribute evenly if all weights are 0
                active_weights = [1.0 / len(active_weights)] * len(active_weights)
            else:
                active_weights = [w / total_active_weight for w in active_weights]

            # Allocate to active layers
            for idx, weight in zip(active_indices, active_weights):
                data = prunable_layers_data[idx]
                max_prunable = data["channels"] - min_channels

                # How much to allocate to this layer
                to_allocate = int(remaining_to_prune * weight)

                # Don't exceed the layer's capacity
                can_allocate = max_prunable - allocated_channels[idx]
                actual_allocation = min(to_allocate, can_allocate)

                allocated_channels[idx] += actual_allocation
                remaining_to_prune -= actual_allocation

        total_allocated = 0
        for i, data in enumerate(prunable_layers_data):
            module = data["module"]
            channels = data["channels"]
            sensitivity = data["sensitivity"]
            channels_to_prune = allocated_channels[i]
            total_allocated += channels_to_prune

            layer_ratio = channels_to_prune / channels if channels > 0 else 0
            layer_quotas[module] = layer_ratio
            layer_sensitivities[module] = sensitivity

            logger.debug(
                f"Layer {module.__class__.__name__}: "
                f"sensitivity={sensitivity:.4f}, "
                f"quota={layer_ratio:.2%}, "
                f"pruning {channels_to_prune}/{channels} channels"
            )

        actual_ratio = total_allocated / total_channels if total_channels > 0 else 0
        logger.info(
            f"Quota allocation: target={global_ratio:.2%}, "
            f"allocated={actual_ratio:.2%}, "
            f"pruned={total_allocated}/{total_channels} channels"
        )

        return layer_quotas, layer_sensitivities


class HybridTaylorMagnitudePruner:
    """
    Hybrid pruner that uses Taylor sensitivity for quota allocation
    and magnitude for channel selection within each layer.
    """

    def __init__(self, config: PruningConfig):
        self.config = config
        self.layer_sensitivities: Dict[nn.Module, float] = {}
        self.layer_quotas: Dict[nn.Module, float] = {}

    def compute_layer_quotas(
        self,
        model: nn.Module,
        taylor_importance: TaylorWeightImportance,
        global_ratio: float,
        alpha: float = 0.7,
        min_channels: int = 8,
    ) -> Dict[nn.Module, float]:
        layer_quotas, layer_sensitivities = LayerQuotaCalculator.compute_layer_quotas(
            model, taylor_importance, global_ratio, alpha, min_channels
        )

        self.layer_quotas = layer_quotas
        self.layer_sensitivities = layer_sensitivities

        return layer_quotas

    def create_hybrid_pruner(
        self,
        model: nn.Module,
        taylor_importance: TaylorWeightImportance,
        example_inputs: torch.Tensor,
        ignored_layers: List[nn.Module],
    ) -> tp.pruner.MagnitudePruner:
        """
        Create a MagnitudePruner with layer-wise quotas from Taylor sensitivity.

        Args:
            model: Model to prune
            taylor_importance: Computed Taylor importance
            example_inputs: Example inputs for dependency graph
            ignored_layers: Layers to ignore during pruning

        Returns:
            Configured MagnitudePruner with hybrid approach
        """
        # Compute layer-wise quotas based on sensitivity
        layer_quotas = self.compute_layer_quotas(
            model=model,
            taylor_importance=taylor_importance,
            global_ratio=self.config.pruning_ratio,
            alpha=getattr(self.config, "sensitivity_alpha", 0.7),
            min_channels=getattr(self.config, "min_out_channels", 8),
        )

        magnitude_importance = tp.importance.MagnitudeImportance(
            p=self.config.importance_norm
        )

        pruner = tp.pruner.MagnitudePruner(
            model=model,
            example_inputs=example_inputs,
            importance=magnitude_importance,
            pruning_ratio=self.config.pruning_ratio,
            pruning_ratio_dict=layer_quotas,
            iterative_steps=self.config.iterative_steps,
            ignored_layers=ignored_layers,
            global_pruning=False,
        )

        logger.info(
            f"Created hybrid pruner: Taylor sensitivity for {len(layer_quotas)} layer quotas, "
            f"Magnitude for channel selection"
        )

        return pruner

    def get_pruning_summary(self) -> Dict:
        summary = {"num_layers": len(self.layer_quotas), "layer_details": []}

        for module, quota in self.layer_quotas.items():
            sensitivity = self.layer_sensitivities.get(module, 0)
            summary["layer_details"].append(
                {
                    "layer_type": module.__class__.__name__,
                    "out_channels": module.out_channels,
                    "sensitivity": float(sensitivity),
                    "pruning_quota": float(quota),
                    "channels_to_prune": int(module.out_channels * quota),
                }
            )

        summary["layer_details"].sort(key=lambda x: x["sensitivity"], reverse=True)
        return summary


def create_hybrid_pruner(
    model: nn.Module,
    config: PruningConfig,
    taylor_importance: TaylorWeightImportance,
    example_inputs: torch.Tensor,
    ignored_layers: List[nn.Module],
) -> Tuple[tp.pruner.MagnitudePruner, Dict]:
    hybrid = HybridTaylorMagnitudePruner(config)
    pruner = hybrid.create_hybrid_pruner(
        model=model,
        taylor_importance=taylor_importance,
        example_inputs=example_inputs,
        ignored_layers=ignored_layers,
    )
    summary = hybrid.get_pruning_summary()
    return pruner, summary


class TrainingStrategy(ABC):
    """Abstract base class for different training strategies"""

    @abstractmethod
    def train_model(
        self,
        trainer: "PruningTrainer",
        model: nn.Module,
        epochs: int,
        step_info: str,
        teacher_model: Optional[nn.Module] = None,
    ) -> nn.Module:
        """Train the model according to the specific strategy"""
        pass


class StandardTrainingStrategy(TrainingStrategy):
    """Standard training without knowledge distillation"""

    def train_model(
        self,
        trainer: "PruningTrainer",
        model: nn.Module,
        epochs: int,
        step_info: str,
        teacher_model: Optional[nn.Module] = None,
    ) -> nn.Module:
        return trainer.fine_tune(model, epochs, step_info, teacher_model=None)


class KDTrainingStrategy(TrainingStrategy):
    """Knowledge distillation training strategy"""

    def train_model(
        self,
        trainer: "PruningTrainer",
        model: nn.Module,
        epochs: int,
        step_info: str,
        teacher_model: Optional[nn.Module] = None,
    ) -> nn.Module:
        return trainer.fine_tune(model, epochs, step_info, teacher_model=teacher_model)


class RefinedKDTrainingStrategy(TrainingStrategy):
    """Two-stage training: standard fine-tuning followed by KD refinement"""

    def __init__(self, config: PruningConfig):
        self.config = config
        self.standard_strategy = StandardTrainingStrategy()
        self.kd_strategy = KDTrainingStrategy()

    def train_model(
        self,
        trainer: "PruningTrainer",
        model: nn.Module,
        epochs: int,
        step_info: str,
        teacher_model: Optional[nn.Module] = None,
    ) -> nn.Module:
        logger.info("Stage 1: Standard fine-tuning")
        model = self.standard_strategy.train_model(
            trainer, model, epochs, f"{step_info} - Standard"
        )

        logger.info("Stage 2: KD refinement")
        kd_epochs = getattr(self.config, "kd_refine_epochs", epochs)
        model = self.kd_strategy.train_model(
            trainer, model, kd_epochs, f"{step_info} - KD Refine", teacher_model
        )

        return model


class TrainingStrategyFactory:
    """Factory for creating training strategies based on configuration"""

    @staticmethod
    def create_strategy(config: PruningConfig) -> TrainingStrategy:
        if not config.enable_kd_lite:
            return StandardTrainingStrategy()

        if getattr(config, "kd_mode", "") == "refine":
            return RefinedKDTrainingStrategy(config)

        return KDTrainingStrategy()


class ModelManager:
    """Manages model loading, saving and directory operations"""

    def __init__(self, config: PruningConfig):
        self.config = config
        self.model_factory = ModelFactory()
        self.pruning_dir = self._create_output_directory()
        self.intermediate_dir = self.pruning_dir / "intermediate"
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)

    def _create_output_directory(self) -> Path:
        config = self.config
        base_name = (
            f"{config.model_name}_{config.pruning_strategy}_{config.pruning_ratio:.2f}"
        )

        if config.enable_kd_lite:
            suffix = (
                "_kd_refine" if getattr(config, "kd_mode", "") == "refine" else "_kd"
            )
            base_name += suffix

        if config.use_timestamp:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_name = f"{timestamp}_{base_name}"
        else:
            dir_name = base_name

        pruning_dir = Path(config.output_dir) / dir_name
        pruning_dir.mkdir(parents=True, exist_ok=True)
        return pruning_dir

    def load_model(self) -> nn.Module:
        set_random_seeds(self.config.seed)
        model = self.model_factory.load_pretrained_model(self.config)
        model.to(self.config.device)
        return model

    def save_intermediate_model(
        self, model: nn.Module, step: int, total_steps: int
    ) -> None:
        intermediate_path = (
            self.intermediate_dir
            / f"{self.config.model_name}_step_{step}_of_{total_steps}.pth"
        )
        model.zero_grad()
        torch.save(model, intermediate_path)
        logger.debug(f"Saved intermediate pruned model: {intermediate_path}")

    def save_final_model(self, model: nn.Module) -> Path:
        model.cpu()
        final_path = self.pruning_dir / "final_model.pth"
        self.model_factory.save_model(model, str(final_path), self.config)
        return final_path

    def save_metrics(self, metrics: "PruningMetrics") -> Path:
        metrics_path = self.pruning_dir / "metrics.json"
        metrics.save_history(str(metrics_path))
        return metrics_path


class PruningStepExecutor:
    """Handles execution of individual pruning steps"""

    def __init__(self, config: PruningConfig, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.training_strategy = TrainingStrategyFactory.create_strategy(config)

    def execute_pruning_step(
        self,
        step: int,
        pruner: tp.pruner.MagnitudePruner,
        model: nn.Module,
        trainer: "PruningTrainer",
        metrics: "PruningMetrics",
        teacher_model: Optional[nn.Module] = None,
    ) -> nn.Module:
        pruner.step()

        example_inputs = self._generate_example_inputs(step)
        step_stats = compute_model_statistics(model, example_inputs)

        if step == 0:
            self.initial_params = step_stats["params"]
        step_stats["compression_ratio"] = step_stats["params"] / self.initial_params

        step_info = f"Step {step + 1}/{self.config.iterative_steps}"
        model = self.training_strategy.train_model(
            trainer, model, self.config.fine_tune_epochs, step_info, teacher_model
        )

        self.model_manager.save_intermediate_model(
            model, step + 1, self.config.iterative_steps
        )

        metrics.add_step(
            step=step + 1,
            train_loss=0.0,
            val_loss=0.0,
            params=step_stats["params"],
            macs=step_stats["macs"],
        )
        if hasattr(trainer, "wandb_tracker"):
            trainer.wandb_tracker.log_pruning_step(
                step=step + 1,
                model_stats=step_stats,
                train_loss=trainer.last_train_loss
                if hasattr(trainer, "last_train_loss")
                else 0.0,
                val_loss=trainer.last_val_loss
                if hasattr(trainer, "last_val_loss")
                else 0.0,
            )

        return model

    def execute_final_training(
        self,
        model: nn.Module,
        trainer: "PruningTrainer",
        teacher_model: Optional[nn.Module] = None,
    ) -> nn.Module:
        logger.info(
            f"\n--- Final Fine-tuning ({self.config.final_fine_tune_epochs} epochs) ---"
        )
        return self.training_strategy.train_model(
            trainer, model, self.config.final_fine_tune_epochs, "Final", teacher_model
        )

    def _generate_example_inputs(self, step: int) -> torch.Tensor:
        torch.manual_seed(self.config.seed + step)
        return torch.randn(*self.config.example_input_size).to(self.config.device)


class StructuredPruner:
    """Main orchestrator for structured pruning with improved modularity"""

    def __init__(self, config: PruningConfig):
        self.config = config
        set_random_seeds(config.seed)

        self.model_manager = ModelManager(config)
        self.trainer = PruningTrainer(config)
        self.metrics = PruningMetrics()
        self.orchestrator = PruningOrchestrator(config)
        self.step_executor = PruningStepExecutor(config, self.model_manager)

        self.wandb_tracker = WandBTracker(
            config=config,
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            tags=config.wandb_tags,
            enabled=config.use_wandb,
        )
        self.trainer.set_wandb_tracker(self.wandb_tracker)
        self.orchestrator.set_wandb_tracker(self.wandb_tracker)

    def prune(self) -> Dict[str, Any]:
        self._log_pruning_info()

        try:
            model, teacher_model, initial_info = self._initialize_models()
            pruner, additional_info = self._setup_pruning(model)

            self.orchestrator.save_reports(
                self.model_manager.pruning_dir, additional_info
            )

            model = self._execute_progressive_pruning(model, pruner, teacher_model)
            model = self.step_executor.execute_final_training(
                model, self.trainer, teacher_model
            )

            return self._finalize_pruning(model, initial_info)

        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            raise
        finally:
            self.trainer.cleanup()

    def _log_pruning_info(self) -> None:
        logger.info(f"Starting structured pruning of {self.config.model_name}")
        logger.info(
            f"Strategy: {self.config.pruning_strategy}, "
            f"Ratio: {self.config.pruning_ratio}, "
            f"Steps: {self.config.iterative_steps}"
        )

        if self.config.enable_kd_lite:
            logger.info(
                f"KD-Lite enabled: Temperature={self.config.kd_temperature}, "
                f"Alpha={self.config.kd_alpha}, Data ratio={self.config.kd_data_ratio}"
            )

    def _initialize_models(self) -> Tuple[nn.Module, Optional[nn.Module], Dict]:
        model = self.model_manager.load_model()

        teacher_model = None
        if self.config.enable_kd_lite:
            logger.info("Saving original model as teacher for KD-Lite")
            teacher_model = self.model_manager.load_model()
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False

        initial_info = self.model_manager.model_factory.get_model_info(model)
        example_inputs = torch.randn(*self.config.example_input_size).to(
            self.config.device
        )
        initial_stats = compute_model_statistics(model, example_inputs)
        initial_info["macs"] = initial_stats["macs"]
        initial_info["macs_million"] = initial_stats["macs_million"]

        logger.info(
            f"Initial model: {initial_info['total_params']:,} parameters, "
            f"{initial_info['model_size_mb']:.2f} MB, "
            f"{initial_info['macs_million']:.2f}M MACs"
        )

        # Log initial model info to WandB
        self.wandb_tracker.log_initial_model_info(initial_info)

        return model, teacher_model, initial_info

    def _setup_pruning(
        self, model: nn.Module
    ) -> Tuple[tp.pruner.MagnitudePruner, Dict]:
        ignored_layers = self.model_manager.model_factory.get_ignored_layers(
            model, self.config
        )

        torch.manual_seed(self.config.seed)
        example_inputs = torch.randn(*self.config.example_input_size).to(
            self.config.device
        )
        self.wandb_tracker.log_model_graph(model, example_inputs)

        pruner, additional_info = self.orchestrator.setup_importance_and_pruner(
            model, example_inputs, ignored_layers
        )
        if "sensitivity_report" in additional_info:
            self.wandb_tracker.log_sensitivity_analysis(
                additional_info["sensitivity_report"]
            )
        elif "hybrid_report" in additional_info:
            self.wandb_tracker.log_sensitivity_analysis(
                additional_info["hybrid_report"],
                layer_quotas=additional_info["hybrid_report"].get("quota_allocation"),
            )
            if "quota_allocation" in additional_info["hybrid_report"]:
                layer_details = additional_info["hybrid_report"][
                    "quota_allocation"
                ].get("layer_details", [])
                if layer_details:
                    self.wandb_tracker.log_layer_pruning_details(layer_details)

        return pruner, additional_info

    def _execute_progressive_pruning(
        self,
        model: nn.Module,
        pruner: tp.pruner.MagnitudePruner,
        teacher_model: Optional[nn.Module],
    ) -> nn.Module:
        logger.info("Starting progressive pruning...")

        with tqdm(
            range(self.config.iterative_steps), desc="Pruning Progress", unit="step"
        ) as pbar:
            for step in pbar:
                pbar.set_description(
                    f"Pruning Step {step + 1}/{self.config.iterative_steps}"
                )

                model = self.step_executor.execute_pruning_step(
                    step, pruner, model, self.trainer, self.metrics, teacher_model
                )

                example_inputs = self.step_executor._generate_example_inputs(step)
                step_stats = compute_model_statistics(model, example_inputs)
                pbar.set_postfix(
                    {
                        "Params": f"{step_stats['params_million']:.2f}M",
                        "MACs": f"{step_stats['macs_million']:.2f}M",
                    }
                )

        return model

    def _finalize_pruning(self, model: nn.Module, initial_info: Dict) -> Dict[str, Any]:
        model.to(self.config.device)
        example_inputs = torch.randn(*self.config.example_input_size).to(
            self.config.device
        )
        final_stats = compute_model_statistics(model, example_inputs)

        final_path = self.model_manager.save_final_model(model)
        metrics_path = self.model_manager.save_metrics(self.metrics)

        final_info = self.model_manager.model_factory.get_model_info(model)
        final_info["macs"] = final_stats["macs"]
        final_info["macs_million"] = final_stats["macs_million"]

        results = self._build_results_dict(
            initial_info, final_info, final_path, metrics_path
        )

        self._log_final_results(results)
        self.wandb_tracker.log_final_results(results)
        initial_stats = {
            "params": initial_info["total_params"],
            "macs": initial_info.get("macs", 0),
        }
        final_stats = {
            "params": final_info["total_params"],
            "macs": final_info.get("macs", 0),
        }
        self.wandb_tracker.create_comparison_chart(initial_stats, final_stats)
        if self.config.wandb_save_model:
            self.wandb_tracker.save_model_checkpoint(
                str(final_path),
                aliases=["latest", f"ratio_{self.config.pruning_ratio}"],
            )

        # Finish WandB run
        self.wandb_tracker.finish()

        return results

    def _build_results_dict(
        self, initial_info: Dict, final_info: Dict, final_path: Path, metrics_path: Path
    ) -> Dict[str, Any]:
        params_reduction = (
            initial_info["total_params"] - final_info["total_params"]
        ) / initial_info["total_params"]

        size_reduction = (
            initial_info["model_size_mb"] - final_info["model_size_mb"]
        ) / initial_info["model_size_mb"]

        macs_reduction = 0
        if initial_info.get("macs", 0) > 0:
            macs_reduction = (
                initial_info["macs"] - final_info.get("macs", 0)
            ) / initial_info["macs"]

        results = {
            "model_name": self.config.model_name,
            "pruning_strategy": self.config.pruning_strategy,
            "pruning_ratio": self.config.pruning_ratio,
            "initial_params": initial_info["total_params"],
            "final_params": final_info["total_params"],
            "params_reduction": params_reduction,
            "initial_size_mb": initial_info["model_size_mb"],
            "final_size_mb": final_info["model_size_mb"],
            "size_reduction": size_reduction,
            "initial_macs": initial_info.get("macs", 0),
            "final_macs": final_info.get("macs", 0),
            "macs_reduction": macs_reduction,
            "output_path": str(final_path),
            "metrics_path": str(metrics_path),
            "pruning_dir": str(self.model_manager.pruning_dir),
            "kd_lite_enabled": self.config.enable_kd_lite,
        }

        if self.config.enable_kd_lite:
            results.update(
                {
                    "kd_temperature": self.config.kd_temperature,
                    "kd_alpha": self.config.kd_alpha,
                    "kd_data_ratio": self.config.kd_data_ratio,
                }
            )

        return results

    def _log_final_results(self, results: Dict[str, Any]) -> None:
        logger.info("\n--- Pruning Complete ---")
        logger.info(
            f"Parameters: {results['initial_params']:,} → {results['final_params']:,} "
            f"({results['params_reduction']:.2%} reduction)"
        )
        logger.info(
            f"Model size: {results['initial_size_mb']:.2f} MB → {results['final_size_mb']:.2f} MB "
            f"({results['size_reduction']:.2%} reduction)"
        )
        logger.info(f"Saved to: {results['output_path']}")


def prune_model(config: PruningConfig) -> Dict[str, Any]:
    set_random_seeds(config.seed)
    pruner = StructuredPruner(config)
    return pruner.prune()
