"""
Helper functions for command line interface
"""

import argparse
from pathlib import Path

from .config import PruningConfig
from .model_configs import AVAILABLE_STRATEGIES, MODEL_CONFIGS


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with all CLI options"""
    parser = argparse.ArgumentParser(
        description="Structured pruning for segmentation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
{chr(10).join(f"  - {name}" for name in MODEL_CONFIGS.keys())}

Available pruning strategies:
{chr(10).join(f"  - {strategy}" for strategy in AVAILABLE_STRATEGIES)}

Examples:
  python -m pruning.cli --model UNET_resnet --strategy magnitude --ratio 0.5
  python -m pruning.cli --model UNET_resnet --strategy magnitude --ratio 0.5 \\
      --enable-kd --kd-alpha 0.7 --fine-tune-epochs 2
        """,
    )

    _add_required_args(parser)
    _add_training_args(parser)
    _add_data_args(parser)
    _add_sensitivity_args(parser)
    _add_kd_args(parser)
    _add_misc_args(parser)

    return parser


def _add_required_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to prune",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=AVAILABLE_STRATEGIES,
        help="Pruning strategy",
    )
    parser.add_argument(
        "--ratio", type=float, required=True, help="Target pruning ratio (0-1)"
    )


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of iterative pruning steps"
    )
    parser.add_argument(
        "--norm",
        type=int,
        choices=[1, 2],
        default=None,
        help="Norm for magnitude importance",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=None,
        help="Epochs for intermediate fine-tuning",
    )
    parser.add_argument(
        "--final-fine-tune-epochs",
        type=int,
        default=None,
        help="Epochs for final fine-tuning",
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate for fine-tuning"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size for fine-tuning"
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=None, help="Early stopping patience"
    )


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root", type=str, default=None, help="Root directory for dataset"
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Number of data loading workers"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for pruned models",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")


def _add_sensitivity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=None,
        help="Number of batches for sensitivity analysis",
    )
    parser.add_argument(
        "--min-out-channels",
        type=int,
        default=None,
        help="Minimum channels to keep per layer",
    )
    parser.add_argument(
        "--flops-alpha",
        type=float,
        default=None,
        help="Weight factor for FLOPs-based allocation",
    )


def _add_kd_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--enable-kd", action="store_true", help="Enable knowledge distillation"
    )
    parser.add_argument(
        "--kd-mode",
        type=str,
        default=None,
        choices=["replace", "refine"],
        help="KD mode: 'replace' or 'refine'",
    )
    parser.add_argument(
        "--kd-temperature",
        type=float,
        default=None,
        help="Temperature for knowledge distillation",
    )
    parser.add_argument(
        "--kd-alpha", type=float, default=None, help="Weight for KD loss vs CE loss"
    )
    parser.add_argument(
        "--kd-data-ratio",
        type=float,
        default=None,
        help="Fraction of training data for KD",
    )
    parser.add_argument(
        "--kd-refine-epochs",
        type=int,
        default=None,
        help="Additional KD refinement epochs",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone during KD training",
    )
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=None,
        help="Weight multiplier for boundary pixels",
    )
    parser.add_argument(
        "--confidence-weight",
        action="store_true",
        help="Enable confidence-based weighting",
    )


def _add_misc_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use-timestamp",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=None,
        help="Enable/disable timestamp prefix in output directory name",
    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate parsed arguments"""
    if args.ratio <= 0 or args.ratio >= 1:
        raise ValueError("Pruning ratio must be between 0 and 1")

    if args.steps and args.steps <= 0:
        raise ValueError("Number of steps must be positive")

    if args.data_root and not Path(args.data_root).exists():
        import logging

        logging.getLogger(__name__).warning(
            f"Data root directory does not exist: {args.data_root}"
        )


def build_config_from_args(args: argparse.Namespace) -> PruningConfig:
    config_kwargs = {
        "model_name": args.model,
        "pruning_strategy": args.strategy,
        "pruning_ratio": args.ratio,
    }
    arg_to_config_mapping = {
        "output_dir": "output_dir",
        "data_root": "data_root",
        "steps": "iterative_steps",
        "norm": "importance_norm",
        "device": "device",
        "batch_size": "batch_size",
        "fine_tune_epochs": "fine_tune_epochs",
        "final_fine_tune_epochs": "final_fine_tune_epochs",
        "lr": "fine_tune_lr",
        "seed": "seed",
        "early_stop_patience": "early_stop_patience",
        "num_workers": "num_workers",
        "use_timestamp": "use_timestamp",
        "calibration_batches": "calibration_batches",
        "min_out_channels": "min_out_channels",
        "flops_alpha": "flops_alpha",
        "kd_mode": "kd_mode",
        "kd_temperature": "kd_temperature",
        "kd_alpha": "kd_alpha",
        "kd_data_ratio": "kd_data_ratio",
        "kd_refine_epochs": "kd_refine_epochs",
        "boundary_weight": "boundary_weight",
    }

    for arg_name, config_name in arg_to_config_mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            config_kwargs[config_name] = value
    if args.enable_kd:
        config_kwargs["enable_kd_lite"] = True
    if args.freeze_backbone:
        config_kwargs["freeze_backbone"] = True
    if args.confidence_weight:
        config_kwargs["confidence_weight"] = True

    return PruningConfig(**config_kwargs)
