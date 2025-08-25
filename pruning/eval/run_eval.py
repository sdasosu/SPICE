#!/usr/bin/env python3
"""
Script to evaluate pruned models
Usage: python -m pruning.eval.run_eval
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EvaluationConfig
from evaluator import PrunedModelEvaluator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from wandb_tracking.wandb_constants import WandBConstants
except ImportError:

    class WandBConstants:
        DEFAULT_PROJECT = "epic-v2"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pruned segmentation models")

    # Basic arguments
    parser.add_argument(
        "--pruned-dir",
        type=str,
        default="pruned_models",
        help="Directory containing pruned models",
    )

    parser.add_argument(
        "--data-root", type=str, default="data", help="Root directory of dataset"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="pruning/eval/results",
        help="Directory to save evaluation results",
    )

    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for evaluation"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Device to use for evaluation",
    )

    parser.add_argument(
        "--single-model",
        type=str,
        help="Evaluate a single model by specifying its directory name",
    )

    # Visualization arguments
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        default=True,
        help="Disable visualization generation (enabled by default)",
    )

    parser.add_argument(
        "--vis-dir",
        type=str,
        default="pruning/eval/outputs",
        help="Directory to save visualizations",
    )

    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=600,
        help="DPI for saved figures",
    )

    parser.add_argument(
        "--figure-formats",
        nargs="+",
        default=["png", "pdf"],
        help="Formats to save figures in",
    )

    parser.add_argument(
        "--advanced-plots",
        action="store_true",
        default=True,
        help="Generate advanced visualizations",
    )

    # WandB arguments
    parser.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        default=True,
        help="Disable WandB tracking (enabled by default)",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default=WandBConstants.DEFAULT_PROJECT,
        help=f"WandB project name (default: {WandBConstants.DEFAULT_PROJECT})",
    )

    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity (team/user)",
    )

    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="WandB run name",
    )

    parser.add_argument(
        "--wandb-tags",
        nargs="+",
        default=["evaluation", "pruning"],
        help="WandB tags for the run",
    )

    # Add evaluation control arguments
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation, only generate visualizations from existing CSV",
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear evaluation cache before running",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create configuration
    config = EvaluationConfig(
        pruned_models_dir=args.pruned_dir,
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        # Visualization settings
        generate_visualizations=args.visualize,
        visualization_dir=args.vis_dir,
        figure_dpi=args.figure_dpi,
        figure_formats=args.figure_formats,
        generate_advanced_plots=args.advanced_plots,
        # WandB settings
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
    )

    # Clear cache if requested
    if args.clear_cache:
        cache_file = Path(config.output_dir) / ".eval_cache.pkl"
        if cache_file.exists():
            cache_file.unlink()
            print("Evaluation cache cleared")

    if args.no_eval:
        csv_file = Path(config.output_dir) / "evaluation_summary.csv"
        if not csv_file.exists():
            print(f"Error: CSV file not found: {csv_file}")
            print("Run evaluation first to generate the CSV file")
            return

        print("Skipping evaluation, generating visualizations from existing CSV...")
        # Initialize evaluator just for visualization generation
        evaluator = PrunedModelEvaluator(config)
        csv_file = Path(config.output_dir) / "evaluation_summary.csv"
        if csv_file.exists():
            evaluator._generate_visualizations(str(csv_file))
        else:
            print(f"CSV file not found: {csv_file}")

        # Log to WandB if enabled
        if config.use_wandb:
            evaluator._log_to_wandb()

        print("Visualizations generated successfully!")
        return

    # Initialize evaluator
    evaluator = PrunedModelEvaluator(config)

    if args.single_model:
        # Evaluate single model
        model_path = Path(args.pruned_dir) / args.single_model / "final_model.pth"
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return

        # Get data loader
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from data.data import get_loaders

        _, val_loader, test_loader = get_loaders(
            data_root=config.data_root,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            img_size=config.img_size,
        )
        eval_loader = test_loader if test_loader is not None else val_loader

        # Evaluate
        results = evaluator.evaluate_single_model(
            str(model_path), eval_loader, {"model_name": args.single_model}
        )

        # Print results
        evaluator._print_results(results)
        evaluator._save_model_results(results, args.single_model)
    else:
        # Evaluate all models
        evaluator.evaluate_all_models()

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
