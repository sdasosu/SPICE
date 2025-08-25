"""
Command line interface for structured pruning
"""

import logging
import sys

from .cli_helpers import build_config_from_args, create_parser, validate_args
from .pruner import prune_model
from .utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = create_parser()
    args = parser.parse_args()
    validate_args(args)
    return args


def main():
    try:
        args = parse_args()
        config = build_config_from_args(args)
        logger.info("Starting structured pruning with configuration:")
        logger.info(f"  Model: {config.model_name}")
        logger.info(f"  Strategy: {config.pruning_strategy}")
        logger.info(f"  Pruning ratio: {config.pruning_ratio}")
        logger.info(f"  Iterative steps: {config.iterative_steps}")
        logger.info(f"  Device: {config.device}")
        logger.info(f"  Output directory: {config.output_dir}")

        if config.enable_kd_lite:
            logger.info("  KD-Lite enabled:")
            logger.info(f"    Temperature: {config.kd_temperature}")
            logger.info(f"    Alpha: {config.kd_alpha}")
            logger.info(f"    Data ratio: {config.kd_data_ratio}")
        results = prune_model(config)
        _print_summary(results)

    except KeyboardInterrupt:
        logger.info("Pruning interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pruning failed: {e}")
        sys.exit(1)


def _print_summary(results):
    print("\n" + "=" * 60)
    print("PRUNING COMPLETE")
    print("=" * 60)
    print(f"Model: {results['model_name']}")
    print(f"Strategy: {results['pruning_strategy']}")
    if results.get("kd_lite_enabled"):
        print(
            f"KD-Lite: Enabled (T={results['kd_temperature']}, α={results['kd_alpha']})"
        )
    print(f"Parameters: {results['initial_params']:,} → {results['final_params']:,}")
    print(f"Reduction: {results['params_reduction']:.2%}")
    print(
        f"Model size: {results['initial_size_mb']:.2f} MB → {results['final_size_mb']:.2f} MB"
    )
    print(f"Size reduction: {results['size_reduction']:.2%}")
    print(f"Output: {results['output_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
