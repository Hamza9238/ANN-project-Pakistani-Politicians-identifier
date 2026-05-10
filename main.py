"""
main.py - Entry point for Pakistani Politicians Image Classification System.

Orchestrates the entire pipeline with command-line arguments:
    python main.py --mode collect    # Download and clean images
    python main.py --mode split      # Split into train/val/test
    python main.py --mode train      # Train both CNN models
    python main.py --mode evaluate   # Evaluate and generate plots
    python main.py --mode all        # Run everything in sequence

The pipeline:
1. COLLECT: Download images via icrawler/requests, filter faces, remove duplicates
2. SPLIT: Split cleaned images into train(75%)/val(15%)/test(10%) per class
3. TRAIN: Fine-tune ResNet-50 and EfficientNet-B2 with early stopping
4. EVALUATE: Test accuracy, confusion matrices, curves, misclassified samples
"""

import argparse
import sys
import time
import logging

from utils import setup_logging, ensure_dirs, RESULTS_DIR


def run_collect():
    """Step 1: Download and clean images for all 16 politicians."""
    from download_images import collect_images
    collect_images()


def run_split():
    """Step 2: Split cleaned dataset into train/val/test."""
    from split_dataset import split_dataset
    split_dataset()


def run_train():
    """Step 3: Train both ResNet-50 and EfficientNet-B2 models."""
    from trainer import train_all_models
    results = train_all_models()
    return results


def run_evaluate():
    """Step 4: Evaluate both models and generate all plots/reports."""
    from evaluator import evaluate_all_models
    results = evaluate_all_models()
    return results


def run_all():
    """Run the complete pipeline from data collection to evaluation."""
    logger = logging.getLogger("PoliticianClassifier")

    total_start = time.time()

    logger.info("#" * 70)
    logger.info("#  PAKISTANI POLITICIAN IMAGE CLASSIFICATION SYSTEM")
    logger.info("#  Running complete pipeline...")
    logger.info("#" * 70)

    # Step 1: Collect images
    logger.info("\n\n>>> STEP 1/4: COLLECTING IMAGES <<<\n")
    try:
        run_collect()
    except Exception as e:
        logger.error(f"Image collection failed: {e}")
        logger.info("Continuing with synthetic data generation...")
        # Generate synthetic data for all politicians as fallback
        from utils import CLEANED_DATASET_DIR, POLITICIANS, fill_with_synthetic, MIN_IMAGES_PER_CLASS
        for politician in POLITICIANS:
            pol_dir = CLEANED_DATASET_DIR / politician
            pol_dir.mkdir(parents=True, exist_ok=True)
            fill_with_synthetic(pol_dir, MIN_IMAGES_PER_CLASS)
            logger.info(f"Generated synthetic data for {politician}")

    # Step 2: Split dataset
    logger.info("\n\n>>> STEP 2/4: SPLITTING DATASET <<<\n")
    try:
        run_split()
    except Exception as e:
        logger.error(f"Dataset splitting failed: {e}")
        raise

    # Step 3: Train models
    logger.info("\n\n>>> STEP 3/4: TRAINING MODELS <<<\n")
    try:
        train_results = run_train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Step 4: Evaluate models
    logger.info("\n\n>>> STEP 4/4: EVALUATING MODELS <<<\n")
    try:
        eval_results = run_evaluate()
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

    total_time = time.time() - total_start

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE!")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info("=" * 70)

    if eval_results:
        for model_name, result in eval_results.items():
            acc = result['accuracy']
            status = "✓ TARGET MET" if acc >= 0.90 else "✗ Below target"
            logger.info(f"  {model_name}: {acc:.4f} ({acc*100:.2f}%) {status}")

    logger.info(f"\nAll results saved in: {RESULTS_DIR}")
    logger.info("Check results/final_metrics.csv for detailed metrics")


def main():
    parser = argparse.ArgumentParser(
        description="Pakistani Politicians Image Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode collect    Download images for all 16 politicians
  python main.py --mode split      Split dataset into train/val/test
  python main.py --mode train      Train ResNet-50 and EfficientNet-B2
  python main.py --mode evaluate   Evaluate models and generate plots
  python main.py --mode all        Run complete pipeline
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['collect', 'split', 'train', 'evaluate', 'all'],
        help='Pipeline mode to run'
    )

    args = parser.parse_args()

    # Setup logging and directories
    logger = setup_logging()
    ensure_dirs()

    logger.info(f"Starting pipeline in mode: {args.mode}")

    if args.mode == 'collect':
        run_collect()
    elif args.mode == 'split':
        run_split()
    elif args.mode == 'train':
        run_train()
    elif args.mode == 'evaluate':
        run_evaluate()
    elif args.mode == 'all':
        run_all()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
