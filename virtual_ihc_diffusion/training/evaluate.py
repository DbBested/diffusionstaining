"""
Evaluation Script for Virtual IHC Staining
Generates predictions and computes metrics on test set
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import torchvision.utils as vutils
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data import get_dataloader
from models import ConditionalLatentDiffusionModel
from training.metrics import MetricsTracker


def evaluate(config: dict, checkpoint_path: str, output_dir: str, num_inference_steps: int = 50):
    """
    Evaluate model on test set

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save outputs
        num_inference_steps: Number of DDIM sampling steps
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    model = ConditionalLatentDiffusionModel(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test data
    test_loader = get_dataloader(config, split="test", shuffle=False)
    print(f"Test samples: {len(test_loader.dataset)}")

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(metrics=config["evaluation"]["metrics"])

    # Evaluate
    all_metrics = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            he_images = batch["he"].to(device)
            ihc_images = batch["ihc"].to(device)

            # Generate IHC predictions
            generated = model.sample(
                he_images,
                num_inference_steps=num_inference_steps,
            )

            # Compute metrics
            batch_size = he_images.shape[0]
            for i in range(batch_size):
                from training.metrics import compute_metrics
                sample_metrics = compute_metrics(
                    generated[i:i+1],
                    ihc_images[i:i+1],
                    metrics=config["evaluation"]["metrics"]
                )
                all_metrics.append(sample_metrics)

                # Save comparison images
                save_comparison(
                    he_images[i],
                    ihc_images[i],
                    generated[i],
                    output_dir,
                    batch_idx * batch_size + i,
                )

            # Update tracker
            metrics_tracker.update(generated, ihc_images)

    # Compute and print statistics
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    avg_metrics = metrics_tracker.compute()
    for metric_name, metric_value in avg_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

        # Compute std
        values = [m[metric_name] for m in all_metrics]
        std = np.std(values)
        print(f"{metric_name} std: {std:.4f}")

    print("=" * 50)

    # Save metrics to file
    metrics_file = output_dir / "metrics.txt"
    with open(metrics_file, "w") as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        for metric_name, metric_value in avg_metrics.items():
            values = [m[metric_name] for m in all_metrics]
            std = np.std(values)
            f.write(f"{metric_name}: {metric_value:.4f} ± {std:.4f}\n")

    print(f"\nMetrics saved to: {metrics_file}")


def save_comparison(he, ihc_real, ihc_gen, output_dir, idx):
    """Save comparison of H&E, real IHC, and generated IHC"""
    # Denormalize from [-1, 1] to [0, 1]
    he = (he + 1) / 2
    ihc_real = (ihc_real + 1) / 2
    ihc_gen = (ihc_gen + 1) / 2

    # Save individual images
    vutils.save_image(he, output_dir / f"{idx:04d}_he.png")
    vutils.save_image(ihc_real, output_dir / f"{idx:04d}_ihc_real.png")
    vutils.save_image(ihc_gen, output_dir / f"{idx:04d}_ihc_generated.png")

    # Save comparison
    comparison = torch.cat([he, ihc_real, ihc_gen], dim=2)
    vutils.save_image(comparison, output_dir / f"{idx:04d}_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Virtual IHC Diffusion Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="eval_outputs", help="Output directory")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of inference steps")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Evaluate
    evaluate(config, args.checkpoint, args.output_dir, args.num_steps)


if __name__ == "__main__":
    main()
