"""
Training Script for Virtual IHC Staining
Implements training loop with checkpointing and validation
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data import get_dataloader
from models import ConditionalLatentDiffusionModel
from training.metrics import MetricsTracker, compute_metrics


class Trainer:
    """Trainer for conditional latent diffusion model"""

    def __init__(self, config: dict, resume: str = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Set random seed
        self.set_seed(config["experiment"]["seed"])

        # Create directories
        self.setup_directories()

        # Create model
        self.model = ConditionalLatentDiffusionModel(config).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create scheduler
        self.lr_scheduler = self._create_lr_scheduler()

        # Create dataloaders
        self.train_loader = get_dataloader(config, split="train", shuffle=True)
        self.val_loader = get_dataloader(config, split="test", shuffle=False)
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")

        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Mixed precision
        self.use_amp = config["training"]["use_amp"]
        self.scaler = GradScaler() if self.use_amp else None

        # Tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Resume from checkpoint
        if resume:
            self.load_checkpoint(resume)

    def set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def setup_directories(self):
        """Create output directories"""
        exp_name = self.config["experiment"]["name"]
        self.output_dir = Path(self.config["paths"]["output_dir"]) / exp_name
        self.checkpoint_dir = Path(self.config["paths"]["checkpoint_dir"]) / exp_name
        self.log_dir = Path(self.config["paths"]["log_dir"]) / exp_name

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self):
        """Create optimizer"""
        lr = float(self.config["training"]["learning_rate"])
        weight_decay = float(self.config["training"]["weight_decay"])

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        return optimizer

    def _create_lr_scheduler(self):
        """Create learning rate scheduler"""
        sched_config = self.config["training"]["lr_scheduler"]
        sched_type = sched_config["type"]

        if sched_type == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(sched_config["T_max"]),
                eta_min=float(sched_config["eta_min"]),
            )
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")

        return scheduler

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")

        for batch_idx, batch in enumerate(pbar):
            he_images = batch["he"].to(self.device)
            ihc_images = batch["ihc"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    noise_pred, noise_target = self.model(ihc_images, he_images)
                    loss = nn.functional.mse_loss(noise_pred, noise_target)

                self.scaler.scale(loss).backward()

                # Gradient clipping
                if "gradient_clip_val" in self.config["training"]:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["training"]["gradient_clip_val"]
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                noise_pred, noise_target = self.model(ihc_images, he_images)
                loss = nn.functional.mse_loss(noise_pred, noise_target)
                loss.backward()

                # Gradient clipping
                if "gradient_clip_val" in self.config["training"]:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["training"]["gradient_clip_val"]
                    )

                self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            self.global_step += 1

            # Logging
            if batch_idx % self.config["training"]["log_interval"] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        metrics_tracker = MetricsTracker(metrics=self.config["evaluation"]["metrics"])

        # Sample images for visualization
        num_samples = min(self.config["evaluation"]["num_samples"], len(self.val_loader))
        sample_idx = 0

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
            he_images = batch["he"].to(self.device)
            ihc_images = batch["ihc"].to(self.device)

            # Compute validation loss
            noise_pred, noise_target = self.model(ihc_images, he_images)
            loss = nn.functional.mse_loss(noise_pred, noise_target)
            total_loss += loss.item()

            # Generate samples
            if batch_idx < num_samples:
                generated = self.model.sample(
                    he_images,
                    num_inference_steps=self.config["evaluation"]["inference_steps"],
                )

                # Compute metrics
                metrics_tracker.update(generated, ihc_images)

                # Save images
                if sample_idx < num_samples:
                    self._save_sample_images(
                        he_images[0],
                        ihc_images[0],
                        generated[0],
                        epoch,
                        sample_idx,
                    )
                    sample_idx += 1

        # Compute average metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = metrics_tracker.compute()

        # Log metrics
        self.writer.add_scalar("val/loss", avg_loss, epoch)
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f"val/{metric_name}", metric_value, epoch)

        print(f"Validation - Loss: {avg_loss:.4f}, {metrics_tracker.get_string()}")

        return avg_loss, metrics

    def _save_sample_images(self, he, ihc_real, ihc_gen, epoch, idx):
        """Save sample images for visualization"""
        import torchvision.utils as vutils

        # Denormalize from [-1, 1] to [0, 1]
        he = (he + 1) / 2
        ihc_real = (ihc_real + 1) / 2
        ihc_gen = (ihc_gen + 1) / 2

        # Concatenate images
        comparison = torch.cat([he, ihc_real, ihc_gen], dim=2)

        # Save
        save_path = self.output_dir / f"epoch_{epoch:03d}_sample_{idx:02d}.png"
        vutils.save_image(comparison, save_path)

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }

        # Save regular checkpoint
        save_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint: {save_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path}")

        # Keep only top-k checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Keep only top-k checkpoints"""
        top_k = self.config["training"]["save_top_k"]
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))

        if len(checkpoints) > top_k:
            for ckpt in checkpoints[:-top_k]:
                ckpt.unlink()

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["val_loss"]

        print(f"Resumed from epoch {self.start_epoch}")

    def train(self):
        """Main training loop"""
        num_epochs = self.config["training"]["num_epochs"]

        for epoch in range(self.start_epoch, num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")

            # Validate
            if (epoch + 1) % self.config["training"]["val_interval"] == 0:
                val_loss, metrics = self.validate(epoch)

                # Save checkpoint
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                if (epoch + 1) % self.config["training"]["save_interval"] == 0 or is_best:
                    self.save_checkpoint(epoch, val_loss, is_best)

            # Step LR scheduler
            self.lr_scheduler.step()

        print("Training complete!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train Virtual IHC Diffusion Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Create trainer
    trainer = Trainer(config, resume=args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
