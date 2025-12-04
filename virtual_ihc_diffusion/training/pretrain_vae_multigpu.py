"""
VAE Training Script using Diffusers AutoencoderKL
Clean implementation without MONAI quirks
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from pathlib import Path
import sys
import torchvision.utils as vutils

sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusers import AutoencoderKL
from data import HEtoIHCDataset


def save_reconstruction_samples(he, ihc, he_recon, ihc_recon, epoch, output_dir):
    """Save sample reconstructions"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Denormalize from [-1, 1] to [0, 1]
    he = (he + 1) / 2
    ihc = (ihc + 1) / 2
    he_recon = (he_recon + 1) / 2
    ihc_recon = (ihc_recon + 1) / 2
    
    # Take first 4 samples
    num_samples = min(4, he.size(0))
    he_samples = he[:num_samples]
    ihc_samples = ihc[:num_samples]
    he_recon_samples = he_recon[:num_samples]
    ihc_recon_samples = ihc_recon[:num_samples]
    
    # Create comparison grid
    he_comparison = torch.cat([he_samples, he_recon_samples], dim=0)
    ihc_comparison = torch.cat([ihc_samples, ihc_recon_samples], dim=0)
    
    # Save
    vutils.save_image(he_comparison, output_dir / f"epoch_{epoch:03d}_he_recon.png", nrow=num_samples)
    vutils.save_image(ihc_comparison, output_dir / f"epoch_{epoch:03d}_ihc_recon.png", nrow=num_samples)


def train_vae(config, num_epochs=50):
    """Train VAE using diffusers"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create VAE
    ae_config = config['model']['autoencoder']
    print("\nCreating VAE...")
    
    vae = AutoencoderKL(
        in_channels=ae_config["in_channels"],
        out_channels=ae_config["out_channels"],
        down_block_types=["DownEncoderBlock2D"] * len(ae_config["channels"]),
        up_block_types=["UpDecoderBlock2D"] * len(ae_config["channels"]),
        block_out_channels=ae_config["channels"],
        layers_per_block=ae_config["num_res_blocks"],
        latent_channels=ae_config["latent_channels"],
        norm_num_groups=32,
        sample_size=ae_config.get("sample_size", 256),
    ).to(device)
    
    num_params = sum(p.numel() for p in vae.parameters())
    print(f"VAE parameters: {num_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(
        vae.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-5),
    )
    
    # Scheduler
    if 'lr_scheduler' in config['training']:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=config['training']['lr_scheduler'].get('eta_min', 0)
        )
    else:
        scheduler = None
    
    # Mixed precision
    use_amp = config['training'].get('use_amp', True)
    scaler = GradScaler() if use_amp else None
    
    # Data
    print("\nLoading data...")
    train_dataset = HEtoIHCDataset(
        root_dir=config['data']['root_dir'],
        split='train',
        image_size=config['data']['image_size'],
        cache_rate=config['data'].get('cache_rate', 0.5),
        augmentation=True
    )
    
    val_dataset = HEtoIHCDataset(
        root_dir=config['data']['root_dir'],
        split='test',
        image_size=config['data']['image_size'],
        cache_rate=0.0,
        augmentation=False
    )
    
    train_loader = DataLoader(
        train_dataset.dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_dataset.dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Logging
    log_dir = Path("logs/vae_diffusers")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    checkpoint_dir = Path("checkpoints/vae_diffusers")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = Path("outputs/vae_diffusers")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training
    best_loss = float('inf')
    global_step = 0
    
    print(f"\nStarting VAE training for {num_epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        # Train
        vae.train()
        train_loss = 0
        train_he_loss = 0
        train_ihc_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch in pbar:
            he = batch["he"].to(device)
            ihc = batch["ihc"].to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=use_amp):
                # Encode and decode H&E
                he_latent_dist = vae.encode(he).latent_dist
                he_latent = he_latent_dist.sample()
                he_recon = vae.decode(he_latent).sample
                
                # Encode and decode IHC
                ihc_latent_dist = vae.encode(ihc).latent_dist
                ihc_latent = ihc_latent_dist.sample()
                ihc_recon = vae.decode(ihc_latent).sample
                
                # Reconstruction loss
                he_loss = nn.functional.l1_loss(he_recon, he)
                ihc_loss = nn.functional.l1_loss(ihc_recon, ihc)
                
                # KL divergence (regularization)
                he_kl = he_latent_dist.kl().mean()
                ihc_kl = ihc_latent_dist.kl().mean()
                
                # Total loss
                recon_loss = he_loss + ihc_loss
                kl_loss = (he_kl + ihc_kl) * 0.00001  # Small weight on KL
                loss = recon_loss + kl_loss
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
                optimizer.step()
            
            train_loss += loss.item()
            train_he_loss += he_loss.item()
            train_ihc_loss += ihc_loss.item()
            global_step += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'he': f'{he_loss.item():.4f}',
                'ihc': f'{ihc_loss.item():.4f}'
            })
            
            if global_step % 10 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/he_loss', he_loss.item(), global_step)
                writer.add_scalar('train/ihc_loss', ihc_loss.item(), global_step)
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        vae.eval()
        val_loss = 0
        val_he_loss = 0
        val_ihc_loss = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")):
                he = batch["he"].to(device)
                ihc = batch["ihc"].to(device)
                
                with autocast(enabled=use_amp):
                    he_latent = vae.encode(he).latent_dist.mode()
                    he_recon = vae.decode(he_latent).sample
                    
                    ihc_latent = vae.encode(ihc).latent_dist.mode()
                    ihc_recon = vae.decode(ihc_latent).sample
                    
                    he_loss = nn.functional.l1_loss(he_recon, he)
                    ihc_loss = nn.functional.l1_loss(ihc_recon, ihc)
                    loss = he_loss + ihc_loss
                
                val_loss += loss.item()
                val_he_loss += he_loss.item()
                val_ihc_loss += ihc_loss.item()
                
                if batch_idx == 0:
                    save_reconstruction_samples(he, ihc, he_recon, ihc_recon, epoch, output_dir)
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        
        writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)
        writer.add_scalar('epoch/val_loss', avg_val_loss, epoch)
        
        # Save checkpoint
        is_best = avg_val_loss < best_loss
        if is_best:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_loss': best_loss,
            }, checkpoint_dir / "vae_best.pth")
            print(f"  ✓ Saved best checkpoint (val_loss={best_loss:.6f})")
        
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_dir / f"vae_epoch_{epoch+1:03d}.pth")
            print(f"  ✓ Saved checkpoint: vae_epoch_{epoch+1:03d}.pth")
        
        if scheduler:
            scheduler.step()
        
        print("=" * 80)
    
    writer.close()
    print("\nVAE training complete!")
    print(f"Best validation loss: {best_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train VAE with Diffusers")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    train_vae(config, num_epochs=args.epochs)


if __name__ == "__main__":
    main()