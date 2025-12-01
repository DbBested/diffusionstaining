"""
VAE Pretraining Script with Multi-GPU Support (DistributedDataParallel)
FIXED VERSION: Respects val_interval config and includes NaN handling
"""

import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
from pathlib import Path
import sys
import torchvision.utils as vutils
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ConditionalLatentDiffusionModel
from data import get_dataloader


class CheckpointedAutoencoder(nn.Module):
    """
    Wrapper that applies gradient checkpointing to the VAE
    Reduces memory usage by ~60% at cost of ~25% slower training
    """
    def __init__(self, base_autoencoder):
        super().__init__()
        self.autoencoder = base_autoencoder
        
    def encode_stage_2_inputs(self, x):
        encoder = self.autoencoder.encode
        
        if hasattr(encoder, 'down_blocks'):
            features = encoder.down_blocks[0](x)
            
            for block in encoder.down_blocks[1:]:
                features = checkpoint(block, features, use_reentrant=False)
            
            if hasattr(encoder, 'mid_block') and encoder.mid_block is not None:
                features = checkpoint(encoder.mid_block, features, use_reentrant=False)
            
            latent = encoder.quant_conv_mu(features)
        else:
            latent = self.autoencoder.encode_stage_2_inputs(x)
        
        return latent
    
    def decode_stage_2_outputs(self, z):
        decoder = self.autoencoder.decode
        
        if hasattr(decoder, 'up_blocks'):
            z = decoder.post_quant_conv(z)
            
            if hasattr(decoder, 'mid_block') and decoder.mid_block is not None:
                z = checkpoint(decoder.mid_block, z, use_reentrant=False)
            
            for block in decoder.up_blocks:
                z = checkpoint(block, z, use_reentrant=False)
            
            reconstruction = decoder.conv_out(z)
        else:
            reconstruction = self.autoencoder.decode_stage_2_outputs(z)
        
        return reconstruction


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
    
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def save_reconstruction_samples(he, ihc, he_recon, ihc_recon, epoch, output_dir, rank):
    """Save sample reconstructions for visual inspection (only on rank 0)"""
    if rank != 0:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Denormalize from [-1, 1] to [0, 1]
    he = (he + 1) / 2
    ihc = (ihc + 1) / 2
    he_recon = (he_recon + 1) / 2
    ihc_recon = (ihc_recon + 1) / 2
    
    # Take first 4 samples from batch
    num_samples = min(4, he.size(0))
    he_samples = he[:num_samples]
    ihc_samples = ihc[:num_samples]
    he_recon_samples = he_recon[:num_samples]
    ihc_recon_samples = ihc_recon[:num_samples]
    
    # Create comparison grid
    he_comparison = torch.cat([he_samples, he_recon_samples], dim=0)
    ihc_comparison = torch.cat([ihc_samples, ihc_recon_samples], dim=0)
    
    # Save grids
    he_save_path = output_dir / f"epoch_{epoch:03d}_he_reconstruction.png"
    ihc_save_path = output_dir / f"epoch_{epoch:03d}_ihc_reconstruction.png"
    
    vutils.save_image(he_comparison, he_save_path, nrow=num_samples)
    vutils.save_image(ihc_comparison, ihc_save_path, nrow=num_samples)
    
    print(f"Saved reconstruction samples to {output_dir}")


def print_memory_stats(rank, local_rank):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(local_rank) / 1e9
        reserved = torch.cuda.memory_reserved(local_rank) / 1e9
        if rank == 0:
            print(f"GPU {local_rank} Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def get_dataloader_distributed(config, split, shuffle, world_size, rank):
    """Get dataloader with distributed sampler"""
    from data import HEtoIHCDataset
    from torch.utils.data import DataLoader
    
    dataset_wrapper = HEtoIHCDataset(
        root_dir=config['data']['root_dir'],
        split=split,
        image_size=config['data']['image_size'],
        cache_rate=config['data'].get('cache_rate', 1.0) if split == 'train' else 0.0,
        augmentation=(split == 'train')
    )
    
    dataset = dataset_wrapper.dataset
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'] // world_size,
        sampler=sampler,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True),
        persistent_workers=config['data'].get('persistent_workers', True),
        prefetch_factor=config['data'].get('prefetch_factor', 2)
    )
    
    return loader, sampler


def check_for_nans(tensor, name, rank):
    """Check tensor for NaN/Inf and print warning"""
    if torch.isnan(tensor).any():
        if rank == 0:
            print(f"⚠️  WARNING: NaN detected in {name}")
        return True
    if torch.isinf(tensor).any():
        if rank == 0:
            print(f"⚠️  WARNING: Inf detected in {name}")
        return True
    return False


def train_vae(config, num_epochs=50, resume_from=None, use_checkpointing=True, 
              gradient_accumulation_steps=1):
    """
    Pretrain VAE with multi-GPU support using DistributedDataParallel
    FIXED: Respects val_interval and includes NaN handling
    """
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"=" * 80)
        print(f"Multi-GPU Training Setup")
        print(f"=" * 80)
        print(f"World size: {world_size} GPUs")
        print(f"Using device: cuda:{local_rank}")
        
        if torch.cuda.is_available():
            for i in range(world_size):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)} - "
                      f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.0f}GB")
    
    # Create model
    if rank == 0:
        print("\nCreating model...")
    
    model = ConditionalLatentDiffusionModel(config).to(device)
    
    # CRITICAL: Initialize weights properly to avoid NaN
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight, gain=0.02)  # Smaller gain for stability
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.autoencoder.apply(init_weights)
    
    # Wrap with gradient checkpointing if enabled
    if use_checkpointing:
        if rank == 0:
            print("✓ Gradient checkpointing ENABLED")
        autoencoder = CheckpointedAutoencoder(model.autoencoder)
    else:
        if rank == 0:
            print("✗ Gradient checkpointing DISABLED")
        autoencoder = model.autoencoder
    
    # Unfreeze VAE parameters
    for param in model.autoencoder.parameters():
        param.requires_grad = True
    
    if rank == 0:
        num_params = sum(p.numel() for p in model.autoencoder.parameters())
        print(f"VAE parameters: {num_params:,}")
    
    # Wrap model with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                   find_unused_parameters=False)
        # Update autoencoder reference
        if use_checkpointing:
            autoencoder.autoencoder = model.module.autoencoder
        else:
            autoencoder = model.module.autoencoder
        
        if rank == 0:
            print(f"✓ Model wrapped with DistributedDataParallel")
    
    # Optimizer - with config settings
    lr = config['training']['learning_rate']
    if lr > 1e-4:  # Safety check for stability
        lr = 1e-4
        if rank == 0:
            print(f"⚠️  Learning rate reduced to {lr} for stability")
    
    if world_size > 1:
        optimizer = torch.optim.AdamW(
            model.module.autoencoder.parameters(),
            lr=lr,
            weight_decay=config['training'].get('weight_decay', 1e-5),
            betas=(0.9, 0.999),
            eps=1e-8
        )
    else:
        optimizer = torch.optim.AdamW(
            model.autoencoder.parameters(),
            lr=lr,
            weight_decay=config['training'].get('weight_decay', 1e-5),
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    # Learning rate scheduler (if configured)
    scheduler = None
    if 'lr_scheduler' in config['training']:
        scheduler_config = config['training']['lr_scheduler']
        scheduler_type = scheduler_config.get('type', 'CosineAnnealingLR')
        
        if scheduler_type == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get('T_max', num_epochs),
                eta_min=scheduler_config.get('eta_min', 0)
            )
            if rank == 0:
                print(f"✓ Using CosineAnnealingLR scheduler (T_max={scheduler_config.get('T_max', num_epochs)})")
        elif scheduler_type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.1)
            )
            if rank == 0:
                print(f"✓ Using StepLR scheduler")
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.1),
                patience=scheduler_config.get('patience', 10)
            )
            if rank == 0:
                print(f"✓ Using ReduceLROnPlateau scheduler")
    
    # Mixed precision scaler
    use_amp = config['training'].get('use_amp', True)
    scaler = GradScaler(init_scale=2048, growth_interval=2000) if use_amp else None
    if rank == 0:
        print(f"Mixed precision training: {'ENABLED' if use_amp else 'DISABLED'}")
    
    # Data - with distributed sampler
    if rank == 0:
        print("\nLoading data...")
    
    train_loader, train_sampler = get_dataloader_distributed(
        config, split="train", shuffle=True, world_size=world_size, rank=rank
    )
    val_loader, val_sampler = get_dataloader_distributed(
        config, split="test", shuffle=False, world_size=world_size, rank=rank
    )
    
    if rank == 0:
        print(f"Train batches: {len(train_loader)} per GPU ({len(train_loader) * world_size} total)")
        print(f"Val batches: {len(val_loader)} per GPU ({len(val_loader) * world_size} total)")
        print(f"Batch size: {config['data']['batch_size'] // world_size} per GPU "
              f"({config['data']['batch_size']} total)")
    
    # Get validation interval from config
    val_interval = config['training'].get('val_interval', 1)
    log_interval = config['training'].get('log_interval', 10)
    gradient_clip_val = config['training'].get('gradient_clip_val', 1.0)
    
    if rank == 0:
        print(f"Validation interval: every {val_interval} epoch(s)")
        print(f"Log interval: every {log_interval} batches")
        print(f"Gradient clip value: {gradient_clip_val}")
    
    # Logging (only on rank 0)
    writer = None
    if rank == 0:
        log_dir = Path("logs/vae_pretrain")
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        
        checkpoint_dir = Path("checkpoints/vae_pretrain")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        output_dir = Path("outputs/vae_pretrain")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training state
    start_epoch = 0
    best_loss = float('inf')
    global_step = 0
    
    # Resume from checkpoint
    if resume_from and rank == 0:
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint_data = torch.load(resume_from, map_location=device)
        if world_size > 1:
            model.module.autoencoder.load_state_dict(checkpoint_data['model_state_dict'])
        else:
            model.autoencoder.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        start_epoch = checkpoint_data['epoch'] + 1
        best_loss = checkpoint_data.get('best_loss', float('inf'))
        global_step = checkpoint_data.get('global_step', 0)
        print(f"Resumed from epoch {start_epoch}")
    
    if rank == 0:
        print("\n" + "=" * 80)
        print(f"STARTING VAE PRETRAINING FOR {num_epochs} EPOCHS")
        print(f"World Size: {world_size} GPUs")
        print(f"Learning Rate: {lr}")
        print(f"Gradient Checkpointing: {'ENABLED' if use_checkpointing else 'DISABLED'}")
        print(f"Gradient Accumulation: {gradient_accumulation_steps}")
        print(f"Validation Interval: {val_interval}")
        print("=" * 80 + "\n")
    
    print_memory_stats(rank, local_rank)
    
    for epoch in range(start_epoch, num_epochs):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        # ==================== Training ====================
        model.train()
        train_loss = 0
        train_he_loss = 0
        train_ihc_loss = 0
        
        optimizer.zero_grad()
        
        # Only show progress bar on rank 0
        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        else:
            pbar = train_loader
        
        for batch_idx, batch in enumerate(pbar):
            he = batch["he"].to(device)
            ihc = batch["ihc"].to(device)
            
            # Check input data for NaN/Inf
            if check_for_nans(he, "input H&E", rank) or check_for_nans(ihc, "input IHC", rank):
                if rank == 0:
                    print(f"Skipping batch {batch_idx} due to NaN/Inf in input")
                continue
            
            # Mixed precision forward pass
            with autocast(enabled=use_amp):
                he_latent = autoencoder.encode_stage_2_inputs(he)
                
                if check_for_nans(he_latent, "H&E latent", rank):
                    if rank == 0:
                        print(f"Skipping batch {batch_idx} - NaN in H&E encoding")
                    continue
                
                he_recon = autoencoder.decode_stage_2_outputs(he_latent)
                
                if check_for_nans(he_recon, "H&E reconstruction", rank):
                    if rank == 0:
                        print(f"Skipping batch {batch_idx} - NaN in H&E decoding")
                    continue
                
                ihc_latent = autoencoder.encode_stage_2_inputs(ihc)
                
                if check_for_nans(ihc_latent, "IHC latent", rank):
                    if rank == 0:
                        print(f"Skipping batch {batch_idx} - NaN in IHC encoding")
                    continue
                
                ihc_recon = autoencoder.decode_stage_2_outputs(ihc_latent)
                
                if check_for_nans(ihc_recon, "IHC reconstruction", rank):
                    if rank == 0:
                        print(f"Skipping batch {batch_idx} - NaN in IHC decoding")
                    continue
                
                # FIXED: Don't divide inside autocast
                he_loss = nn.functional.l1_loss(he_recon, he)
                ihc_loss = nn.functional.l1_loss(ihc_recon, ihc)
                loss = he_loss + ihc_loss
            
            # Check loss for NaN
            if check_for_nans(loss, "loss", rank):
                if rank == 0:
                    print(f"Skipping batch {batch_idx} - NaN in loss")
                continue
            
            # Scale loss for gradient accumulation AFTER autocast
            scaled_loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            # Update weights
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                
                # Check gradients for NaN
                has_nan_grad = False
                if world_size > 1:
                    params = model.module.autoencoder.parameters()
                else:
                    params = model.autoencoder.parameters()
                
                for param in params:
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    if rank == 0:
                        print(f"⚠️  NaN in gradients at batch {batch_idx}, skipping update")
                    optimizer.zero_grad()
                    continue
                
                # Gradient clipping
                if world_size > 1:
                    torch.nn.utils.clip_grad_norm_(model.module.autoencoder.parameters(), gradient_clip_val)
                else:
                    torch.nn.utils.clip_grad_norm_(model.autoencoder.parameters(), gradient_clip_val)
                
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            # Track losses
            batch_loss = loss.item()
            train_loss += batch_loss
            train_he_loss += he_loss.item()
            train_ihc_loss += ihc_loss.item()
            global_step += 1
            
            # Update progress bar (rank 0 only)
            if rank == 0:
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'he': f'{he_loss.item():.4f}',
                    'ihc': f'{ihc_loss.item():.4f}'
                })
            
            # Log to tensorboard (rank 0 only)
            if rank == 0 and writer and batch_idx % log_interval == 0:
                writer.add_scalar('train/loss', batch_loss, global_step)
                writer.add_scalar('train/he_loss', he_loss.item(), global_step)
                writer.add_scalar('train/ihc_loss', ihc_loss.item(), global_step)
        
        # Average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_train_he = train_he_loss / len(train_loader)
        avg_train_ihc = train_ihc_loss / len(train_loader)
        
        # ==================== Validation ====================
        # Only validate every val_interval epochs or on last epoch
        should_validate = (epoch + 1) % val_interval == 0 or (epoch + 1) == num_epochs
        
        if should_validate:
            model.eval()
            val_loss = 0
            val_he_loss = 0
            val_ihc_loss = 0
            
            with torch.no_grad():
                if rank == 0:
                    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                else:
                    val_pbar = val_loader
                
                for batch_idx, batch in enumerate(val_pbar):
                    he = batch["he"].to(device)
                    ihc = batch["ihc"].to(device)
                    
                    with autocast(enabled=use_amp):
                        he_latent = autoencoder.encode_stage_2_inputs(he)
                        he_recon = autoencoder.decode_stage_2_outputs(he_latent)
                        
                        ihc_latent = autoencoder.encode_stage_2_inputs(ihc)
                        ihc_recon = autoencoder.decode_stage_2_outputs(ihc_latent)
                        
                        he_loss = nn.functional.l1_loss(he_recon, he)
                        ihc_loss = nn.functional.l1_loss(ihc_recon, ihc)
                        loss = he_loss + ihc_loss
                    
                    val_loss += loss.item()
                    val_he_loss += he_loss.item()
                    val_ihc_loss += ihc_loss.item()
                    
                    # Save samples (rank 0 only)
                    if batch_idx == 0:
                        save_reconstruction_samples(he, ihc, he_recon, ihc_recon, epoch, 
                                                  Path("outputs/vae_pretrain"), rank)
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_he = val_he_loss / len(val_loader)
            avg_val_ihc = val_ihc_loss / len(val_loader)
        else:
            # Skip validation this epoch
            avg_val_loss = None
            avg_val_he = None
            avg_val_ihc = None
        
        # Print summary (rank 0 only)
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {avg_train_loss:.6f} (H&E: {avg_train_he:.6f}, IHC: {avg_train_ihc:.6f})")
            if should_validate:
                print(f"  Val Loss:   {avg_val_loss:.6f} (H&E: {avg_val_he:.6f}, IHC: {avg_val_ihc:.6f})")
            else:
                next_val_epoch = ((epoch // val_interval) + 1) * val_interval
                print(f"  Val Loss:   Skipped (next validation at epoch {next_val_epoch})")
            print_memory_stats(rank, local_rank)
            
            # Log epoch metrics
            if writer:
                writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)
                writer.add_scalar('epoch/train_he_loss', avg_train_he, epoch)
                writer.add_scalar('epoch/train_ihc_loss', avg_train_ihc, epoch)
                if should_validate:
                    writer.add_scalar('epoch/val_loss', avg_val_loss, epoch)
                    writer.add_scalar('epoch/val_he_loss', avg_val_he, epoch)
                    writer.add_scalar('epoch/val_ihc_loss', avg_val_ihc, epoch)
        
        # ==================== Checkpointing (rank 0 only) ====================
        if rank == 0 and should_validate:
            is_best = avg_val_loss < best_loss
            if is_best:
                best_loss = avg_val_loss
                best_path = Path("checkpoints/vae_pretrain/vae_best.pth")
                
                if world_size > 1:
                    state_dict = model.module.autoencoder.state_dict()
                else:
                    state_dict = model.autoencoder.state_dict()
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'best_loss': best_loss,
                    'global_step': global_step,
                }, best_path)
                print(f"  ✓ Saved best checkpoint (val_loss={best_loss:.6f})")
            
            # Save periodic checkpoints based on save_interval
            save_interval = config['training'].get('save_interval', 10)
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = Path(f"checkpoints/vae_pretrain/vae_epoch_{epoch+1:03d}.pth")
                
                if world_size > 1:
                    state_dict = model.module.autoencoder.state_dict()
                else:
                    state_dict = model.autoencoder.state_dict()
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss if should_validate else None,
                    'best_loss': best_loss,
                    'global_step': global_step,
                }, checkpoint_path)
                print(f"  ✓ Saved checkpoint: {checkpoint_path.name}")
            
            print("=" * 80)
        
        # Step learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau needs validation loss
                if should_validate:
                    scheduler.step(avg_val_loss)
            else:
                scheduler.step()
            
            # Log current learning rate
            if rank == 0 and writer:
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('train/learning_rate', current_lr, epoch)
    
    if rank == 0 and writer:
        writer.close()
        
        print("\n" + "=" * 80)
        print("VAE PRETRAINING COMPLETE!")
        print(f"Best validation loss: {best_loss:.6f}")
        print(f"Best checkpoint: checkpoints/vae_pretrain/vae_best.pth")
        print("=" * 80)
    
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU VAE Pretraining")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    train_vae(
        config,
        num_epochs=args.epochs,
        resume_from=args.resume,
        use_checkpointing=not args.no_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )


if __name__ == "__main__":
    main()