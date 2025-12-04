"""
Conditional Latent Diffusion Model using Hugging Face Diffusers
Replaces MONAI implementation with industry-standard components
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DModel, DDIMScheduler, DDPMScheduler
from pathlib import Path


class ConditionalLatentDiffusionModel(nn.Module):
    """
    Latent Diffusion Model with H&E conditioning using Diffusers
    
    Architecture:
    1. AutoencoderKL: Compress images to latent space (from diffusers)
    2. UNet2DConditionModel: Denoise latent with H&E guidance (from diffusers)
    3. DDIM/DDPM Scheduler: Control diffusion process (from diffusers)
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Build autoencoder (VAE)
        self.autoencoder = self._build_autoencoder()
        
        # Load pretrained VAE weights
        vae_checkpoint_path = Path("checkpoints/vae_diffusers/vae_best.pth")
        if vae_checkpoint_path.exists():
            print(f"Loading pretrained VAE from {vae_checkpoint_path}")
            checkpoint = torch.load(vae_checkpoint_path, map_location='cpu')
            missing, unexpected = self.autoencoder.load_state_dict(
                checkpoint['model_state_dict'], 
                strict=False
            )
            print("✓ Pretrained VAE loaded successfully")
            if missing:
                print(f"  Warning: {len(missing)} missing keys")
            if unexpected:
                print(f"  Warning: {len(unexpected)} unexpected keys")
            if 'val_loss' in checkpoint:
                print(f"  VAE validation loss: {checkpoint['val_loss']:.4f}")
        else:
            print(f"⚠ WARNING: No pretrained VAE found at {vae_checkpoint_path}")
            print("  Training will use randomly initialized VAE!")

        # Build diffusion UNet
        self.unet = self._build_unet()
        
        # Fix UNet initialization
        self._fix_unet_initialization()

        # Build scheduler
        self.scheduler = self._build_scheduler()

        # Freeze autoencoder during diffusion training
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        # Classifier-free guidance settings
        self.use_cfg = config["model"]["conditioning"]["use_classifier_free_guidance"]
        self.guidance_scale = config["model"]["conditioning"]["guidance_scale"]
        self.uncond_prob = config["model"]["conditioning"]["unconditional_prob"]
        
        # VAE scaling factor (critical for SD VAE!)
        self.vae_scale_factor = 0.18215

    def _build_autoencoder(self) -> AutoencoderKL:
        """Build AutoencoderKL using diffusers"""
        ae_config = self.config["model"]["autoencoder"]
        
        # Diffusers AutoencoderKL config
        autoencoder = AutoencoderKL(
            in_channels=ae_config["in_channels"],
            out_channels=ae_config["out_channels"],
            down_block_types=["DownEncoderBlock2D"] * len(ae_config["channels"]),
            up_block_types=["UpDecoderBlock2D"] * len(ae_config["channels"]),
            block_out_channels=ae_config["channels"],
            layers_per_block=ae_config["num_res_blocks"],
            latent_channels=ae_config["latent_channels"],
            norm_num_groups=32,
            sample_size=ae_config.get("sample_size", 256),
        )
        
        return autoencoder

    def _build_unet(self) -> UNet2DModel:
        """Build UNet2DConditionModel using diffusers"""
        unet_config = self.config["model"]["unet"]
        
        # Use UNet2DModel instead of UNet2DConditionModel since we don't use cross-attention
        from diffusers import UNet2DModel
        
        unet = UNet2DModel(
            in_channels=unet_config["in_channels"],
            out_channels=unet_config["out_channels"],
            down_block_types=tuple(["DownBlock2D"] * len(unet_config["channels"])),
            up_block_types=tuple(["UpBlock2D"] * len(unet_config["channels"])),
            block_out_channels=tuple(unet_config["channels"]),
            layers_per_block=unet_config["num_res_blocks"],
            attention_head_dim=unet_config["num_head_channels"],
            norm_num_groups=32,
        )
        
        return unet

    def _build_scheduler(self):
        """Build diffusion scheduler using diffusers"""
        sched_config = self.config["model"]["scheduler"]
        sched_type = sched_config["type"]

        common_args = {
            "num_train_timesteps": sched_config["num_train_timesteps"],
            "beta_start": sched_config["beta_start"],
            "beta_end": sched_config["beta_end"],
            "beta_schedule": "scaled_linear",  # Diffusers uses beta_schedule instead of schedule
            "clip_sample": sched_config.get("clip_sample", False),
        }

        if sched_type == "DDIM":
            scheduler = DDIMScheduler(**common_args)
        elif sched_type == "DDPM":
            scheduler = DDPMScheduler(**common_args)
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")

        return scheduler
    def _fix_unet_initialization(self):
        """
        Fix UNet output layer initialization.
        Diffusers initializes conv_out too conservatively (std~0.012),
        causing outputs to be 3x weaker than they should be.
        """
        with torch.no_grad():
            # Scale up the final conv layer by 3x
            self.unet.conv_out.weight.data *= 2.5
            if self.unet.conv_out.bias is not None:
                self.unet.conv_out.bias.data *= 2.5
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space"""
        latent_dist = self.autoencoder.encode(x)
        z = latent_dist.latent_dist.mode()
        # NO SCALING - use raw VAE latents
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image space"""
        # NO SCALING - use raw VAE latents
        x = self.autoencoder.decode(z).sample
        return x

    def get_conditioning(
        self,
        he_images: torch.Tensor,
        apply_uncond: bool = False,
    ) -> torch.Tensor:
        """
        Encode H&E images as conditioning
        
        Args:
            he_images: H&E input images [B, 3, H, W]
            apply_uncond: Whether to use unconditional (null) conditioning
        
        Returns:
            Conditioning latent [B, C, H', W']
        """
        if apply_uncond:
            # Return ACTUAL zeros in latent space (don't encode zero image!)
            batch_size = he_images.shape[0]
            device = he_images.device
            # Latent space: [B, 4, 32, 32] for 256x256 images with 8x downsampling
            cond = torch.zeros(
                batch_size, 
                self.config["model"]["autoencoder"]["latent_channels"], 
                he_images.shape[2] // 8,
                he_images.shape[3] // 8,
                device=device
            )
            return cond
        else:
            with torch.no_grad():
                cond = self.encode(he_images)
            return cond

    def forward(
        self,
        ihc_images: torch.Tensor,
        he_images: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass
        
        Args:
            ihc_images: Target IHC images [B, 3, H, W]
            he_images: Conditional H&E images [B, 3, H, W]
            timesteps: Optional timesteps [B]
        
        Returns:
            noise: Predicted noise
            target: Target noise
        """
        batch_size = ihc_images.shape[0]
        device = ihc_images.device

        # Encode IHC target to latent
        ihc_latent = self.encode(ihc_images)

        # Sample timesteps if not provided
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
            ).long()

        # Sample noise
        noise = torch.randn_like(ihc_latent)

        # Add noise to latent (forward diffusion)
        noisy_latent = self.scheduler.add_noise(
            original_samples=ihc_latent,
            noise=noise,
            timesteps=timesteps,
        )

        # Get conditioning (with classifier-free guidance)
        if self.use_cfg and self.training:
            # Randomly drop conditioning
            uncond_mask = torch.rand(batch_size, device=device) < self.uncond_prob
            he_cond = []
            for i in range(batch_size):
                if uncond_mask[i]:
                    cond = self.get_conditioning(he_images[i:i+1], apply_uncond=True)
                else:
                    cond = self.get_conditioning(he_images[i:i+1], apply_uncond=False)
                he_cond.append(cond)
            he_cond = torch.cat(he_cond, dim=0)
        else:
            he_cond = self.get_conditioning(he_images, apply_uncond=False)

        # Concatenate noisy latent with conditioning
        unet_input = torch.cat([noisy_latent, he_cond], dim=1)

        # Predict noise - diffusers UNet expects (sample, timestep, encoder_hidden_states)
        # For non-cross-attention case, encoder_hidden_states can be None
        noise_pred = self.unet(
            sample=unet_input,
            timestep=timesteps,
        ).sample

        return noise_pred, noise

    @torch.no_grad()
    def sample(
        self,
        he_images: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Sample IHC images from H&E conditioning
        
        Args:
            he_images: H&E input images [B, 3, H, W]
            num_inference_steps: Number of DDIM steps
            guidance_scale: Classifier-free guidance scale (override config)
        
        Returns:
            Generated IHC images [B, 3, H, W]
        """
        device = he_images.device
        batch_size = he_images.shape[0]

        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        # Get conditioning
        he_cond = self.get_conditioning(he_images, apply_uncond=False)

        # Get unconditional for CFG
        if self.use_cfg and guidance_scale > 1.0:
            uncond = self.get_conditioning(he_images, apply_uncond=True)

        # Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Calculate latent size based on VAE downsampling
        # Diffusers VAE uses fixed 8x downsampling
        latent_channels = self.config["model"]["autoencoder"]["latent_channels"]
        latent_size = he_images.shape[2] // 8  # Diffusers VAE always does 8x

        # Random latent initialization
        latent = torch.randn(
            (batch_size, latent_channels, latent_size, latent_size),
            device=device,
        )

        # Denoising loop
        for t in self.scheduler.timesteps:
            # Expand timestep for batch
            timestep = t.unsqueeze(0).repeat(batch_size).to(device)

            # Conditional prediction
            unet_input = torch.cat([latent, he_cond], dim=1)
            noise_pred_cond = self.unet(
                sample=unet_input,
                timestep=timestep,
            ).sample

            # Classifier-free guidance
            if self.use_cfg and guidance_scale > 1.0:
                unet_input_uncond = torch.cat([latent, uncond], dim=1)
                noise_pred_uncond = self.unet(
                    sample=unet_input_uncond,
                    timestep=timestep,
                ).sample
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                noise_pred = noise_pred_cond

            # Denoise step
            latent = self.scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=latent,
            )[0]

        # Decode latent to image
        ihc_images = self.decode(latent)

        return ihc_images


if __name__ == "__main__":
    # Test model creation
    import yaml

    test_config = {
        "model": {
            "autoencoder": {
                "in_channels": 3,
                "out_channels": 3,
                "channels": [128, 256, 512, 512],
                "latent_channels": 4,
                "num_res_blocks": 2,
                "sample_size": 256,
            },
            "unet": {
                "in_channels": 8,  # 4 (latent) + 4 (conditioning)
                "out_channels": 4,
                "channels": [256, 512, 768, 1024],
                "num_head_channels": 64,
                "num_res_blocks": 2,
            },
            "scheduler": {
                "type": "DDIM",
                "num_train_timesteps": 1000,
                "beta_start": 0.0015,
                "beta_end": 0.0195,
                "clip_sample": False,
            },
            "conditioning": {
                "use_classifier_free_guidance": True,
                "guidance_scale": 7.5,
                "unconditional_prob": 0.1,
            },
        }
    }

    print("Creating model...")
    model = ConditionalLatentDiffusionModel(test_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test forward pass
    batch_size = 2
    he = torch.randn(batch_size, 3, 256, 256)
    ihc = torch.randn(batch_size, 3, 256, 256)

    noise_pred, noise = model(ihc, he)
    print(f"Noise prediction shape: {noise_pred.shape}")

    # Test sampling
    model.eval()
    samples = model.sample(he, num_inference_steps=10)
    print(f"Generated samples shape: {samples.shape}")
    print("Model test successful!")