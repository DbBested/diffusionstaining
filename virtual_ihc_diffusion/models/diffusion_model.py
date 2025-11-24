"""
Conditional Latent Diffusion Model for H&E to IHC Translation
Uses MONAI's generative model components
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler, DDPMScheduler


class ConditionalLatentDiffusionModel(nn.Module):
    """
    Latent Diffusion Model with H&E conditioning

    Architecture:
    1. AutoencoderKL: Compress images to latent space
    2. Conditional UNet: Denoise latent with H&E guidance
    3. DDIM/DDPM Scheduler: Control diffusion process
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Build autoencoder (VAE)
        self.autoencoder = self._build_autoencoder()

        # Build diffusion UNet
        self.unet = self._build_unet()

        # Build scheduler
        self.scheduler = self._build_scheduler()

        # Freeze autoencoder during diffusion training
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        # Classifier-free guidance settings
        self.use_cfg = config["model"]["conditioning"]["use_classifier_free_guidance"]
        self.guidance_scale = config["model"]["conditioning"]["guidance_scale"]
        self.uncond_prob = config["model"]["conditioning"]["unconditional_prob"]

    def _build_autoencoder(self) -> AutoencoderKL:
        """Build AutoencoderKL for latent compression"""
        ae_config = self.config["model"]["autoencoder"]

        autoencoder = AutoencoderKL(
            spatial_dims=ae_config["spatial_dims"],
            in_channels=ae_config["in_channels"],
            out_channels=ae_config["out_channels"],
            channels=ae_config["channels"],
            latent_channels=ae_config["latent_channels"],
            num_res_blocks=ae_config["num_res_blocks"],
            attention_levels=ae_config["attention_levels"],
        )

        return autoencoder

    def _build_unet(self) -> DiffusionModelUNet:
        """Build conditional UNet for denoising"""
        unet_config = self.config["model"]["unet"]

        unet = DiffusionModelUNet(
            spatial_dims=unet_config["spatial_dims"],
            in_channels=unet_config["in_channels"],
            out_channels=unet_config["out_channels"],
            channels=unet_config["channels"],
            attention_levels=unet_config["attention_levels"],
            num_head_channels=unet_config["num_head_channels"],
            num_res_blocks=unet_config["num_res_blocks"],
        )

        return unet

    def _build_scheduler(self):
        """Build diffusion scheduler (DDIM or DDPM)"""
        sched_config = self.config["model"]["scheduler"]
        sched_type = sched_config["type"]

        common_args = {
            "num_train_timesteps": sched_config["num_train_timesteps"],
            "schedule": sched_config["schedule"],
            "beta_start": sched_config["beta_start"],
            "beta_end": sched_config["beta_end"],
        }

        if sched_type == "DDIM":
            scheduler = DDIMScheduler(**common_args)
        elif sched_type == "DDPM":
            scheduler = DDPMScheduler(**common_args)
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")

        return scheduler

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space"""
        z = self.autoencoder.encode_stage_2_inputs(x)
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image space"""
        x = self.autoencoder.decode_stage_2_outputs(z)
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
            # Return zeros for unconditional training
            with torch.no_grad():
                dummy = torch.zeros_like(he_images)
                cond = self.encode(dummy)
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
                self.scheduler.num_train_timesteps,
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
            uncond_mask = torch.rand(batch_size) < self.uncond_prob
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

        # Predict noise
        noise_pred = self.unet(unet_input, timesteps=timesteps)

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

        # Random latent initialization
        latent = torch.randn(
            (batch_size, self.config["model"]["autoencoder"]["latent_channels"],
             he_images.shape[2] // 4, he_images.shape[3] // 4),
            device=device,
        )

        # Denoising loop
        for t in self.scheduler.timesteps:
            timestep = torch.tensor([t] * batch_size, device=device)

            # Conditional prediction
            unet_input = torch.cat([latent, he_cond], dim=1)
            noise_pred_cond = self.unet(unet_input, timesteps=timestep)

            # Classifier-free guidance
            if self.use_cfg and guidance_scale > 1.0:
                unet_input_uncond = torch.cat([latent, uncond], dim=1)
                noise_pred_uncond = self.unet(unet_input_uncond, timesteps=timestep)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                noise_pred = noise_pred_cond

            # Denoise step
            step_output = self.scheduler.step(noise_pred, t, latent)
            latent = step_output[0] if isinstance(step_output, tuple) else step_output.prev_sample

        # Decode latent to image
        ihc_images = self.decode(latent)

        return ihc_images


if __name__ == "__main__":
    # Test model creation
    import yaml

    with open("../configs/baseline.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Creating model...")
    model = ConditionalLatentDiffusionModel(config)
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
