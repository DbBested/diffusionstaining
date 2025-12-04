"""
Comprehensive diffusion model debugging
Checks every single step to find where things break
"""

import yaml
import torch
import torch.nn as nn
from models import ConditionalLatentDiffusionModel

print("="*80)
print("COMPREHENSIVE DIFFUSION DEBUG")
print("="*80)

# Load config
with open('configs/prod.yaml') as f:
    config = yaml.safe_load(f)

# Create model
model = ConditionalLatentDiffusionModel(config)
model.eval()

print("\n1. MODEL ARCHITECTURE CHECK")
print("-"*80)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
unet_params = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)

print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
print(f"UNet params: {unet_params:,}")

if unet_params < 10_000_000:
    print(f"⚠️  WARNING: UNet might be too small!")
else:
    print(f"✅ UNet size looks reasonable")

print("\n2. VAE LATENT SPACE CHECK")
print("-"*80)

# Test VAE with real-looking data
test_img = (torch.rand(2, 3, 256, 256) * 2) - 1  # Uniform [-1, 1]

with torch.no_grad():
    latent = model.encode(test_img)
    recon = model.decode(latent)

print(f"Image -> Latent -> Image:")
print(f"  Input: shape={test_img.shape}, mean={test_img.mean():.3f}, std={test_img.std():.3f}")
print(f"  Latent: shape={latent.shape}, mean={latent.mean():.3f}, std={latent.std():.3f}")
print(f"  Recon: shape={recon.shape}, mean={recon.mean():.3f}, std={recon.std():.3f}")
print(f"  Recon error: {(recon - test_img).abs().mean():.4f}")

if latent.std() < 0.5 or latent.std() > 2.0:
    print(f"⚠️  WARNING: Latent std={latent.std():.3f} is unusual (expected ~1.0)")
else:
    print(f"✅ Latent std looks good")

print("\n3. NOISE SCHEDULER CHECK")
print("-"*80)

timesteps = torch.tensor([0, 250, 500, 750, 999])
clean_latent = torch.randn(1, 4, 32, 32)
noise = torch.randn_like(clean_latent)

print(f"Clean latent std: {clean_latent.std():.4f}")
print(f"Noise std: {noise.std():.4f}")

for t in timesteps:
    noisy = model.scheduler.add_noise(clean_latent, noise, t.unsqueeze(0))
    print(f"  t={t:4d}: noisy_std={noisy.std():.4f}")

print("\n4. CONDITIONING CHECK")
print("-"*80)

he_img = (torch.rand(2, 3, 256, 256) * 2) - 1
ihc_img = (torch.rand(2, 3, 256, 256) * 2) - 1

with torch.no_grad():
    he_cond = model.get_conditioning(he_img, apply_uncond=False)
    uncond = model.get_conditioning(he_img, apply_uncond=True)

print(f"H&E conditioning:")
print(f"  Conditional: mean={he_cond.mean():.3f}, std={he_cond.std():.3f}")
print(f"  Unconditional: mean={uncond.mean():.3f}, std={uncond.std():.3f}")

if uncond.abs().max() > 0.01:
    print(f"⚠️  WARNING: Unconditional should be ~zero!")
else:
    print(f"✅ Unconditional conditioning is zero")

print("\n5. FORWARD PASS - DETAILED BREAKDOWN")
print("-"*80)

with torch.no_grad():
    # Encode target
    ihc_latent = model.encode(ihc_img)
    print(f"Step 1 - IHC latent: mean={ihc_latent.mean():.3f}, std={ihc_latent.std():.3f}")
    
    # Sample timestep
    t = torch.tensor([500, 500])
    print(f"Step 2 - Timesteps: {t}")
    
    # Sample noise
    noise = torch.randn_like(ihc_latent)
    print(f"Step 3 - Noise: mean={noise.mean():.3f}, std={noise.std():.3f}")
    
    # Add noise
    noisy_latent = model.scheduler.add_noise(ihc_latent, noise, t)
    print(f"Step 4 - Noisy latent: mean={noisy_latent.mean():.3f}, std={noisy_latent.std():.3f}")
    
    # Get conditioning
    he_cond = model.get_conditioning(he_img, apply_uncond=False)
    print(f"Step 5 - Conditioning: mean={he_cond.mean():.3f}, std={he_cond.std():.3f}")
    
    # Concatenate
    unet_input = torch.cat([noisy_latent, he_cond], dim=1)
    print(f"Step 6 - UNet input: shape={unet_input.shape}, mean={unet_input.mean():.3f}, std={unet_input.std():.3f}")
    
    # UNet prediction
    noise_pred = model.unet(sample=unet_input, timestep=t).sample
    print(f"Step 7 - UNet output: mean={noise_pred.mean():.3f}, std={noise_pred.std():.3f}")

print(f"\n{'='*80}")
print("NOISE PREDICTION ANALYSIS:")
print(f"{'='*80}")
print(f"Target noise std:    {noise.std():.4f}")
print(f"Predicted noise std: {noise_pred.std():.4f}")
print(f"Ratio: {noise.std() / noise_pred.std():.2f}x")

if abs(noise_pred.std() - noise.std()) > 0.3:
    print(f"\n❌ MISMATCH: UNet predicts noise that's {noise.std() / noise_pred.std():.1f}x too weak")
    print("\nPossible causes:")
    print("  1. UNet initialization - weights start too small")
    print("  2. UNet architecture - not enough capacity")
    print("  3. Training issue - loss function or optimizer problem")
    print("  4. Timestep embedding - not working correctly")
else:
    print("\n✅ Noise prediction scale looks good!")

print("\n6. UNET ARCHITECTURE DETAILS")
print("-"*80)

print(f"UNet structure:")
print(f"  Input channels: {config['model']['unet']['in_channels']}")
print(f"  Output channels: {config['model']['unet']['out_channels']}")
print(f"  Block channels: {config['model']['unet']['channels']}")
print(f"  Attention head dim: {config['model']['unet']['num_head_channels']}")
print(f"  Layers per block: {config['model']['unet']['num_res_blocks']}")

# Check first and last layer
first_conv = None
last_conv = None
for name, module in model.unet.named_modules():
    if isinstance(module, nn.Conv2d):
        if first_conv is None:
            first_conv = (name, module)
        last_conv = (name, module)

if first_conv:
    print(f"\nFirst conv layer: {first_conv[0]}")
    print(f"  Weight std: {first_conv[1].weight.std():.6f}")
if last_conv:
    print(f"Last conv layer: {last_conv[0]}")
    print(f"  Weight std: {last_conv[1].weight.std():.6f}")

print("\n7. TRAINING STEP SIMULATION")
print("-"*80)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

# Simulate one training step
he_img = (torch.rand(2, 3, 256, 256) * 2) - 1
ihc_img = (torch.rand(2, 3, 256, 256) * 2) - 1

optimizer.zero_grad()
noise_pred, noise_target = model(ihc_img, he_img)

loss = nn.functional.mse_loss(noise_pred, noise_target)
loss.backward()

# Check gradients
total_grad_norm = 0
for p in model.parameters():
    if p.grad is not None:
        total_grad_norm += p.grad.norm().item() ** 2
total_grad_norm = total_grad_norm ** 0.5

print(f"Loss: {loss.item():.4f}")
print(f"Gradient norm: {total_grad_norm:.4f}")

if total_grad_norm < 0.001:
    print("⚠️  WARNING: Gradients are very small!")
elif total_grad_norm > 100:
    print("⚠️  WARNING: Gradients are very large!")
else:
    print("✅ Gradient norm looks reasonable")

print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

# Analyze the mismatch
ratio = noise.std() / noise_pred.std()
if ratio > 2.5:
    print(f"\n🔴 CRITICAL ISSUE: {ratio:.1f}x noise prediction mismatch")
    print("\nMost likely causes (in order):")
    print("  1. UNet initialization is wrong (weights too small at init)")
    print("  2. UNet output layer needs different initialization")
    print("  3. Need to check timestep embedding")
    print("\nNext debugging steps:")
    print("  A. Check UNet weight initialization")
    print("  B. Try training for 5 epochs and see if ratio improves")
    print("  C. Check if loss is actually decreasing")
elif ratio > 1.5:
    print(f"\n🟡 MODERATE ISSUE: {ratio:.1f}x mismatch")
    print("  This might improve with training")
else:
    print(f"\n🟢 LOOKS GOOD: Only {ratio:.1f}x mismatch")

