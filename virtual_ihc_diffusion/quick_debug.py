import yaml
import torch
from models import ConditionalLatentDiffusionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open('configs/prod.yaml') as f:
    config = yaml.safe_load(f)

model = ConditionalLatentDiffusionModel(config).to(device)
model.eval()

# Quick test
test_img = (torch.rand(2, 3, 256, 256, device=device) * 2) - 1

with torch.no_grad():
    latent = model.encode(test_img)
    print(f"Latent: mean={latent.mean():.3f}, std={latent.std():.3f}")
    
    he_cond = model.get_conditioning(test_img, apply_uncond=False)
    uncond = model.get_conditioning(test_img, apply_uncond=True)
    print(f"Conditional: std={he_cond.std():.3f}")
    print(f"Unconditional: std={uncond.std():.3f}")
    
    noise_pred, noise_target = model(test_img, test_img)
    print(f"Noise pred std: {noise_pred.std():.4f}")
    print(f"Noise target std: {noise_target.std():.4f}")
    print(f"Ratio: {noise_target.std() / noise_pred.std():.2f}x")
