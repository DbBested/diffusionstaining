"""Fix UNet output layer initialization"""
import torch

# Load model
import yaml
from models import ConditionalLatentDiffusionModel

with open('configs/prod.yaml') as f:
    config = yaml.safe_load(f)

model = ConditionalLatentDiffusionModel(config)

# Check current output layer
print("Before fix:")
print(f"  conv_out weight std: {model.unet.conv_out.weight.std():.6f}")

# Rescale the output layer by 3x to compensate
with torch.no_grad():
    model.unet.conv_out.weight.data *= 3.0
    if model.unet.conv_out.bias is not None:
        model.unet.conv_out.bias.data *= 3.0

print("After fix:")
print(f"  conv_out weight std: {model.unet.conv_out.weight.std():.6f}")

# Test it
device = torch.device("cuda")
model = model.to(device)
model.eval()

test_img = (torch.rand(2, 3, 256, 256, device=device) * 2) - 1

with torch.no_grad():
    noise_pred, noise_target = model(test_img, test_img)
    
print(f"\nNoise prediction test:")
print(f"  Pred std: {noise_pred.std():.4f}")
print(f"  Target std: {noise_target.std():.4f}")
print(f"  Ratio: {noise_target.std() / noise_pred.std():.2f}x")

if abs(noise_pred.std() - noise_target.std()) < 0.2:
    print("\n✅ FIXED! Noise predictions now match!")
else:
    print(f"\n⚠️  Still {noise_target.std() / noise_pred.std():.1f}x off")
