"""Test with real training data"""
import yaml
import torch
from models import ConditionalLatentDiffusionModel
from data import get_dataloader

device = torch.device("cuda")

with open('configs/prod.yaml') as f:
    config = yaml.safe_load(f)

# Reduce workers for quick test
config['data']['num_workers'] = 2
config['data']['batch_size'] = 4

print("Loading real data...")
train_loader = get_dataloader(config, split="train", shuffle=True)
batch = next(iter(train_loader))

he = batch['he'].to(device)
ihc = batch['ihc'].to(device)

print(f"Real data loaded: {he.shape}")
print(f"  H&E range: [{he.min():.2f}, {he.max():.2f}]")
print(f"  IHC range: [{ihc.min():.2f}, {ihc.max():.2f}]")

# Load model
model = ConditionalLatentDiffusionModel(config).to(device)
model.eval()

# Test forward pass
with torch.no_grad():
    noise_pred, noise_target = model(ihc, he)
    loss = torch.nn.functional.mse_loss(noise_pred, noise_target)

print(f"\nForward pass on real data:")
print(f"  Noise pred std: {noise_pred.std():.4f}")
print(f"  Noise target std: {noise_target.std():.4f}")
print(f"  Loss: {loss.item():.4f}")
print(f"  Ratio: {noise_target.std() / noise_pred.std():.2f}x")

# Test sampling
print(f"\nTesting sampling...")
with torch.no_grad():
    generated = model.sample(he[:2], num_inference_steps=10)

print(f"  Generated: {generated.shape}")
print(f"  Range: [{generated.min():.2f}, {generated.max():.2f}]")
print(f"  Mean: {generated.mean():.2f}, Std: {generated.std():.2f}")

if loss.item() < 2.0:
    print("\n✅ ALL CHECKS PASSED - READY TO TRAIN!")
else:
    print(f"\n⚠️  Loss is high: {loss.item():.2f}")
