import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

# Now the import will work
from models import ConditionalLatentDiffusionModel

# Rest of your code...
with open("configs/dev.yaml", "r") as f:
    config = yaml.safe_load(f)

model = ConditionalLatentDiffusionModel(config)

# Test reconstruction quality
he = torch.randn(1, 3, 256, 256)
latent = model.encode(he)
recon = model.decode(latent)

print(f"Input range: [{he.min():.3f}, {he.max():.3f}]")
print(f"Latent range: [{latent.min():.3f}, {latent.max():.3f}]")
print(f"Recon range: [{recon.min():.3f}, {recon.max():.3f}]")
print(f"Reconstruction MSE: {((he - recon)**2).mean():.6f}")

# If MSE > 0.1, your VAE is random/untrained!