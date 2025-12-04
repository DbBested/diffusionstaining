import yaml
import torch
from models import ConditionalLatentDiffusionModel
from data import get_dataloader

device = torch.device("cuda")

with open('configs/prod.yaml') as f:
    config = yaml.safe_load(f)

config['data']['num_workers'] = 2
config['data']['batch_size'] = 2

# Load model
model = ConditionalLatentDiffusionModel(config).to(device)

# Load checkpoint (if exists)
import glob
checkpoints = sorted(glob.glob('checkpoints/he_to_ihc_sd_vae/checkpoint_epoch_*.pth'))
if checkpoints:
    latest = checkpoints[-1]
    print(f"Loading {latest}")
    ckpt = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Epoch: {ckpt['epoch']}")
    print(f"  Loss: {ckpt.get('val_loss', 'N/A')}")

model.eval()

# Get real data
loader = get_dataloader(config, split="test", shuffle=False)
batch = next(iter(loader))
he = batch['he'].to(device)
ihc = batch['ihc'].to(device)

print(f"\nTesting sampling with different guidance scales:")

for guidance in [1.0, 3.0, 7.5]:
    with torch.no_grad():
        generated = model.sample(he[:1], num_inference_steps=50, guidance_scale=guidance)
    
    print(f"\nGuidance={guidance}:")
    print(f"  Range: [{generated.min():.2f}, {generated.max():.2f}]")
    print(f"  Mean: {generated.mean():.2f}, Std: {generated.std():.2f}")
    print(f"  Unique values: {generated.flatten().unique().shape[0]}")

# Check latent space during sampling
print(f"\n\nChecking latent space:")
with torch.no_grad():
    he_latent = model.encode(he[:1])
    ihc_latent = model.encode(ihc[:1])
    
print(f"  H&E latent std: {he_latent.std():.3f}")
print(f"  IHC latent std: {ihc_latent.std():.3f}")
    
# Decode them back
with torch.no_grad():
    he_recon = model.decode(he_latent)
    ihc_recon = model.decode(ihc_latent)
    
print(f"  H&E recon range: [{he_recon.min():.2f}, {he_recon.max():.2f}]")
print(f"  IHC recon range: [{ihc_recon.min():.2f}, {ihc_recon.max():.2f}]")
