with open('models/diffusion_model.py', 'r') as f:
    content = f.read()

# Remove scaling from encode - keep raw latents
new_encode = '''    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space"""
        # Diffusers VAE returns a distribution, we take the mode
        latent_dist = self.autoencoder.encode(x)
        z = latent_dist.latent_dist.mode()
        # Don't use SD VAE scaling factor - it breaks our latent space
        # Instead, we'll normalize to std=1.0 manually
        z = z / z.std() * 1.0
        return z'''

# Find and replace encode
import re
pattern = r'(@torch\.no_grad\(\)\s+def encode\(self, x: torch\.Tensor\) -> torch\.Tensor:.*?return z)'
match = re.search(pattern, content, re.DOTALL)
if match:
    content = content.replace(match.group(1), new_encode.strip())

# Remove scaling from decode
new_decode = '''    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image space"""
        # No scaling - latents are already normalized
        x = self.autoencoder.decode(z).sample
        return x'''

pattern = r'(@torch\.no_grad\(\)\s+def decode\(self, z: torch\.Tensor\) -> torch\.Tensor:.*?return x)'
match = re.search(pattern, content, re.DOTALL)
if match:
    content = content.replace(match.group(1), new_decode.strip())

with open('models/diffusion_model.py', 'w') as f:
    f.write(content)

print("✅ Removed SD VAE scaling and added manual normalization")
