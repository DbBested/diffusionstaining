#!/bin/bash
#SBATCH -p pg_tata  
#SBATCH -N 1
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=192G
#SBATCH -t 24:00:00
#SBATCH -J diffusion_l40s
#SBATCH -o logs/diffusion_train_%j.out
#SBATCH -e logs/diffusion_train_%j.err

echo "============================================================================"
echo "🚀 H&E to IHC Diffusion Training: 4x L40S GPUs"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start: $(date)"
echo "Config: diffusion_prod_l40s.yaml"
echo "VAE Checkpoint: checkpoints/vae_diffusers/vae_best.pth"
echo "============================================================================"

# Environment setup
source ~/micromamba/etc/profile.d/conda.sh
conda activate monai-r7

# Memory optimization for L40S
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to project directory
cd /orcd/home/002/tomli/diffusionstaining/virtual_ihc_diffusion || exit 1

# Python environment info
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "MONAI: $(python -c 'import monai; print(monai.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

echo "Checking GPUs..."
python -c "import torch; print(f'PyTorch sees {torch.cuda.device_count()} GPUs')"
python -c "import torch; [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo ""

# Verify VAE checkpoint exists
echo "Checking VAE checkpoint..."
if [ -f "checkpoints/vae_diffusers/vae_best.pth" ]; then
    python -c "
import torch
ckpt = torch.load('checkpoints/vae_diffusers/vae_best.pth', map_location='cpu')
print(f'✓ VAE checkpoint found')
print(f'  Validation loss: {ckpt[\"val_loss\"]:.4f}')
print(f'  Trained epochs: {ckpt[\"epoch\"]+1}')
"
else
    echo "❌ ERROR: VAE checkpoint not found!"
    echo "   Expected: checkpoints/vae_diffusers/vae_best.pth"
    echo "   Train VAE first before starting diffusion training!"
    exit 1
fi

echo ""
echo "============================================================================"
echo "Starting Diffusion Training..."
echo "============================================================================"
echo ""

# Create directories
mkdir -p logs checkpoints outputs

# Run training
python training/train.py --config configs/prod.yaml

echo ""
echo "============================================================================"
echo "Training Complete"
echo "End: $(date)"
echo "============================================================================"