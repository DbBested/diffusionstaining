#!/bin/bash
#SBATCH -p pg_tata  
#SBATCH -N 1
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=128G
#SBATCH -t 6:00:00
#SBATCH -J vae_diffusers
#SBATCH -o logs/vae_diffusers_%j.out
#SBATCH -e logs/vae_diffusers_%j.err

echo "============================================================================"
echo "🚀 VAE Training with Diffusers: 2x L40S"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "============================================================================"

# Environment setup
source ~/micromamba/etc/profile.d/conda.sh
conda activate monai-r7

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to project directory
cd /orcd/home/002/tomli/diffusionstaining/virtual_ihc_diffusion || exit 1

# Check environment
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "Diffusers: $(python -c 'import diffusers; print(diffusers.__version__)')"
echo ""

echo "Checking GPUs..."
python -c "import torch; print(f'PyTorch sees {torch.cuda.device_count()} GPUs')"
echo ""

# Create directories
mkdir -p logs checkpoints/vae_diffusers outputs/vae_diffusers

echo "============================================================================"
echo "Starting VAE Training..."
echo "============================================================================"
echo ""

# Run training
python training/pretrain_vae_multigpu.py \
    --config configs/full_vae_h200_max.yaml \
    --epochs 50

echo ""
echo "============================================================================"
echo "Training Complete"
echo "End: $(date)"
echo "============================================================================"