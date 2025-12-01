#!/bin/bash
#SBATCH -p pg_tata  
#SBATCH -n 8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -J virtual_ihc_fast
#SBATCH -o logs/vae_pretrain_%j.out  
#SBATCH -e logs/vae_pretrain_%j.err  

# ============================================================================
# VAE Pretraining Job Script
# ============================================================================

echo "============================================================================"
echo "Job Information"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Started at: $(date)"
echo "Working directory: $(pwd)"
echo ""

# ============================================================================
# Environment Setup
# ============================================================================

echo "============================================================================"
echo "Setting up environment"
echo "============================================================================"

# Load any required modules (uncomment if needed)
# module load cuda/11.8

# Activate conda environment
echo "Activating conda environment: monai-r7"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate monai-r7

# Verify environment
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)'; then
    echo "GPU device: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# ============================================================================
# Navigate to project directory
# ============================================================================

cd /orcd/home/002/tomli/diffusionstaining/virtual_ihc_diffusion || exit 1
echo "Project directory: $(pwd)"
echo ""

# ============================================================================
# Create necessary directories
# ============================================================================

echo "Creating output directories..."
mkdir -p logs
mkdir -p checkpoints/vae_pretrain
mkdir -p outputs/vae_pretrain
echo ""

# ============================================================================
# Run VAE Pretraining
# ============================================================================

echo "============================================================================"
echo "Starting VAE Pretraining"
echo "============================================================================"
echo "Config: configs/full_vae.yaml"
echo "Epochs: 50"
echo ""

python training/pretrain_vae.py \
    --config configs/full_vae.yaml \
    --epochs 50
EXIT_CODE=$?

# ============================================================================
# Job Summary
# ============================================================================

echo ""
echo "============================================================================"
echo "Job Summary"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS ✓"
    echo ""
    echo "Next steps:"
    echo "1. Check training curves: tensorboard --logdir=logs/vae_pretrain"
    echo "2. View sample reconstructions in: outputs/vae_pretrain/"
    echo "3. Best checkpoint saved at: checkpoints/vae_pretrain/vae_best.pth"
    echo "4. Update your diffusion training to load the pretrained VAE"
else
    echo "Status: FAILED ✗"
    echo "Check error log: logs/vae_pretrain_${SLURM_JOB_ID}.err"
fi

echo "============================================================================"

exit $EXIT_CODE