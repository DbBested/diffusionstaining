#!/bin/bash
#SBATCH -p pg_tata  
#SBATCH -N 1
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -J vae_l40s_4gpu
#SBATCH -o logs/vae_pretrain_%j.out  
#SBATCH -e logs/vae_pretrain_%j.err  

echo "============================================================================"
echo "🚀 Multi-GPU VAE Training: 4x L40S"
echo "============================================================================"

source ~/micromamba/etc/profile.d/conda.sh
conda activate monai-r7

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /orcd/home/002/tomli/diffusionstaining/virtual_ihc_diffusion || exit 1

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

echo "Checking GPUs..."
python -c "import torch; print(f'PyTorch sees {torch.cuda.device_count()} GPUs')"
python -c "import torch; [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo ""

mkdir -p logs checkpoints/vae_pretrain outputs/vae_pretrain

# Parse arguments - check if RESUME_CHECKPOINT was passed
RESUME_FLAG=""

if [ -n "$RESUME_CHECKPOINT" ]; then
    # Explicit checkpoint path provided
    if [ "$RESUME_CHECKPOINT" = "auto" ]; then
        # Auto-detect latest
        LATEST_CHECKPOINT=$(ls -t checkpoints/vae_pretrain/vae_epoch_*.pth 2>/dev/null | head -1)
        if [ -n "$LATEST_CHECKPOINT" ]; then
            RESUME_FLAG="--resume $LATEST_CHECKPOINT"
            echo "📂 Auto-detected checkpoint: $LATEST_CHECKPOINT"
        elif [ -f "checkpoints/vae_pretrain/vae_best.pth" ]; then
            RESUME_FLAG="--resume checkpoints/vae_pretrain/vae_best.pth"
            echo "📂 Auto-detected checkpoint: checkpoints/vae_pretrain/vae_best.pth"
        else
            echo "⚠️  No checkpoint found for auto-resume"
        fi
    else
        # Use the specified checkpoint
        RESUME_FLAG="--resume $RESUME_CHECKPOINT"
        echo "📂 Using specified checkpoint: $RESUME_CHECKPOINT"
    fi
else
    echo "🆕 Training from scratch (no resume)"
fi

# Show checkpoint info if resuming
if [ -n "$RESUME_FLAG" ]; then
    python -c "
import torch
import sys
checkpoint_path = '$RESUME_FLAG'.replace('--resume ', '')
try:
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    print(f'')
    print(f'Checkpoint info:')
    print(f'  Epoch: {ckpt[\"epoch\"] + 1}')
    print(f'  Train loss: {ckpt.get(\"train_loss\", 0):.4f}')
    print(f'  Val loss: {ckpt.get(\"val_loss\", 0):.4f}')
    print(f'  Best loss: {ckpt.get(\"best_loss\", float(\"inf\")):.4f}')
    print(f'')
    print(f'Will continue from epoch {ckpt[\"epoch\"] + 2} to 50')
except Exception as e:
    print(f'⚠️  Could not load checkpoint info: {e}')
    sys.exit(1)
"
    if [ $? -ne 0 ]; then
        echo "Error loading checkpoint. Aborting."
        exit 1
    fi
fi

echo ""
echo "Starting training..."
echo ""

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    training/pretrain_vae_multigpu.py \
    --config configs/full_vae_h200_max.yaml \
    --epochs 50 \
    $RESUME_FLAG \
    --gradient-accumulation-steps 1

echo "Finished: $(date)"