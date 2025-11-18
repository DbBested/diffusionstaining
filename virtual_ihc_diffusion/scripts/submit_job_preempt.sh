#!/bin/bash
#SBATCH -p mit_preemptable         # Preemptable partition (lower priority, longer time)
#SBATCH -n 8                       # Number of CPU cores
#SBATCH --gres=gpu:1               # Request 1 GPU (any available)
#SBATCH --mem=32G                  # Memory
#SBATCH -t 12:00:00                # Time limit (12 hours)
#SBATCH --requeue                  # Auto-requeue if preempted
#SBATCH -J virtual_ihc_v0.1        # Job name
#SBATCH -o logs/train_%j.out       # Standard output log
#SBATCH -e logs/train_%j.err       # Standard error log

# Print job information
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs: $SLURM_GPUS"
echo "CPUs: $SLURM_CPUS_ON_NODE"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"
echo "=================================="

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export CUDA_VISIBLE_DEVICES=0

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate monai-r7

# Print Python and CUDA info
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "MONAI version: $(python -c 'import monai; print(monai.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo "=================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Change to project directory
cd /orcd/home/002/tomli/diffusionstaining/virtual_ihc_diffusion

# Run training
python training/train.py --config configs/baseline.yaml

# Print end time
echo "=================================="
echo "End Time: $(date)"
echo "Job completed!"
echo "=================================="
