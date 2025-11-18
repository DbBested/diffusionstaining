#!/bin/bash
#SBATCH -p mit_normal_gpu          # GPU partition
#SBATCH -n 8                       # Number of CPU cores
#SBATCH --gres=gpu:h100:1          # Request 1 H100 GPU (can also use l40s)
#SBATCH --mem=64G                  # Memory
#SBATCH -t 06:00:00                # Time limit (6 hours)
#SBATCH -J virtual_ihc             # Job name
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

# Load modules (adjust based on ORCD environment)
# module load python/3.10
# module load cuda/12.1

# Activate virtual environment
# Adjust this path to your actual venv location
source ~/venv_ihc/bin/activate

# Print Python and CUDA info
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
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
