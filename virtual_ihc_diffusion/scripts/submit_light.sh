#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -n 4                       # Reduced CPUs
#SBATCH --gres=gpu:1               # Any GPU (no specific type)
#SBATCH --mem=16G                  # Reduced memory
#SBATCH -t 06:00:00                # 6 hours
#SBATCH --requeue
#SBATCH -J virtual_ihc_v0.1
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $SLURM_GPUS"
echo "Start: $(date)"
echo "=================================="

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export CUDA_VISIBLE_DEVICES=0

eval "$(micromamba shell hook --shell bash)"
micromamba activate monai-r7

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "MONAI: $(python -c 'import monai; print(monai.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo "=================================="

mkdir -p logs
cd /orcd/home/002/tomli/diffusionstaining/virtual_ihc_diffusion

python training/train.py --config configs/baseline.yaml

echo "=================================="
echo "End: $(date)"
echo "=================================="
