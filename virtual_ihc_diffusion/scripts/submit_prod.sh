#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -J virtual_ihc_full
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $SLURM_GPUS"
echo "Start: $(date)"
echo "Config: prod.yaml (optimized for full quality)"
echo "=================================="

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

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

python training/train.py --config configs/prod.yaml

echo "=================================="
echo "End: $(date)"
echo "=================================="
