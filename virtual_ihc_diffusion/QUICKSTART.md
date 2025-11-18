# Quick Start Guide

Get started with virtual IHC staining in 5 minutes!

## Prerequisites

- Access to MIT ORCD cluster
- Python 3.10+
- BCI dataset downloaded and organized

## Step-by-Step Setup

### 1. Clone and Navigate

```bash
cd /orcd/home/002/tomli/diffusionstaining/virtual_ihc_diffusion
```

### 2. Set Up Environment

```bash
# Make setup script executable
chmod +x scripts/setup_environment.sh

# Run setup
bash scripts/setup_environment.sh
```

This will:
- Create virtual environment at `~/venv_ihc`
- Install all dependencies
- Create necessary directories

### 3. Verify Dataset

Ensure your data is organized as:

```
/orcd/home/002/tomli/orcd/scratch/data/bci/
├── train/
│   ├── HE/     # H&E training images
│   └── IHC/    # IHC training images
└── test/
    ├── HE/     # H&E test images
    └── IHC/    # IHC test images
```

### 4. Test Dataset Loading (Optional)

```bash
source ~/venv_ihc/bin/activate
cd data
python dataset.py
```

### 5. Submit Training Job

```bash
# Make submit script executable
chmod +x scripts/submit_job.sh

# Submit to SLURM
sbatch scripts/submit_job.sh
```

### 6. Monitor Training

```bash
# Check job status
squeue -u $USER

# Watch training logs
tail -f logs/train_<JOB_ID>.out

# View TensorBoard (if using port forwarding)
tensorboard --logdir logs/
```

## Expected Timeline

- **Setup**: 10-15 minutes
- **Training (v0.1)**: 2-4 hours on H100 GPU
- **Evaluation**: 5-10 minutes

## First Results

After training completes:

1. **Check metrics** in logs: `logs/train_<JOB_ID>.out`
2. **View samples**: `outputs/virtual_ihc_v0.1/`
3. **TensorBoard**: Training curves and validation samples

## Evaluate Best Model

```bash
python training/evaluate.py \
    --config configs/baseline.yaml \
    --checkpoint checkpoints/virtual_ihc_v0.1/checkpoint_best.pth \
    --output_dir eval_outputs
```

Results will be saved in `eval_outputs/` with:
- Comparison images (H&E | Real IHC | Generated IHC)
- Metrics summary (`metrics.txt`)

## Document Results

Update `docs/version_history.md` with:
- PSNR and SSIM values
- Visual quality observations
- Next improvement ideas

## Next Steps

Based on v0.1 results:

1. **If metrics are low (<25 dB PSNR)**:
   - Check data loading (visualize samples)
   - Verify loss is decreasing
   - Try lower learning rate

2. **If metrics are good (>25 dB PSNR)**:
   - Proceed to v0.2: increase resolution to 512x512
   - Add perceptual loss
   - Implement Fast-DDPM

3. **Document everything** in version_history.md

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `configs/baseline.yaml`
- Try: `batch_size: 4` or `batch_size: 2`

### Dataset Not Found
- Verify paths in `configs/baseline.yaml`
- Check `data_root` points to correct directory
- Ensure paired images have matching filenames

### Slow Training
- Verify GPU is being used: check logs for CUDA device
- Ensure `cache_rate: 1.0` for fast data loading
- Check disk I/O is not bottleneck

### Poor Visual Quality
- Training may need more epochs
- Try different guidance scales: 3.0, 5.0, 7.5
- Check if VAE weights are frozen (they should be)

## Getting Help

1. Check logs: `logs/train_<JOB_ID>.out` and `logs/train_<JOB_ID>.err`
2. Review TensorBoard for training curves
3. Verify dataset with test script
4. Check ORCD documentation: https://orcd-docs.mit.edu/

## Quick Commands Reference

```bash
# Activate environment
source ~/venv_ihc/bin/activate

# Submit job
sbatch scripts/submit_job.sh

# Check jobs
squeue -u $USER

# Cancel job
scancel <JOB_ID>

# View logs
tail -f logs/train_<JOB_ID>.out

# Evaluate
python training/evaluate.py --config configs/baseline.yaml --checkpoint <PATH>
```

Happy training!
