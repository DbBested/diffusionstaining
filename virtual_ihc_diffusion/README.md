# Virtual IHC Staining using Diffusion Models

Transform H&E stained histopathology images to virtual IHC (Immunohistochemical) staining using conditional latent diffusion models.

## Overview

This project implements a state-of-the-art diffusion model for medical image-to-image translation, specifically targeting the challenging task of virtual staining. The model learns to generate realistic IHC-stained images from H&E-stained tissue sections.

### Key Features

- **Latent Diffusion Architecture**: Efficient training with AutoencoderKL compression
- **MONAI Integration**: Medical imaging-specific transforms and utilities
- **Classifier-Free Guidance**: Improved generation quality with conditioning
- **Fast Inference**: DDIM sampling with 10-50 steps
- **Comprehensive Evaluation**: PSNR, SSIM, and visual quality metrics
- **SLURM Support**: Ready for MIT ORCD cluster deployment

## Project Structure

```
virtual_ihc_diffusion/
├── data/                  # Dataset loading with MONAI transforms
│   ├── __init__.py
│   └── dataset.py
├── models/                # Diffusion model architecture
│   ├── __init__.py
│   └── diffusion_model.py
├── training/              # Training and evaluation
│   ├── __init__.py
│   ├── train.py          # Main training script
│   ├── evaluate.py       # Evaluation script
│   └── metrics.py        # PSNR, SSIM metrics
├── scripts/               # SLURM job scripts
│   └── submit_job.sh     # MIT ORCD submission script
├── configs/               # Configuration files
│   └── baseline.yaml     # Baseline configuration
├── docs/                  # Documentation
│   └── version_history.md
├── logs/                  # Training logs (TensorBoard)
├── checkpoints/           # Model checkpoints
├── outputs/               # Generated samples
└── requirements.txt       # Python dependencies
```

## Setup

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv ~/venv_ihc
source ~/venv_ihc/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Download

Download the BCI (Breast Cancer Immunohistochemical) dataset:

1. Visit https://bci.grand-challenge.org/
2. Register and join the challenge
3. Download the dataset
4. Organize as follows:

```
/orcd/home/002/tomli/orcd/scratch/data/bci/
├── train/
│   ├── HE/     # H&E training images
│   └── IHC/    # IHC training images
└── test/
    ├── HE/     # H&E test images
    └── IHC/    # IHC test images
```

## Usage

### Training

#### Local Training

```bash
python training/train.py --config configs/baseline.yaml
```

#### SLURM Cluster (MIT ORCD)

```bash
# Submit job
sbatch scripts/submit_job.sh

# Monitor job
squeue -u $USER

# Check logs
tail -f logs/train_<job_id>.out
```

### Evaluation

```bash
python training/evaluate.py \
    --config configs/baseline.yaml \
    --checkpoint checkpoints/virtual_ihc_v0.1/checkpoint_best.pth \
    --output_dir eval_outputs \
    --num_steps 50
```

### Monitoring

View training progress with TensorBoard:

```bash
tensorboard --logdir logs/
```

## Configuration

Edit `configs/baseline.yaml` to modify:

- **Model architecture**: UNet channels, attention levels, latent dimensions
- **Training**: Batch size, learning rate, epochs, augmentation
- **Evaluation**: Metrics, inference steps, guidance scale
- **Data**: Image size, cache rate, dataset paths

## Model Architecture

```
H&E Input [3, 256, 256]
    ↓
AutoencoderKL Encoder
    ↓
Latent [4, 64, 64] → Concatenate ← Conditioning Latent
    ↓
Add Noise (Training) / Random Noise (Inference)
    ↓
Conditional UNet (Denoising)
    ↓
DDIM Scheduler (50 steps)
    ↓
AutoencoderKL Decoder
    ↓
IHC Output [3, 256, 256]
```

## Metrics

- **PSNR** (Peak Signal-to-Noise Ratio): Measures pixel-level accuracy
  - Target: >28 dB (excellent), >25 dB (good)
- **SSIM** (Structural Similarity Index): Measures perceptual similarity
  - Target: >0.85 (excellent), >0.80 (good)

## Results

See `docs/version_history.md` for detailed results of each model version.

### Current Best Model: v0.1 (Baseline)

- PSNR: TBD
- SSIM: TBD
- Inference time: TBD seconds/image

## Iterative Development

This project follows an iterative improvement cycle:

1. **Train**: Submit SLURM job
2. **Evaluate**: Compute metrics and generate samples
3. **Document**: Update `version_history.md` with results
4. **Improve**: Analyze results and implement enhancements
5. **Repeat**: Train next version

### Planned Improvements

- [ ] Increase resolution to 512x512
- [ ] Add perceptual loss (VGG)
- [ ] Implement Fast-DDPM (10 steps)
- [ ] Color normalization (Macenko/Reinhard)
- [ ] Stronger data augmentation
- [ ] Test different guidance scales
- [ ] Full resolution (1024x1024)

## References

1. **Fast-DDPM**: Fast Denoising Diffusion Probabilistic Models for Medical Image-to-Image Generation
   - Paper: https://arxiv.org/abs/2405.14802

2. **BCI Dataset**: Breast Cancer Immunohistochemical Image Generation
   - Website: https://bci.grand-challenge.org/

3. **MONAI**: Medical Open Network for AI
   - Docs: https://docs.monai.io/

4. **DDIM**: Denoising Diffusion Implicit Models
   - Paper: https://arxiv.org/abs/2010.02502

## License

MIT License (or specify your license)

## Contact

For questions or issues, please contact [your email] or open an issue on GitHub.

## Acknowledgments

- BCI dataset creators
- MONAI framework developers
- MIT ORCD computing resources
