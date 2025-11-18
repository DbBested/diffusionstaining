# Virtual IHC Staining Research Summary

**Date**: November 17, 2025
**Mission**: Build a diffusion model that transforms H&E stained images to virtual IHC staining

---

## 1. Dataset Selection

### Recommended Dataset: **BCI (Breast Cancer Immunohistochemical)**

**Rationale**:
- **Size**: 4,873 registered H&E-HER2 IHC image pairs (3,896 train, 977 test)
- **Resolution**: 1024×1024 pixels - ideal for diffusion training
- **Quality**: First publicly available dataset of registered H&E and HER2 IHC tile pairs
- **Accessibility**: Available at https://bci.grand-challenge.org/

**Alternative Datasets**:
1. **HER2match** (2025, arXiv:2506.18484) - Same tissue sections (not consecutive), superior alignment
   - Status: May require access request
   - Advantage: Better visual quality in published results

2. **ACROBAT** (2023-2024) - 4,212 WSIs with H&E and multiple IHC stains
   - Size: ~448 GB across 7 ZIP archives
   - Advantage: Multiple IHC markers (ER, PGR, HER2, KI67)

**Action**: Start with BCI dataset for rapid prototyping, consider HER2match for v2.0

---

## 2. Diffusion Model Architecture

### Recommended Approach: **Latent Diffusion Model (LDM) with Conditioning**

**Key Research Findings** (2024):

1. **Fast-DDPM** (May 2024, arXiv:2405.14802)
   - Reduces training and sampling from 1,000 to 10 steps
   - 100x faster than standard DDPM
   - Outperforms DDPM on medical image-to-image tasks
   - **Decision**: Use this for production model

2. **DDIM vs DDPM**:
   - DDIM: Deterministic, faster sampling, better for our application
   - DDPM: More stochastic, slower
   - **Decision**: Implement DDIM for inference speed

3. **Conditioning Strategy**:
   - Use H&E images as conditional input
   - Concatenate with noisy latent (channel-wise)
   - Apply classifier-free guidance for better fidelity

### Architecture Components:

```
1. AutoencoderKL (VAE) - MONAI implementation
   - Compress 1024x1024 → 128x128 latent space
   - Reduces computation by 64x

2. Conditional UNet - MONAI DiffusionModelUNet
   - Input: Noisy latent + H&E condition
   - Output: Predicted noise
   - Attention layers for spatial coherence

3. DDIM Scheduler - MONAI DDIMScheduler
   - 10-50 diffusion steps (Fast-DDPM approach)
   - Linear or cosine noise schedule
```

**Loss Function**:
- L2 loss on predicted noise (standard DDPM)
- Optional: Perceptual loss (VGG) for texture fidelity
- Optional: L1 loss in pixel space for color accuracy

---

## 3. MONAI Implementation Details

### Available MONAI Components (v1.4+):

1. **Networks**:
   - `AutoencoderKL` - Latent diffusion VAE
   - `DiffusionModelUNet` - Conditional UNet
   - `ControlNetLatentDiffusionInferer` - For inference

2. **Schedulers**:
   - `DDPMScheduler` - Standard diffusion
   - `DDIMScheduler` - Fast deterministic sampling
   - `RectifiedFlowScheduler` - 33x faster (MONAI 1.5+)

3. **Transforms** (Medical-specific):
   - `LoadImage` - Medical image formats
   - `ScaleIntensity` - Normalize to [-1, 1]
   - `RandRotate`, `RandFlip` - Data augmentation
   - `RandAdjustContrast` - Stain variation robustness
   - `CacheDataset` - Fast data loading

4. **Training Utilities**:
   - `LatentDiffusionInferer` - Training wrapper
   - Automatic mixed precision (AMP)
   - Distributed training support

**Tutorial Reference**:
- MONAI Tutorials: `/generation` directory
- DiMEDIA 2024 Tutorial (ISBI): Diffusion models theory and tricks

---

## 4. MIT ORCD SLURM Configuration

### GPU Resources:

**Partition Options**:

1. **mit_normal_gpu** (Recommended for development)
   - Max time: 6 hours
   - Max GPUs: 2
   - Max cores: 32
   - GPU types: L40S, H100, H200
   - Priority: Normal

2. **mit_preemptable** (For long training runs)
   - Max time: 48 hours
   - Max GPUs: 4
   - Max cores: 1024
   - GPU types: L40S, H100, H200, A100
   - Priority: Low (preemptable)
   - Use `--requeue` flag for auto-restart

**Recommended GPU**:
- **H100** or **L40S** for training (40-80GB VRAM)
- 1-2 GPUs sufficient for 1024x1024 images with latent diffusion

### SLURM Script Template:

```bash
#!/bin/bash
#SBATCH -p mit_normal_gpu          # Partition
#SBATCH -n 8                       # CPU cores
#SBATCH --gres=gpu:h100:1          # 1 H100 GPU
#SBATCH --mem=64G                  # RAM
#SBATCH -t 06:00:00                # 6 hours
#SBATCH -o logs/train_%j.out       # Output log
#SBATCH -e logs/train_%j.err       # Error log

# Environment setup
module load python/3.10
source ~/venv/bin/activate

# Application
python training/train.py --config configs/baseline.yaml
```

---

## 5. Implementation Plan

### Phase 1: Baseline Model (v0.1)

**Goal**: Simple working diffusion model with basic metrics

**Architecture**:
- AutoencoderKL (4x downsampling)
- Simple conditional UNet
- DDIM with 50 steps
- Image size: 256x256 (for speed)

**Training**:
- Batch size: 4-8
- Learning rate: 1e-4
- Optimizer: AdamW
- Epochs: 50-100
- Loss: MSE on noise prediction

**Evaluation**:
- PSNR (target: >25 dB)
- SSIM (target: >0.80)
- Visual inspection

**Timeline**: 1-2 days

---

### Phase 2: Enhanced Model (v0.2-v0.5)

**Improvements to test**:
1. Increase resolution to 512x512
2. Add perceptual loss (VGGLoss)
3. Implement Fast-DDPM (10 steps)
4. Add color normalization (Macenko/Reinhard)
5. Stronger data augmentation
6. Classifier-free guidance

**Expected**:
- PSNR: 26-28 dB
- SSIM: 0.82-0.85

**Timeline**: 1-2 weeks

---

### Phase 3: Production Model (v1.0)

**Final optimizations**:
1. Full resolution (1024x1024)
2. Ensemble prediction
3. Post-processing refinement
4. Attention visualization
5. Comprehensive evaluation

**Target**:
- PSNR: >28 dB
- SSIM: >0.85
- Clinically realistic images

**Timeline**: 2-4 weeks

---

## 6. Evaluation Metrics

### Quantitative Metrics:

1. **PSNR (Peak Signal-to-Noise Ratio)**
   - Measures pixel-level accuracy
   - Target: >28 dB (excellent), >25 dB (good)
   - Formula: 20 * log10(MAX / sqrt(MSE))

2. **SSIM (Structural Similarity Index)**
   - Measures perceptual similarity
   - Target: >0.85 (excellent), >0.80 (good)
   - Captures luminance, contrast, structure

3. **LPIPS (Learned Perceptual Image Patch Similarity)**
   - Deep feature similarity
   - Lower is better
   - Target: <0.2

### Qualitative Metrics:

1. Color fidelity (brown DAB staining)
2. Nuclear detail preservation
3. Background consistency
4. Absence of artifacts

---

## 7. Dataset Download Instructions

### BCI Dataset:

1. Visit https://bci.grand-challenge.org/
2. Register account
3. Join challenge
4. Download from dataset page
5. Organize as:
   ```
   /orcd/home/002/tomli/orcd/scratch/data/bci/
   ├── train/
   │   ├── HE/
   │   └── IHC/
   └── test/
       ├── HE/
       └── IHC/
   ```

---

## 8. Key References

1. **Fast-DDPM**: https://arxiv.org/abs/2405.14802
2. **BCI Dataset**: https://bci.grand-challenge.org/
3. **GANs vs Diffusion for Virtual Staining**: https://arxiv.org/abs/2506.18484
4. **MONAI Generative Models**: https://github.com/Project-MONAI/tutorials
5. **MIT ORCD Docs**: https://orcd-docs.mit.edu/

---

## 9. Success Criteria

- ✅ PSNR > 28 dB
- ✅ SSIM > 0.85
- ✅ Realistic brown DAB staining
- ✅ Sharp nuclear details
- ✅ No visual artifacts
- ✅ Consistent with real IHC images
- ✅ Complete documentation
- ✅ Reproducible results

---

**Next Step**: Create project structure and begin implementation of baseline model (v0.1)
