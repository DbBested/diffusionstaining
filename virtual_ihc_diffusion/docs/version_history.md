# Version History - Virtual IHC Staining

This document tracks all iterations, improvements, and results of the virtual IHC staining model.

---

## Version 0.1 - Baseline Model

**Date**: TBD

### Changes:
- Initial implementation of conditional latent diffusion model
- AutoencoderKL for latent compression (4x downsampling)
- Basic conditional UNet with attention
- DDIM scheduler with 50 inference steps
- Image resolution: 256x256
- Simple MSE loss on noise prediction
- Basic data augmentation (rotation, flip, zoom, contrast)

### Configuration:
- Batch size: 8
- Learning rate: 1e-4
- Optimizer: AdamW
- Epochs: 100
- Scheduler: CosineAnnealingLR

### Rationale:
Start with a simple, working baseline to establish the training pipeline and verify that the approach works. Using smaller images (256x256) for faster iteration during initial development.

### Results:
- PSNR: TBD
- SSIM: TBD
- Training time: TBD
- Visual quality: TBD

### Observations:
TBD

### Next Steps:
1. Increase image resolution to 512x512
2. Add perceptual loss (VGG) for better texture fidelity
3. Implement Fast-DDPM (10 steps) for faster training
4. Test different guidance scales
5. Add color normalization preprocessing

---

## Version 0.2 - Enhanced Resolution

**Date**: TBD

### Changes:
TBD

### Rationale:
TBD

### Results:
- PSNR: TBD
- SSIM: TBD

### Observations:
TBD

### Next Steps:
TBD

---

## Template for Future Versions

```markdown
## Version X.X - Description

**Date**: YYYY-MM-DD

### Changes:
- List all changes made
- Architecture modifications
- Hyperparameter adjustments
- New techniques added

### Rationale:
Why these changes were made based on previous results

### Results:
- PSNR: X.XX dB
- SSIM: X.XX
- Other metrics
- Training time: X hours
- Inference time: X seconds/image

### Observations:
- What worked well
- What didn't work
- Unexpected behaviors
- Visual quality notes

### Next Steps:
- Planned improvements for next version
- Hypotheses to test
```

---

## Notes

- Target metrics: PSNR > 28 dB, SSIM > 0.85
- All experiments tracked with TensorBoard
- Checkpoints saved in `checkpoints/`
- Sample outputs saved in `outputs/`
