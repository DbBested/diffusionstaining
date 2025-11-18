# Virtual IHC Staining using Diffusion Models

A research project implementing conditional latent diffusion models for transforming H&E stained histopathology images to virtual IHC (Immunohistochemical) staining.

## Overview

This project uses state-of-the-art diffusion models (DDPM/DDIM) with MONAI framework to perform medical image-to-image translation, specifically targeting the challenging task of virtual histological staining. The model learns to generate realistic IHC-stained images from H&E-stained tissue sections.

## Features

- **Latent Diffusion Model**: Efficient conditional diffusion with AutoencoderKL
- **MONAI Integration**: Medical imaging-specific transforms and training utilities
- **Classifier-Free Guidance**: Enhanced generation quality with conditional guidance
- **Fast Inference**: DDIM sampling with 10-50 steps
- **Comprehensive Evaluation**: PSNR, SSIM metrics and visual quality assessment
- **MIT ORCD SLURM Support**: Production-ready cluster deployment scripts
- **Iterative Development**: Complete framework for continuous improvement

## Project Structure

```
diffusionstaining/
├── virtual_ihc_diffusion/     # Main implementation
│   ├── data/                  # Dataset loading
│   ├── models/                # Diffusion architecture
│   ├── training/              # Training & evaluation
│   ├── scripts/               # SLURM job scripts
│   ├── configs/               # Configuration files
│   ├── docs/                  # Documentation
│   └── README.md              # Detailed project README
├── RESEARCH_SUMMARY.md        # Research findings and plan
└── README.md                  # This file
```

## Quick Start

See [`virtual_ihc_diffusion/QUICKSTART.md`](virtual_ihc_diffusion/QUICKSTART.md) for detailed setup instructions.

### TL;DR

```bash
# 1. Navigate to project
cd virtual_ihc_diffusion

# 2. Setup environment
bash scripts/setup_environment.sh

# 3. Submit training job
sbatch scripts/submit_job.sh

# 4. Monitor training
tail -f logs/train_*.out
```

## Research Foundation

This project is built on comprehensive research of:

1. **Datasets**: BCI, ACROBAT, HER2match datasets for H&E/IHC pairs
2. **Models**: Fast-DDPM, DDIM, Latent Diffusion for medical imaging
3. **Framework**: MONAI v1.4+ with generative models support
4. **Infrastructure**: MIT ORCD GPU cluster (H100, L40S, A100)

See [`RESEARCH_SUMMARY.md`](RESEARCH_SUMMARY.md) for complete research findings.

## Target Metrics

- **PSNR**: >28 dB (excellent), >25 dB (good)
- **SSIM**: >0.85 (excellent), >0.80 (good)
- **Visual Quality**: Realistic brown DAB staining, sharp nuclear details

## Iterative Development Workflow

This project follows a rigorous iterative improvement cycle:

1. **Research** → Find datasets, review papers, plan approach
2. **Implement** → Build model, data pipeline, training loop
3. **Train** → Submit SLURM job, monitor progress
4. **Evaluate** → Compute metrics, generate samples
5. **Document** → Record results in version_history.md
6. **Improve** → Analyze, hypothesize, implement changes
7. **Repeat** → Train next version

Each version is fully documented with changes, rationale, results, and next steps.

## Documentation

- [`virtual_ihc_diffusion/README.md`](virtual_ihc_diffusion/README.md) - Detailed project documentation
- [`virtual_ihc_diffusion/QUICKSTART.md`](virtual_ihc_diffusion/QUICKSTART.md) - Quick start guide
- [`virtual_ihc_diffusion/docs/version_history.md`](virtual_ihc_diffusion/docs/version_history.md) - Version tracking
- [`RESEARCH_SUMMARY.md`](RESEARCH_SUMMARY.md) - Research findings and plan

## Key Technologies

- **PyTorch** 2.0+ - Deep learning framework
- **MONAI** 1.4+ - Medical imaging AI toolkit
- **DDIM/DDPM** - Diffusion model schedulers
- **MIT ORCD** - SLURM-based GPU cluster
- **TensorBoard** - Training monitoring

## Current Status

- ✅ Research phase complete
- ✅ Implementation complete
- ✅ Ready for training v0.1 (baseline)
- ⏳ Awaiting dataset download
- ⏳ Initial training run
- ⏳ Iterative improvement

## Next Steps

1. **Download BCI dataset** from https://bci.grand-challenge.org/
2. **Run baseline training** (v0.1)
3. **Evaluate and document** results
4. **Iterate and improve** based on metrics

## References

- **Fast-DDPM**: https://arxiv.org/abs/2405.14802
- **BCI Dataset**: https://bci.grand-challenge.org/
- **MONAI**: https://docs.monai.io/
- **MIT ORCD**: https://orcd-docs.mit.edu/

## License

MIT License
