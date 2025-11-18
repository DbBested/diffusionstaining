# Training Scripts and Utilities

This directory contains automation scripts for managing iterative training experiments.

## Available Scripts

### 1. `monitor_training.sh` - Real-time Job Monitoring

Monitor SLURM job status and training progress.

```bash
./scripts/monitor_training.sh <JOB_ID>
```

**Features:**
- Displays job status (PENDING, RUNNING, COMPLETED, FAILED)
- Shows last 50 lines of training logs
- Extracts metrics (PSNR, SSIM, loss)
- Highlights errors if present

**Example:**
```bash
./scripts/monitor_training.sh 6470504
```

---

### 2. `analyze_results.py` - Results Analysis

Analyze completed training runs and compare versions.

```bash
# Analyze specific job
python scripts/analyze_results.py --job_id 6470504

# Compare all versions
python scripts/analyze_results.py --compare

# Analyze latest run
python scripts/analyze_results.py
```

**Features:**
- Extracts metrics from logs (PSNR, SSIM, loss)
- Performance assessment (target vs actual)
- Version comparison table
- Identifies best performing model

**Output Example:**
```
============================================================
TRAINING RESULTS SUMMARY - Job 6470504
============================================================

Status: COMPLETED
Epochs: 200/200

📊 Best PSNR:  26.3421 dB
   Final PSNR: 26.1234 dB
📊 Best SSIM:  0.8234
   Final SSIM: 0.8156

📉 Final Val Loss: 0.001234
⏱️  Training Time: 02:34:56

============================================================

🎯 Performance Assessment:
  ⚠️  Good PSNR (25-28 dB) - Room for improvement
  ⚠️  Good SSIM (0.80-0.85) - Room for improvement
```

---

### 3. `submit_next_version.sh` - Auto-Submit Next Version

Submit training jobs for different experimental configurations.

```bash
./scripts/submit_next_version.sh <config_name>
```

**Available Configs:**
- `baseline` - v0.1 (256px, standard settings)
- `v0.2_hires` - Higher resolution (512px)
- `v0.3_augmented` - Enhanced augmentation
- `v0.4_fast` - Fast-DDPM (100 timesteps)
- `v0.5_combined` - Best of all versions

**Example:**
```bash
# Submit v0.2 (higher resolution)
./scripts/submit_next_version.sh v0.2_hires

# Submit v0.4 (fast training)
./scripts/submit_next_version.sh v0.4_fast
```

---

### 4. `quick_status.sh` - Dashboard Overview

Quick status check of all training jobs.

```bash
./scripts/quick_status.sh
```

**Shows:**
- Current SLURM jobs
- Recent training runs with metrics
- Available configurations
- Recent checkpoints
- Quick command reference

**Example Output:**
```
=========================================
Virtual IHC Training - Quick Status
=========================================

📊 Current SLURM Jobs:
-------------------------------------
JOBID      PARTITION  NAME           STATE     TIME
6470504    mit_preem  virtual_ihc_v0.1  RUNNING   01:23:45

📝 Recent Training Runs:
-------------------------------------
Job 6470504: running | PSNR: 25.4 | SSIM: 0.81
Job 6470123: ✅ completed | PSNR: 24.1 | SSIM: 0.79

🔧 Available Configurations:
-------------------------------------
baseline
v0.2_hires
v0.3_augmented
v0.4_fast
v0.5_combined
```

---

### 5. `prepare_dataset.py` - Dataset Preparation

Prepare and organize training data with train/test split.

```bash
python scripts/prepare_dataset.py
```

**Features:**
- Automatically finds paired H&E/IHC images
- Creates 80/20 train/test split
- Renames files for consistency
- Reports matching statistics

---

## Typical Workflow

### 1. Submit Initial Training (v0.1)

```bash
# Already submitted: Job 6470504
squeue -u $USER
```

### 2. Monitor Progress

```bash
# Check status
./scripts/quick_status.sh

# Monitor specific job
./scripts/monitor_training.sh 6470504

# Follow logs in real-time
tail -f logs/train_6470504.out
```

### 3. Analyze Results

```bash
# Wait for completion
./scripts/monitor_training.sh 6470504

# Analyze results
python scripts/analyze_results.py --job_id 6470504
```

### 4. Submit Next Version

Based on v0.1 results, choose next experiment:

```bash
# If metrics are low: Try higher resolution
./scripts/submit_next_version.sh v0.2_hires

# If training is slow: Try fast-DDPM
./scripts/submit_next_version.sh v0.4_fast

# If ready for best model: Try combined
./scripts/submit_next_version.sh v0.5_combined
```

### 5. Compare All Versions

```bash
python scripts/analyze_results.py --compare
```

---

## Configuration Files

Located in `configs/`:

### `baseline.yaml` (v0.1)
- Resolution: 256x256
- Epochs: 200
- Timesteps: 1000
- **Use for**: Quick baseline

### `v0.2_hires.yaml`
- Resolution: 512x512  ⬆️
- Epochs: 150
- Timesteps: 1000
- **Use for**: Better detail preservation

### `v0.3_augmented.yaml`
- Resolution: 256x256
- Guidance scale: 5.0 (vs 7.5)
- Unconditional prob: 0.15 (vs 0.10)
- **Use for**: Better generalization

### `v0.4_fast.yaml`
- Resolution: 256x256
- Timesteps: 100 (vs 1000) ⚡
- Inference steps: 10 (vs 50)
- **Use for**: Faster training/inference

### `v0.5_combined.yaml`
- Resolution: 512x512
- Timesteps: 100
- Optimized architecture
- **Use for**: Best overall performance

---

## Tips

1. **Always check job status** before submitting new jobs:
   ```bash
   ./scripts/quick_status.sh
   ```

2. **Monitor GPU usage**:
   ```bash
   sacct -j <JOB_ID> --format=JobID,ReqGRES,Elapsed,State
   ```

3. **Check for errors**:
   ```bash
   tail -50 logs/train_<JOB_ID>.err
   ```

4. **Compare versions** after each run:
   ```bash
   python scripts/analyze_results.py --compare
   ```

5. **Document results** in `docs/version_history.md` after each version

---

## Troubleshooting

### Job stuck in PENDING
```bash
squeue -j <JOB_ID>  # Check reason
# Usually: (ReqNodeNotAvail, Reserved for maintenance)
# Solution: Wait for maintenance to complete
```

### Out of memory
- Reduce `batch_size` in config
- Reduce `image_size`
- Set `cache_rate: 0.0`

### Low metrics
- Try different config (v0.2, v0.3, v0.4)
- Check visual outputs in `outputs/`
- Ensure data is correctly loaded

---

## File Permissions

Make scripts executable:
```bash
chmod +x scripts/*.sh
```
