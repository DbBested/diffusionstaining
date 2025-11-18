#!/bin/bash
# Auto-submit next training version based on config
# Usage: ./submit_next_version.sh v0.2_hires

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <config_name>"
    echo ""
    echo "Available configs:"
    ls -1 configs/*.yaml | grep -v test.yaml | xargs -n 1 basename
    exit 1
fi

CONFIG_NAME=$1

# Add .yaml extension if not provided
if [[ ! "$CONFIG_NAME" =~ \.yaml$ ]]; then
    CONFIG_NAME="${CONFIG_NAME}.yaml"
fi

CONFIG_PATH="configs/${CONFIG_NAME}"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "=================================="
echo "Submitting Training Job"
echo "=================================="
echo "Config: $CONFIG_PATH"
echo ""

# Extract version name from config
VERSION=$(grep "name:" $CONFIG_PATH | head -1 | awk '{print $2}' | tr -d '"')
echo "Version: $VERSION"
echo ""

# Create temporary submission script
TMP_SCRIPT=$(mktemp /tmp/submit_XXXXX.sh)
cat > $TMP_SCRIPT << 'EOFSCRIPT'
#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
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

EOFSCRIPT

# Add the specific config to the script
echo "python training/train.py --config $CONFIG_PATH" >> $TMP_SCRIPT

cat >> $TMP_SCRIPT << 'EOFSCRIPT'

echo "=================================="
echo "End Time: $(date)"
echo "Job completed!"
echo "=================================="
EOFSCRIPT

# Set job name from version
chmod +x $TMP_SCRIPT

# Submit job
JOB_ID=$(sbatch --job-name="$VERSION" $TMP_SCRIPT | awk '{print $4}')

echo "✅ Job submitted: $JOB_ID"
echo ""
echo "Monitor with:"
echo "  ./scripts/monitor_training.sh $JOB_ID"
echo "  squeue -j $JOB_ID"
echo "  tail -f logs/train_${JOB_ID}.out"
echo ""

# Clean up
rm $TMP_SCRIPT

# Save job info
echo "$JOB_ID:$VERSION:$CONFIG_PATH:$(date)" >> logs/job_history.txt

echo "Job history updated"
