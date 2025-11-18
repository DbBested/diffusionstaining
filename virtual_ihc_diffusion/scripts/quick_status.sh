#!/bin/bash
# Quick status check for all training jobs

echo "========================================="
echo "Virtual IHC Training - Quick Status"
echo "========================================="
echo ""

# Check running jobs
echo "📊 Current SLURM Jobs:"
echo "-------------------------------------"
squeue -u $USER -o "%.10i %.12P %.20j %.8T %.10M %.6D %R" 2>/dev/null || echo "No jobs found"
echo ""

# Check recent logs
echo "📝 Recent Training Runs:"
echo "-------------------------------------"
if [ -d "logs" ]; then
    for log in $(ls -t logs/train_*.out 2>/dev/null | head -5); do
        job_id=$(basename $log | sed 's/train_//' | sed 's/.out//')

        # Extract key info
        if [ -f "$log" ]; then
            version=$(grep "virtual_ihc" $log | head -1 | awk -F'/' '{print $NF}' || echo "unknown")
            status="running"
            if grep -q "Job completed!" $log; then
                status="✅ completed"
            elif grep -q "Error\|Traceback" $log; then
                status="❌ failed"
            fi

            # Get metrics if available
            psnr=$(grep "PSNR:" $log | tail -1 | awk '{print $2}' || echo "N/A")
            ssim=$(grep "SSIM:" $log | tail -1 | awk '{print $2}' || echo "N/A")

            echo "Job $job_id: $status | PSNR: $psnr | SSIM: $ssim"
        fi
    done
else
    echo "No logs directory found"
fi
echo ""

# Check available configs
echo "🔧 Available Configurations:"
echo "-------------------------------------"
ls -1 configs/*.yaml 2>/dev/null | grep -v test.yaml | sed 's/configs\///' | sed 's/.yaml//' || echo "No configs found"
echo ""

# Check checkpoints
echo "💾 Recent Checkpoints:"
echo "-------------------------------------"
if [ -d "checkpoints" ]; then
    find checkpoints -name "checkpoint_best.pth" -o -name "checkpoint_epoch_*.pth" 2>/dev/null | head -5 || echo "No checkpoints found"
else
    echo "No checkpoints directory found"
fi
echo ""

echo "========================================="
echo "Quick Commands:"
echo "  Monitor job:    ./scripts/monitor_training.sh <JOB_ID>"
echo "  Analyze results: python scripts/analyze_results.py --compare"
echo "  Submit next:     ./scripts/submit_next_version.sh <CONFIG>"
echo "  Follow logs:     tail -f logs/train_<JOB_ID>.out"
echo "========================================="
