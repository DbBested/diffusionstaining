#!/bin/bash
# Automated training monitor
# Checks SLURM job status and displays relevant logs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================="
echo "Virtual IHC Training Monitor"
echo -e "==================================${NC}\n"

# Check if job ID provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}Usage: $0 <job_id>${NC}"
    echo ""
    echo "Current jobs for $USER:"
    squeue -u $USER -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"
    exit 1
fi

JOB_ID=$1

echo -e "${BLUE}Monitoring Job ID: $JOB_ID${NC}\n"

# Function to get job status
get_job_status() {
    squeue -j $JOB_ID -h -o "%T" 2>/dev/null || echo "NOT_FOUND"
}

# Function to get job info
get_job_info() {
    sacct -j $JOB_ID --format=JobID,JobName,State,Elapsed,ReqGRES,NodeList -P 2>/dev/null | head -2
}

# Main monitoring loop
STATUS=$(get_job_status)

echo -e "${BLUE}Job Status:${NC} $STATUS"
echo ""

# Display job info
echo -e "${BLUE}Job Information:${NC}"
get_job_info
echo ""

# Check log files
LOG_FILE="logs/train_${JOB_ID}.out"
ERR_FILE="logs/train_${JOB_ID}.err"

if [ "$STATUS" == "RUNNING" ]; then
    echo -e "${GREEN}Job is RUNNING!${NC}\n"

    if [ -f "$LOG_FILE" ]; then
        echo -e "${BLUE}Last 50 lines of training log:${NC}"
        echo -e "${YELLOW}--------------------------------${NC}"
        tail -50 "$LOG_FILE"
        echo -e "${YELLOW}--------------------------------${NC}\n"

        # Extract metrics if available
        echo -e "${BLUE}Recent metrics:${NC}"
        grep -E "PSNR|SSIM|loss|Epoch" "$LOG_FILE" | tail -20 || echo "No metrics found yet"
    else
        echo -e "${YELLOW}Log file not created yet${NC}"
    fi

    if [ -f "$ERR_FILE" ] && [ -s "$ERR_FILE" ]; then
        echo -e "\n${RED}Errors detected:${NC}"
        tail -20 "$ERR_FILE"
    fi

elif [ "$STATUS" == "PENDING" ]; then
    echo -e "${YELLOW}Job is PENDING - waiting for resources${NC}"
    squeue -j $JOB_ID -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"

elif [ "$STATUS" == "COMPLETED" ]; then
    echo -e "${GREEN}Job COMPLETED!${NC}\n"

    if [ -f "$LOG_FILE" ]; then
        echo -e "${BLUE}Final metrics:${NC}"
        grep -E "PSNR|SSIM|best" "$LOG_FILE" | tail -30

        echo -e "\n${BLUE}Training completed at:${NC}"
        grep "End Time" "$LOG_FILE" || echo "Timestamp not found"
    fi

elif [ "$STATUS" == "FAILED" ]; then
    echo -e "${RED}Job FAILED!${NC}\n"

    if [ -f "$LOG_FILE" ]; then
        echo -e "${RED}Last 30 lines of log:${NC}"
        tail -30 "$LOG_FILE"
    fi

    if [ -f "$ERR_FILE" ]; then
        echo -e "\n${RED}Error log:${NC}"
        cat "$ERR_FILE"
    fi

else
    echo -e "${YELLOW}Job status: $STATUS${NC}"
    if [ -f "$LOG_FILE" ]; then
        tail -30 "$LOG_FILE"
    fi
fi

echo ""
echo -e "${BLUE}To follow logs in real-time:${NC}"
echo -e "  tail -f $LOG_FILE"
echo ""
echo -e "${BLUE}To check GPU usage:${NC}"
echo -e "  sacct -j $JOB_ID --format=JobID,JobName,ReqGRES,Elapsed,State"
