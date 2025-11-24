#!/bin/bash
# Continuous live monitoring of training job
# Usage: ./monitor_live.sh <job_id>

if [ -z "$1" ]; then
    echo "Usage: $0 <job_id>"
    exit 1
fi

JOB_ID=$1
LOG_FILE="logs/train_${JOB_ID}.out"
ERR_FILE="logs/train_${JOB_ID}.err"

echo "=================================="
echo "Live Training Monitor - Job $JOB_ID"
echo "Press Ctrl+C to exit"
echo "=================================="
echo ""

# Check if job exists
squeue -j $JOB_ID &>/dev/null
if [ $? -ne 0 ]; then
    sacct -j $JOB_ID --format=JobID,State,Elapsed 2>/dev/null | head -2
    echo ""
fi

# Follow the log file if it exists
if [ -f "$LOG_FILE" ]; then
    echo "Following log: $LOG_FILE"
    echo "=================================="
    tail -f "$LOG_FILE"
else
    echo "Waiting for log file to appear: $LOG_FILE"
    echo "Checking job status every 5 seconds..."
    echo ""
    while true; do
        STATUS=$(squeue -j $JOB_ID -h -o "%T" 2>/dev/null || echo "COMPLETED/FAILED")
        echo "[$(date +%H:%M:%S)] Job Status: $STATUS"
        
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "Log file appeared! Following..."
            echo "=================================="
            tail -f "$LOG_FILE"
            break
        fi
        sleep 5
    done
fi
