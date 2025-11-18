#!/usr/bin/env python
"""
Analyze training results and generate summary report
Usage: python scripts/analyze_results.py --job_id 6470504
"""

import argparse
import re
from pathlib import Path
import sys

def parse_log_file(log_file):
    """Extract metrics from training log"""
    if not log_file.exists():
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    results = {
        'job_id': log_file.stem.replace('train_', ''),
        'epochs_completed': 0,
        'best_psnr': None,
        'best_ssim': None,
        'final_train_loss': None,
        'final_val_loss': None,
        'training_time': None,
        'status': 'unknown'
    }

    # Extract epochs
    epoch_matches = re.findall(r'Epoch (\d+)/(\d+)', content)
    if epoch_matches:
        results['epochs_completed'] = int(epoch_matches[-1][0])
        results['total_epochs'] = int(epoch_matches[-1][1])

    # Extract PSNR
    psnr_matches = re.findall(r'PSNR[:\s]+([0-9.]+)', content)
    if psnr_matches:
        psnr_values = [float(x) for x in psnr_matches]
        results['best_psnr'] = max(psnr_values)
        results['final_psnr'] = psnr_values[-1]

    # Extract SSIM
    ssim_matches = re.findall(r'SSIM[:\s]+([0-9.]+)', content)
    if ssim_matches:
        ssim_values = [float(x) for x in ssim_matches]
        results['best_ssim'] = max(ssim_values)
        results['final_ssim'] = ssim_values[-1]

    # Extract losses
    val_loss_matches = re.findall(r'Loss[:\s]+([0-9.]+)', content)
    if val_loss_matches:
        results['final_val_loss'] = float(val_loss_matches[-1])

    # Check completion status
    if 'Job completed!' in content:
        results['status'] = 'completed'
    elif 'Error' in content or 'Traceback' in content:
        results['status'] = 'failed'
    else:
        results['status'] = 'running'

    # Extract training time
    time_matches = re.findall(r'Elapsed[:\s]+([0-9:]+)', content)
    if time_matches:
        results['training_time'] = time_matches[-1]

    return results


def print_summary(results):
    """Print formatted summary"""
    print("=" * 60)
    print(f"TRAINING RESULTS SUMMARY - Job {results['job_id']}")
    print("=" * 60)
    print()

    print(f"Status: {results['status'].upper()}")
    print(f"Epochs: {results['epochs_completed']}/{results.get('total_epochs', 'N/A')}")
    print()

    if results['best_psnr']:
        print(f"📊 Best PSNR:  {results['best_psnr']:.4f} dB")
        print(f"   Final PSNR: {results.get('final_psnr', 'N/A'):.4f} dB" if results.get('final_psnr') else "")
    else:
        print("📊 PSNR: Not available yet")

    if results['best_ssim']:
        print(f"📊 Best SSIM:  {results['best_ssim']:.4f}")
        print(f"   Final SSIM: {results.get('final_ssim', 'N/A'):.4f}" if results.get('final_ssim') else "")
    else:
        print("📊 SSIM: Not available yet")

    print()

    if results['final_val_loss']:
        print(f"📉 Final Val Loss: {results['final_val_loss']:.6f}")

    if results['training_time']:
        print(f"⏱️  Training Time: {results['training_time']}")

    print()
    print("=" * 60)

    # Provide recommendations
    if results['best_psnr']:
        print("\n🎯 Performance Assessment:")
        if results['best_psnr'] > 28:
            print("  ✅ Excellent PSNR (>28 dB) - Target achieved!")
        elif results['best_psnr'] > 25:
            print("  ⚠️  Good PSNR (25-28 dB) - Room for improvement")
        else:
            print("  ❌ Low PSNR (<25 dB) - Needs significant improvement")

        if results['best_ssim']:
            if results['best_ssim'] > 0.85:
                print("  ✅ Excellent SSIM (>0.85) - Target achieved!")
            elif results['best_ssim'] > 0.80:
                print("  ⚠️  Good SSIM (0.80-0.85) - Room for improvement")
            else:
                print("  ❌ Low SSIM (<0.80) - Needs significant improvement")


def compare_versions(log_dir):
    """Compare multiple training versions"""
    log_files = sorted(Path(log_dir).glob("train_*.out"))

    if len(log_files) < 2:
        print("Not enough completed runs to compare")
        return

    print("\n" + "=" * 80)
    print("VERSION COMPARISON")
    print("=" * 80)
    print(f"{'Job ID':<12} {'Status':<12} {'Epochs':<10} {'Best PSNR':<12} {'Best SSIM':<12}")
    print("-" * 80)

    all_results = []
    for log_file in log_files:
        results = parse_log_file(log_file)
        if results:
            all_results.append(results)
            psnr_str = f"{results['best_psnr']:.4f}" if results['best_psnr'] else "N/A"
            ssim_str = f"{results['best_ssim']:.4f}" if results['best_ssim'] else "N/A"
            epochs_str = f"{results['epochs_completed']}/{results.get('total_epochs', '?')}"
            print(f"{results['job_id']:<12} {results['status']:<12} {epochs_str:<10} {psnr_str:<12} {ssim_str:<12}")

    print("-" * 80)

    # Find best performing version
    completed_results = [r for r in all_results if r['status'] == 'completed' and r['best_psnr']]
    if completed_results:
        best_result = max(completed_results, key=lambda x: x['best_psnr'])
        print(f"\n🏆 Best performing version: Job {best_result['job_id']}")
        print(f"   PSNR: {best_result['best_psnr']:.4f} dB, SSIM: {best_result['best_ssim']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('--job_id', type=str, help='Specific job ID to analyze')
    parser.add_argument('--compare', action='store_true', help='Compare all versions')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory containing logs')

    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    if args.compare:
        compare_versions(log_dir)
    elif args.job_id:
        log_file = log_dir / f"train_{args.job_id}.out"
        results = parse_log_file(log_file)
        if results:
            print_summary(results)
        else:
            print(f"❌ Log file not found: {log_file}")
    else:
        # Analyze most recent job
        log_files = sorted(log_dir.glob("train_*.out"))
        if log_files:
            latest_log = log_files[-1]
            print(f"Analyzing latest job: {latest_log.stem}")
            results = parse_log_file(latest_log)
            if results:
                print_summary(results)
        else:
            print("❌ No log files found in", log_dir)
            print("Usage: python scripts/analyze_results.py --job_id <JOB_ID>")


if __name__ == "__main__":
    main()
