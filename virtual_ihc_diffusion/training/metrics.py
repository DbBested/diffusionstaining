"""
Evaluation Metrics for Virtual IHC Staining
Implements PSNR, SSIM, and other quality metrics
"""

import torch
import torch.nn.functional as F
from typing import Tuple
import numpy as np


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,  # [-1, 1] range
) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio

    Args:
        pred: Predicted images [B, C, H, W]
        target: Target images [B, C, H, W]
        data_range: Range of data (2.0 for [-1, 1])

    Returns:
        PSNR value (higher is better)
    """
    mse = F.mse_loss(pred, target, reduction='mean')
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20 * torch.log10(data_range / torch.sqrt(mse))


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 2.0,
) -> torch.Tensor:
    """
    Structural Similarity Index Measure

    Args:
        pred: Predicted images [B, C, H, W]
        target: Target images [B, C, H, W]
        window_size: Gaussian window size
        data_range: Range of data (2.0 for [-1, 1])

    Returns:
        SSIM value (higher is better, range [0, 1])
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Create Gaussian window
    channel = pred.size(1)
    window = _create_gaussian_window(window_size, channel).to(pred.device)

    # Compute mean
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute variance and covariance
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def _create_gaussian_window(window_size: int, channel: int) -> torch.Tensor:
    """Create Gaussian window for SSIM"""
    def _gaussian(window_size, sigma):
        x = torch.arange(window_size).float() - window_size // 2
        gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        return gauss / gauss.sum()

    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    metrics: list = ["PSNR", "SSIM"],
) -> dict:
    """
    Compute multiple metrics

    Args:
        pred: Predicted images [B, C, H, W] in range [-1, 1]
        target: Target images [B, C, H, W] in range [-1, 1]
        metrics: List of metrics to compute

    Returns:
        Dictionary of metric values
    """
    results = {}

    if "PSNR" in metrics:
        results["PSNR"] = psnr(pred, target).item()

    if "SSIM" in metrics:
        results["SSIM"] = ssim(pred, target).item()

    return results


class MetricsTracker:
    """Track and aggregate metrics over epochs"""

    def __init__(self, metrics: list = ["PSNR", "SSIM"]):
        self.metrics = metrics
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.values = {metric: [] for metric in self.metrics}

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metrics with new batch"""
        batch_metrics = compute_metrics(pred, target, self.metrics)
        for metric, value in batch_metrics.items():
            self.values[metric].append(value)

    def compute(self) -> dict:
        """Compute average metrics"""
        results = {}
        for metric, values in self.values.items():
            if len(values) > 0:
                results[metric] = np.mean(values)
            else:
                results[metric] = 0.0
        return results

    def get_string(self) -> str:
        """Get formatted string of metrics"""
        results = self.compute()
        return ", ".join([f"{k}: {v:.4f}" for k, v in results.items()])


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")

    # Create synthetic data
    pred = torch.randn(4, 3, 256, 256) * 0.5
    target = pred + torch.randn_like(pred) * 0.1  # Add noise

    # Compute metrics
    metrics = compute_metrics(pred, target)
    print(f"PSNR: {metrics['PSNR']:.2f} dB")
    print(f"SSIM: {metrics['SSIM']:.4f}")

    # Test tracker
    tracker = MetricsTracker()
    for _ in range(5):
        tracker.update(pred, target)

    print(f"\nAggregated: {tracker.get_string()}")
    print("Metrics test successful!")
