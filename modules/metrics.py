# =============================================================================
# Module: metrics.py
# Purpose: Image quality metrics + PATENT INNOVATION #6 — Edge Preservation Score
#
# Standard metrics:
#   • MSE  — Mean Squared Error (lower is better)
#   • PSNR — Peak Signal-to-Noise Ratio in dB (higher is better)
#   • SSIM — Structural Similarity Index (higher is better, max 1.0)
#
# Novel metric:
#   • EPS  — Edge Preservation Score (higher is better, max 1.0)
#     Measures how faithfully edges are maintained after denoising
#     by comparing Canny edge maps of original and denoised images.
# =============================================================================

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_metric


def compute_mse(original: np.ndarray, processed: np.ndarray) -> float:
    """Mean Squared Error between two images."""
    return float(np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2))


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB). Returns 100.0 for identical images."""
    mse = compute_mse(original, processed)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(255.0 ** 2 / mse))


def compute_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """Structural Similarity Index (range [−1, 1], higher is better)."""
    if len(original.shape) == 3:
        return float(ssim_metric(original, processed, channel_axis=2, data_range=255))
    return float(ssim_metric(original, processed, data_range=255))


def edge_preservation_score(original: np.ndarray, denoised: np.ndarray) -> float:
    """
    ┌─────────────────────────────────────────────────────────────────────┐
    │  PATENT INNOVATION: Edge Preservation Score (EPS)                  │
    │                                                                     │
    │  Measures how well edges from the original image are preserved     │
    │  after denoising. Uses Canny edge detection on both images and     │
    │  computes the overlap ratio:                                       │
    │                                                                     │
    │    EPS = |intersection(edges_orig, edges_denoised)|                │
    │          ÷ |edges_orig|                                            │
    │                                                                     │
    │  - EPS = 1.0 → all original edges are perfectly preserved          │
    │  - EPS = 0.0 → no original edges remain after denoising            │
    │                                                                     │
    │  This metric fills a gap in traditional quality measures (MSE,     │
    │  PSNR, SSIM) which do not specifically assess structural edge      │
    │  fidelity — a critical quality factor in medical imaging,          │
    │  surveillance, and industrial inspection.                          │
    └─────────────────────────────────────────────────────────────────────┘
    """
    # Convert to grayscale
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = original

    if len(denoised.shape) == 3:
        den_gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    else:
        den_gray = denoised

    # Compute Canny edges for both
    edges_orig = cv2.Canny(orig_gray, 50, 150)
    edges_den = cv2.Canny(den_gray, 50, 150)

    # Count edge pixels in original
    orig_edge_count = np.sum(edges_orig > 0)

    if orig_edge_count == 0:
        return 1.0  # No edges to preserve → trivially perfect

    # Count intersection (overlap) — original edges that also appear in denoised
    intersection = np.sum((edges_orig > 0) & (edges_den > 0))

    eps = float(intersection / orig_edge_count)
    return round(eps, 4)


def compute_all_metrics(original: np.ndarray, processed: np.ndarray) -> dict:
    """
    Compute all four quality metrics between original and processed images.

    Returns dict:
        {MSE: float, PSNR: float, SSIM: float, EPS: float}
    """
    return {
        "MSE": round(compute_mse(original, processed), 4),
        "PSNR": round(compute_psnr(original, processed), 4),
        "SSIM": round(compute_ssim(original, processed), 4),
        "EPS": round(edge_preservation_score(original, processed), 4),
    }


def compute_basic_metrics(original: np.ndarray, processed: np.ndarray) -> dict:
    """Compute MSE, PSNR, SSIM only (without EPS, for speed in training loops)."""
    return {
        "MSE": round(compute_mse(original, processed), 4),
        "PSNR": round(compute_psnr(original, processed), 4),
        "SSIM": round(compute_ssim(original, processed), 4),
    }
