# =============================================================================
# Module: noise_detection.py
# Purpose: PATENT INNOVATION #1 — Automatic Noise Profile Detection
#
# Before applying any filter, this module characterises the noise present in
# an image by analysing statistical features. The detected profile directly
# drives downstream filtering strategy (regions, weights, fusion).
#
# Returns:
#   noise_type         — "gaussian" | "impulse" | "mixed"
#   noise_intensity    — float in [0, 1]  (0 = clean, 1 = heavy noise)
#   distribution_pattern — "uniform" | "clustered" | "sparse"
#
# Statistical features used:
#   • Global variance & standard deviation
#   • Histogram kurtosis & skewness (via scipy.stats)
#   • Salt/pepper pixel ratio (extreme-value count)
#   • Laplacian variance (focus/noise energy)
#   • Edge density (Canny)
#   • Local variance spatial analysis
# =============================================================================

import numpy as np
import cv2
from scipy import stats


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB image to grayscale; pass through if already single-channel."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _salt_pepper_ratio(gray: np.ndarray) -> float:
    """
    Fraction of pixels at extreme values (0 or 255).
    High ratio indicates impulse (salt & pepper) noise.
    """
    total = gray.size
    salt = np.sum(gray == 255)
    pepper = np.sum(gray == 0)
    return float((salt + pepper) / total)


def _laplacian_variance(gray: np.ndarray) -> float:
    """
    Variance of the Laplacian — measures high-frequency energy.
    Higher values indicate more noise or more texture/edges.
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap))


def _edge_density(gray: np.ndarray) -> float:
    """
    Fraction of pixels detected as edges by Canny.
    Very high edge density in a noisy image suggests noise rather than real structure.
    """
    edges = cv2.Canny(gray, 50, 150)
    return float(np.sum(edges > 0) / edges.size)


def _local_variance_map(gray: np.ndarray, window_size: int = 15) -> np.ndarray:
    """
    Compute local variance using a sliding window.
    Returns a variance map the same size as the input.
    """
    gray_f = gray.astype(np.float64)
    kernel = np.ones((window_size, window_size), dtype=np.float64) / (window_size ** 2)
    local_mean = cv2.filter2D(gray_f, -1, kernel)
    local_sq_mean = cv2.filter2D(gray_f ** 2, -1, kernel)
    local_var = local_sq_mean - local_mean ** 2
    return np.clip(local_var, 0, None)


def _distribution_pattern(local_var: np.ndarray) -> str:
    """
    Analyse the spatial distribution of noise by examining the local variance map.
    - "uniform"   : variance is evenly spread across the image
    - "clustered" : high-variance regions are concentrated in blobs
    - "sparse"    : noise is scattered in isolated pixels
    """
    # Threshold at the 75th percentile to find "high variance" regions
    thresh = np.percentile(local_var, 75)
    high_var_mask = (local_var > thresh).astype(np.uint8) * 255

    # Count connected components (blobs)
    num_labels, labels, region_stats, _ = cv2.connectedComponentsWithStats(
        high_var_mask, connectivity=8
    )
    num_regions = num_labels - 1  # exclude background

    if num_regions == 0:
        return "uniform"

    # Average region area
    if num_regions > 0:
        areas = region_stats[1:, cv2.CC_STAT_AREA]
        avg_area = np.mean(areas)
    else:
        avg_area = 0

    total_pixels = local_var.size
    coverage = np.sum(high_var_mask > 0) / total_pixels

    # Heuristic classification
    if coverage > 0.6:
        return "uniform"
    elif avg_area > total_pixels * 0.01:
        return "clustered"
    else:
        return "sparse"


def compute_noise_features(image: np.ndarray) -> dict:
    """
    Extract a comprehensive set of statistical features from the image
    that characterise the noise properties.

    Returns a dictionary of named features used for both noise detection
    and as input to the ML weight optimizer.
    """
    gray = _to_gray(image)
    gray_f = gray.astype(np.float64)

    # Global statistics
    global_var = float(np.var(gray_f))
    global_std = float(np.std(gray_f))
    img_mean = float(np.mean(gray_f))

    # Histogram shape statistics (kurtosis & skewness)
    hist_kurtosis = float(stats.kurtosis(gray_f.ravel(), fisher=True))
    hist_skewness = float(stats.skew(gray_f.ravel()))

    # Impulse noise indicator
    sp_ratio = _salt_pepper_ratio(gray)

    # High-frequency energy
    lap_var = _laplacian_variance(gray)

    # Edge density
    edge_dens = _edge_density(gray)

    # Local variance statistics
    local_var = _local_variance_map(gray)
    mean_local_var = float(np.mean(local_var))

    # Distribution pattern
    dist_pattern = _distribution_pattern(local_var)

    return {
        "global_variance": global_var,
        "global_std": global_std,
        "image_mean": img_mean,
        "histogram_kurtosis": hist_kurtosis,
        "histogram_skewness": hist_skewness,
        "salt_pepper_ratio": sp_ratio,
        "laplacian_variance": lap_var,
        "edge_density": edge_dens,
        "mean_local_variance": mean_local_var,
        "distribution_pattern": dist_pattern,
    }


def detect_noise_profile(image: np.ndarray) -> dict:
    """
    ┌─────────────────────────────────────────────────────────────────────┐
    │  PATENT INNOVATION: Automatic Noise Profile Detection              │
    │                                                                     │
    │  Analyses an image and returns a structured noise profile that      │
    │  drives all downstream processing decisions.                        │
    │                                                                     │
    │  Decision logic:                                                    │
    │  1. Salt/pepper ratio > 0.02  →  impulse component detected        │
    │  2. Laplacian variance high + mesokurtic → Gaussian component      │
    │  3. Both present → "mixed"                                         │
    │  4. Intensity = normalised combination of std & Laplacian variance │
    └─────────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR, uint8).

    Returns
    -------
    dict with keys:
        noise_type          : str   — "gaussian", "impulse", or "mixed"
        noise_intensity     : float — 0.0 (clean) to 1.0 (heavy noise)
        distribution_pattern: str   — "uniform", "clustered", or "sparse"
        features            : dict  — raw statistical features (for ML pipeline)
    """
    features = compute_noise_features(image)

    # ── Noise type classification ─────────────────────────────────────────
    sp_ratio = features["salt_pepper_ratio"]
    lap_var = features["laplacian_variance"]
    kurtosis = features["histogram_kurtosis"]

    has_impulse = sp_ratio > 0.02     # significant extreme-value pixels
    has_gaussian = lap_var > 200.0    # significant high-frequency energy

    if has_impulse and has_gaussian:
        noise_type = "mixed"
    elif has_impulse:
        noise_type = "impulse"
    else:
        noise_type = "gaussian"

    # ── Noise intensity estimation (normalised to [0, 1]) ────────────────
    # Combine standard deviation and Laplacian variance as energy proxies
    std_norm = min(features["global_std"] / 80.0, 1.0)     # σ=80 → max
    lap_norm = min(features["laplacian_variance"] / 5000.0, 1.0)  # 5000 → max
    sp_norm = min(sp_ratio / 0.20, 1.0)                      # 20% → max

    if noise_type == "impulse":
        intensity = 0.3 * std_norm + 0.2 * lap_norm + 0.5 * sp_norm
    elif noise_type == "mixed":
        intensity = 0.3 * std_norm + 0.3 * lap_norm + 0.4 * sp_norm
    else:
        intensity = 0.5 * std_norm + 0.5 * lap_norm

    intensity = round(float(np.clip(intensity, 0.0, 1.0)), 4)

    return {
        "noise_type": noise_type,
        "noise_intensity": intensity,
        "distribution_pattern": features["distribution_pattern"],
        "features": features,
    }
