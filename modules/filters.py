# =============================================================================
# Module: filters.py
# Purpose: PATENT INNOVATION #3 — Adaptive Multi-Filter Pipeline
#
# Contains:
#   • Classical filter wrappers (Gaussian, Median, Bilateral)
#   • Region-aware adaptive denoising that applies DIFFERENT filters
#     to different structural regions of the image, driven by the
#     detected noise profile.
#
# Strategy:
#   Smooth regions  → Gaussian filter  (strong smoothing for flat areas)
#   Edge regions    → Bilateral filter (edge-preserving denoising)
#   Texture regions → Median filter    (removes impulse noise, preserves texture)
#
# For impulse/mixed noise, a median pre-pass is applied first to remove
# salt & pepper artifacts before region-specific filtering.
# =============================================================================

import numpy as np
import cv2
from .utils import ensure_odd


# ── Classical Filter Wrappers ─────────────────────────────────────────────────

def apply_gaussian(image: np.ndarray, ksize: int = 5, sigma: float = 1.5) -> np.ndarray:
    """
    Apply Gaussian blur filter.
    Best for: additive Gaussian noise in smooth/homogeneous regions.
    """
    ksize = ensure_odd(ksize)
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_median(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Apply Median blur filter.
    Best for: impulse (salt & pepper) noise — replaces each pixel with
    the median of its neighbourhood, effectively removing outlier pixels.
    """
    ksize = ensure_odd(ksize)
    return cv2.medianBlur(image, ksize)


def apply_bilateral(image: np.ndarray, d: int = 9,
                     sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """
    Apply Bilateral filter.
    Best for: edge regions — smooths noise while preserving strong edges
    via joint spatial-intensity weighting.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


# ── Adaptive Region-Aware Denoising ──────────────────────────────────────────

def adaptive_region_denoising(
    image: np.ndarray,
    region_masks: dict,
    noise_profile: dict,
    filter_params: dict,
) -> np.ndarray:
    """
    ┌─────────────────────────────────────────────────────────────────────┐
    │  PATENT INNOVATION: Adaptive Multi-Filter Pipeline                 │
    │                                                                     │
    │  Applies DIFFERENT filters to different regions of the image       │
    │  based on the structural segmentation and noise profile.           │
    │                                                                     │
    │  Pipeline:                                                          │
    │  1. If impulse/mixed noise detected → median pre-pass              │
    │  2. Smooth regions → Gaussian filter (strong noise reduction)      │
    │  3. Edge regions → Bilateral filter (preserve edges)               │
    │  4. Texture regions → light Median filter (preserve detail)        │
    │  5. Composite using region masks                                   │
    │                                                                     │
    │  Filter parameters adapt based on noise intensity:                 │
    │  - Higher noise → larger kernels, stronger filtering               │
    │  - Lower noise → smaller kernels, gentler filtering                │
    └─────────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    image : np.ndarray
        Noisy input image (BGR, uint8).
    region_masks : dict
        Output of segment_image_regions() with smooth/edge/texture masks.
    noise_profile : dict
        Output of detect_noise_profile() with noise type/intensity.
    filter_params : dict
        User-specified filter parameters from the sidebar:
        {g_ksize, g_sigma, m_ksize, b_d, b_sc, b_ss}

    Returns
    -------
    np.ndarray
        Composited denoised image with per-region filtering.
    """
    noise_type = noise_profile["noise_type"]
    intensity = noise_profile["noise_intensity"]

    # Adapt filter strength based on noise intensity
    g_ksize = filter_params.get("g_ksize", 5)
    g_sigma = filter_params.get("g_sigma", 1.5)
    m_ksize = filter_params.get("m_ksize", 5)
    b_d = filter_params.get("b_d", 9)
    b_sc = filter_params.get("b_sc", 75)
    b_ss = filter_params.get("b_ss", 75)

    # Scale parameters with noise intensity (stronger filtering for heavier noise)
    intensity_scale = 1.0 + intensity * 0.5
    g_ksize_adapted = ensure_odd(max(3, int(g_ksize * intensity_scale)))
    m_ksize_adapted = ensure_odd(max(3, int(m_ksize * intensity_scale)))
    g_sigma_adapted = g_sigma * intensity_scale
    b_sc_adapted = b_sc * intensity_scale
    b_ss_adapted = b_ss * intensity_scale

    working = image.copy()

    # ── Step 1: Median pre-pass for impulse noise ─────────────────────────
    if noise_type in ("impulse", "mixed"):
        working = apply_median(working, m_ksize_adapted)

    # ── Step 2: Apply region-specific filters ─────────────────────────────
    smooth_filtered = apply_gaussian(working, g_ksize_adapted, g_sigma_adapted)
    edge_filtered = apply_bilateral(working, b_d, b_sc_adapted, b_ss_adapted)
    texture_filtered = apply_median(working, ensure_odd(max(3, m_ksize - 2)))  # lighter

    # ── Step 3: Composite using masks ─────────────────────────────────────
    result = np.zeros_like(image)
    smooth_mask = region_masks["smooth_mask"]
    edge_mask = region_masks["edge_mask"]
    texture_mask = region_masks["texture_mask"]

    # Apply each filtered version where its region mask is True
    for c in range(image.shape[2] if len(image.shape) == 3 else 1):
        if len(image.shape) == 3:
            channel_s = smooth_filtered[:, :, c]
            channel_e = edge_filtered[:, :, c]
            channel_t = texture_filtered[:, :, c]
            result[:, :, c][smooth_mask] = channel_s[smooth_mask]
            result[:, :, c][edge_mask] = channel_e[edge_mask]
            result[:, :, c][texture_mask] = channel_t[texture_mask]
        else:
            result[smooth_mask] = smooth_filtered[smooth_mask]
            result[edge_mask] = edge_filtered[edge_mask]
            result[texture_mask] = texture_filtered[texture_mask]

    return result
