# =============================================================================
# Module: fusion.py
# Purpose: PATENT INNOVATION #5 — Multi-Filter Fusion Engine
#
# Instead of selecting a single filter output, this engine blends
# multiple filter outputs using learned weights:
#
#   final = w1 * gaussian + w2 * median + w3 * bilateral
#
# Weights are predicted by the ML optimizer and normalised to sum to 1.
# An extended version applies different weight vectors per region.
# =============================================================================

import numpy as np


def weighted_filter_fusion(
    gaussian_img: np.ndarray,
    median_img: np.ndarray,
    bilateral_img: np.ndarray,
    weights: list | np.ndarray,
) -> np.ndarray:
    """
    ┌─────────────────────────────────────────────────────────────────────┐
    │  PATENT INNOVATION: Multi-Filter Fusion Engine                     │
    │                                                                     │
    │  Intelligently combines three filter outputs using ML-predicted    │
    │  weights. The weighted fusion produces a result superior to any    │
    │  individual filter by leveraging the strengths of each.            │
    │                                                                     │
    │  final_pixel = w1 * gaussian + w2 * median + w3 * bilateral       │
    └─────────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    gaussian_img, median_img, bilateral_img : np.ndarray
        Filter outputs (same shape, uint8).
    weights : array-like of 3 floats
        Blend weights [w_gaussian, w_median, w_bilateral].
        Will be normalised to sum to 1.0.

    Returns
    -------
    np.ndarray (uint8) — fused result image.
    """
    w = np.array(weights, dtype=np.float64)
    # Normalise weights to sum to 1.0 (safety)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

    fused = (
        w[0] * gaussian_img.astype(np.float64)
        + w[1] * median_img.astype(np.float64)
        + w[2] * bilateral_img.astype(np.float64)
    )
    return np.clip(fused, 0, 255).astype(np.uint8)


def region_aware_fusion(
    gaussian_img: np.ndarray,
    median_img: np.ndarray,
    bilateral_img: np.ndarray,
    global_weights: list | np.ndarray,
    region_masks: dict,
) -> np.ndarray:
    """
    Extended fusion: applies different weight emphasis per region type.

    Region-specific weight modulation:
        Smooth  → boost Gaussian weight   (best noise reducer in flat areas)
        Edge    → boost Bilateral weight   (best edge preserver)
        Texture → boost Median weight      (best impulse noise remover)

    The modulation adds a bias to the global weights for each region,
    then re-normalises per region.

    Parameters
    ----------
    global_weights : array-like of 3 floats
        Base weights from ML optimizer.
    region_masks : dict
        Keys: smooth_mask, edge_mask, texture_mask (boolean arrays).

    Returns
    -------
    np.ndarray (uint8) — region-aware fused image.
    """
    w = np.array(global_weights, dtype=np.float64)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

    # Region-specific weight biases (additive adjustment)
    bias_smooth = np.array([0.15, -0.05, -0.10])   # favour Gaussian
    bias_edge = np.array([-0.10, -0.05, 0.15])     # favour Bilateral
    bias_texture = np.array([-0.05, 0.15, -0.10])   # favour Median

    # Compute per-region weights (clip to [0, 1], re-normalise)
    def _safe_weights(base, bias):
        adj = np.clip(base + bias, 0.01, 1.0)
        return adj / adj.sum()

    w_smooth = _safe_weights(w, bias_smooth)
    w_edge = _safe_weights(w, bias_edge)
    w_texture = _safe_weights(w, bias_texture)

    # Prepare float images
    g = gaussian_img.astype(np.float64)
    m = median_img.astype(np.float64)
    b = bilateral_img.astype(np.float64)

    result = np.zeros_like(g)
    smooth_mask = region_masks["smooth_mask"]
    edge_mask = region_masks["edge_mask"]
    texture_mask = region_masks["texture_mask"]

    # Apply weighted fusion per region
    result[smooth_mask] = (
        w_smooth[0] * g[smooth_mask]
        + w_smooth[1] * m[smooth_mask]
        + w_smooth[2] * b[smooth_mask]
    )
    result[edge_mask] = (
        w_edge[0] * g[edge_mask]
        + w_edge[1] * m[edge_mask]
        + w_edge[2] * b[edge_mask]
    )
    result[texture_mask] = (
        w_texture[0] * g[texture_mask]
        + w_texture[1] * m[texture_mask]
        + w_texture[2] * b[texture_mask]
    )

    return np.clip(result, 0, 255).astype(np.uint8)
