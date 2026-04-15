# =============================================================================
# Module: region_segmentation.py
# Purpose: PATENT INNOVATION #2 — Region-Aware Image Segmentation
#
# Instead of treating the entire image uniformly, this module segments the
# image into three structural region types:
#
#   • Smooth regions  — flat, homogeneous areas (best for Gaussian filtering)
#   • Edge regions    — strong gradient boundaries (need edge-preserving filtering)
#   • Texture regions — high-frequency detail areas (need gentle filtering)
#
# The segmentation masks drive the adaptive multi-filter pipeline, allowing
# different filters (or filter weights) to be applied per region.
#
# Techniques used:
#   • Canny edge detection
#   • Gradient magnitude (Sobel)
#   • Local variance analysis
#   • Morphological cleanup
# =============================================================================

import numpy as np
import cv2


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if needed."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _local_variance(gray: np.ndarray, window_size: int = 15) -> np.ndarray:
    """Compute local variance map using a box filter approach."""
    gray_f = gray.astype(np.float64)
    kernel = np.ones((window_size, window_size), np.float64) / (window_size ** 2)
    local_mean = cv2.filter2D(gray_f, -1, kernel)
    local_sq_mean = cv2.filter2D(gray_f ** 2, -1, kernel)
    return np.clip(local_sq_mean - local_mean ** 2, 0, None)


def segment_image_regions(image: np.ndarray) -> dict:
    """
    ┌─────────────────────────────────────────────────────────────────────┐
    │  PATENT INNOVATION: Region-Aware Image Segmentation                │
    │                                                                     │
    │  Segments an image into smooth, edge, and texture regions using     │
    │  a combination of edge detection, gradient analysis, and local      │
    │  variance thresholding.                                             │
    │                                                                     │
    │  Algorithm:                                                         │
    │  1. Canny edge detection → dilated to form edge_mask               │
    │  2. Local variance thresholding → high variance & not edge =       │
    │     texture_mask                                                    │
    │  3. Remaining pixels → smooth_mask                                 │
    │  4. Morphological cleanup to reduce noise in masks                 │
    │  5. Guarantee: smooth ∪ edge ∪ texture = full image (no gaps)      │
    └─────────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR, uint8).

    Returns
    -------
    dict with keys:
        smooth_mask  : np.ndarray (bool) — flat homogeneous regions
        edge_mask    : np.ndarray (bool) — strong gradient boundaries
        texture_mask : np.ndarray (bool) — high-frequency texture areas
        region_stats : dict              — percentage coverage of each region
    """
    gray = _to_gray(image)
    h, w = gray.shape

    # ── Step 1: Edge detection ────────────────────────────────────────────
    # Use Canny with automatic thresholds based on median intensity
    median_val = np.median(gray)
    low_thresh = max(0, int(0.5 * median_val))
    high_thresh = min(255, int(1.5 * median_val))
    edges = cv2.Canny(gray, low_thresh, high_thresh)

    # Dilate edges to create a band around boundaries
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_dilated = cv2.dilate(edges, edge_kernel, iterations=1)
    edge_mask = edge_dilated > 0

    # ── Step 2: Texture detection via local variance ─────────────────────
    local_var = _local_variance(gray, window_size=15)

    # Adaptive threshold: texture = high local variance but not edge
    var_threshold = np.percentile(local_var, 70)  # top 30% variance
    high_var_mask = local_var > var_threshold
    texture_mask_raw = high_var_mask & (~edge_mask)

    # Morphological cleanup — remove small spurious regions
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    texture_clean = cv2.morphologyEx(
        texture_mask_raw.astype(np.uint8) * 255, cv2.MORPH_OPEN, morph_kernel
    )
    texture_clean = cv2.morphologyEx(texture_clean, cv2.MORPH_CLOSE, morph_kernel)
    texture_mask = texture_clean > 0

    # ── Step 3: Smooth mask = everything else ─────────────────────────────
    smooth_mask = (~edge_mask) & (~texture_mask)

    # ── Step 4: Ensure mutual exclusivity & completeness ─────────────────
    # Priority order: edge > texture > smooth (edges are most critical)
    texture_mask = texture_mask & (~edge_mask)
    smooth_mask = (~edge_mask) & (~texture_mask)

    # ── Compute region statistics ─────────────────────────────────────────
    total_pixels = h * w
    region_stats = {
        "smooth_pct": round(float(np.sum(smooth_mask)) / total_pixels * 100, 1),
        "edge_pct": round(float(np.sum(edge_mask)) / total_pixels * 100, 1),
        "texture_pct": round(float(np.sum(texture_mask)) / total_pixels * 100, 1),
    }

    return {
        "smooth_mask": smooth_mask,
        "edge_mask": edge_mask,
        "texture_mask": texture_mask,
        "region_stats": region_stats,
    }


def visualise_regions(image: np.ndarray, masks: dict) -> np.ndarray:
    """
    Create a colour-coded overlay visualisation of the segmented regions.

    Colours:
        Smooth  → Blue   (255, 180, 0)   in BGR
        Edge    → Red    (0, 0, 255)      in BGR
        Texture → Green  (0, 220, 100)    in BGR

    Returns a blended image (original + semi-transparent overlay).
    """
    overlay = image.copy()
    alpha = 0.35  # overlay transparency

    # Apply colour for each region
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()

    # Blue for smooth
    overlay[masks["smooth_mask"]] = [255, 180, 0]
    # Red for edges
    overlay[masks["edge_mask"]] = [0, 0, 255]
    # Green for texture
    overlay[masks["texture_mask"]] = [0, 220, 100]

    blended = cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)
    return blended
