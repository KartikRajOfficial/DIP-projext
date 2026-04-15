# =============================================================================
# Module: utils.py
# Purpose: Shared utility functions — image loading, noise injection, helpers.
# =============================================================================

import numpy as np
import cv2


def ensure_odd(value: int) -> int:
    """Return value if already odd, else return value + 1 (kernels must be odd)."""
    return value if value % 2 == 1 else value + 1


def load_demo_image() -> np.ndarray:
    """
    Generate a synthetic demo image with gradients and geometric shapes.
    Returns a 256×256×3 BGR image (uint8).
    """
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Smooth gradient background
    for i in range(256):
        img[i, :] = [int(50 + i * 0.6), int(80 + i * 0.4), int(120 + i * 0.3)]
    # Circle (smooth region)
    cv2.circle(img, (128, 100), 60, (220, 180, 80), -1)
    # Rectangle (edge-rich region)
    cv2.rectangle(img, (50, 160), (200, 230), (80, 160, 220), -1)
    # Text overlay (fine texture)
    cv2.putText(img, "DIP", (90, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    return img


# ── Noise Injection Functions ─────────────────────────────────────────────────

def add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Add zero-mean Gaussian noise with standard deviation `sigma`.
    Models sensor noise, thermal noise, and other additive noise sources.
    """
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.05) -> np.ndarray:
    """
    Replace a fraction (`amount`) of pixels with salt (255) or pepper (0).
    Models impulse noise from transmission errors or dead pixels.
    """
    noisy = image.copy()
    total = image.size
    # Salt pixels (set to 255)
    salt_coords = [np.random.randint(0, d, int(total * amount / 2)) for d in image.shape]
    noisy[salt_coords[0], salt_coords[1]] = 255
    # Pepper pixels (set to 0)
    pepper_coords = [np.random.randint(0, d, int(total * amount / 2)) for d in image.shape]
    noisy[pepper_coords[0], pepper_coords[1]] = 0
    return noisy


def add_mixed_noise(image: np.ndarray, sigma: float = 15.0, amount: float = 0.03) -> np.ndarray:
    """
    Apply both Gaussian and Salt & Pepper noise simultaneously.
    Models real-world scenarios where multiple noise sources co-exist
    (e.g., sensor noise + transmission errors).

    This is a NOVEL combination that the adaptive system must handle
    by detecting each noise component and applying targeted filtering.
    """
    # First: Gaussian noise layer
    noisy = add_gaussian_noise(image, sigma)
    # Second: Impulse noise layer on top
    noisy = add_salt_pepper_noise(noisy, amount)
    return noisy
