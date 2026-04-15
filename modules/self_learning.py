# =============================================================================
# Module: self_learning.py
# Purpose: PATENT INNOVATION #7 — Self-Learning Feedback System
#
# Implements a feedback loop where the system learns from each processed image:
#
#   1. After denoising, store the feature vector, optimal weights, and metrics
#   2. Persist to a JSON file (feedback_data/learning_history.json)
#   3. On subsequent runs, the ML model is trained on BOTH synthetic data AND
#      historical feedback data — creating a system that improves over time
#
# This is a core patent claim: the denoising system accumulates experience
# and becomes more accurate with each image it processes.
# =============================================================================

import json
import os
import time
import numpy as np

# Feedback data directory (relative to project root)
FEEDBACK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "feedback_data")
HISTORY_FILE = os.path.join(FEEDBACK_DIR, "learning_history.json")


def _ensure_dir():
    """Create feedback data directory if it doesn't exist."""
    os.makedirs(FEEDBACK_DIR, exist_ok=True)


def save_feedback(
    features: np.ndarray,
    optimal_weights: np.ndarray,
    metrics: dict,
) -> None:
    """
    ┌─────────────────────────────────────────────────────────────────────┐
    │  PATENT INNOVATION: Self-Learning Feedback Storage                 │
    │                                                                     │
    │  Persists the feature vector, predicted/optimal weights, and       │
    │  quality metrics after each image processing session.              │
    │                                                                     │
    │  This creates a growing knowledge base that the ML model can       │
    │  leverage for continuous improvement.                               │
    └─────────────────────────────────────────────────────────────────────┘
    """
    _ensure_dir()

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "features": features.tolist(),
        "optimal_weights": optimal_weights.tolist(),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }

    # Load existing history or start fresh
    history = load_feedback_history()
    history.append(record)

    # Write back (keep last 500 records to prevent unbounded growth)
    history = history[-500:]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def load_feedback_history() -> list[dict]:
    """
    Load all past feedback records from disk.
    Returns an empty list if no history exists yet.
    """
    _ensure_dir()
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def augment_training_data(
    synthetic_X: np.ndarray,
    synthetic_Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ┌─────────────────────────────────────────────────────────────────────┐
    │  PATENT INNOVATION: Self-Learning Data Augmentation                │
    │                                                                     │
    │  Merges historical feedback records with synthetic training data   │
    │  to create a richer, experience-augmented training set.            │
    │                                                                     │
    │  Over time, this allows the model to learn from REAL images        │
    │  in addition to synthetic noise simulations.                       │
    └─────────────────────────────────────────────────────────────────────┘
    """
    history = load_feedback_history()
    if not history:
        return synthetic_X, synthetic_Y

    hist_X, hist_Y = [], []
    for record in history:
        try:
            feat = record["features"]
            weights = record["optimal_weights"]
            if len(feat) == synthetic_X.shape[1] and len(weights) == synthetic_Y.shape[1]:
                hist_X.append(feat)
                hist_Y.append(weights)
        except (KeyError, TypeError):
            continue

    if not hist_X:
        return synthetic_X, synthetic_Y

    hist_X = np.array(hist_X, dtype=np.float64)
    hist_Y = np.array(hist_Y, dtype=np.float64)

    # Concatenate: synthetic + historical
    X_combined = np.vstack([synthetic_X, hist_X])
    Y_combined = np.vstack([synthetic_Y, hist_Y])

    return X_combined, Y_combined


def get_learning_stats() -> dict:
    """
    Return summary statistics about the self-learning system.

    Returns dict:
        total_samples    : int   — total feedback records stored
        avg_psnr         : float — average PSNR across all feedback
        avg_ssim         : float — average SSIM across all feedback
        first_session    : str   — timestamp of first feedback
        last_session     : str   — timestamp of most recent feedback
    """
    history = load_feedback_history()
    if not history:
        return {
            "total_samples": 0,
            "avg_psnr": 0.0,
            "avg_ssim": 0.0,
            "first_session": "N/A",
            "last_session": "N/A",
        }

    psnr_values = []
    ssim_values = []
    for rec in history:
        m = rec.get("metrics", {})
        if "PSNR" in m:
            psnr_values.append(m["PSNR"])
        if "SSIM" in m:
            ssim_values.append(m["SSIM"])

    return {
        "total_samples": len(history),
        "avg_psnr": round(float(np.mean(psnr_values)), 2) if psnr_values else 0.0,
        "avg_ssim": round(float(np.mean(ssim_values)), 4) if ssim_values else 0.0,
        "first_session": history[0].get("timestamp", "N/A"),
        "last_session": history[-1].get("timestamp", "N/A"),
    }


def clear_feedback_history() -> None:
    """Delete all stored feedback data (reset the learning system)."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
