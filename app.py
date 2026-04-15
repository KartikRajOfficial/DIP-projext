# =============================================================================
# Adaptive Image Denoising Framework — Patent-Worthy Streamlit Dashboard
# =============================================================================
#
# SYSTEM PIPELINE:
#   Input Image → Noise Profile Detection → Region Segmentation →
#   Multi-Filter Processing → ML Weight Optimization → Adaptive Filter Fusion →
#   Edge Preservation Evaluation → Self-Learning Model Update → Final Output
#
# PATENT INNOVATIONS:
#   1. Automatic noise profile detection (type, intensity, distribution)
#   2. Region-aware adaptive filtering (smooth/edge/texture segmentation)
#   3. ML-based filter weight optimization (multi-output regression)
#   4. Multi-filter fusion engine (weighted blending)
#   5. Edge Preservation Score (EPS) metric
#   6. Self-learning feedback system (continuous improvement)
# =============================================================================

import io
import warnings
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# ── Import patent-worthy modules ─────────────────────────────────────────────
from modules.utils import (
    load_demo_image, add_gaussian_noise, add_salt_pepper_noise, add_mixed_noise,
)
from modules.noise_detection import detect_noise_profile
from modules.region_segmentation import segment_image_regions, visualise_regions
from modules.filters import (
    apply_gaussian, apply_median, apply_bilateral, adaptive_region_denoising,
)
from modules.fusion import weighted_filter_fusion, region_aware_fusion
from modules.metrics import compute_all_metrics, compute_basic_metrics
from modules.ml_optimizer import (
    extract_ml_features, build_weight_training_data,
    train_weight_predictor, predict_filter_weights,
    build_classifier_training_data, train_classifier, predict_best_filter,
)
from modules.self_learning import (
    save_feedback, augment_training_data, get_learning_stats, clear_feedback_history,
)

warnings.filterwarnings("ignore")


# ── Premium Dark UI CSS ──────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --accent: #4A9EFF;
    --accent-glow: rgba(74, 158, 255, 0.15);
    --bg: #0A0E17;
    --bg-secondary: #0E1220;
    --card: #141929;
    --card-hover: #1A2035;
    --border: #1E2642;
    --text-primary: #E8ECF4;
    --text-secondary: #8892A4;
    --success: #34D399;
    --warning: #FBBF24;
    --danger: #F87171;
    --purple: #A78BFA;
}

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, var(--bg) 0%, var(--bg-secondary) 100%);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0C1018 0%, #111827 100%);
    border-right: 1px solid var(--border);
}

h1 {
    background: linear-gradient(135deg, #4A9EFF 0%, #A78BFA 50%, #34D399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

h2, h3 {
    color: var(--accent) !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, var(--card) 0%, #1A1F35 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 6px 0;
    text-align: center;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}
.metric-card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 20px var(--accent-glow);
    transform: translateY(-2px);
}
.metric-card .label {
    font-size: 0.72rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 500;
}
.metric-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent);
    margin-top: 4px;
}

/* Section headers */
.section-header {
    background: linear-gradient(135deg, rgba(74, 158, 255, 0.08) 0%, rgba(167, 139, 250, 0.05) 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 24px;
    margin: 24px 0 16px 0;
}
.section-header .title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--accent);
}
.section-header .subtitle {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 4px;
}

/* Noise profile badge */
.noise-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.noise-gaussian { background: rgba(74, 158, 255, 0.15); color: #4A9EFF; border: 1px solid rgba(74, 158, 255, 0.3); }
.noise-impulse  { background: rgba(248, 113, 113, 0.15); color: #F87171; border: 1px solid rgba(248, 113, 113, 0.3); }
.noise-mixed    { background: rgba(251, 191, 36, 0.15);  color: #FBBF24; border: 1px solid rgba(251, 191, 36, 0.3);  }

/* Weight bar */
.weight-bar-container {
    background: var(--card);
    border-radius: 8px;
    overflow: hidden;
    height: 28px;
    display: flex;
    margin: 8px 0;
    border: 1px solid var(--border);
}
.weight-bar-segment {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 600;
    color: #fff;
    min-width: 30px;
}

/* Best badge */
.best-badge {
    background: linear-gradient(135deg, #0D2145 0%, #1A3A6B 100%);
    border: 2px solid var(--accent);
    border-radius: 14px;
    padding: 10px 18px;
    display: inline-block;
    font-size: 0.9rem;
    color: var(--accent);
    font-weight: 600;
    box-shadow: 0 0 30px var(--accent-glow);
}

/* Learning indicator */
.learning-indicator {
    background: linear-gradient(135deg, rgba(52, 211, 153, 0.08) 0%, rgba(52, 211, 153, 0.03) 100%);
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 12px;
    padding: 14px 20px;
    margin: 8px 0;
}
.learning-indicator .stat {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--success);
}
.learning-indicator .desc {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Intensity gauge */
.gauge-bg {
    background: var(--card);
    border-radius: 8px;
    height: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
    margin: 6px 0;
}
.gauge-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}

[data-testid="stDataFrame"] { border-radius: 8px; }
</style>
"""


# ── UI Helper Functions ──────────────────────────────────────────────────────

def metric_card(label: str, value: str) -> str:
    """Generate an HTML metric card."""
    return f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>'


def section_header(title: str, subtitle: str) -> str:
    """Generate a styled section header."""
    return f'<div class="section-header"><div class="title">{title}</div><div class="subtitle">{subtitle}</div></div>'


def noise_badge(noise_type: str) -> str:
    """Generate a coloured noise type badge."""
    css_class = f"noise-{noise_type}"
    label = noise_type.upper()
    return f'<span class="noise-badge {css_class}">{label}</span>'


def intensity_gauge(value: float) -> str:
    """Generate a horizontal intensity gauge bar."""
    pct = int(value * 100)
    if value < 0.3:
        color = "#34D399"
    elif value < 0.6:
        color = "#FBBF24"
    else:
        color = "#F87171"
    return (
        f'<div class="gauge-bg">'
        f'<div class="gauge-fill" style="width: {pct}%; background: linear-gradient(90deg, {color}88, {color});"></div>'
        f'</div>'
        f'<div style="font-size:0.75rem; color:#8892A4; text-align:right;">{pct}%</div>'
    )


def weight_bar(weights: list, labels: list) -> str:
    """Generate a stacked weight bar visualisation."""
    colors = ["#4A9EFF", "#FBBF24", "#A78BFA"]
    segments = ""
    for w, lbl, col in zip(weights, labels, colors):
        pct = max(w * 100, 5)  # min width for visibility
        segments += f'<div class="weight-bar-segment" style="width: {pct}%; background: {col};">{lbl} {w:.0%}</div>'
    return f'<div class="weight-bar-container">{segments}</div>'


def dark_bar_chart(labels, values, title, color) -> plt.Figure:
    """Create a dark-themed bar chart for metric comparison."""
    fig, ax = plt.subplots(figsize=(4, 3), facecolor="#0A0E17")
    ax.set_facecolor("#141929")
    bars = ax.bar(labels, values, color=color, edgecolor="#1E2642", width=0.5, zorder=3)
    ax.set_title(title, color="#4A9EFF", fontsize=11, pad=10, fontweight=600)
    ax.tick_params(colors="#8892A4", labelsize=8)
    ax.grid(axis="y", color="#1E2642", linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E2642")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(max(values), 1),
                f"{val:.2f}", ha="center", va="bottom", color="#E8ECF4", fontsize=8, fontweight=500)
    fig.tight_layout()
    return fig


# ── Main Application ─────────────────────────────────────────────────────────

def main() -> None:
    """Entry point — renders the full patent-worthy Streamlit dashboard."""
    st.set_page_config(
        page_title="Adaptive Denoising Framework | Patent-Worthy ML Pipeline",
        page_icon="🔬",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Title ─────────────────────────────────────────────────────────────
    st.markdown("# 🔬 Adaptive Image Denoising Framework")
    st.caption(
        "Patent-worthy intelligent denoising with noise profiling · region-aware filtering · "
        "ML weight optimization · multi-filter fusion · self-learning feedback"
    )

    # ══════════════════════════════════════════════════════════════════════
    # SIDEBAR CONTROLS
    # ══════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("### ⚙️ Controls")

        # Image source
        use_demo = st.checkbox("Use demo image", value=True)
        uploaded = st.file_uploader("Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"])

        st.markdown("---")
        st.markdown("### 🔊 Noise Settings")
        noise_type = st.selectbox("Noise type", ["Gaussian", "Salt & Pepper", "Mixed"])
        noise_sigma = st.slider("Gaussian σ (intensity)", 5, 80, 25, step=5)
        sp_amount = st.slider("Salt & Pepper amount", 0.01, 0.20, 0.05, step=0.01)

        st.markdown("---")
        st.markdown("### 🔧 Filter Parameters")

        with st.expander("Gaussian Filter", expanded=False):
            g_ksize = st.slider("Kernel size", 3, 21, 5, step=2, key="gk")
            g_sigma = st.slider("Sigma (σ)", 0.5, 5.0, 1.5, step=0.5, key="gs")

        with st.expander("Median Filter", expanded=False):
            m_ksize = st.slider("Kernel size", 3, 21, 5, step=2, key="mk")

        with st.expander("Bilateral Filter", expanded=False):
            b_d = st.slider("Diameter (d)", 5, 25, 9, step=2)
            b_sc = st.slider("Sigma colour", 10, 150, 75, step=5)
            b_ss = st.slider("Sigma space", 10, 150, 75, step=5)

        st.markdown("---")
        st.markdown("### 🤖 ML Settings")
        model_type = st.selectbox("Legacy classifier", ["KNN", "SVM"])
        enable_learning = st.checkbox("Enable self-learning", value=True)

        st.markdown("---")
        run_btn = st.button("▶  Run Full Pipeline", use_container_width=True)

        # Self-learning controls
        if enable_learning:
            stats = get_learning_stats()
            st.markdown("---")
            st.markdown("### 🧠 Learning Status")
            st.caption(f"📊 {stats['total_samples']} samples learned")
            if stats["total_samples"] > 0:
                st.caption(f"📈 Avg PSNR: {stats['avg_psnr']:.1f} dB")
                st.caption(f"📐 Avg SSIM: {stats['avg_ssim']:.4f}")
            if st.button("🗑️ Reset Learning Data", use_container_width=True):
                clear_feedback_history()
                st.success("Learning history cleared!")
                st.rerun()

    if not run_btn:
        return

    # ══════════════════════════════════════════════════════════════════════
    # LOAD IMAGE
    # ══════════════════════════════════════════════════════════════════════
    if uploaded is not None and not use_demo:
        pil_img = Image.open(uploaded).convert("RGB").resize((256, 256))
        original = np.array(pil_img)
    else:
        original = load_demo_image()

    filter_params = {
        "g_ksize": g_ksize, "g_sigma": g_sigma,
        "m_ksize": m_ksize,
        "b_d": b_d, "b_sc": b_sc, "b_ss": b_ss,
    }

    # ══════════════════════════════════════════════════════════════════════
    # INJECT NOISE
    # ══════════════════════════════════════════════════════════════════════
    if noise_type == "Gaussian":
        noisy = add_gaussian_noise(original, noise_sigma)
    elif noise_type == "Salt & Pepper":
        noisy = add_salt_pepper_noise(original, sp_amount)
    else:  # Mixed
        noisy = add_mixed_noise(original, noise_sigma, sp_amount)

    # ══════════════════════════════════════════════════════════════════════
    # PIPELINE STAGE 1: NOISE PROFILE DETECTION
    # ══════════════════════════════════════════════════════════════════════
    with st.spinner("🔍 Detecting noise profile..."):
        noise_profile = detect_noise_profile(noisy)

    # ══════════════════════════════════════════════════════════════════════
    # PIPELINE STAGE 2: REGION SEGMENTATION
    # ══════════════════════════════════════════════════════════════════════
    with st.spinner("🧩 Segmenting image regions..."):
        region_masks = segment_image_regions(noisy)
        region_vis = visualise_regions(noisy, region_masks)

    # ══════════════════════════════════════════════════════════════════════
    # PIPELINE STAGE 3: MULTI-FILTER PROCESSING
    # ══════════════════════════════════════════════════════════════════════
    with st.spinner("🔧 Applying filters..."):
        gaussian_img = apply_gaussian(noisy, g_ksize, g_sigma)
        median_img = apply_median(noisy, m_ksize)
        bilateral_img = apply_bilateral(noisy, b_d, b_sc, b_ss)
        adaptive_img = adaptive_region_denoising(noisy, region_masks, noise_profile, filter_params)

    # ══════════════════════════════════════════════════════════════════════
    # PIPELINE STAGE 4: ML WEIGHT OPTIMIZATION
    # ══════════════════════════════════════════════════════════════════════
    with st.spinner("🤖 Training ML models & predicting weights..."):
        # Compute per-filter metrics for feature extraction
        g_metrics = compute_all_metrics(original, gaussian_img)
        m_metrics = compute_all_metrics(original, median_img)
        b_metrics = compute_all_metrics(original, bilateral_img)

        # Build weight training data
        X_train, Y_train = build_weight_training_data(original, filter_params)

        # Augment with self-learning history if enabled
        if enable_learning:
            X_train, Y_train = augment_training_data(X_train, Y_train)

        # Train weight predictor
        weight_model, weight_scaler = train_weight_predictor(X_train, Y_train)

        # Extract features for current image
        ml_features = extract_ml_features(
            noisy, noise_profile,
            g_metrics["PSNR"], m_metrics["PSNR"], b_metrics["PSNR"],
        )

        # Predict optimal weights
        predicted_weights = predict_filter_weights(weight_model, weight_scaler, ml_features)

        # Legacy classifier
        noisy_basic_metrics = compute_basic_metrics(original, noisy)
        X_clf, y_clf = build_classifier_training_data(original, filter_params)
        clf, clf_scaler = train_classifier(X_clf, y_clf, model_type)
        filter_names = ["Gaussian", "Median", "Bilateral"]
        best_single_filter = predict_best_filter(clf, clf_scaler, noisy, noisy_basic_metrics, filter_names)

    # ══════════════════════════════════════════════════════════════════════
    # PIPELINE STAGE 5: ADAPTIVE FILTER FUSION
    # ══════════════════════════════════════════════════════════════════════
    with st.spinner("⚗️ Fusing filter outputs..."):
        fused_global = weighted_filter_fusion(gaussian_img, median_img, bilateral_img, predicted_weights)
        fused_region = region_aware_fusion(gaussian_img, median_img, bilateral_img, predicted_weights, region_masks)

    # ══════════════════════════════════════════════════════════════════════
    # PIPELINE STAGE 6: METRICS COMPUTATION
    # ══════════════════════════════════════════════════════════════════════
    noisy_metrics = compute_all_metrics(original, noisy)
    adaptive_metrics = compute_all_metrics(original, adaptive_img)
    fused_global_metrics = compute_all_metrics(original, fused_global)
    fused_region_metrics = compute_all_metrics(original, fused_region)

    # ══════════════════════════════════════════════════════════════════════
    # PIPELINE STAGE 7: SELF-LEARNING UPDATE
    # ══════════════════════════════════════════════════════════════════════
    if enable_learning:
        save_feedback(ml_features, predicted_weights, fused_region_metrics)

    # ██████████████████████████████████████████████████████████████████████
    # DASHBOARD RENDERING
    # ██████████████████████████████████████████████████████████████████████

    # ── SECTION 1: Input Images ───────────────────────────────────────────
    st.markdown(section_header(
        "📷 Section 1 — Input Images",
        "Original and noisy image comparison with baseline metrics"
    ), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(original, caption="Original", use_container_width=True)
    with c2:
        st.image(noisy, caption=f"Noisy ({noise_type})", use_container_width=True)
    with c3:
        st.markdown("#### Baseline Noise Metrics")
        st.markdown(metric_card("MSE", f"{noisy_metrics['MSE']:.2f}"), unsafe_allow_html=True)
        st.markdown(metric_card("PSNR", f"{noisy_metrics['PSNR']:.2f} dB"), unsafe_allow_html=True)
        st.markdown(metric_card("SSIM", f"{noisy_metrics['SSIM']:.4f}"), unsafe_allow_html=True)
        st.markdown(metric_card("EPS", f"{noisy_metrics['EPS']:.4f}"), unsafe_allow_html=True)

    # ── SECTION 2: Noise Profile Analysis ─────────────────────────────────
    st.markdown(section_header(
        "🔍 Section 2 — Noise Profile Detection",
        "Automatic characterisation of noise type, intensity, and spatial distribution"
    ), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Detected Noise Type**")
        st.markdown(noise_badge(noise_profile["noise_type"]), unsafe_allow_html=True)
        st.markdown("")
        st.markdown("**Distribution Pattern**")
        st.markdown(f'`{noise_profile["distribution_pattern"]}`')
    with c2:
        st.markdown("**Noise Intensity**")
        st.markdown(intensity_gauge(noise_profile["noise_intensity"]), unsafe_allow_html=True)
        st.markdown("")
        st.markdown("**Intensity Value**")
        st.markdown(f'`{noise_profile["noise_intensity"]:.4f}`')
    with c3:
        st.markdown("**Statistical Features**")
        feats = noise_profile["features"]
        feat_df = pd.DataFrame({
            "Feature": [
                "Global Variance", "Global Std Dev", "Laplacian Var",
                "Histogram Kurtosis", "S&P Ratio", "Edge Density",
            ],
            "Value": [
                f"{feats['global_variance']:.1f}",
                f"{feats['global_std']:.2f}",
                f"{feats['laplacian_variance']:.1f}",
                f"{feats['histogram_kurtosis']:.3f}",
                f"{feats['salt_pepper_ratio']:.4f}",
                f"{feats['edge_density']:.4f}",
            ],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ── SECTION 3: Region Segmentation ────────────────────────────────────
    st.markdown(section_header(
        "🧩 Section 3 — Region-Aware Segmentation",
        "Image segmented into smooth (blue), edge (red), and texture (green) regions"
    ), unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.image(region_vis, caption="Region Segmentation Overlay", use_container_width=True)
    with c2:
        rs = region_masks["region_stats"]
        st.markdown(metric_card("🔵 Smooth Regions", f"{rs['smooth_pct']}%"), unsafe_allow_html=True)
        st.markdown(metric_card("🔴 Edge Regions", f"{rs['edge_pct']}%"), unsafe_allow_html=True)
        st.markdown(metric_card("🟢 Texture Regions", f"{rs['texture_pct']}%"), unsafe_allow_html=True)
        st.markdown("")
        st.caption("Each region type receives a different filtering strategy for optimal denoising.")

    # ── SECTION 4: Individual Filter Outputs ──────────────────────────────
    st.markdown(section_header(
        "🔧 Section 4 — Individual Filter Outputs",
        "Results from each classical filter with full quality metrics"
    ), unsafe_allow_html=True)

    filter_data = {
        "Gaussian": (gaussian_img, g_metrics),
        "Median": (median_img, m_metrics),
        "Bilateral": (bilateral_img, b_metrics),
    }

    cols = st.columns(3)
    for col, (name, (img, m)) in zip(cols, filter_data.items()):
        with col:
            st.image(img, caption=f"{name} Filter", use_container_width=True)
            st.markdown(metric_card("MSE", f"{m['MSE']:.2f}"), unsafe_allow_html=True)
            st.markdown(metric_card("PSNR", f"{m['PSNR']:.2f} dB"), unsafe_allow_html=True)
            st.markdown(metric_card("SSIM", f"{m['SSIM']:.4f}"), unsafe_allow_html=True)
            st.markdown(metric_card("EPS", f"{m['EPS']:.4f}"), unsafe_allow_html=True)

    # ── SECTION 5: ML Weight Optimization ─────────────────────────────────
    st.markdown(section_header(
        "🤖 Section 5 — ML-Predicted Filter Weights",
        "Multi-output regression predicts optimal blend weights for filter fusion"
    ), unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**Predicted Optimal Weights**")
        st.markdown(
            weight_bar(predicted_weights.tolist(), ["Gauss", "Median", "Bilat"]),
            unsafe_allow_html=True,
        )
        st.markdown("")

        # Weight comparison table
        weight_df = pd.DataFrame({
            "Filter": ["Gaussian", "Median", "Bilateral"],
            "ML Weight": [f"{w:.3f}" for w in predicted_weights],
            "Equal Weight": ["0.333", "0.333", "0.333"],
        })
        st.dataframe(weight_df, use_container_width=True, hide_index=True)

    with c2:
        st.markdown(f'<div class="best-badge">🏆 {model_type} recommends: <b>{best_single_filter}</b></div>',
                     unsafe_allow_html=True)
        st.markdown("")
        st.caption("The legacy classifier picks a single filter, while the ML optimizer predicts blend weights for superior results.")

        # Weight pie chart
        fig, ax = plt.subplots(figsize=(3, 3), facecolor="#0A0E17")
        colors_pie = ["#4A9EFF", "#FBBF24", "#A78BFA"]
        wedges, texts, autotexts = ax.pie(
            predicted_weights, labels=["G", "M", "B"],
            colors=colors_pie, autopct="%1.0f%%",
            textprops={"color": "#E8ECF4", "fontsize": 9, "fontweight": 600},
            wedgeprops={"edgecolor": "#1E2642", "linewidth": 1.5},
        )
        for t in autotexts:
            t.set_fontsize(8)
        ax.set_title("Weight Distribution", color="#4A9EFF", fontsize=10, fontweight=600)
        fig.tight_layout()
        st.pyplot(fig)

    # ── SECTION 6: Fusion Results ─────────────────────────────────────────
    st.markdown(section_header(
        "⚗️ Section 6 — Adaptive Fusion Results",
        "Comparison of fusion methods: region-aware adaptive, ML-fused, and region-aware fused"
    ), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(adaptive_img, caption="Region-Aware Adaptive", use_container_width=True)
        st.markdown(metric_card("PSNR", f"{adaptive_metrics['PSNR']:.2f} dB"), unsafe_allow_html=True)
        st.markdown(metric_card("SSIM", f"{adaptive_metrics['SSIM']:.4f}"), unsafe_allow_html=True)
        st.markdown(metric_card("EPS", f"{adaptive_metrics['EPS']:.4f}"), unsafe_allow_html=True)
    with c2:
        st.image(fused_global, caption="ML Global Fusion", use_container_width=True)
        st.markdown(metric_card("PSNR", f"{fused_global_metrics['PSNR']:.2f} dB"), unsafe_allow_html=True)
        st.markdown(metric_card("SSIM", f"{fused_global_metrics['SSIM']:.4f}"), unsafe_allow_html=True)
        st.markdown(metric_card("EPS", f"{fused_global_metrics['EPS']:.4f}"), unsafe_allow_html=True)
    with c3:
        st.image(fused_region, caption="ML Region-Aware Fusion", use_container_width=True)
        st.markdown(metric_card("PSNR", f"{fused_region_metrics['PSNR']:.2f} dB"), unsafe_allow_html=True)
        st.markdown(metric_card("SSIM", f"{fused_region_metrics['SSIM']:.4f}"), unsafe_allow_html=True)
        st.markdown(metric_card("EPS", f"{fused_region_metrics['EPS']:.4f}"), unsafe_allow_html=True)

    # ── SECTION 7: Edge Preservation Analysis ─────────────────────────────
    st.markdown(section_header(
        "📐 Section 7 — Edge Preservation Analysis",
        "Comparison of edge preservation across all methods using the novel EPS metric"
    ), unsafe_allow_html=True)

    # Compute edge maps for visualisation
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
    fused_gray = cv2.cvtColor(fused_region, cv2.COLOR_BGR2GRAY) if len(fused_region.shape) == 3 else fused_region
    edges_orig = cv2.Canny(orig_gray, 50, 150)
    edges_fused = cv2.Canny(fused_gray, 50, 150)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(edges_orig, caption="Original Edges", use_container_width=True)
    with c2:
        st.image(edges_fused, caption="Fused Output Edges", use_container_width=True)
    with c3:
        # EPS comparison chart
        all_method_names = ["Gaussian", "Median", "Bilateral", "Adaptive", "Global\nFusion", "Region\nFusion"]
        all_eps = [
            g_metrics["EPS"], m_metrics["EPS"], b_metrics["EPS"],
            adaptive_metrics["EPS"], fused_global_metrics["EPS"], fused_region_metrics["EPS"],
        ]
        st.pyplot(dark_bar_chart(all_method_names, all_eps, "Edge Preservation Score (EPS) ↑", "#A78BFA"))

    # ── SECTION 8: Comprehensive Comparison ───────────────────────────────
    st.markdown(section_header(
        "📊 Section 8 — Full Metrics Comparison",
        "Complete comparison table and charts across all denoising methods"
    ), unsafe_allow_html=True)

    # Comparison charts
    all_names = ["Gaussian", "Median", "Bilateral", "Adaptive", "Fusion\n(Global)", "Fusion\n(Region)"]
    all_metrics = [g_metrics, m_metrics, b_metrics, adaptive_metrics, fused_global_metrics, fused_region_metrics]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.pyplot(dark_bar_chart(all_names, [m["PSNR"] for m in all_metrics], "PSNR (dB) ↑", "#4A9EFF"))
    with c2:
        st.pyplot(dark_bar_chart(all_names, [m["MSE"] for m in all_metrics], "MSE ↓", "#F87171"))
    with c3:
        st.pyplot(dark_bar_chart(all_names, [m["SSIM"] for m in all_metrics], "SSIM ↑", "#34D399"))

    # Full comparison table
    rows = [
        {"Method": "Noisy (baseline)", **noisy_metrics},
        {"Method": "Gaussian Filter", **g_metrics},
        {"Method": "Median Filter", **m_metrics},
        {"Method": "Bilateral Filter", **b_metrics},
        {"Method": "Adaptive Region-Aware", **adaptive_metrics},
        {"Method": "ML Global Fusion", **fused_global_metrics},
        {"Method": "ML Region-Aware Fusion", **fused_region_metrics},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── SECTION 9: Self-Learning Status ───────────────────────────────────
    if enable_learning:
        st.markdown(section_header(
            "🧠 Section 9 — Self-Learning Status",
            "The system accumulates experience from each processed image to improve future predictions"
        ), unsafe_allow_html=True)

        stats = get_learning_stats()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                '<div class="learning-indicator">'
                f'<div class="stat">{stats["total_samples"]}</div>'
                '<div class="desc">Samples Learned</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                '<div class="learning-indicator">'
                f'<div class="stat">{stats["avg_psnr"]:.1f} dB</div>'
                '<div class="desc">Avg PSNR</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                '<div class="learning-indicator">'
                f'<div class="stat">{stats["avg_ssim"]:.4f}</div>'
                '<div class="desc">Avg SSIM</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                '<div class="learning-indicator">'
                f'<div class="stat">{stats["last_session"]}</div>'
                '<div class="desc">Last Session</div></div>',
                unsafe_allow_html=True,
            )

    # ── Download Best Result ──────────────────────────────────────────────
    st.markdown("---")
    best_pil = Image.fromarray(fused_region)
    buf = io.BytesIO()
    best_pil.save(buf, format="PNG")
    st.download_button(
        label="⬇️  Download Best Denoised Image (Region-Aware Fusion)",
        data=buf.getvalue(),
        file_name="denoised_region_fusion.png",
        mime="image/png",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
