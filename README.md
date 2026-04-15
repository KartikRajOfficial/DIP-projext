# Adaptive Image Denoising using Hybrid Filtering with Machine Learning Optimization

## A Patent-Worthy Intelligent Image Denoising Framework

A **Streamlit** web application implementing an advanced adaptive image denoising
framework that combines classical image-processing filters with machine learning
optimization, region-aware segmentation, multi-filter fusion, and a self-learning
feedback loop.

---

## 📋 Requirements

### System Requirements

| Requirement | Version |
|---|---|
| **Operating System** | Windows 10/11, macOS, or Linux |
| **Python** | 3.10 or higher |
| **RAM** | Minimum 4 GB |
| **Disk Space** | ~500 MB (for Python + dependencies) |
| **Browser** | Chrome, Firefox, or Edge (for Streamlit UI) |

### Python Dependencies

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | Latest | Web UI framework |
| `opencv-python` | Latest | Image filters (GaussianBlur, medianBlur, bilateralFilter), edge detection |
| `numpy` | Latest | Numerical arrays & noise generation |
| `scikit-learn` | Latest | ML models (Ridge, KNN, SVM, MultiOutputRegressor, StandardScaler) |
| `scikit-image` | Latest | SSIM metric |
| `scipy` | Latest | Statistical functions (kurtosis, skewness) |
| `matplotlib` | Latest | Charts & visualisations |
| `Pillow` | Latest | Image I/O |
| `pandas` | Latest | Metrics tables |

All dependencies are listed in `requirements.txt`.

---

## 🚀 Deployment & Setup (Step-by-Step)

### Step 1 — Open Terminal

Open **Command Prompt** or **PowerShell** on Windows.

Navigate to the project folder:

```bash
cd C:\Users\karti\OneDrive\Desktop\DIP Project
```

### Step 2 — Create a Virtual Environment (One-Time Only)

```bash
python -m venv .venv
```

This creates an isolated Python environment inside the `.venv` folder.

### Step 3 — Activate the Virtual Environment

**Windows (Command Prompt):**
```bash
.venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

> After activation, you should see `(.venv)` at the beginning of your terminal prompt.

### Step 4 — Install All Dependencies (One-Time Only)

```bash
pip install -r requirements.txt
```

This installs all 9 required libraries listed above.

### Step 5 — Run the Application

```bash
python -m streamlit run app.py
```

The app will automatically open in your default browser at:

```
http://localhost:8501
```

### Step 6 — Stop the Application

Press `Ctrl + C` in the terminal to stop the Streamlit server.

---

## 🖥️ How to Use the Application

1. Open the **sidebar** on the left side of the screen.
2. **"Use demo image"** is checked by default. Uncheck it and upload your own PNG/JPG if you want.
3. Choose a **noise type**: `Gaussian`, `Salt & Pepper`, or `Mixed`.
4. Adjust the **noise intensity** sliders.
5. Optionally adjust individual **filter parameters** (kernel sizes, sigma values).
6. Select an **ML classifier** (`KNN` or `SVM`) for the legacy comparison.
7. Enable or disable **self-learning** (enabled by default).
8. Click **▶ Run Full Pipeline**.
9. Wait ~20 seconds — all **9 dashboard sections** will render.
10. Scroll down through the results.
11. Click the **download button** at the bottom to save the best denoised image.

---

## 📊 Dashboard Sections (9 Sections)

| Section | What It Shows |
|---|---|
| **1. Input Images** | Original image + Noisy image side-by-side with baseline MSE/PSNR/SSIM/EPS metrics |
| **2. Noise Profile Detection** | Detected noise type badge, intensity gauge bar, distribution pattern, statistical feature table |
| **3. Region Segmentation** | Colour-coded overlay — Smooth (blue), Edge (red), Texture (green) — with percentage breakdown |
| **4. Filter Outputs** | Individual Gaussian, Median, Bilateral filter results with full metrics per filter |
| **5. ML Weight Optimization** | Predicted blend weights as stacked bar + pie chart, legacy single-filter recommendation |
| **6. Fusion Results** | 3 methods compared — Region-Aware Adaptive, ML Global Fusion, ML Region-Aware Fusion |
| **7. Edge Preservation** | Edge maps (original vs denoised), EPS comparison bar chart across all methods |
| **8. Full Comparison** | PSNR/MSE/SSIM bar charts + complete metrics table for all 6 denoising methods |
| **9. Self-Learning Status** | Number of samples learned, average PSNR/SSIM, last session timestamp |

---

## 📁 Project Structure

```
DIP Project/
│
├── app.py                          ← Main Streamlit application (9-section dashboard)
│
├── modules/                        ← Modular architecture (8 modules)
│   ├── __init__.py                 ← Package initialisation
│   ├── utils.py                    ← Image loading, noise injection (Gaussian/S&P/Mixed)
│   ├── noise_detection.py          ← Innovation #1: Noise Profile Detection
│   ├── region_segmentation.py      ← Innovation #2: Region-Aware Segmentation
│   ├── filters.py                  ← Innovation #3: Adaptive Multi-Filter Pipeline
│   ├── fusion.py                   ← Innovation #5: Multi-Filter Fusion Engine
│   ├── metrics.py                  ← Innovation #6: Edge Preservation Score (EPS)
│   ├── ml_optimizer.py             ← Innovation #4: ML Weight Optimization
│   └── self_learning.py            ← Innovation #7: Self-Learning Feedback System
│
├── feedback_data/                  ← Auto-created directory for self-learning data
│   └── learning_history.json       ← Accumulated feedback records (grows with usage)
│
├── test_pipeline.py                ← End-to-end test script
├── requirements.txt                ← Python dependencies
└── README.md                       ← This file
```

---

## 🔬 System Pipeline

```
Input Image
    ↓
🔍 Noise Profile Detection (type, intensity, distribution)
    ↓
🧩 Region Segmentation (smooth / edge / texture masks)
    ↓
🔧 Multi-Filter Processing (Gaussian · Median · Bilateral)
    ↓
🤖 ML Weight Optimization (Ridge Regression → blend weights)
    ↓
⚗️ Adaptive Filter Fusion (global + region-aware weighted blend)
    ↓
📐 Edge Preservation Evaluation (EPS metric)
    ↓
🧠 Self-Learning Model Update (feedback stored for next run)
    ↓
✅ Final Denoised Image
```

---

## 🏆 Patent-Worthy Innovations (7 Novelties)

### Innovation 1 — Automatic Noise Profile Detection
- **Module:** `noise_detection.py`
- **Function:** `detect_noise_profile(image)`
- **What it does:** Analyses the noisy image using 6 statistical features (variance, kurtosis, Laplacian variance, S&P ratio, edge density, local variance) to detect:
  - Noise type: Gaussian / Impulse / Mixed
  - Noise intensity: 0.0 (clean) to 1.0 (heavy)
  - Distribution pattern: uniform / clustered / sparse
- **Why novel:** Existing denoisers blindly apply filters. This system diagnoses first, then treats.

### Innovation 2 — Region-Aware Image Segmentation
- **Module:** `region_segmentation.py`
- **Function:** `segment_image_regions(image)`
- **What it does:** Segments the image into 3 structural regions:
  - Smooth regions (flat, homogeneous)
  - Edge regions (strong gradients)
  - Texture regions (high-frequency detail)
- **Why novel:** Enables different treatment for different parts of the same image.

### Innovation 3 — Adaptive Multi-Filter Pipeline
- **Module:** `filters.py`
- **Function:** `adaptive_region_denoising(image, region_masks, noise_profile, filter_params)`
- **What it does:** Assigns different filters to different regions:
  - Smooth → Gaussian filter
  - Edge → Bilateral filter
  - Texture → Light Median filter
- **Why novel:** Noise-profile-driven + region-aware + intensity-adaptive filter assignment.

### Innovation 4 — ML-Based Filter Weight Optimization
- **Module:** `ml_optimizer.py`
- **Function:** `predict_filter_weights(model, scaler, features)`
- **What it does:** Instead of picking ONE best filter, predicts optimal blend weights `[w₁, w₂, w₃]` using multi-output Ridge regression trained on grid-searched optimal weights.
- **Why novel:** Continuous weight prediction (regression) instead of discrete filter selection (classification).

### Innovation 5 — Multi-Filter Fusion Engine
- **Module:** `fusion.py`
- **Functions:** `weighted_filter_fusion()` + `region_aware_fusion()`
- **What it does:** Combines filter outputs: `result = w₁·Gaussian + w₂·Median + w₃·Bilateral`
  - Global fusion applies weights uniformly
  - Region-aware fusion modulates weights per region type
- **Why novel:** Spatially varying fusion that adapts to image structure.

### Innovation 6 — Edge Preservation Score (EPS)
- **Module:** `metrics.py`
- **Function:** `edge_preservation_score(original, denoised)`
- **What it does:** Measures edge fidelity: `EPS = |Canny(orig) ∩ Canny(denoised)| / |Canny(orig)|`
  - EPS = 1.0 → all edges preserved
  - EPS = 0.0 → all edges destroyed
- **Why novel:** MSE/PSNR/SSIM don't measure structural edge fidelity. EPS fills this gap.

### Innovation 7 — Self-Learning Feedback System
- **Module:** `self_learning.py`
- **Functions:** `save_feedback()` + `augment_training_data()`
- **What it does:** After every image processed:
  1. Stores features, weights, and metrics to disk
  2. Next run merges historical data with synthetic training data
  3. ML model retrains on the combined dataset
- **Why novel:** Creates a self-improving system that gets better with every use.

---

## 🧪 How the ML Models Work

### Weight Predictor (Multi-Output Regression)

1. **Training data:** For 12 noise configurations (Gaussian at 5 levels + S&P at 4 levels + Mixed at 3 levels):
   - Inject noise → apply all 3 filters → grid-search weight combinations
   - Find `[w₁, w₂, w₃]` that maximises PSNR
   - Extract 12-dimensional feature vector

2. **Self-learning:** Historical feedback records merged with synthetic data.

3. **Model:** `MultiOutputRegressor(Ridge)` predicts 3 weights from 12 features.

4. **Inference:** Weights are clipped to `[0, 1]` and normalised to sum to 1.0.

### Legacy Classifier (KNN / SVM)

- Predicts the single best filter based on 5 noisy-image features.
- Retained for comparison against the multi-filter fusion approach.

---

## 🧪 Running Tests

To verify the entire pipeline works:

```bash
python test_pipeline.py
```

Expected output:
```
=== All modules imported successfully ===
1. Demo image loaded: (256, 256, 3)
2. Noisy image created: (256, 256, 3)
3. Noise profile: type=gaussian, intensity=0.7237, dist=sparse
...
=== ALL TESTS PASSED ===
```

---

## 💡 Demo Tips for Presentation

1. **First run:** Use **Gaussian** noise → click **Run Full Pipeline** → show all 9 sections
2. **Second run:** Switch to **Mixed** noise → run again → show how:
   - Section 2 detects different noise type
   - Section 5 shows different ML weights
   - Section 9 shows **"1 sample learned"** (self-learning in action)
3. **Third run:** Upload a **real photo** → show it works on actual images
4. **Show improvement:** Compare the PSNR values — the adaptive/fusion methods beat individual filters

---

## ⚠️ Important Notes

- **No deep learning** — no TensorFlow, PyTorch, or Keras required.
- All images are resized to **256×256** for consistent processing.
- Kernel sizes are automatically forced to **odd numbers**.
- Self-learning history is capped at **500 records** to prevent unbounded growth.
- The `feedback_data/` directory is created automatically on first run.
- The application runs entirely **offline** — no internet required after setup.

---

