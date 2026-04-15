"""End-to-end test of the full denoising pipeline."""
import sys
sys.path.insert(0, ".")

from modules.utils import load_demo_image, add_gaussian_noise, add_salt_pepper_noise, add_mixed_noise
from modules.noise_detection import detect_noise_profile
from modules.region_segmentation import segment_image_regions, visualise_regions
from modules.filters import apply_gaussian, apply_median, apply_bilateral, adaptive_region_denoising
from modules.fusion import weighted_filter_fusion, region_aware_fusion
from modules.metrics import compute_all_metrics, compute_basic_metrics
from modules.ml_optimizer import (
    extract_ml_features, build_weight_training_data,
    train_weight_predictor, predict_filter_weights,
    build_classifier_training_data, train_classifier, predict_best_filter,
)
from modules.self_learning import get_learning_stats

print("=== All modules imported successfully ===")

# 1. Load image
img = load_demo_image()
print(f"1. Demo image loaded: {img.shape}")

# 2. Add noise
noisy = add_gaussian_noise(img, 25)
print(f"2. Noisy image created: {noisy.shape}")

# 3. Noise profile detection
profile = detect_noise_profile(noisy)
print(f"3. Noise profile: type={profile['noise_type']}, intensity={profile['noise_intensity']:.4f}, dist={profile['distribution_pattern']}")

# 4. Region segmentation
regions = segment_image_regions(noisy)
rs = regions["region_stats"]
print(f"4. Regions: smooth={rs['smooth_pct']}%, edge={rs['edge_pct']}%, texture={rs['texture_pct']}%")

# 5. Visualise regions
vis = visualise_regions(noisy, regions)
print(f"5. Region visualisation: {vis.shape}")

# 6. Apply filters
g = apply_gaussian(noisy, 5, 1.5)
m = apply_median(noisy, 5)
b = apply_bilateral(noisy, 9, 75, 75)
print(f"6. Filters applied: G={g.shape}, M={m.shape}, B={b.shape}")

# 7. Adaptive region denoising
fp = {"g_ksize": 5, "g_sigma": 1.5, "m_ksize": 5, "b_d": 9, "b_sc": 75, "b_ss": 75}
adaptive = adaptive_region_denoising(noisy, regions, profile, fp)
print(f"7. Adaptive denoising: {adaptive.shape}")

# 8. Compute metrics
g_metrics = compute_all_metrics(img, g)
m_metrics = compute_all_metrics(img, m)
b_metrics = compute_all_metrics(img, b)
a_metrics = compute_all_metrics(img, adaptive)
print(f"8. Metrics computed:")
print(f"   Gaussian:  PSNR={g_metrics['PSNR']:.2f}, SSIM={g_metrics['SSIM']:.4f}, EPS={g_metrics['EPS']:.4f}")
print(f"   Median:    PSNR={m_metrics['PSNR']:.2f}, SSIM={m_metrics['SSIM']:.4f}, EPS={m_metrics['EPS']:.4f}")
print(f"   Bilateral: PSNR={b_metrics['PSNR']:.2f}, SSIM={b_metrics['SSIM']:.4f}, EPS={b_metrics['EPS']:.4f}")
print(f"   Adaptive:  PSNR={a_metrics['PSNR']:.2f}, SSIM={a_metrics['SSIM']:.4f}, EPS={a_metrics['EPS']:.4f}")

# 9. ML feature extraction
ml_feats = extract_ml_features(noisy, profile, g_metrics["PSNR"], m_metrics["PSNR"], b_metrics["PSNR"])
print(f"9. ML features: shape={ml_feats.shape}")

# 10. Build training data + train weight predictor
X, Y = build_weight_training_data(img, fp)
print(f"10. Training data: X={X.shape}, Y={Y.shape}")

model, scaler = train_weight_predictor(X, Y)
weights = predict_filter_weights(model, scaler, ml_feats)
print(f"11. Predicted weights: G={weights[0]:.3f}, M={weights[1]:.3f}, B={weights[2]:.3f}, sum={sum(weights):.3f}")

# 12. Fusion
fused_global = weighted_filter_fusion(g, m, b, weights)
fused_region = region_aware_fusion(g, m, b, weights, regions)
fg_metrics = compute_all_metrics(img, fused_global)
fr_metrics = compute_all_metrics(img, fused_region)
print(f"12. Fusion results:")
print(f"    Global:  PSNR={fg_metrics['PSNR']:.2f}, SSIM={fg_metrics['SSIM']:.4f}, EPS={fg_metrics['EPS']:.4f}")
print(f"    Region:  PSNR={fr_metrics['PSNR']:.2f}, SSIM={fr_metrics['SSIM']:.4f}, EPS={fr_metrics['EPS']:.4f}")

# 13. Legacy classifier
noisy_basic = compute_basic_metrics(img, noisy)
X_clf, y_clf = build_classifier_training_data(img, fp)
clf, clf_scaler = train_classifier(X_clf, y_clf, "KNN")
best = predict_best_filter(clf, clf_scaler, noisy, noisy_basic, ["Gaussian", "Median", "Bilateral"])
print(f"13. Legacy classifier recommends: {best}")

# 14. Self-learning stats
stats = get_learning_stats()
print(f"14. Learning stats: {stats['total_samples']} samples")

print("\n=== ALL TESTS PASSED ===")
