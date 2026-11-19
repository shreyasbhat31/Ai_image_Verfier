import cv2
import numpy as np


def analyze_tier2_patches(
    spatial_tensor: np.ndarray,
    frequency_map: np.ndarray,
    patch_size: int = 16,
) -> dict:
    """
    Recalibrated patch-based forensic heuristic for prototype demonstration.
    Separates structured edge/contrast detail from high-frequency anomalies
    and flags low-texture / minimalist images where pixel rules are less reliable.
    """
    h, w, _ = spatial_tensor.shape
    gray = cv2.cvtColor(spatial_tensor.astype(np.float32), cv2.COLOR_RGB2GRAY)

    # Local high-frequency detail (Laplacian) and directional gradients (Sobel)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(gx**2 + gy**2)

    # Normalize frequency_map to 0.0 - 1.0
    freq_max = float(frequency_map.max())
    freq_norm = (frequency_map / freq_max) if freq_max > 0 else np.zeros_like(frequency_map)

    rows = h // patch_size
    cols = w // patch_size
    patch_scores = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            r0, r1 = r * patch_size, (r + 1) * patch_size
            c0, c1 = c * patch_size, (c + 1) * patch_size

            patch_gray = gray[r0:r1, c0:c1]
            patch_lap = laplacian[r0:r1, c0:c1]
            patch_edge = edge_mag[r0:r1, c0:c1]
            patch_freq = freq_norm[r0:r1, c0:c1]

            contrast = float(np.std(patch_gray))
            lap_mean = float(np.mean(patch_lap))
            edge_mean = float(np.mean(patch_edge))
            freq_mean = float(np.mean(patch_freq))

            # Ratio of fine Laplacian detail to broader edge structure
            # (High when micro-fluctuations exist without strong structured edges)
            micro_to_edge = lap_mean / (edge_mean + 1e-4)
            micro_norm = min(micro_to_edge / 1.5, 1.0)

            # Frequency energy scaled down so natural scenery doesn't hit 0.6 automatically
            freq_component = max(0.0, (freq_mean - 0.35) / 0.65)

            # Slight penalty only when contrast is very low AND micro-detail is high
            low_contrast_anomaly = (1.0 - min(contrast * 4.0, 1.0)) * min(lap_mean * 3.0, 1.0)

            raw_score = (0.45 * freq_component) + (0.35 * micro_norm) + (0.20 * low_contrast_anomaly)
            patch_scores[r, c] = float(np.clip(raw_score, 0.0, 1.0))

    overall_score = float(np.mean(patch_scores))
    max_patch_score = float(np.max(patch_scores))
    global_contrast = float(np.std(gray))
    global_edges = float(np.mean(edge_mag))

    # Detect minimalist / low-texture images (very low contrast or very few edges)
    is_minimalist = (global_contrast < 0.14) or (global_edges < 0.04)

    heatmap_unit = cv2.resize(patch_scores, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.clip(heatmap_unit * 255.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    base_uint8 = np.clip(spatial_tensor * 255.0, 0, 255).astype(np.uint8)
    overlay_rgb = cv2.addWeighted(base_uint8, 0.55, heatmap_rgb, 0.45, 0)

    if is_minimalist:
        grade = "Minimalist / Low-Texture (Low Heuristic Confidence)"
    elif overall_score < 0.30:
        grade = "Low Anomaly (Natural-Like)"
    elif overall_score < 0.58:
        grade = "Moderate Anomaly (Worth Inspecting)"
    else:
        grade = "High Anomaly (Strong Signal)"

    return {
        "patch_grid_shape": patch_scores.shape,
        "patch_scores": patch_scores,
        "overall_score": overall_score,
        "max_patch_score": max_patch_score,
        "grade": grade,
        "is_minimalist": is_minimalist,
        "heatmap_rgb": heatmap_rgb,
        "overlay_rgb": overlay_rgb,
    }