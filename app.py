import os
from pathlib import Path
from PIL import Image
import streamlit as st
from c2pa_check import inspect_c2pa                         #for the inspect_c2pa file and function
from image_features import extract_image_data               #for the image_features file and function
from forensics_tier2 import analyze_tier2_patches           #for the forensics_tier2 file and function


# 1. Page setup
st.set_page_config(page_title="AI Image Segregation Prototype", layout="wide")
st.title("Two-Tier AI Image Verification Prototype")
st.caption("Tier 1: C2PA Provenance | Tier 2: Spatial & Frequency Forensics")

# 2. Create a local folder for raw uploaded files
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 3. File uploader widget
uploaded_file = st.file_uploader(
    "Upload an image to inspect",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    # 4. Save raw bytes without re-encoding (preserves C2PA metadata)
    save_path = UPLOAD_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Saved raw file to: `{save_path}`")

    # 5. Open with Pillow for display / later tensor conversion
    image = Image.open(save_path)
    st.image(image, caption=f"Uploaded: {uploaded_file.name} ({image.size[0]}x{image.size[1]})", width=400)

    # Placeholder where Tier 1 & Tier 2 will plug in next
    c2pa_result = inspect_c2pa(save_path)

    st.subheader("Tier 1: C2PA Provenance Check")                                    #First: Tier 1
    if c2pa_result["status"] == "verified":
        st.success(c2pa_result["message"])
        st.json(c2pa_result["details"])
    elif c2pa_result["status"] == "invalid_or_tampered":
        st.warning(c2pa_result["message"])
        st.json(c2pa_result["details"])
    else:
        st.info(c2pa_result["message"])

    st.subheader("Tier 2 (Step 3): Image-to-Data Conversion & Signal Summary")       #Second: Tier 2
    features = extract_image_data(save_path)

    tensor = features["spatial_tensor"]
    freq_map = features["frequency_map"]

    # 1. Key numbers / metrics for the demo
    mean_brightness = float(tensor.mean())
    red_mean = float(tensor[:, :, 0].mean())
    green_mean = float(tensor[:, :, 1].mean())
    blue_mean = float(tensor[:, :, 2].mean())
    high_freq_energy = float(freq_map.mean())
    max_freq_peak = float(freq_map.max())

    # Simple descriptive "Signal Grade" based on mean brightness (0.0 to 1.0)
    if mean_brightness < 0.25:
        brightness_grade = "Dark / Low-Key"
    elif mean_brightness <= 0.75:
        brightness_grade = "Balanced / Mid-Tone"
    else:
        brightness_grade = "Bright / High-Key"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tensor Shape", str(features["tensor_shape"]))
    m2.metric("Mean Brightness", f"{mean_brightness:.3f}", brightness_grade)
    m3.metric("Avg Frequency Energy", f"{high_freq_energy:.2f}")
    m4.metric("Peak Spectrum Value", f"{max_freq_peak:.2f}")

    # 2. Side-by-side images + a real graph of RGB channel averages
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("Normalized RGB Image (0.0–1.0)")
        st.image(tensor, use_container_width=True)
    with col2:
        st.caption("2D Frequency Spectrum (Log Magnitude)")
        st.image(features["frequency_vis"], clamp=True, use_container_width=True)
    with col3:
        st.caption("Average RGB Channel Intensity")
        channel_data = {
            "Red": [red_mean],
            "Green": [green_mean],
            "Blue": [blue_mean],
        }
        st.bar_chart(channel_data)

    st.subheader("Tier 2 (Step 4): Patch-Based Forensic Analysis & Heatmap")
    tier2 = analyze_tier2_patches(tensor, freq_map, patch_size=16)

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Patch Grid", str(tier2["patch_grid_shape"]))
    t2.metric("Tier-2 Score", f"{tier2['overall_score']:.3f}", tier2["grade"])
    t3.metric("Peak Patch Score", f"{tier2['max_patch_score']:.3f}")
    t4.metric("Patch Size", "16 x 16 px")

    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        st.caption("Patch Anomaly Heatmap")
        st.image(tier2["heatmap_rgb"], use_container_width=True)
    with h_col2:
        st.caption("Heatmap Overlay on Image")
        st.image(tier2["overlay_rgb"], use_container_width=True)
    with h_col3:
        st.caption("Patch Score Distribution (Row Averages)")
        row_means = tier2["patch_scores"].mean(axis=1)
        st.line_chart(row_means)
    
    st.divider()
    st.markdown("## Final Verification Summary & Conclusion")

    t1_status = c2pa_result.get("status")
    claim_gen = (c2pa_result.get("details") or {}).get("claim_generator")

    if t1_status == "verified":
        verdict = "Provenance Verified (Tier 1)"
        conclusion = (
            "Conclusion: Cryptographic C2PA provenance is present and valid"
            + (f" (signed/generated via {claim_gen}). " if claim_gen else ". ")
            + f"Tier-2 forensic anomaly score is {tier2['overall_score']:.3f} ({tier2['grade']})."
        )
        banner_fn = st.success

    elif t1_status == "untrusted_cert":
        verdict = "C2PA Manifest Detected (Untrusted Local Cert)"
        conclusion = (
            "Conclusion: A C2PA manifest was found"
            + (f" (Generator: {claim_gen}), " if claim_gen else ", ")
            + "indicating provenance metadata exists, though the certificate is untrusted locally. "
            + f"Tier-2 score is {tier2['overall_score']:.3f} ({tier2['grade']})."
        )
        banner_fn = st.warning

    elif t1_status == "invalid_or_tampered":
        verdict = "Manifest Validation Warning"
        conclusion = (
            "Conclusion: A C2PA manifest is present, but validation reported warnings or issues. "
            + f"Tier-2 score is {tier2['overall_score']:.3f} ({tier2['grade']})."
        )
        banner_fn = st.error

    else:
        if tier2.get("is_minimalist"):
            verdict = "Inconclusive — Low-Texture / Minimalist Image"
            conclusion = (
                "Conclusion: No C2PA manifest was found, and pixel/frequency heuristics have low "
                f"confidence on low-grain or minimalist imagery (Tier-2 score: {tier2['overall_score']:.3f})."
            )
            banner_fn = st.warning
        elif tier2["overall_score"] >= 0.58:
            verdict = "Elevated Tier-2 Forensic Anomaly"
            conclusion = (
                f"Conclusion: No C2PA manifest was found, and Tier-2 patch analysis shows elevated "
                f"irregularity (Score: {tier2['overall_score']:.3f}, Peak: {tier2['max_patch_score']:.3f})."
            )
            banner_fn = st.warning
        else:
            verdict = "No Manifest Found — Low Tier-2 Anomaly"
            conclusion = (
                f"Conclusion: No C2PA manifest was found, and Tier-2 patch analysis shows low "
                f"anomaly ({tier2['overall_score']:.3f} — {tier2['grade']})."
            )
            banner_fn = st.info

    # Big, high-visibility conclusion box
    st.markdown(
        f"""
        <div style="padding: 18px 22px; border-radius: 10px; border: 1px solid rgba(150,150,150,0.3); margin-bottom: 16px;">
            <div style="font-size: 1.35rem; font-weight: 700; margin-bottom: 8px;">{verdict}</div>
            <div style="font-size: 1.08rem; line-height: 1.55;">{conclusion}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Larger, styled summary table
    t1_msg = c2pa_result.get("message", "N/A")
    t2_summary = f"{tier2['overall_score']:.3f} — {tier2['grade']}"

    st.markdown("### Detailed Stage Breakdown")
    st.markdown(
            f"""
    | Stage | Findings & Conclusion |
    | :--- | :--- |
    | **Tier 1: C2PA Provenance** | {t1_msg} |
    | **Tier 2: Patch Forensics** | {t2_summary} |
    | **Final Conclusion** | {conclusion} |
    """
        )
    table_html = """
    
    """
    
    table_html += "</tbody></table>"

    st.markdown(table_html, unsafe_allow_html=True)
        
    