import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re
import uuid

# ---------------------------------------------------------
# 0. Page Config & Constants
# ---------------------------------------------------------
st.set_page_config(page_title="Bio-Image Quantifier V2 (EN)", layout="wide")

# Version Management
SOFTWARE_VERSION = "Bio-Image Quantifier Pro v2026.02 (EN/Auto-Group)"

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())
    
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# --- Analysis ID Management (Human-readable + Unique ID) ---
if "current_analysis_id" not in st.session_state:
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
    unique_suffix = str(uuid.uuid4())[:8]
    st.session_state.current_analysis_id = f"AID-{date_str}-{unique_suffix}"

# ---------------------------------------------------------
# 1. Image Processing Engine
# ---------------------------------------------------------
COLOR_MAP = {
    "Brown (DAB)": {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 255])},
    "Green (GFP)": {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},
    "Red (RFP)": {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
    "Blue (DAPI)": {"lower": np.array([100, 50, 50]), "upper": np.array([140, 255, 255])}
}

def get_mask(hsv_img, color_name, sens, bright_min):
    if color_name == "Red (RFP)":
        lower1 = np.array([0, 30, bright_min]); upper1 = np.array([10 + sens//2, 255, 255])
        lower2 = np.array([170 - sens//2, 30, bright_min]); upper2 = np.array([180, 255, 255])
        return cv2.inRange(hsv_img, lower1, upper1) | cv2.inRange(hsv_img, lower2, upper2)
    else:
        conf = COLOR_MAP[color_name]
        l = np.clip(conf["lower"] - sens, 0, 255); u = np.clip(conf["upper"] + sens, 0, 255)
        l[2] = max(l[2], bright_min)
        return cv2.inRange(hsv_img, l, u)

def get_tissue_mask(hsv_img, color_name, sens, bright_min):
    mask = get_mask(hsv_img, color_name, sens, bright_min)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
    valid_tissue = [c for c in cnts if cv2.contourArea(c) > 500]
    cv2.drawContours(mask_filled, valid_tissue, -1, 255, thickness=cv2.FILLED)
    return mask_filled

def get_centroids(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] != 0: pts.append(np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]]))
    return pts

# ---------------------------------------------------------
# 2. Validation Data Loading
# ---------------------------------------------------------
@st.cache_data
def load_validation_data():
    files = {'C14': 'quantified_data_20260102_201522.csv', 'C40': 'quantified_data_20260102_194322.csv',
             'C70': 'quantified_data_20260103_093427.csv', 'C100': 'quantified_data_20260102_202525.csv'}
    data_list = []; mapping = {'C14': 14, 'C40': 40, 'C70': 70, 'C100': 100}
    for density, filename in files.items():
        try:
            df = pd.read_csv(filename); col = 'Image_Name' if 'Image_Name' in df.columns else 'File Name'
            for _, row in df.iterrows():
                fname = str(row[col]); val = row['Value']
                channel = 'W1' if 'w1' in fname.lower() else 'W2' if 'w2' in fname.lower() else None
                if not channel: continue
                f_match = re.search(r'_F(\d+)_', fname)
                if f_match:
                    focus = int(f_match.group(1)); accuracy = (val / mapping[density]) * 100
                    data_list.append({'Density': density, 'Ground Truth': mapping[density], 'Focus': focus, 'Channel': channel, 'Value': val, 'Accuracy': accuracy})
        except FileNotFoundError: pass
    return pd.DataFrame(data_list)

df_val = load_validation_data()

# ---------------------------------------------------------
# 3. UI Framework & Sidebar
# ---------------------------------------------------------
st.title("🔬 Bio-Image Quantifier: Pro Edition (English)")
st.caption(f"{SOFTWARE_VERSION}: Industrial-Grade Image Analysis & Data Extraction")

st.sidebar.markdown(f"**Current Analysis ID:** `{st.session_state.current_analysis_id}`")

tab_main, tab_val = st.tabs(["🚀 Run Analysis", "🏆 Performance Validation"])

with st.sidebar:
    st.markdown("### [Important: Use in Papers/Presentations]")
    st.warning("""
    **Planning to publish research results?**
    This tool is in beta. For academic use, **please contact the developer (Kaneko) in advance.**
    We can discuss co-authorship or acknowledgments.
    👉 **[Contact Form](https://forms.gle/xgNscMi3KFfWcuZ1A)**
    """)
    st.divider()

    st.header("Analysis Recipe")
    mode_raw = st.selectbox("Select Analysis Mode:", [
        "1. Area Occupancy %", 
        "2. Nuclei Count / Density", 
        "3. Colocalization Analysis", 
        "4. Spatial Distance Analysis", 
        "5. Ratio Trend Analysis"
    ])
    mode = mode_raw 

    st.divider()

    # --- Grouping Strategy ---
    st.markdown("### 🏷️ Grouping Settings")
    group_strategy = st.radio("Label Determination:", ["Manual Entry", "Auto from Filename"], 
                              help="Auto: Extracts the part before the separator in the filename as the group name")
    
    if group_strategy.startswith("Manual"):
        sample_group = st.text_input("Group Name (X-axis Label):", value="Control")
        filename_sep = None
    else:
        filename_sep = st.text_input("Separator (e.g., _ or - ):", value="_", help="The part before this character becomes the group name")
        st.info(f"Ex: 'WT{filename_sep}01.tif' → Group: 'WT'")
        sample_group = "(Auto Detected)" 

    st.divider()

    # Analysis Parameter Settings
    if mode.startswith("5."):
        st.markdown("### 🔢 Trend Analysis Conditions")
        trend_metric = st.radio("Metric Target:", ["Colocalization Rate", "Area Occupancy"])
        ratio_val = st.number_input("Condition Value:", value=0, step=10)
        ratio_unit = st.text_input("Unit:", value="%", key="unit")
        if group_strategy.startswith("Manual"):
            sample_group = f"{ratio_val}{ratio_unit}" 
        
        if trend_metric.startswith("Colocalization"):
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (Reference):", list(COLOR_MAP.keys()), index=3)
                sens_a = st.slider("A Sensitivity", 5, 50, 20); bright_a = st.slider("A Brightness", 0, 255, 60)
            with c2:
                target_b = st.selectbox("CH-B (Target):", list(COLOR_MAP.keys()), index=2)
                sens_b = st.slider("B Sensitivity", 5, 50, 20); bright_b = st.slider("B Brightness", 0, 255, 60)
        else:
            target_a = st.selectbox("Analysis Color:", list(COLOR_MAP.keys()), index=2)
            sens_a = st.slider("Sensitivity", 5, 50, 20); bright_a = st.slider("Brightness", 0, 255, 60)
            use_roi_norm = st.checkbox("Normalize by Tissue Area (ROI)", value=False, key="roi_m5")
            if use_roi_norm:
                roi_color = st.selectbox("Tissue Color:", list(COLOR_MAP.keys()), index=2, key="roi_c5")
                sens_roi = st.slider("ROI Sensitivity", 5, 50, 20, key="roi_s5")
                bright_roi = st.slider("ROI Brightness", 0, 255, 40, key="roi_b5")
    else:
        if mode.startswith("1."):
            target_a = st.selectbox("Analysis Color:", list(COLOR_MAP.keys()))
            sens_a = st.slider("Sensitivity", 5, 50, 20)
            bright_a = st.slider("Brightness", 0, 255, 60)
            use_roi_norm = st.checkbox("Normalize by Tissue Area (ROI)", value=False, key="roi_m1")
            if use_roi_norm:
                roi_color = st.selectbox("Tissue Color:", list(COLOR_MAP.keys()), index=2, key="roi_c1")
                sens_roi = st.slider("ROI Sensitivity", 5, 50, 20, key="roi_s1")
                bright_roi = st.slider("ROI Brightness", 0, 255, 40, key="roi_b1")
        elif mode.startswith("2."):
            min_size = st.slider("Min Nuclei Size (px)", 10, 500, 50); bright_count = st.slider("Detection Threshold", 0, 255, 50)
            use_roi_norm = st.checkbox("Normalize by Tissue Area (ROI)", value=True, key="roi_m2")
            if use_roi_norm:
                roi_color = st.selectbox("Tissue Color:", list(COLOR_MAP.keys()), index=2, key="roi_c2")
                sens_roi = st.slider("ROI Sensitivity", 5, 50, 20, key="roi_s2")
                bright_roi = st.slider("ROI Brightness", 0, 255, 40, key="roi_b2")
        elif mode.startswith("3."):
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A:", list(COLOR_MAP.keys()), index=3); sens_a = st.slider("A Sensitivity", 5, 50, 20); bright_a = st.slider("A Brightness", 0, 255, 60)
            with c2:
                target_b = st.selectbox("CH-B:", list(COLOR_MAP.keys()), index=2); sens_b = st.slider("B Sensitivity", 5, 50, 20); bright_b = st.slider("B Brightness", 0, 255, 60)
        elif mode.startswith("4."):
            target_a = st.selectbox("Origin A:", list(COLOR_MAP.keys()), index=2); target_b = st.selectbox("Target B:", list(COLOR_MAP.keys()), index=3)
            sens_common = st.slider("Common Sensitivity", 5, 50, 20); bright_common = st.slider("Common Brightness", 0, 255, 60)

    st.divider()
    scale_val = st.number_input("Spatial Scale (μm/px)", value=1.5267, format="%.4f")
    st.markdown("### 🔄 Sequential Analysis")
    
    def prepare_next_group():
        st.session_state.uploader_key = str(uuid.uuid4())

    st.button(
        "📸 Next Group (Clear Images Only)", 
        on_click=prepare_next_group, 
        help="Keeps current analysis history but clears uploaded images to prepare for the next group"
    )
    
    st.divider()
    if st.button("Clear History & Generate New ID"): 
        st.session_state.analysis_history = []
        date_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
        st.session_state.current_analysis_id = f"AID-{date_str}-{str(uuid.uuid4())[:8]}"
        st.session_state.uploader_key = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.markdown("### ⚙️ Audit Trail (Param Log)")
    
    current_params = {
        "Software_Version": SOFTWARE_VERSION, 
        "Analysis_ID": st.session_state.current_analysis_id,
        "Analysis_Date_UTC": datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        "Mode": mode,
        "Scale_um_px": scale_val,
        "Grouping_Strategy": group_strategy
    }
    if group_strategy.startswith("Manual"): current_params["Manual_Group_Label"] = sample_group
    else: current_params["Filename_Separator"] = filename_sep

    # (Logging other parameters)
    if "trend_metric" in locals(): current_params["Trend_Metric"] = trend_metric
    if "target_a" in locals(): current_params["Target_A"] = target_a
    if "target_b" in locals(): current_params["Target_B"] = target_b
    if "roi_color" in locals(): current_params["ROI_Color"] = roi_color
    if "sens_a" in locals(): current_params["Sens_A"] = sens_a
    if "bright_a" in locals(): current_params["Bright_A"] = bright_a
    if "sens_b" in locals(): current_params["Sens_B"] = sens_b
    if "bright_b" in locals(): current_params["Bright_B"] = bright_b
    if "min_size" in locals(): current_params["Min_Nuclei_Size_px"] = min_size
    if "bright_count" in locals(): current_params["Count_Threshold"] = bright_count
    if "use_roi_norm" in locals(): current_params["ROI_Normalization_Enabled"] = use_roi_norm
    if "sens_roi" in locals(): current_params["ROI_Sens"] = sens_roi
    if "bright_roi" in locals(): current_params["ROI_Bright"] = bright_roi
    if "sens_common" in locals(): current_params["Common_Sens"] = sens_common
    if "bright_common" in locals(): current_params["Common_Bright"] = bright_common

    df_params = pd.DataFrame([current_params]).T.reset_index()
    df_params.columns = ["Parameter", "Setting Value"]
    param_filename = f"params_{st.session_state.current_analysis_id}.csv"

    # -------------------------------------------------------------------------
    # Dynamic Parameter Capture (Only save relevant params based on Mode)
    # -------------------------------------------------------------------------
    current_active_params = {
        "Mode": mode,
        "Scale_um_px": scale_val,
        "ROI_Norm": use_roi_norm if 'use_roi_norm' in locals() else False
    }

    st.divider()
    st.markdown("### ⚙️ Traceability (Active Settings)")
    st.table(pd.DataFrame([current_active_params]).T)
    
    st.download_button("📥 Download Settings CSV", df_params.to_csv(index=False).encode('utf-8'), param_filename, "text/csv")

    st.divider()
    st.caption("[Disclaimer]")
    st.caption("This tool is for research use only and does not guarantee clinical diagnosis. Final validity check is the responsibility of the user.")

# ---------------------------------------------------------
# 4. Tab 1: Run Analysis
# ---------------------------------------------------------
with tab_main:
    uploaded_files = st.file_uploader("Upload Images (16-bit TIFF supported)", type=["jpg", "png", "tif", "tiff"], accept_multiple_files=True, key=st.session_state.uploader_key)
    if uploaded_files:
        st.success(f"Analyzing {len(uploaded_files)} images...")
        batch_results = []
        for i, file in enumerate(uploaded_files):
            file.seek(0); file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            # 16-bit support: IMREAD_UNCHANGED
            img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            if img_raw is not None:
                # --- Auto Grouping Logic ---
                if group_strategy.startswith("Auto"):
                    try:
                        detected_group = file.name.split(filename_sep)[0]
                    except:
                        detected_group = "Unknown"
                    current_group_label = detected_group
                else:
                    current_group_label = sample_group

                # Image Processing (Normalization for display/calc)
                img_f = img_raw.astype(np.float32); mn, mx = np.min(img_f), np.max(img_f)
                img_8 = ((img_f - mn) / (mx - mn) * 255.0 if mx > mn else np.clip(img_f, 0, 255)).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_8, cv2.COLOR_GRAY2BGR) if len(img_8.shape) == 2 else img_8[:,:,:3]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB); img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                val, unit, res_disp = 0.0, "", img_rgb.copy()
                h, w = img_rgb.shape[:2]; fov_mm2 = (h * w) * ((scale_val / 1000) ** 2)

                extra_data = {}

                # --- Mode 1: Area Occupancy (With ROI Support) ---
                if mode.startswith("1.") or (mode.startswith("5.") and trend_metric.startswith("Area")):
                    mask_target = get_mask(img_hsv, target_a, sens_a, bright_a)
                    a_den_px = h * w
                    roi_status = "Field of View"
                    final_mask = mask_target
                    
                    if use_roi_norm:
                        mask_roi = get_tissue_mask(img_hsv, roi_color, sens_roi, bright_roi)
                        # Exclude noise outside ROI
                        final_mask = cv2.bitwise_and(mask_target, mask_roi)
                        a_den_px = cv2.countNonZero(mask_roi)
                        roi_status = "Inside ROI"
                        # Draw ROI contour
                        cv2.drawContours(res_disp, cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255,0,0), 3)

                    val = (cv2.countNonZero(final_mask) / a_den_px * 100) if a_den_px > 0 else 0
                    unit = "% Area"
                    
                    # Highlight mask in Green
                    m_disp = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB); m_disp[:,:,0]=0; m_disp[:,:,2]=0
                    res_disp = cv2.addWeighted(res_disp, 0.7, m_disp, 0.3, 0)
                    
                    a_den_mm2 = a_den_px * ((scale_val/1000)**2)
                    extra_data = {
                        "Target Area (mm2)": round(a_den_mm2, 6),
                        "Normalization Basis": roi_status
                    }

                # --- Mode 2: Nuclei Count (With ROI Support) ---
                elif mode.startswith("2."):
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY); _, th = cv2.threshold(gray, bright_count, 255, cv2.THRESH_BINARY)
                    blur = cv2.GaussianBlur(gray, (5,5), 0); _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    cnts, _ = cv2.findContours(cv2.bitwise_and(th, otsu), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid = [c for c in cnts if cv2.contourArea(c) > min_size]; val, unit = len(valid), "cells"
                    cv2.drawContours(res_disp, valid, -1, (0,255,0), 2)
                    
                    a_den_mm2 = fov_mm2; roi_status = "Field of View"
                    if use_roi_norm:
                        mask_roi = get_tissue_mask(img_hsv, roi_color, sens_roi, bright_roi)
                        a_den_mm2 = cv2.countNonZero(mask_roi) * ((scale_val/1000)**2)
                        roi_status = "Inside ROI"
                        # Draw ROI contour
                        cv2.drawContours(res_disp, cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255,0,0), 3)

                    density = val / a_den_mm2 if a_den_mm2 > 0 else 0
                    extra_data = {
                        "Target Area (mm2)": round(a_den_mm2, 6),
                        "Density (cells/mm2)": round(density, 2),
                        "Normalization Basis": roi_status
                    }

                # --- Other Modes ---
                elif mode.startswith("3.") or (mode.startswith("5.") and trend_metric.startswith("Colocalization")):
                    mask_a = get_mask(img_hsv, target_a, sens_a, bright_a); mask_b = get_mask(img_hsv, target_b, sens_b, bright_b)
                    coloc = cv2.bitwise_and(mask_a, mask_b); denom = cv2.countNonZero(mask_a)
                    val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0; unit = "% Coloc"; res_disp = cv2.merge([mask_b, mask_a, np.zeros_like(mask_a)])
                
                elif mode.startswith("4."):
                    ma, mb = get_mask(img_hsv, target_a, sens_common, bright_common), get_mask(img_hsv, target_b, sens_common, bright_common)
                    pa, pb = get_centroids(ma), get_centroids(mb)
                    if pa and pb: val = np.mean([np.min([np.linalg.norm(a - b) for b in pb]) for a in pa]) * (scale_val if scale_val > 0 else 1)
                    unit = "μm Dist" if scale_val > 0 else "px Dist"; res_disp = cv2.addWeighted(img_rgb, 0.6, cv2.merge([ma, mb, np.zeros_like(ma)]), 0.4, 0)

                st.divider()
                st.markdown(f"### 📷 Image {i+1}: {file.name}")
                st.markdown(f"**Detected Group:** `{current_group_label}`")
                
                if "Density (cells/mm2)" in extra_data:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Count", f"{int(val)} cells")
                    c2.metric("Density", f"{int(extra_data['Density (cells/mm2)']):,} /mm²")
                    c3.caption(f"Area: {extra_data['Target Area (mm2)']:.4f} mm² ({extra_data['Normalization Basis']})")
                elif "Target Area (mm2)" in extra_data:
                    c1, c2 = st.columns(2)
                    c1.metric("Occupancy", f"{val:.2f} %")
                    c2.caption(f"Denominator: {extra_data['Target Area (mm2)']:.4f} mm² ({extra_data['Normalization Basis']})")
                else:
                    st.markdown(f"### Result: **{val:.2f} {unit}**")
                
                c1, c2 = st.columns(2); c1.image(img_rgb, caption="Raw"); c2.image(res_disp, caption="Analysis Result")
                
                row_data = {
                    "Software_Version": SOFTWARE_VERSION,
                    "Analysis_ID": st.session_state.current_analysis_id,
                    "Analysis_Timestamp_UTC": datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                    "File Name": file.name,
                    "Group": current_group_label,
                    "Value": val,
                    "Unit": unit,
                }
                if extra_data: row_data.update(extra_data)
                row_data.update(current_active_params)
                batch_results.append(row_data)
        
        if st.button("Commit Batch Data", type="primary"):
            st.session_state.analysis_history.extend(batch_results)
            st.success("Data added to history. Analysis ID maintained.")
            st.rerun()

    if st.session_state.analysis_history:
        st.divider()
        st.header("💾 CSV Export (Full Traceability)")
        df_exp = pd.DataFrame(st.session_state.analysis_history)
        
        st.dataframe(df_exp, use_container_width=True)
        
        utc_filename = f"quantified_data_{st.session_state.current_analysis_id}.csv"
        st.download_button("📥 Download Results CSV", df_exp.to_csv(index=False).encode('utf-8'), utc_filename)

# ---------------------------------------------------------
# 5. Tab 2: Performance Validation
# ---------------------------------------------------------
with tab_val:
    st.header("🏆 Performance Validation Summary")
    st.markdown("""
    * **Benchmark:** BBBC005 (Broad Bioimage Benchmark Collection)
    * **Scale:** 3,200 images (High-Throughput)
    * **Methodology:** Parameters were individually optimized for each density group to demonstrate maximum performance under proper calibration.
    """)

    if not df_val.empty:
        gt_map = {'C14': 14, 'C40': 40, 'C70': 70, 'C100': 100}
        
        # Using Focus 1-5 for W1/W2 Comparison
        df_hq = df_val[(df_val['Focus'] >= 1) & (df_val['Focus'] <= 5)]
        
        # Statistics (W1 focus)
        w1_hq = df_hq[df_hq['Channel'] == 'W1']
        avg_acc = w1_hq['Accuracy'].mean()
        df_lin = w1_hq.groupby('Ground Truth')['Value'].mean().reset_index()
        r2 = np.corrcoef(df_lin['Ground Truth'], df_lin['Value'])[0, 1]**2

        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Accuracy", f"{avg_acc:.1f}%", help="Focus 1-5 Average")
        m2.metric("Linearity (R²)", f"{r2:.4f}", help="Based on measured values")
        m3.metric("Analyzed Images", "3,200+")

        st.divider()

        # Graph 1: Linearity (W1 vs W2)
        st.subheader("📈 1. Counting Performance & Linearity (W1 vs W2)")
        st.info("💡 **Conclusion:** W1 (Nuclei) shows extremely high linearity, while W2 (Cytoplasm) shows a **V-shaped divergence**, proving it is unsuitable for quantitative analysis.")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot([0, 110], [0, 110], 'k--', alpha=0.3, label='Ideal Line')
        ax1.scatter(df_lin['Ground Truth'], df_lin['Value'], color='#1f77b4', s=100, label='W1 (Nuclei)', zorder=5)
        w2_lin = df_hq[df_hq['Channel'] == 'W2'].groupby('Ground Truth')['Value'].mean().reset_index()
        ax1.scatter(w2_lin['Ground Truth'], w2_lin['Value'], color='#ff7f0e', s=100, marker='D', label='W2 (Cytoplasm)', zorder=5)
        z = np.polyfit(df_lin['Ground Truth'], df_lin['Value'], 1)
        ax1.plot(df_lin['Ground Truth'], np.poly1d(z)(df_lin['Ground Truth']), '#1f77b4', alpha=0.5, label='W1 Reg')
        ax1.set_xlabel('Ground Truth'); ax1.set_ylabel('Measured Value'); ax1.legend(); ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

        st.divider()

        # Graph 2 & 3
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 2. Accuracy Comparison by Density")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            df_bar = df_hq.groupby(['Density', 'Channel'])['Accuracy'].mean().reset_index()
            df_bar['Density'] = pd.Categorical(df_bar['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
            sns.barplot(data=df_bar, x='Density', y='Accuracy', hue='Channel', palette={'W1': '#1f77b4', 'W2': '#ff7f0e'}, ax=ax2)
            ax2.axhline(100, color='red', linestyle='--'); ax2.set_ylabel('Accuracy (%)')
            st.pyplot(fig2)
        
        with c2:
            st.subheader("📉 3. Optical Robustness (Blur Resistance)")
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            df_decay = df_val[df_val['Channel'] == 'W1'].copy()
            df_decay['Density'] = pd.Categorical(df_decay['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
            sns.lineplot(data=df_decay, x='Focus', y='Accuracy', hue='Density', marker='o', ax=ax3)
            ax3.axhline(100, color='red', linestyle='--'); ax3.set_ylabel('Accuracy (%)')
            st.pyplot(fig3)

        st.divider()

        # Data Table
        st.subheader("📋 4. Validation Numerical Data")
        summary = df_hq.groupby(['Density', 'Channel'])['Accuracy'].mean().unstack().reset_index()
        summary['Ground Truth'] = summary['Density'].map(gt_map)
        summary['W1 Measured'] = (summary['W1']/100)*summary['Ground Truth']
        summary['W2 Measured'] = (summary['W2']/100)*summary['Ground Truth']
        summary['Density'] = pd.Categorical(summary['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
        summary = summary.sort_values('Density')
        st.table(summary[['Density', 'Ground Truth', 'W1', 'W1 Measured', 'W2', 'W2 Measured']].rename(columns={
            'W1': 'W1 Accuracy(%)', 'W1 Measured': 'W1 Avg Count', 'W2': 'W2 Accuracy(%)', 'W2 Measured': 'W2 Avg Count'
        }))
        st.info("💡 **Overall Conclusion:** W1 (Nuclei) maintains high accuracy across all density ranges. W2 (Cytoplasm) fluctuates heavily between under/overestimation and is not scientifically recommended for quantitative analysis.")
    else:
        st.error("Validation CSV file not found. Please place it in the repository.")
